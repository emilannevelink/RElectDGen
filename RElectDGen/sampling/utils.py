import os
import numpy as np
import torch

from ase.io import read
from nequip.data import AtomicData

def sort_by_uncertainty(traj, embeddings, UQ, max_samples, min_uncertainty=0.04, max_uncertainty=np.inf):

    if UQ.__class__.__name__ in ['Nequip_ensemble_NN']:
        uncertainties, calc_inds = embedding_downselect(traj, embeddings, UQ, min_uncertainty=min_uncertainty, max_uncertainty=max_uncertainty)
    else:
        uncertainties, calc_inds = uncertainty_downselect(traj, embeddings, UQ, min_uncertainty=min_uncertainty, max_uncertainty=max_uncertainty)

    traj_sorted = []
    uncertainties_sorted = []
    embedding_sorted = []
    ind_sorted = np.argsort(uncertainties)[::-1][:max_samples]
    for ind in ind_sorted:
        traj_sorted.append(traj[calc_inds[ind]])
        uncertainties_sorted.append(uncertainties[ind])
        embedding_sorted.append(embeddings[calc_inds[ind]])

    print(len(uncertainties_sorted),uncertainties_sorted)
    if len(uncertainties_sorted)>0:
        print(np.mean(uncertainties_sorted), np.std(uncertainties_sorted))

    return traj_sorted, embedding_sorted

def uncertainty_downselect(traj, embeddings, UQ, min_uncertainty=0.04, max_uncertainty=np.inf):
    calc_inds = []
    uncertainties = []
    # embedding_distances = {}
    keep_embeddings = {}
    if hasattr(UQ,'test_embeddings'):
        if isinstance(UQ.test_embeddings, dict):
            for key in UQ.MLP_config.get('chemical_symbol_to_type'): 
                keep_embeddings[key] = torch.empty((0,UQ.test_embeddings[key].shape[-1])).to(UQ.device)
    for i, (embedding_i, atoms) in enumerate(zip(embeddings,traj)):
        
        active_uncertainty = UQ.predict_uncertainty(atoms, embedding_i, extra_embeddings=keep_embeddings, type='std').detach().cpu().numpy()
        active_uncertainty = active_uncertainty.sum(axis=-1)
        # data = UQ.transform(AtomicData.from_ase(atoms=atoms,r_max=UQ.r_max, self_interaction=UQ.self_interaction))
        # active_uncertainty = UQ.predict_uncertainty(data['atom_types'], embedding_i, extra_embeddings=keep_embeddings, type='std').detach().cpu().numpy()

        if np.any(active_uncertainty>min_uncertainty) and np.all(active_uncertainty<max_uncertainty):
            calc_inds.append(i)
            uncertainties.append(float(active_uncertainty.max()))
            if hasattr(UQ,'test_embeddings'):
                if isinstance(UQ.test_embeddings, dict):
                    for key in UQ.MLP_config.get('chemical_symbol_to_type'): 
                        mask = np.array(atoms.get_chemical_symbols()) == key
                        keep_embeddings[key] = torch.cat([keep_embeddings[key],embedding_i[mask].to(UQ.device)])

    return uncertainties, calc_inds

def embedding_downselect(traj, embeddings, UQ, min_uncertainty=0.04, max_uncertainty=np.inf):
    calc_inds = []
    uncertainties = []
    embedding_distances = {}
    keep_embeddings = {}
    for key in UQ.MLP_config.get('chemical_symbol_to_type'): 
        keep_embeddings[key] = torch.empty((0,UQ.latent_size)).to(UQ.device)
        embed_distances = torch.cdist(UQ.train_embeddings[key],UQ.train_embeddings[key],p=2)
        mask = embed_distances!=0
        embedding_distances[key] = embed_distances[mask].mean()

    for i, (embedding_i, atoms) in enumerate(zip(embeddings,traj)):
        
        active_uncertainty = UQ.predict_uncertainty(atoms, embedding_i, extra_embeddings=keep_embeddings, type='std').detach().cpu().numpy()
        active_uncertainty = active_uncertainty.sum(axis=-1)

        if np.any(active_uncertainty>min_uncertainty) and np.all(active_uncertainty<max_uncertainty):
            add_atoms = False
            for key in UQ.MLP_config.get('chemical_symbol_to_type'): 
                mask = np.array(atoms.get_chemical_symbols()) == key
                if mask.sum()>0:
                    dataset_embeddings = torch.cat([
                        UQ.train_embeddings[key][:,:UQ.latent_size],
                        UQ.test_embeddings[key][:,:UQ.latent_size],
                        keep_embeddings[key]
                    ], dim=0)
                    embed_distances = torch.cdist(embedding_i[mask],dataset_embeddings,p=2)
                    add_atoms = add_atoms or embed_distances.max()>embedding_distances[key]

            if add_atoms:
                calc_inds.append(i)
                uncertainties.append(float(active_uncertainty.max()))
                for key in UQ.MLP_config.get('chemical_symbol_to_type'): 
                    mask = np.array(atoms.get_chemical_symbols()) == key
                    keep_embeddings[key] = torch.cat([keep_embeddings[key],embedding_i[mask]])
            
    return uncertainties, calc_inds


def sample_from_dataset(config):
    trajectory_file_name = os.path.join(
        config.get('data_directory'),
        config.get('trajectory_file')
    )
    if os.path.isfile(trajectory_file_name):
        traj = read(trajectory_file_name, index=':')
        max_samples = int(min([0.1*len(traj), config.get('max_samples')]))
        n_adversarial_samples = int(config.get('n_adversarial_samples',2*max_samples))
        
        traj_indices = torch.randperm(len(traj))[:2*n_adversarial_samples].numpy()
        traj_adv = [traj[i] for i in traj_indices]
    else:
        traj_adv = []

    return traj_adv