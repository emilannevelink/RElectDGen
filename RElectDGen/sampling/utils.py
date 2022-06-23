import os
import numpy as np
import torch

from ase.io import read
from nequip.data import AtomicData

def sort_by_uncertainty(traj, embeddings, UQ, max_samples, min_uncertainty=0.04, max_uncertainty=np.inf):

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