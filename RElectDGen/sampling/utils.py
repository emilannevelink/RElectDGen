import os
import numpy as np
import torch

from ase.io import read
from nequip.data import AtomicData

def sort_by_uncertainty(traj, embeddings, UQ, max_samples, min_uncertainty=0.04, max_uncertainty=np.inf):

    if UQ.__class__.__name__ in ['Nequip_ensemble_NN']:
        print('finetune downselect', flush = True)
        uncertainties, calc_inds = finetune_downselect(traj, embeddings, UQ, min_uncertainty=min_uncertainty, max_uncertainty=max_uncertainty)
    else:
        print('uncertainty downselect', flush = True)
        uncertainties, calc_inds = uncertainty_downselect(traj, embeddings, UQ, min_uncertainty=min_uncertainty, max_uncertainty=max_uncertainty)

    calc_inds_sorted = []
    traj_sorted = []
    uncertainties_sorted = []
    embedding_sorted = []
    ind_sorted = np.argsort(uncertainties)[::-1][:max_samples]
    for ind in ind_sorted:
        calc_inds_sorted.append(calc_inds[ind])
        traj_sorted.append(traj[calc_inds[ind]])
        uncertainties_sorted.append(uncertainties[ind])
        embedding_sorted.append(embeddings[calc_inds[ind]])

    print(calc_inds)
    print(calc_inds_sorted)
    print(len(uncertainties_sorted),uncertainties_sorted)
    if len(uncertainties_sorted)>0:
        print(np.mean(uncertainties_sorted), np.std(uncertainties_sorted))

    return traj_sorted, embedding_sorted, calc_inds_sorted

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
        # embedding_distances[key] = embed_distances[mask].mean()
        # set distance to the 90th percentile
        embedding_distances[key] = embed_distances[mask].sort().values[int(mask.sum()*0.5)] 

    print(embedding_distances)
    for i, (embedding_i, atoms) in enumerate(zip(embeddings,traj)):
        
        active_uncertainty = UQ.predict_uncertainty(atoms, embedding_i, extra_embeddings=keep_embeddings, type='std').detach().cpu().numpy()
        active_uncertainty = active_uncertainty.sum(axis=-1)

        if np.any(active_uncertainty>min_uncertainty) and np.all(active_uncertainty<max_uncertainty):
            add_atoms = False
            embed_distances_i = []
            if not embedding_i.device.type ==UQ.device:
                embedding_i = embedding_i.to(UQ.device)
            for key in UQ.MLP_config.get('chemical_symbol_to_type'): 
                mask = torch.tensor(np.array(atoms.get_chemical_symbols()) == key, device=UQ.device)
                if mask.sum()>0:
                    dataset_embeddings = torch.cat([
                        UQ.train_embeddings[key][:,:UQ.latent_size],
                        UQ.test_embeddings[key][:,:UQ.latent_size],
                        keep_embeddings[key]
                    ], dim=0)
                    embed_distances = torch.cdist(embedding_i[mask],dataset_embeddings,p=2)
                    atom_emb_dis = embed_distances.min(dim=1).values.max()
                    if len(calc_inds) == 0 and atom_emb_dis < embedding_distances[key]:
                        embedding_distances[key] = atom_emb_dis/2.
                    add_atoms = add_atoms or atom_emb_dis>embedding_distances[key]
                    embed_distances_i.append(atom_emb_dis)

                # print(key, embedding_distances[key])
                # print(embed_distances)
                # print(embedding_i[mask], flush=True)

            if add_atoms:
                print('embedding distance')
                print(embedding_distances)
                print(embed_distances_i, flush=True)
                calc_inds.append(i)
                uncertainties.append(float(active_uncertainty.max()))
                for key in UQ.MLP_config.get('chemical_symbol_to_type'): 
                    mask = torch.tensor(np.array(atoms.get_chemical_symbols()) == key, device=UQ.device)
                    keep_embeddings[key] = torch.cat([keep_embeddings[key],embedding_i[mask]])
            else:
                print(f'Embedding of {i} was too close')
                print(embedding_distances)
                print(embed_distances_i, flush=True)
            
    return uncertainties, calc_inds

def finetune_downselect(traj, embeddings, UQ, min_uncertainty=0.04, max_uncertainty=np.inf):
    calc_inds = []
    uncertainties = []
    embedding_distances = {}
    keep_embeddings = {}
    keep_energies = {}
    for key in UQ.MLP_config.get('chemical_symbol_to_type'): 
        keep_embeddings[key] = torch.empty((0,UQ.latent_size+UQ.natoms)).to(UQ.device)
        keep_energies[key] = torch.empty(0, device=UQ.device)

    for i, (embedding_i, atoms) in enumerate(zip(embeddings,traj)):
        
        active_uncertainty = UQ.predict_uncertainty(atoms, embedding_i, extra_embeddings=keep_embeddings, type='std').detach().cpu().numpy()
        active_uncertainty = active_uncertainty.sum(axis=-1)

        if np.any(active_uncertainty>min_uncertainty) and np.all(active_uncertainty<max_uncertainty):
            if not embedding_i.device.type ==UQ.device:
                embedding_i = embedding_i.to(UQ.device)
            
            calc_inds.append(i)
            uncertainties.append(float(active_uncertainty.max()))
            
            data = UQ.transform_data_input(atoms)
            out = UQ.model(data)
            atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze(),num_classes=UQ.natoms).to(UQ.device)
            for key in UQ.MLP_config.get('chemical_symbol_to_type'): 
                mask = torch.tensor(np.array(atoms.get_chemical_symbols()) == key, device=UQ.device)

                NN_inputs = torch.hstack([embedding_i[mask], atom_one_hot[mask]])
                keep_embeddings[key] = torch.cat([keep_embeddings[key],NN_inputs])
                uncertainty_training = UQ.config.get('uncertainty_training','energy')
                if uncertainty_training=='energy':
                    keep_energies[key] = torch.cat([keep_energies[key], out['atomic_energy'].detach()[mask]])
                elif uncertainty_training=='forces':
                    keep_energies[key] = torch.cat([keep_energies[key], out['forces'].detach()[mask].norm(dim=1).unsqueeze(1)])

            print(i, flush=True)
            UQ.fine_tune(keep_embeddings, keep_energies)
            active_uncertainty_after = UQ.predict_uncertainty(atoms, embedding_i, extra_embeddings=keep_embeddings, type='std').detach().cpu().numpy()
            active_uncertainty_after = active_uncertainty_after.sum(axis=-1)
            ind_max = np.argmax(active_uncertainty)
            print(active_uncertainty[ind_max], active_uncertainty_after[ind_max], flush=True)
            print(active_uncertainty[ind_max] > active_uncertainty_after[ind_max], flush=True)
            
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
        traj_md = [traj[i] for i in traj_indices]
    else:
        traj_md = []

    adversarial_trajectroy = os.path.join(
        config.get('data_directory'),
        config.get('adversarial_trajectroy','')
    )
    if os.path.isfile(adversarial_trajectroy):
        try:
            traj = read(adversarial_trajectroy, index=':')
            max_samples = int(min([0.1*len(traj), config.get('max_samples')]))
            n_adversarial_samples = int(config.get('n_adversarial_samples',2*max_samples))
            
            traj_indices = torch.randperm(len(traj))[:2*n_adversarial_samples].numpy()
            traj_adv = [traj[i] for i in traj_indices]
        except Exception as e:
            print(e)
            traj_adv = []
    else:
        traj_adv = []

    traj = traj_md + traj_adv
    return traj