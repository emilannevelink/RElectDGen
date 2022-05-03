import numpy as np
import torch

from nequip.data import AtomicData

def sort_by_uncertainty(traj, embeddings, UQ, max_samples, min_uncertainty=0.04):

    calc_inds = []
    uncertainties = []
    # embedding_distances = {}
    keep_embeddings = {}
    if isinstance(UQ.test_embeddings, dict):
        for key in UQ.MLP_config.get('chemical_symbol_to_type'): 
            keep_embeddings[key] = torch.empty((0,UQ.test_embeddings[key].shape[-1])).to(UQ.device)
    for i, (embedding_i, atoms) in enumerate(zip(embeddings,traj)):
        
        active_uncertainty = UQ.predict_uncertainty(atoms, embedding_i, extra_embeddings=keep_embeddings, type='std').detach().cpu().numpy()
        # data = UQ.transform(AtomicData.from_ase(atoms=atoms,r_max=UQ.r_max, self_interaction=UQ.self_interaction))
        # active_uncertainty = UQ.predict_uncertainty(data['atom_types'], embedding_i, extra_embeddings=keep_embeddings, type='std').detach().cpu().numpy()

        if np.any(active_uncertainty>min_uncertainty):
            calc_inds.append(i)
            uncertainties.append(float(active_uncertainty.max()))
            if isinstance(UQ.test_embeddings, dict):
                for key in UQ.MLP_config.get('chemical_symbol_to_type'): 
                    mask = np.array(atoms.get_chemical_symbols()) == key
                    keep_embeddings[key] = torch.cat([keep_embeddings[key],embedding_i[mask]])

    traj_sorted = []
    embedding_sorted = []
    ind_sorted = np.argsort(uncertainties)[:max_samples]
    for ind in ind_sorted:
        traj_sorted.append(traj[calc_inds[ind]])
        embedding_sorted.append(embeddings[calc_inds[ind]])

    return traj_sorted, embedding_sorted