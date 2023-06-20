import torch
import numpy as np


from RElectDGen.uncertainty.models import uncertainty_base


def subsample_uncertain(UQ,traj_uncertain,minimum_uncertainty_cutoffs,maximim_uncertainty_cutoffs,max_add,method=None):

    if method is None:
        if 'distance' in UQ.__class__.__name__:
            method = 'finetune'
        else:
            method = 'statistics'
    
    if method == 'random':
        traj_add = random_subsample(traj_uncertain,max_add)
    elif method == 'statistics':
        traj_add = subsample_statistics(UQ,traj_uncertain,minimum_uncertainty_cutoffs,maximim_uncertainty_cutoffs,max_add)
    elif method == 'finetune':
        traj_add = finetune_subsample(UQ,traj_uncertain,minimum_uncertainty_cutoffs,maximim_uncertainty_cutoffs,max_add)
    else:
        raise ValueError(f'Method {method} is not defined')

    return traj_add

def random_subsample(traj_uncertain,max_add):
    add_indices = np.random.permutation(len(traj_uncertain))[:max_add]
    traj_add = [traj_uncertain[ind] for ind in add_indices]
    return traj_add

def subsample_statistics(
    UQ: uncertainty_base,
    traj_uncertain: list,
    minimum_uncertainty_cutoffs: dict,
    maximum_uncertainty_cutoffs: dict,
    max_add: int,
):
    
    calc_inds = []
    uncertainties = []

    type_to_chemical_symbol = {}
    for (key,val) in UQ.chemical_symbol_to_type.items():
        type_to_chemical_symbol[val] = key
    
    for i, atoms in enumerate(traj_uncertain):
        if len(calc_inds) >= max_add:
            break

        out = UQ.predict_uncertainty(atoms)
        uncertainty = out['uncertainties'].detach().cpu().numpy().sum(axis=-1)
        
        max_ind = np.argmax(uncertainty)
        atom_type = int(out['atom_types'][max_ind])
        symbol = type_to_chemical_symbol[atom_type]
        unc_value = uncertainty[max_ind]
        if unc_value>minimum_uncertainty_cutoffs[symbol] and unc_value<maximum_uncertainty_cutoffs[symbol]:
            calc_inds.append(int(i))
            uncertainties.append(unc_value)
        
    
    print(uncertainties)
    sorted_ind = np.argsort(uncertainties)[::-1]
    calc_inds = np.array(calc_inds)[sorted_ind][:max_add]
    traj_add = [traj_uncertain[ind] for ind in calc_inds]
    return traj_add

def finetune_subsample(
    UQ: uncertainty_base,
    traj_uncertain: list,
    minimum_uncertainty_cutoffs: dict,
    maximum_uncertainty_cutoffs: dict,
    max_add: int,
):
    calc_inds = []
    uncertainties = []
    keep_embeddings = {}
    for key in UQ.MLP_config.get('chemical_symbol_to_type'): 
        keep_embeddings[key] = torch.empty((0,UQ.latent_size)).to(UQ.device)

    type_to_chemical_symbol = {}
    for (key,val) in UQ.chemical_symbol_to_type.items():
        type_to_chemical_symbol[val] = key

    ndiff = 0
    for i, atoms in enumerate(traj_uncertain):
        if len(calc_inds) >= max_add+ndiff:
            break
        
        out = UQ.predict_uncertainty(atoms,extra_embeddings=keep_embeddings)
        uncertainty = out['uncertainties'].detach().cpu().numpy().sum(axis=-1)
        append_embedding = False

        max_ind = np.argmax(uncertainty)
        atom_type = int(out['atom_types'][max_ind])
        symbol = type_to_chemical_symbol[atom_type]
        unc_value = uncertainty[max_ind]


        if unc_value>minimum_uncertainty_cutoffs[symbol] and unc_value<maximum_uncertainty_cutoffs[symbol]:
            calc_inds.append(int(i))
            uncertainties.append(unc_value)
            append_embedding = True
            
            if not np.isclose(unc_value,atoms.info['uncertainties'][max_ind]):
                ndiff += 0.75
        
        if append_embedding:
            for key in UQ.MLP_config.get('chemical_symbol_to_type'): 
                embedding_i = UQ.atom_embedding.detach().to(UQ.device)
                # print(data['atom_types'])
                # print(embedding_i)
                mask = (out['atom_types'] == UQ.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                # print(keep_embeddings[key])
                # print(embedding_i[mask])
                keep_embeddings[key] = torch.cat([keep_embeddings[key],embedding_i[mask]])
    
    print(f'Sampled up to {i} of Uncertain trajectory: ',len(traj_uncertain))
    print(uncertainties)
    sorted_ind = np.argsort(uncertainties)[::-1]
    calc_inds = np.array(calc_inds)[sorted_ind][:max_add]
    uncertainties = np.array(uncertainties)[sorted_ind][:max_add]
    print(uncertainties)
    print(calc_inds)
    traj_add = [traj_uncertain[ind] for ind in calc_inds]
    return traj_add
    