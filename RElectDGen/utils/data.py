import numpy as np
from ase import neighborlist

def reduce_trajectory(traj,config, MLP_config):
    reduce_traj_type = config.get('reduce_traj_type','finite-isolated-free_H')
    combined_size = len(traj)
    if 'finite' in reduce_traj_type:
        traj_reduced = reduce_traj_finite(traj)
        print('Reduce finite ', len(traj), len(traj_reduced),flush=True)
    else:
        traj_reduced = traj

    if 'isolated' in reduce_traj_type:
        _, traj_reduced = reduce_traj_isolated(traj_reduced,MLP_config.get('r_max'))
        print('Reduce isolated ', len(traj), len(traj_reduced),flush=True)
    
    if 'free_H' in reduce_traj_type:
        _, traj_reduced = reduce_traj_free_H(traj_reduced)
        print('Reduce free Hydrogen ', len(traj), len(traj_reduced),flush=True)

    return traj_reduced


def reduce_traj_finite(traj, keep = 'finite'):

    if keep == 'finite':
        pbc_sum = 0
    else:
        pbc_sum = 3

    traj_reduced = []
    for atoms in traj:
        if sum(atoms.pbc) == pbc_sum:
            traj_reduced.append(atoms)
        elif sum(atoms.pbc) == 2:
            traj_reduced.append(atoms)
        elif sum(atoms.pbc) == 3 and np.all(atoms.get_atomic_numbers()==3):
            traj_reduced.append(atoms)

    return traj_reduced

def reduce_traj_isolated(traj, cutoff, test=False):
    if test:
        traj_removed = []
    ind_reduced = []
    traj_reduced = []
    for i, atoms in enumerate(traj):
        src, dst, d = neighborlist.neighbor_list('ijd',atoms,cutoff)
        unique, counts = np.unique(src, return_counts=True)
        if len(unique) == len(atoms):
            ind_reduced.append(i)
            traj_reduced.append(atoms)
        elif test:
            traj_removed.append(atoms)
        # elif sum(counts==1)>0:
        #     print('done')

    if test:
        return ind_reduced, traj_reduced, traj_removed
    else:
        return ind_reduced, traj_reduced

def reduce_traj_free_H(traj, H_cutoff=1.5, test=False):
    if test:
        traj_removed = []
    ind_reduced = []
    traj_reduced = []
    for i, atoms in enumerate(traj):
        src, dst, d = neighborlist.neighbor_list('ijd',atoms,H_cutoff)
        unique, counts = np.unique(src, return_counts=True)
        H_indices = np.where(np.array(atoms.get_chemical_symbols())=='H')[0]
        H_attached = np.intersect1d(unique,H_indices,return_indices=True)
        if len(H_attached) == len(H_indices):
            ind_reduced.append(i)
            traj_reduced.append(atoms)
        elif test:
            traj_removed.append(atoms)
        # elif sum(counts==1)>0:
        #     print('done')

    if test:
        return ind_reduced, traj_reduced, traj_removed
    else:
        return ind_reduced, traj_reduced
