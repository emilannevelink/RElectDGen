import numpy as np

def reduce_traj(traj, keep = 'finite'):

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