import copy
import os
import numpy as np

from ase.io import read
from ase.parallel import world

from RElectDGen.structure.build import get_initial_structure

def reassign_cluster_charges(atoms, clusters, FragmentDB, run_dir = ''):
    nclusters = clusters.n_components
    initial_charges = copy.copy(atoms.get_initial_charges())
    initial_supercell_charge = initial_charges.sum()
    
    
    if FragmentDB is not None:
        fragment_db = FragmentDB.fragment_db 
        
        cluster_fragments = []
        for cluster_ind in range(nclusters):
            cluster = atoms[clusters.component_list==cluster_ind]

            if not FragmentDB.is_valid_cluster(cluster): # most likely signifies it comes from a larger molecule
                
                cluster_fragments.append([cluster,cluster_ind])
                
        # replace charge on atoms
        n_unknown = 0
        for i, (cluster_i, ind_i) in enumerate(cluster_fragments):
            db_ind = FragmentDB.is_valid_fragment(cluster_i,raw=True)

            #look-up fragment id in fragment database
            if db_ind.sum()==1:
                fragment_name = fragment_db['fragment_name'][db_ind].values[0]
                print(fragment_name)
                fragment_filename = os.path.join(FragmentDB.fragment_dir,f'fragment_{fragment_name}.json')
                cluster_fragment = read(fragment_filename)

                indices = clusters.component_list==ind_i

                atoms.arrays['initial_charges'][indices] = cluster_fragment.get_initial_charges()
            elif db_ind.sum() == 0:
                n_unknown += 1
                unknown_fragment_file = os.path.join(run_dir,'unknown_fragments.txt')
                f = open(unknown_fragment_file,'a')
                f.write(str(cluster_i)+str(cluster_i.get_initial_charges())+str(atoms)+'\n')
                f.close()
            else:
                print('Fragment DB retrieves too many results ', db_ind.sum())
                print(fragment_db['fragment_name'][db_ind])

        final_charges = atoms.get_initial_charges()
        final_supercell_charge = final_charges.sum()
        
        if not np.isclose(initial_supercell_charge,final_supercell_charge,atol=1e-3):
            print('Reassign charges changed the total supercell charge, changing back')
            print('Number of unknown clusters: ', n_unknown)
            atoms.set_initial_charges(initial_charges)
        
    else:
        print('Fragment DB is none')

    return atoms

def extend_z(atoms, config):
    if atoms.get_pbc()[2] or sum(atoms.get_pbc())!=2:
        return atoms

    atoms_copy = copy.deepcopy(atoms)
    vacuum = config.get('vacuum')
    atoms_copy.center(vacuum = vacuum,axis=2)

    z_tolerance = config.get('z_tolerance',0)

    initial_structure = get_initial_structure(config)

    z_difference_0 = atoms_copy.get_cell()[2,2] - initial_structure.get_cell()[2,2]
    z_difference_rel = atoms_copy.get_cell()[2,2] - atoms.get_cell()[2,2]
    if world.rank == 0:
        print(z_difference_0,flush=True)
    if z_difference_0 > 0 and z_difference_0 < z_tolerance:
        if world.rank == 0:
            print(f'Changed supercell z dimension by {z_difference_rel}', flush=True)
        return atoms_copy
    elif z_difference_0 < 0:
        atoms_copy.set_cell(initial_structure.get_cell())
        if world.rank == 0:
            print('Reset cell to initial cell')
        return atoms_copy
    elif z_difference_0 > z_tolerance:
        cell = initial_structure.get_cell()
        cell[2,2] += z_tolerance
        atoms_copy.set_cell(cell)
        if world.rank == 0:
            print('Set cell to max')
        return atoms_copy
    else:
        return atoms
    