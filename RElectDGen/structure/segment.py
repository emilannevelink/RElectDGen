import time
import psutil
import os, gc

import ase
from ase.io import read, write

import pandas as pd
import numpy as np

import uuid
from ase import neighborlist
from scipy import sparse
from ase import Atoms
import multiprocessing as mp
import copy

from .build import create_slab

def findclusters(
    atoms: ase.Atoms,
    cutoff = None,
):

    if cutoff is None:
        cutoff = 1.1*neighborlist.natural_cutoffs(atoms)
    elif isinstance(cutoff,float) or isinstance(cutoff,int):
        cutoff = [cutoff for _ in atoms]
    elif isinstance(cutoff,list):
        assert len(cutoff) == len(atoms)


    nblist = neighborlist.NeighborList(cutoff,skin=0,self_interaction=False,bothways=True)
    nblist.update(atoms)

    matrix = nblist.get_connectivity_matrix()

    n_components, component_list = sparse.csgraph.connected_components(matrix)

    return n_components, component_list

def clusterstocalculate(
    atoms: ase.Atoms,
    uncertain_indices: list,
    n_components: int,
    component_list: list,
    max_cluster_size: int = 40,
    vacuum_size: float = 2.0
):

    if len(atoms)<=max_cluster_size:
        if len(uncertain_indices)>0:
            atoms.wrap()
            return [[atoms], [0]]
        else:
            return [[],[]]
    
    n_components_natural, component_list_natural = findclusters(atoms)

    molecules = []
    clusters = []
    atom_indices = []
    for idx in uncertain_indices:
        if idx < len(atoms):
            molIdx = component_list[idx]
            if molIdx not in molecules:
                molecule = atoms[[ i for i in range(len(component_list)) if component_list[i] == molIdx ]]
                atoms1 = molecule.copy()
                arg = np.argwhere(atoms1.positions.std(axis=0)>atoms1.cell.diagonal()/4)
                if arg.shape[0]>0:
                    arg = arg[0]
                    atoms1.positions[:,arg]+=atoms1.cell.diagonal()[arg]/2
                    atoms1.wrap()
                atoms1.center(vacuum=vacuum_size)
                if np.any(molecule.cell<atoms1.cell):
                    molecule.wrap()
                else:
                    molecule = atoms1
                del atoms1
                # molecule.center()#vacuum=vacuum_size)
                if len(molecule)>max_cluster_size:
                    print(f'Cluster {molIdx} has {len(molecule)} atoms greater than max cluster size of {max_cluster_size}',flush=True)

                    molIdx_natural = component_list_natural[idx]

                else:
                    clusters.append(molecule)
                    atom_indices.append(idx)
                molecules.append(molIdx)

    return clusters, atom_indices


class segment_atoms():
    def __init__(self, 
        atoms: ase.Atoms, 
        cutoff = None,
        config = None) -> None:
        
        self.atoms = atoms
        self.cutoff = cutoff
        self.config = config

        self.segment_type = config.get('segment_type','uncertain')
        self.n_components_natural, self.component_list_natural, self.matrix_natural = self.findclusters()

        self.n_components, self.component_list, self.matrix = self.findclusters(cutoff=self.cutoff)
        

    def findclusters(self,
        cutoff = None,
    ):

        atoms = self.atoms#.copy()
        if cutoff is None:
            cutoff = np.array(neighborlist.natural_cutoffs(atoms))*1.2
        elif isinstance(cutoff,float) or isinstance(cutoff,int):
            cutoff = [cutoff for _ in atoms]
        elif isinstance(cutoff,list):
            assert len(cutoff) == len(atoms)


        nblist = neighborlist.NeighborList(cutoff,skin=0,self_interaction=False,bothways=True)
        nblist.update(atoms)

        matrix = nblist.get_connectivity_matrix()

        n_components, component_list = sparse.csgraph.connected_components(matrix)

        return n_components, component_list, matrix

    def clusterstocalculate(self,
        uncertain_indices: list,
        min_cluster_size: int = 20,
        max_cluster_size: int = 40,
        vacuum_size: float = 2.0,
    ):

        max_volume_per_atom = self.config.get('max_volume_per_atom',150)
        if len(self.atoms)<=max_cluster_size:
            if len(uncertain_indices)>0:
                self.atoms.wrap()
                atoms = Atoms(self.atoms)
                atoms.calc = None
                return [[atoms], [0]]
            else:
                return [[],[]]

        
        molecules = []
        clusters = []
        atom_indices = []
        for idx in uncertain_indices:
            molIdx = self.component_list_natural[idx]

            if molIdx not in molecules:
                indices_add = [ i for i in range(len(self.component_list_natural)) if self.component_list_natural[i] == molIdx ]
                slab_add, mixture_add = self.segment_slab_mixture(indices_add)
                

                build=True
                add_slab = False
                add_bulk = False
                build_ind = 0
                if idx in mixture_add:
                    cluster_indices = list(mixture_add)
                    slab_indices = []
                    if len(slab_add)>0 and (len(cluster_indices)) <= max_cluster_size / 2:
                        add_slab = True
                        slab_indices += list(slab_add)
                elif idx in slab_add:
                    print('check segment bulk', flush = True)
                    cluster, cluster_indices = self.segment_bulk(slab_add,idx)
                    build = False
                    add_bulk = True
                
                lithium_ind = np.argwhere(self.atoms[cluster_indices].get_atomic_numbers()==3).flatten()
                if len(lithium_ind)>500:
                    print('wrong')

                while build: #len(cluster_indices) < min_cluster_size and build:
                    if build_ind>100:
                        print(build_ind,flush=True)
                    build_ind+=1
                    total_indices = cluster_indices + slab_indices
                    neighbor_atoms = np.unique(np.concatenate([self.matrix.getcol(i).nonzero()[0] for i in cluster_indices]))
                    neighbor_atoms = [ind for ind in neighbor_atoms if ind not in total_indices]

                    if len(neighbor_atoms) == 0:
                        build = False
                    else:

                        if self.segment_type == 'uncertain':
                            # get neighbor atom with the lowest index in uncertain indices
                            neigh_uncertain = [np.argwhere(neigh_ind == np.array(uncertain_indices))[0,0] for neigh_ind in neighbor_atoms if neigh_ind in uncertain_indices]
                            if len(neigh_uncertain)>0:
                                atom_ind = neighbor_atoms[np.argmin(neigh_uncertain)]
                                molIdx_add = self.component_list_natural[atom_ind]
                            
                                # add natural molecule that atom is in to cluster indices                             
                                indices_add = [ i for i in range(len(self.component_list_natural)) if self.component_list_natural[i] == molIdx_add ]
                            else:
                                indices_add = []
                        
                        
                        if len(indices_add)>0:
                            slab_add, mixture_add = self.segment_slab_mixture(indices_add)
                            
                            lithium_ind = np.argwhere(self.atoms[mixture_add].get_atomic_numbers()==3).flatten()
                            if len(lithium_ind)>500:
                                print('wrong')

                            if (len(cluster_indices) + len(mixture_add)) > max_cluster_size / (2  - (not add_slab)):
                                build = False
                            elif atom_ind in mixture_add:
                                cluster_indices += list(mixture_add)
                                molecules.append(molIdx_add)

                                if len(slab_add)>0 and (len(cluster_indices)) <= max_cluster_size / 2:
                                    add_slab = True
                                    slab_indices += list(slab_add)
                            elif atom_ind in slab_add:
                                if len(cluster_indices)==0:
                                    print('check segment bulk', flush = True)
                                    cluster, cluster_indices = self.segment_bulk(slab_add,atom_ind)
                                    build = False
                                    add_bulk = True
                                elif (len(cluster_indices)) <= max_cluster_size / 2:
                                    add_slab = True
                                    slab_indices += list(slab_add)
                                    molecules.append(molIdx_add)
                                else:
                                    build = False

                            lithium_ind = np.argwhere(self.atoms[cluster_indices].get_atomic_numbers()==3).flatten()
                            if len(lithium_ind)>500:
                                print('wrong')

                        else:
                            build = False
                        
                if not add_bulk:           
                    cluster = self.atoms[cluster_indices]

                if add_slab and len(slab_indices)>0:
                    cluster, cluster_indices = self.segment_slab(cluster, cluster_indices, slab_indices)
                    # write('test_cluster.xyz',cluster)
                    # print('done')
                elif add_bulk:
                    cluster.pbc = True
                else:
                    cluster = self.reduce_mixture_size(cluster,vacuum_size)
                    cluster.pbc = True

                lithium_ind = np.argwhere(cluster.get_atomic_numbers()==3).flatten()
                if len(lithium_ind)>500:
                    print('wrong, too many lithium atoms',flush=True)
                else:

                    # try:
                    #     data = AtomicData.from_ase(atoms=cluster, r_max=4)
                    #     save = True
                    # except Exception as e:
                    #     print(e)
                    #     save = False

                    if len(cluster)>1 and cluster.get_volume()/len(cluster)<max_volume_per_atom:
                        cluster.arrays['cluster_indices'] = np.array(cluster_indices,dtype=int)
                        clusters.append(cluster)
                        atom_indices.append(idx)
                        molecules.append(molIdx)
                    else:
                        if len(cluster)==1:
                            print(f'wrong, not enough atoms around {idx}',flush=True)
                        elif cluster.get_volume()/len(cluster)>max_volume_per_atom:
                            print(f'wrong, cluster volume too large volume/atom: {cluster.get_volume()/len(cluster)}',flush=True)
                        # elif not save:
                        #     print(f'problem with AtomicData {idx}',flush=True)
                        #     save = True
            if len(clusters)>=self.config.get('max_clusters',10):
                break

        return clusters, atom_indices

    def reduce_mixture_size(self,cluster,vacuum_size):

        atoms1 = cluster.copy()
        divisor = 2
        # arg = np.argwhere(atoms1.positions.std(axis=0)>cluster.cell.diagonal()/5)
        
        bools = self.axis_crossing_boundary(atoms1)
        while bools.sum()>0:
            atoms1.positions[:,bools]+=atoms1.cell.diagonal()[bools]/divisor
            atoms1.wrap()
            bools = self.axis_crossing_boundary(atoms1)
            divisor*=2
            if divisor >=8:
                break

        # while arg.shape[0]>0:
        #     arg = arg[0]
        #     atoms1.positions[:,arg]+=atoms1.cell.diagonal()[arg]/divisor
        #     atoms1.wrap()
        #     arg = np.argwhere(atoms1.positions.std(axis=0)>cluster.cell.diagonal()/5)
        #     divisor*=2
        #     if divisor >=8:
        #         break
        
        atoms2 = atoms1.copy()
        atoms1.center(vacuum=vacuum_size)
        shrink_ind = np.argwhere(np.any(cluster.cell>atoms1.cell,axis=1)).flatten()
        # atoms1.center(vacuum=vacuum_size,axis=shrink_ind)
        
        if len(shrink_ind)==0:
            cluster.wrap()
        elif len(shrink_ind)==3:
            cluster = atoms1
        else:
            # adjust_inds = [i for i in [0,1,2] if i not in shrink_ind]
            # atoms1.cell[adjust_inds] = cluster.cell[adjust_inds]
            # atoms1.center(vacuum=vacuum_size,axis=shrink_ind)
            atoms2.center(vacuum_size,axis=shrink_ind)
            cluster = atoms2
        
        del atoms1, atoms2
        return cluster

    def axis_crossing_boundary(self,atoms):
        bools = [True, True, True]
        bools = []
        for ind in range(len(atoms)):
            Di = atoms.get_distances(np.arange(len(atoms)),ind, mic=True,vector=True)
            Df = atoms.get_distances(np.arange(len(atoms)),ind, mic=False,vector=True)
            bools.append(np.all(np.isclose(Di,Df),axis=0))

        bools = np.array(bools).sum(axis=0)>len(atoms)*0.6
        # np.any([np.isclose(Di,Df),np.isclose(Di+Df,atoms.cell.diagonal())])
        # bools = np.all([bools, np.all(np.isclose(Di,Df),axis=0)],axis=0)

        bools = np.invert(bools)
        return bools

    def segment_slab_mixture(self, cluster_indices):

        cluster = self.atoms[cluster_indices]

        lithium_ind = np.argwhere(cluster.get_atomic_numbers()==3).flatten()
        if len(lithium_ind)>5 and self.config.get('slab_config') is not None:

            lithium_cluster = cluster[lithium_ind]

            src, dst, d = neighborlist.neighbor_list('ijd',lithium_cluster,4)


            lattice_site_tolerance = 0.1
            #BCC nearest neighbor atoms
            ind_BCC_nn = np.argwhere(np.isclose(d,2.967,atol=lattice_site_tolerance)).flatten()

            #BCC second nearest neighbor atoms
            ind_BCC_2nn = np.argwhere(np.isclose(d,3.426,atol=lattice_site_tolerance)).flatten()

            #todo add FCC nearest neighbor distance

            #determine which lithiums are part of slab and mixture
            slab_ind = lithium_ind[np.unique(np.concatenate([src[ind_BCC_nn],src[ind_BCC_2nn]]))]
            mixture_ind = np.delete(np.arange(len(cluster)),slab_ind)

            slab_indices = np.array(cluster_indices)[slab_ind]
            mixture_indices = np.array(cluster_indices)[mixture_ind]

        else:
            slab_indices = np.array(cluster_indices)[[]]
            mixture_indices = np.array(cluster_indices)
        
        return slab_indices, mixture_indices
            
    
    def segment_slab(self, cluster, cluster_indices, slab_indices):

        cluster_reduce = self.reduce_mixture_size(cluster,1)

        if np.any(cluster_reduce.cell<cluster.cell):
            reduce_centerofmass = cluster_reduce.get_positions().mean(axis=0)
            centerofmass = (reduce_centerofmass - cluster_reduce[0].position) + cluster[0].position
            centerofmass = centerofmass%cluster.cell.diagonal()
        else:
            centerofmass = cluster.get_center_of_mass()
        
        xy_centerofmass = np.concatenate([centerofmass[:2],[0]])
        total_indices = cluster_indices + slab_indices
        slab_cluster = self.atoms[total_indices]
        slab_cluster.center(about=xy_centerofmass)
        slab = self.atoms[slab_indices]

        slab_seed = np.argmin(np.linalg.norm(slab.positions-centerofmass,axis=1))

        D = slab.get_distances(np.arange(len(slab)),slab_seed, mic=True,vector=True)
        
        pure_slab = create_slab(self.config.get('slab_config'))
        cell = pure_slab.cell.diagonal()

        
        for i, n_planes in enumerate(self.config.get('slab_config')['supercell_size']):
            hist, bin_edges = np.histogram(D[:,i],bins=self.config.get('supercell_size')[i]*10)
            
            bin_indices, bin_centers = self.coarsen_histogram(D[:,i],hist, bin_edges)

            n_basis = max([1,int(np.round(len(bin_indices)/self.config.get('supercell_size')[i],0))])
            n_planes *= n_basis
            
            bin_seed_ind = np.argsort(np.abs(bin_centers))[:n_planes]

            min_bin_indices = np.concatenate(np.array(bin_indices,dtype=object)[bin_seed_ind]).astype(int)
            if i == 0:
                keep_indices = min_bin_indices
            else:
                keep_indices = np.intersect1d(keep_indices,min_bin_indices)
        
        slab_keep = slab[keep_indices]
        
        reduced_slab_cluster = slab_keep + self.atoms[cluster_indices]

        cell_new = pure_slab.get_cell()
        cell_new[2,2] += 2*self.config.get('vacuum')
        reduced_slab_cluster.set_cell(cell_new)
        reduced_slab_cluster.center()

        #remove overlapping atoms
        overlap = self.config.get('overlap_radius',0.5)
        edge_vec = []

        keep_indices = list(keep_indices)
        while len(edge_vec)>0:
            delete_indices = []
            src, dst, edge_vec = neighborlist.neighbor_list('ijD',reduced_slab_cluster,overlap)
            if len(edge_vec)>0:
                print('Check that deleting overlaping atoms is working properly', flush=True)
            for i, (src_i, dst_i, vec) in enumerate(zip(src,dst,edge_vec)):
                if dst_i in delete_indices:
                    break
                reduced_slab_cluster.positions[dst_i] -= vec/2
                delete_indices.append(src_i)
            
            [keep_indices.pop(ii) for ii in delete_indices]
            del reduced_slab_cluster[delete_indices]
        
        reduced_slab_cluster.wrap()
        reduced_slab_cluster.pbc = [True,True,False]
        return reduced_slab_cluster, keep_indices + cluster_indices

        # src, dst, edge_vec = neighborlist.neighbor_list('ijD',slab,np.linalg.norm(pure_slab.get_cell().diagonal()))

        # # determine how to reduce configuration...
        # # start with mixture atoms
        # print('not done')
        # slab_indices = np.array(cluster_indices)[slab_ind]
        # mixture_indices = np.array(cluster_indices)[mixture_ind]
        # neighbor_atoms = np.unique(np.concatenate([self.matrix.getcol(i).nonzero()[0] for i in mixture_indices]))
        
        # slab_seed = np.intersect1d(slab_indices,neighbor_atoms)
        # slab_nuclei = slab_seed
        # for i in range(3):
        #     neighbor_slab = np.unique(np.concatenate([self.matrix.getcol(i).nonzero()[0] for i in slab_nuclei]))
        #     mask =  self.atoms[neighbor_slab].get_atomic_numbers()==3
        #     slab_nuclei = np.concatenate([slab_nuclei,neighbor_slab[mask]])

        # lithium_slab = self.atoms[slab_nuclei]
        # # lithium_slab.center(about=np.array(self.atoms[slab_seed].positions.mean(axis=0)))
        # # lithium_slab.center(about=np.array([0,0,self.atoms[slab_seed].positions.mean(axis=0)[2]]))
        # center = self.atoms[slab_seed].positions.mean(axis=0)-lithium_slab.get_center_of_mass()
        # lithium_slab.center(about=center)
        # lithium_slab.wrap()
        # lithium_slab.write('test_lithium_slab.xyz')

        # pure_slab = create_slab(self.config.get('slab_config'))

        # pure_slab.write('test_pure_slab.xyz')

        # #match_seed
        # ind_location = np.array([*pure_slab.get_cell().diagonal()[:2]/2,pure_slab.get_cell().diagonal()[-2]])
        # ind_seed = np.argmin(np.linalg.norm(pure_slab.positions-ind_location,axis=1))

        # #match_bottom
        # ind_location = np.array([*pure_slab.get_cell().diagonal()[:2]/2,0])
        # ind_bottom = np.argmin(np.linalg.norm(pure_slab.positions-ind_location,axis=1))

    def coarsen_histogram(self, d, hist, bin_edges):

        #find the proper bin edges
        idx = np.where(hist!=0)[0]
        indices = np.where(np.diff(idx)!=1)[0]
        coarse_edges = []
        bin_edgesi = [bin_edges[0]-0.01]
        for ind in indices:
            bin_edgesi.append(bin_edges[idx[ind]+1])
            coarse_edges.append(bin_edgesi)
            bin_edgesi = [bin_edges[idx[ind]+1]]
        bin_edgesi.append(bin_edges[-1]+0.01)
        coarse_edges.append(bin_edgesi)

        #segment D into the bin edges
        bin_indices = [np.argwhere(np.logical_and(d>edge[0],d<edge[1])).flatten() for edge in coarse_edges]

        #find bin centers from bin_indices
        bin_centers = [np.mean(d[indices]) for indices in bin_indices]

        return bin_indices, bin_centers

    def segment_bulk(self,slab_indices, atom_ind):

        config = self.config.get('slab_config').copy()
        config['vacuum'] = 0
        config['zperiodic'] = True
        pure_slab = create_slab(config)
        cell = pure_slab.cell.diagonal()

        slab = self.atoms[slab_indices]
        slab_seed = int(np.argwhere(slab_indices==atom_ind))
        D = slab.get_distances(np.arange(len(slab)),slab_seed, mic=True,vector=True)

        for i, n_planes in enumerate(self.config.get('slab_config')['supercell_size']):
            hist, bin_edges = np.histogram(D[:,i],bins=self.config.get('supercell_size')[i]*10)
            
            bin_indices, bin_centers = self.coarsen_histogram(D[:,i],hist, bin_edges)

            n_basis = max([1,int(np.round(len(bin_indices)/self.config.get('supercell_size')[i],0))])
            n_planes *= n_basis
            bin_seed_ind = np.argsort(np.abs(bin_centers))[:n_planes]

            min_bin_indices = np.concatenate(np.array(bin_indices,dtype=object)[bin_seed_ind]).astype(int)
            if i == 0:
                keep_indices = min_bin_indices
            else:
                keep_indices = np.intersect1d(keep_indices,min_bin_indices)
        
        
        # keep_indices = np.argwhere(np.logical_and(
        #     np.logical_and(
        #         np.logical_and(D[:,0]>-cell[0]/2,D[:,0]<=cell[0]/2),
        #         np.logical_and(D[:,1]>-cell[1]/2,D[:,1]<=cell[1]/2)),
        #         np.logical_and(D[:,2]>-cell[2]/2,D[:,2]<=cell[2]/2))
        # ).flatten()
        
        bulk_keep = slab[keep_indices]
        
        cell_new = pure_slab.get_cell()
        bulk_keep.set_cell(cell_new)
        bulk_keep.center()
        bulk_keep.wrap()

        #remove overlapping atoms
        overlap = self.config.get('overlap_radius',0.5)
        edge_vec = []

        while len(edge_vec)>0:
            delete_indices = []
            src, dst, edge_vec = neighborlist.neighbor_list('ijD',bulk_keep,overlap)
            if len(edge_vec)>0:
                print('Check that deleting overlaping atoms is working properly', flush=True)
            for i, (src_i, dst_i, vec) in enumerate(zip(src,dst,edge_vec)):
                if dst_i in delete_indices:
                    break
                bulk_keep.positions[dst_i] -= vec/2
                delete_indices.append(src_i)
                # keep_indices.remove(src_i)
            
            [keep_indices.pop(ii) for ii in delete_indices]
            del bulk_keep[delete_indices]

        bulk_keep.pbc=True
        return bulk_keep, keep_indices



def clusters_from_traj(
    traj,
    uncertainties,
    uncertainty_thresholds: list,
    cutoff = None,
    min_cluster_size: int = 20,
    max_cluster_size: int = 40,
    vacuum_size: float = 2.0,
    sorted: bool = True,
    config = None,
):

    ncores = mp.cpu_count()
    cores = int(config.get('cores',ncores))
    traj_cores = max(int(cores/4),1) # int(config.get('trajectory_cores',cores))
    print(traj_cores, ncores, flush=True)

    traj_filename = f'tmp.{uuid.uuid4().hex}.traj'
    natoms = len(traj)
    write(traj_filename,traj)
    del traj

    generator = ((traj_filename, i, cutoff, config, uncertainties[i], uncertainty_thresholds, min_cluster_size, max_cluster_size, vacuum_size) 
                for i in range(natoms))
    
    
    results = send_to_multiprocessing(cluster_from_atoms, generator, traj_cores)

    os.remove(traj_filename)

    clusters_all = []

    df_ind = pd.DataFrame({'traj_ind': pd.Series(dtype=int),
                            'atom_ind': pd.Series(dtype=int),
                            'uncertainty': pd.Series(dtype=float)})


    for clusters, df in results:
        df_ind = df_ind.append(df,ignore_index=True)

        clusters_all += clusters


    if sorted:
        df_ind = df_ind.sort_values('uncertainty')[::-1]
        # cluster_ind = np.argsort(df_ind['uncertainty'].values)[::-1]
        clusters_all = [clusters_all[i] for i in df_ind.index]

    return clusters_all, df_ind

def cluster_from_atoms(args):
    traj_filename, i, cutoff, config, uncertainty, uncertainty_thresholds, min_cluster_size, max_cluster_size, vacuum_size = args

    atoms = read(traj_filename,index=i)
    max_uncertainty, min_uncertainty = uncertainty_thresholds
    
    process = psutil.Process(os.getpid())
    print(i,f'used {psutil.virtual_memory().used/1024**3} GB', f'total {psutil.virtual_memory().total/1024**3} GB', psutil.virtual_memory().percent, flush=True)
    segment = segment_atoms(atoms,cutoff,config)

    uncertain_indices = np.where(np.logical_and(uncertainty>min_uncertainty,uncertainty<max_uncertainty))[0]
    uncertain_indices = uncertain_indices[np.argsort(uncertainty[uncertain_indices]).flipud()]


    if not isinstance(uncertain_indices,np.ndarray):
        uncertain_indices = np.array([uncertain_indices])

    # clusters, atom_indices = clusterstocalculate(atoms,uncertain_indices,n_components,component_list,max_cluster_size,vacuum_size)

    clusters, atom_indices = segment.clusterstocalculate(uncertain_indices,min_cluster_size,max_cluster_size,vacuum_size)

    cluster_uncertainties = uncertainty[atom_indices]

    data = {
        'traj_ind': i*np.ones(len(atom_indices)),
        'atom_ind': atom_indices,
        'uncertainty': cluster_uncertainties,
    }
    df = pd.DataFrame(data)

    return clusters, df

def send_to_multiprocessing(func, gen, ncores):

    def produce(semaphore, generator):
        for gen in generator:
            semaphore.acquire()
            yield gen

    def consume(semaphore):#, result, results):
        semaphore.release()

    pool = mp.Pool(ncores)
    print('memory error doesnt occurs when creating pool',flush=True)    
    # results = pool.imap_unordered(cluster_from_atoms, generator)
    results = []
    
    ii = 0
    print(ii,f'used {psutil.virtual_memory().used/1024**3} GB', f'total {psutil.virtual_memory().total/1024**3} GB', psutil.virtual_memory().percent, flush=True)

    semaphore_1 = mp.Semaphore(ncores)
    for result in pool.imap_unordered(func, produce(semaphore_1, gen)):
        # consume(semaphore_1)#,result,results)
        ii+=1
        results.append(result)
        # if ii%100 == 0:
        gc.collect()
        print(ii,f'used {psutil.virtual_memory().used/1024**3} GB', f'total {psutil.virtual_memory().total/1024**3} GB', psutil.virtual_memory().percent, flush=True)
        
        semaphore_1.release()
        # print('Elapsed time: ', time.time()-start)
    pool.close()
    print('waiting for pool to join',flush=True)
    pool.join()
    print('pools joined',flush=True)
    
    gc.collect()
    print(ii,f'used {psutil.virtual_memory().used/1024**3} GB', f'total {psutil.virtual_memory().total/1024**3} GB', psutil.virtual_memory().percent, flush=True)

    return results