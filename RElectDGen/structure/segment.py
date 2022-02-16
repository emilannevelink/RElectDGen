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
from ..calculate.calculator import nn_from_results
from nequip.data import AtomicData
from nequip.data.transforms import TypeMapper

class segment_atoms():
    def __init__(self, 
        atoms: ase.Atoms,
        slab_config: dict, 
        main_supercell_size: list,
        cutoff: float = 2.0,
        segment_type: str = 'distance',
        max_volume_per_atom: int = 150,
        min_cluster_size: int = 20,
        max_cluster_size: int = 50,
        max_samples: int = 10,
        vacuum: float = 2.0,
        overlap_radius: float = 0.5,
        max_electrons: int = 400,
        run_dir: str = '',
        fragment_dir: str = '',
        ) -> None:
        
        self.atoms = atoms
        self.cutoff = cutoff
        self.segment_type = segment_type
        self.max_volume_per_atom = max_volume_per_atom
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.max_samples = max_samples
        self.slab_config = slab_config
        self.main_supercell_size = main_supercell_size
        self.vacuum = vacuum
        self.overlap_radius = overlap_radius
        self.max_electrons = max_electrons
        self.run_dir = run_dir
        self.fragment_dir = fragment_dir
        # ('/Users/emil/GITHUB/RElectDGen/' + 
        # 'tests/structure/reassign_charge/data/')

        filename = os.path.join(self.fragment_dir,'fragment_db.csv')
        if os.path.isfile(filename):
            self.fragment_db = pd.read_csv(filename)
        else:
            self.fragment_db = None

        if len(self.atoms)>self.max_cluster_size:
            self.n_components_natural, self.component_list_natural, self.matrix_natural = self.findclusters(natural = True)

            self.reassign_cluster_charges()

            self.n_components, self.component_list, self.matrix = self.findclusters(cutoff=self.cutoff)
            self.matrixnonzero = self.matrix.nonzero()
            
            self.nlithiums = (self.atoms.get_atomic_numbers()==3).sum()

            if self.segment_type == 'embedding':
                calc_nn, model_load, MLP_config = nn_from_results()
                self.model = copy.copy(model_load)
                self.transform = TypeMapper(chemical_symbol_to_type=MLP_config.get('chemical_symbol_to_type'))
                self.r_max = MLP_config.get('r_max')
                print('loaded model',self.model, flush=True)

    def findclusters(self,
        atoms = None,
        cutoff = None,
        nlithium = 4,
        natural = False,
    ):
        if atoms is None:
            atoms = self.atoms#.copy()
        
        
        if cutoff is None:
            cutoff = np.array(neighborlist.natural_cutoffs(atoms))*1.3
            cutoff[atoms.get_atomic_numbers()==3]*=1.3
        elif isinstance(cutoff,float) or isinstance(cutoff,int):
            cutoff = [cutoff for _ in atoms]
        elif isinstance(cutoff,list):
            assert len(cutoff) == len(atoms)


        nblist = neighborlist.NeighborList(cutoff,skin=0,self_interaction=False,bothways=True)
        nblist.update(atoms)

        matrix = nblist.get_connectivity_matrix()

        n_components, component_list = sparse.csgraph.connected_components(matrix)
        
        if natural:
            #Separate molecules from lithium slab
            start_time = time.time()
            print('natural')
            clusters_to_check = [(atoms[component_list ==i].get_atomic_numbers()==3).sum()>nlithium for i in np.arange(n_components)]
            
            for cluster_index, check in enumerate(clusters_to_check):
                if check:
                    matrix = self.separate_cluster(cluster_index, matrix, n_components, component_list, atoms, cutoff)

            time_separate = time.time()
            print('Time to separate ', time_separate-start_time)
            n_components, component_list = sparse.csgraph.connected_components(matrix)


            # Make sure they are smallest connected 
            clusters_to_check = [not self.is_valid_cluster(atoms[component_list ==i]) for i in np.arange(n_components)]
            for cluster_index, check in enumerate(clusters_to_check):
                if check:
                    matrix = self.split_molecules_fragments(cluster_index, matrix, n_components, component_list, atoms, cutoff)

            time_split = time.time()
            print('Time to fragment ', time_split- time_separate)

            n_components, component_list = sparse.csgraph.connected_components(matrix)

        return n_components, component_list, matrix

    def separate_cluster(self,
        cluster_index,
        matrix,
        n_components,
        component_list,
        atoms,
        cutoff):

        cluster_indices = np.argwhere(component_list==cluster_index).flatten()

        slab_indices, mixture_indices = self.segment_slab_mixture(cluster_indices)
        

        if len(slab_indices)>0 and len(mixture_indices)>0:

            separate_ind = mixture_indices if len(mixture_indices)<len(slab_indices) else slab_indices

            nblist = neighborlist.NeighborList(cutoff[separate_ind],skin=0,self_interaction=False,bothways=True)

            sub_cluster = atoms[separate_ind]
            nblist.update(sub_cluster)
            sub_matrix = nblist.get_connectivity_matrix()

            matrix[separate_ind] = 0
            matrix[:,separate_ind] = 0
            indices = np.array(list(sub_matrix.keys()))
            matrix[separate_ind[indices[:,0]],separate_ind[indices[:,1]]] = 1

        return matrix #, n_components, component_list

    def split_molecules_fragments(self, cluster_index, matrix, n_components, component_list, atoms, cutoff):

        cluster_indices = np.argwhere(component_list==cluster_index).flatten()

        fragments, cluster_indices = self.fragments_from_cluster(atoms, cluster_indices, cutoff)
        
        cluster = atoms[cluster_indices]
        if self.is_valid_cluster(cluster) or self.is_valid_fragment(cluster):
            fragments += [cluster_indices]
            cluster_indices = []
        elif len(cluster_indices)>0:
            lithium_indices = np.argwhere(cluster.get_atomic_numbers()==3).flatten()
            
            fragments += [np.array([ind]) for ind in cluster_indices[lithium_indices] if self.is_valid_cluster(atoms[[ind]])]
            cluster_indices = np.delete(cluster_indices, lithium_indices)

            fragments_stripped, cluster_indices = self.fragments_from_cluster(atoms, cluster_indices, cutoff)

            fragments = fragments + fragments_stripped

        
        #Change matrix only if cluster was able to be decomposed into fragments
        if len(cluster_indices) == 0:
            for fragment in fragments:
                #Remove previous connectivity
                matrix[fragment] = 0
                matrix[:,fragment] = 0
                cluster = atoms[fragment]
                cluster_nblist = neighborlist.NeighborList(cutoff[fragment],skin=0,self_interaction=False,bothways=True)
                cluster_nblist.update(cluster)
                cluster_matrix = cluster_nblist.get_connectivity_matrix()

                #Add connectivity of the fragment
                for src, dst in cluster_matrix.keys():
                    matrix[fragment[src],fragment[dst]] = 1

        return matrix

    def fragments_from_cluster(self, atoms, cluster_indices, cutoff):
        fragments = []
        cluster = atoms[cluster_indices]
        # cutoff = np.array(neighborlist.natural_cutoffs(cluster))
        if self.is_valid_cluster(cluster):
            fragments = [cluster_indices]
            cluster_indices = np.array([],dtype=int)
        elif self.is_valid_fragment(cluster):
            fragments = [cluster_indices]
            cluster_indices = np.array([],dtype=int)
        else:
            max_iterations = len(cluster_indices)
            iterations = 0
            
            while len(cluster_indices)>0 and iterations < max_iterations:
                iterations += 1
                cluster = atoms[cluster_indices]
                cluster_nblist = neighborlist.NeighborList(cutoff[cluster_indices],skin=0,self_interaction=False,bothways=True)
                cluster_nblist.update(cluster)
                cluster_matrix = cluster_nblist.get_connectivity_matrix().asformat("array")

                leafs = np.argwhere(cluster_matrix.sum(axis=0)==1)
                
                for leaf_index in leafs.flatten():
                    fragment = [leaf_index]
                    fragment_iterations = 0
                    max_fragment_iterations = len(cluster_indices)
                    while (
                        not (
                            self.is_valid_fragment(cluster[fragment]) or 
                            self.is_valid_cluster(cluster[fragment])
                        ) 
                        and len(fragment)<len(cluster)
                        and fragment_iterations<max_fragment_iterations
                        ):
                        
                        fragment_iterations+=1
                        potential_additions = np.argwhere(cluster_matrix[fragment])
                        sorted_by_connections = np.argsort(cluster_matrix[potential_additions[:,1]].sum(axis=1))
                        for add_ind in sorted_by_connections:
                            if potential_additions[add_ind,1] not in fragment:
                                fragment.append(potential_additions[add_ind,1])
                                break
                        # fragment += indices[indices[:,0]==minimum_connection,1].tolist()
                        # fragment = np.unique(fragment).tolist()

                    if self.is_valid_cluster(cluster[fragment]) or self.is_valid_fragment(cluster[fragment]):
                        fragments.append(cluster_indices[fragment])
                        cluster_indices = np.delete(cluster_indices,fragment)
                        break

        return fragments, cluster_indices

    def is_valid_cluster(self, cluster):

        total_charge = cluster.get_initial_charges().sum()
        integer_charge = np.round(total_charge,0) == np.round(total_charge,2)

        return integer_charge

    def is_valid_fragment(self, cluster,raw=False):

        if self.fragment_db is None:
            return False
        else:
            charge_i = cluster.get_initial_charges().sum()
                    
            db_ind = np.isclose(self.fragment_db['origin_charge'],charge_i)

            atom_symbols = np.array(cluster.get_chemical_symbols())
            for i, sym in enumerate(np.unique(atom_symbols)):
                col = 'n_' + sym
                
                nsym = np.sum(atom_symbols==sym)
                try:
                    db_ind = np.logical_and(db_ind,self.fragment_db[col] == nsym)
                except KeyError as e:
                    print(e)
                    print(cluster)
            if raw:
                return db_ind
            else:
                if db_ind.sum()==1:
                    return True
                else:
                    return False

    def reassign_cluster_charges(self):
        nclusters = self.n_components_natural
        initial_charges = copy.copy(self.atoms.get_initial_charges())
        initial_supercell_charge = initial_charges.sum()
        
        
        if self.fragment_db is not None:
            fragment_db = self.fragment_db 
            
            cluster_fragments = []
            for cluster_ind in range(nclusters):
                cluster = self.atoms[self.component_list_natural==cluster_ind]

                if not self.is_valid_cluster(cluster): # most likely signifies it comes from a larger molecule
                    
                    cluster_fragments.append([cluster,cluster_ind])
                    
            # replace charge on atoms
            for i, (cluster_i, ind_i) in enumerate(cluster_fragments):
                db_ind = self.is_valid_fragment(cluster_i,raw=True)

                #look-up fragment id in fragment database
                if db_ind.sum()==1:
                    fragment_name = fragment_db['fragment_name'][db_ind].values[0]
                    print(fragment_name)
                    fragment_filename = os.path.join(self.fragment_dir,f'fragment_{fragment_name}.json')
                    cluster_fragment = read(fragment_filename)

                    indices = self.component_list_natural==ind_i

                    self.atoms.arrays['initial_charges'][indices] = cluster_fragment.get_initial_charges()
                elif db_ind.sum() == 0:
                    unknown_fragment_file = os.path.join(self.run_dir,'unknown_fragments.txt')
                    f = open(unknown_fragment_file,'a')
                    f.write(str(cluster_i)+str(cluster_i.get_initial_charges())+str(self.atoms)+'\n')
                    f.close()
                else:
                    print('Fragment DB retrieves too many results ', db_ind.sum())
                    print(fragment_db['fragment_name'][db_ind])

            final_charges = self.atoms.get_initial_charges()
            final_supercell_charge = final_charges.sum()
            
            if not np.isclose(initial_supercell_charge,final_supercell_charge,atol=1e-3):
                print('Reassign charges changed the total supercell charge, changing back')
                self.atoms.set_initial_charges(initial_charges)
            
        else:
            print('Fragment DB is none')

    def clusterstocalculate(self,
        uncertain_indices: list,
    ):
        
        if len(self.atoms)<=self.max_cluster_size:
            if len(uncertain_indices)>0:
                self.atoms.wrap()
                atoms = Atoms(self.atoms)
                atoms.calc = None
                atoms.arrays['cluster_indices'] = np.array(range(len(atoms)),dtype=int)
                return [[atoms], [uncertain_indices[0]]]
            else:
                return [[],[]]

        
        molecules = []
        clusters = []
        atom_indices = []

        for idx in uncertain_indices:
            start_time = time.time()

            molIdx = self.component_list_natural[idx]

            if molIdx not in molecules:
                
                indices_add = [ i for i in range(len(self.component_list_natural)) if self.component_list_natural[i] == molIdx ]
                slab_add, mixture_add = self.segment_slab_mixture(indices_add)
                

                build=True
                add_slab = False
                add_bulk = False
                build_ind = 0
                idx_add = [idx]
                mol_add = [molIdx]
                if idx in mixture_add:
                    
                    cluster_indices = list(mixture_add)
                    slab_indices = list(slab_add)
                    if len(slab_indices)>0:
                        print('weird')
                        
                elif idx in slab_add:
                    cluster_indices = list(mixture_add)
                    slab_indices = list(slab_add)
                    add_slab = True
                    if len(cluster_indices)>0:
                        print('weird')
                
                
                while build: #len(cluster_indices) < min_cluster_size and build:
                    if build_ind>100:
                        print(build_ind,flush=True)
                    build_ind+=1
                    total_indices = cluster_indices + slab_indices
                    neighbor_list_i = [self.matrixnonzero[0][self.matrixnonzero[1]==i] for i in total_indices]
                    neighbor_atoms = np.unique(np.concatenate(neighbor_list_i))
                    neighbor_atoms = [ind for ind in neighbor_atoms if ind not in total_indices]

                    if len(neighbor_atoms) == 0:
                        build = False
                    else:
                        
                        indices_add, atom_ind, molIdx_add = self.next_cluster(idx,neighbor_atoms,cluster_indices,uncertain_indices)
                        idx_add.append(atom_ind)
                        mol_add.append(molIdx_add)

                        if len(indices_add)>0:
                            slab_add, mixture_add = self.segment_slab_mixture(indices_add)

                            if ((len(cluster_indices) + len(mixture_add)) > self.max_cluster_size / (2  - (not add_slab)) or
                                self.atoms[cluster_indices+list(mixture_add)].get_atomic_numbers().sum() > self.max_electrons/ (2  - (not add_slab)) ):
                                build = False
                            elif atom_ind in mixture_add:
                                
                                # molecules.append(molIdx_add)
                                if len(slab_add)>0:
                                    print('does this even happen???')
                                    ind_mixture = np.argwhere(mixture_add==idx)[0,0]

                                    n_components, component_list, matrix = self.findclusters(atoms=self.atoms[mixture_add])
                                    mol_mixture = component_list[ind_mixture]
                                    mol_indices = [ i for i in range(len(component_list)) if component_list[i] == mol_mixture ]
                                    cluster_indices += list(mixture_add[mol_indices])
                                    
                                    if (len(cluster_indices)) <= self.max_cluster_size / 2:
                                        add_slab = True
                                        slab_indices += list(slab_add)
                                else:
                                    cluster_indices += list(mixture_add)

                            elif atom_ind in slab_add:
                                add_slab = True
                                slab_indices += list(slab_add)
                                # molecules.append(molIdx_add)
                                # if len(slab_indices)>0.75*self.nlithiums:
                                #     build = False

                        else:
                            build = False
                        
                if add_slab:
                    
                    if len(cluster_indices)>0:
                        print(len(cluster_indices),flush=True)
                        pure_slab = create_slab(self.slab_config)
                        cell = pure_slab.cell.diagonal()
                        min_cluster = np.absolute(self.atoms[cluster_indices].positions[:,2]-self.atoms[idx].position[2]).min()
                        if min_cluster <= cell[2]/2:
                            print('segment slab', flush=True)
                            cluster, cluster_indices = self.segment_slab(cluster_indices, slab_indices)
                        else:
                            # Add bulk
                            print('segment bulk', flush=True)
                            cluster, cluster_indices = self.segment_bulk(slab_indices,idx)
                            cluster.pbc = True
                    else:
                        # Add bulk
                        print('segment bulk', flush=True)
                        cluster, cluster_indices = self.segment_bulk(slab_indices,idx)
                        cluster.pbc = True
                    
                else:
                    cluster = self.atoms[cluster_indices]
                    cluster = self.reduce_mixture_size(cluster)
                    cluster.pbc = False

                lithium_ind = np.argwhere(cluster.get_atomic_numbers()==3).flatten()
                if len(lithium_ind)>500:
                    print('wrong, too many lithium atoms',flush=True)
                    print(add_slab,len(cluster_indices), len(lithium_ind), flush=True)
                else:

                    if (len(cluster)>1 and 
                            cluster.get_volume()/len(cluster)<self.max_volume_per_atom and
                            np.isclose(cluster.get_initial_charges().sum().round(),cluster.get_initial_charges().sum().round(2))):
                        
                        cluster.arrays['cluster_indices'] = np.array(cluster_indices,dtype=int)
                        clusters.append(cluster)
                        atom_indices.append(idx)
                        molecules.append(molIdx)
                        print('Added atoms', idx_add)
                        print('Added molecules', mol_add)
                    else:
                        if len(cluster)==1:
                            print(f'wrong, not enough atoms around {idx}',flush=True)
                        elif cluster.get_volume()/len(cluster)>self.max_volume_per_atom:
                            print(f'wrong, cluster volume too large volume/atom: {cluster.get_volume()/len(cluster)}',flush=True)
                        elif not np.isclose(cluster.get_initial_charges().sum().round(),cluster.get_initial_charges().sum().round(2)):
                            print(f'wrong, cluster doesnt have whole number charge')
                        
                        print('Didnt add atoms', idx_add)
                        print('Didnt add molecules', mol_add)
                        
            if len(clusters)>=self.max_samples:
                break

        return clusters, atom_indices

    def next_cluster(self,idx,neighbor_atoms,cluster_indices,uncertain_indices):
        # print('getting next cluster', self.segment_type,flush=True)
        if self.segment_type == 'uncertain':
            # get neighbor atom with the lowest index in uncertain indices (uncertain indices are sorted low to high)
            neigh_uncertain = [np.argwhere(neigh_ind == np.array(uncertain_indices))[0,0] for neigh_ind in neighbor_atoms if neigh_ind in uncertain_indices]
            if len(neigh_uncertain)>0:
                atom_ind = neighbor_atoms[np.argmin(neigh_uncertain)]
                molIdx_add = self.component_list_natural[atom_ind]
                
                # add natural molecule that atom is in to cluster indices                             
                indices_add = [ i for i in range(len(self.component_list_natural)) if self.component_list_natural[i] == molIdx_add ]
            else:
                indices_add = []
        
        elif self.segment_type == 'distance':
            # get the neighbor atom that is closest
            Di = self.atoms.get_distances(neighbor_atoms,idx, mic=True)
            atom_ind = neighbor_atoms[np.argmin(Di)]
            molIdx_add = self.component_list_natural[atom_ind]
            
            indices_add = [ i for i in range(len(self.component_list_natural)) if self.component_list_natural[i] == molIdx_add ]

        elif self.segment_type == 'embedding':
            # print('in embedding', flush=True)
            data = self.transform(AtomicData.from_ase(atoms=self.atoms,r_max=self.r_max))
            # print('transformed data',data, flush=True)
            print('loaded model', flush=True)
            out = self.model(AtomicData.to_AtomicDataDict(data))
            print('calculated model', flush=True)
            embeddingi = out['node_features'][idx].detach().numpy()
            # print('got node embedding', flush=True)
            embed_dist = []
            #get possible added molecules
            close_clusters = np.unique(self.component_list_natural[neighbor_atoms])
            for i, cc in enumerate(close_clusters):
                # print('going through close clusters', i, flush=True)
                indices_add = [ i for i in range(len(self.component_list_natural)) if self.component_list_natural[i] == cc ]
                ind_tmp = cluster_indices + indices_add
                cluster_tmp = self.atoms[ind_tmp]
                
                data = self.transform(AtomicData.from_ase(atoms=cluster_tmp,r_max=self.r_max))
                out = self.model(AtomicData.to_AtomicDataDict(data))
                # print('got node embedding', flush=True)
                tmp_idx = np.argwhere(ind_tmp==idx)[0,0]
                embeddingj = out['node_features'][tmp_idx].detach().numpy()
                embed_dist.append(np.linalg.norm(embeddingi-embeddingj))
            
            molIdx_add = close_clusters[np.argmin(embed_dist)]
            
            neigh_mol_indices = np.array(neighbor_atoms)[np.argwhere(self.component_list_natural[neighbor_atoms]==molIdx_add).flatten()]
            Di = self.atoms.get_distances(neigh_mol_indices,idx, mic=True)
            atom_ind = neigh_mol_indices[np.argmin(Di)]
            
            indices_add = [ i for i in range(len(self.component_list_natural)) if self.component_list_natural[i] == molIdx_add ]

        return indices_add, atom_ind, molIdx_add

    def reduce_mixture_size(self,cluster):

        atoms1 = copy.deepcopy(cluster)
        divisor = 2
        # arg = np.argwhere(atoms1.positions.std(axis=0)>cluster.cell.diagonal()/5)
        
        bools = self.axis_crossing_boundary(atoms1)
        while bools.sum()>0:
            atoms1 = copy.deepcopy(cluster)
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
        
        atoms2 = copy.deepcopy(atoms1)
        atoms1.center(vacuum=self.vacuum)
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
            atoms2.center(self.vacuum,axis=shrink_ind)
            cluster = atoms2
        
        del atoms1, atoms2
        return cluster

    def axis_crossing_boundary(self,atoms):
        bools = [True, True, True]
        bools = []
        for ind in range(len(atoms)):
            Di = atoms.get_distances(np.arange(len(atoms)),ind, mic=True,vector=True)
            Df = atoms.get_distances(np.arange(len(atoms)),ind, mic=False,vector=True)
            greater_cutoff = np.logical_and(np.abs(Di)>2*self.cutoff,np.abs(Df)>2*self.cutoff)
            bools.append(np.all(np.logical_or(np.isclose(Di,Df),greater_cutoff),axis=0))

        bools = np.all(bools,axis=0)#np.array(bools).sum(axis=0)>len(atoms)*0.6
        # np.any([np.isclose(Di,Df),np.isclose(Di+Df,atoms.cell.diagonal())])
        # bools = np.all([bools, np.all(np.isclose(Di,Df),axis=0)],axis=0)

        bools = np.invert(bools)
        return bools

    def segment_slab_mixture(self, cluster_indices):

        cluster = self.atoms[cluster_indices]

        lithium_ind = np.argwhere(cluster.get_atomic_numbers()==3).flatten()
        if len(lithium_ind)>5 and self.slab_config is not None:

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
            
    
    def segment_slab(self, cluster_indices, slab_indices):

        cluster = self.atoms[cluster_indices]
        cluster_reduce = self.reduce_mixture_size(cluster)

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
        
        pure_slab = create_slab(self.slab_config)
        cell = pure_slab.cell.diagonal()

        
        for i, n_planes in enumerate(self.slab_config['supercell_size']):
            hist, bin_edges = np.histogram(D[:,i],bins=self.main_supercell_size[i]*30)
            
            bin_indices, bin_centers = self.coarsen_histogram(D[:,i],hist, bin_edges,self.atoms.cell.diagonal()[i],self.main_supercell_size[i])
            print(len(bin_centers),self.main_supercell_size[i])
            n_basis = max([1,int(np.round(len(bin_indices)/self.main_supercell_size[i],0))])
            n_planes *= n_basis
            
            bin_seed_ind = np.argsort(np.abs(bin_centers))[:n_planes]

            # import matplotlib.pyplot as plt
            # plt.hist(D[:,i],bins=bin_edges)

            min_bin_indices = np.concatenate(np.array(bin_indices,dtype=object)[bin_seed_ind]).astype(int)
            if i == 0:
                keep_indices = min_bin_indices
            else:
                keep_indices = np.intersect1d(keep_indices,min_bin_indices)
        
        slab_keep = slab[keep_indices]
        
        reduced_slab_cluster = slab_keep + self.atoms[cluster_indices]

        cell_new = pure_slab.get_cell()
        reduced_slab_cluster.set_cell(cell_new)
        reduced_slab_cluster.center()
        reduced_slab_cluster.center(vacuum=4,axis=2)

        #remove overlapping atoms
        edge_vec = []

        keep_indices = list(keep_indices)
        while len(edge_vec)>0:
            delete_indices = []
            src, dst, edge_vec = neighborlist.neighbor_list('ijD',reduced_slab_cluster,self.overlap_radius)
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

    def coarsen_histogram(self, d, hist, bin_edges,max_dist,expected_bins):

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

        def check_condition(n_coarse):
            mean = np.mean(n_coarse)
            condition = np.all(n_coarse>mean/2)
            # if np.round(mean,0)==mean:
            #     condition = np.all(n_coarse == np.mean(n_coarse))
            # else:
            #     condition = np.all(np.logical_or(
            #         n_coarse==np.ceil(mean),
            #         n_coarse==np.floor(mean)
            #     ))
            return condition

        n_coarse = [len(bi) for bi in bin_indices]
        while not check_condition(n_coarse) and len(bin_indices)>expected_bins:
            for i, val in enumerate(n_coarse):
                if val == np.min(n_coarse):
                    dist = np.abs(bin_centers-bin_centers[i])
                    dist = np.min([dist,np.abs(max_dist-dist)],axis=0)
                    concat_ind = np.argsort(dist)[1]
                    bin_indices[concat_ind] = np.concatenate([bin_indices[concat_ind],bin_indices[i]])
                    bin_indices.pop(i)
                    bin_centers.pop(i)

                    n_coarse = [len(bi) for bi in bin_indices]
                    break
        

        return bin_indices, bin_centers

    def segment_bulk(self,slab_indices, atom_ind):

        config = self.slab_config.copy()
        config['vacuum'] = 0
        config['zperiodic'] = True
        pure_slab = create_slab(config)
        cell = pure_slab.cell.diagonal()

        slab = self.atoms[slab_indices]
        slab_seed = int(np.argwhere(slab_indices==atom_ind))
        D = slab.get_distances(np.arange(len(slab)),slab_seed, mic=True,vector=True)

        for i, n_planes in enumerate(self.slab_config['supercell_size']):
            hist, bin_edges = np.histogram(D[:,i],bins=self.main_supercell_size[i]*30)
            
            bin_indices, bin_centers = self.coarsen_histogram(D[:,i],hist, bin_edges,self.atoms.cell.diagonal()[i],self.main_supercell_size[i])
            print(len(bin_centers),self.main_supercell_size[i])
            n_basis = max([1,int(np.round(len(bin_indices)/self.main_supercell_size[i],0))])
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
        
        z_bins = np.array(bin_centers)[bin_seed_ind]
        z_span = z_bins.max()-z_bins.min()
        z_cell = z_span*len(bin_seed_ind)/(len(bin_seed_ind)-1)
        bulk_keep = slab[keep_indices]
        
        cell_new = pure_slab.get_cell()
        cell_new[2,2] = z_cell
        bulk_keep.set_cell(cell_new)
        bulk_keep.center()
        bulk_keep.wrap()

        #remove overlapping atoms
        edge_vec = []

        while len(edge_vec)>0:
            delete_indices = []
            src, dst, edge_vec = neighborlist.neighbor_list('ijD',bulk_keep,self.overlap_radius)
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
    uncertainty_thresholds: list = [0.2,0.01],
    slab_config: dict = {},
    supercell_size: list = [1,1,1],
    cutoff: float = 2.0,
    min_cluster_size: int = 20,
    max_cluster_size: int = 40,
    sorted: bool = True,
    cores: int = 1,
    segment_type: str = 'uncertain',
    max_volume_per_atom: int = 150,
    max_samples: int = 10,
    molecule_vacuum: float = 2.0,
    overlap_radius: float = 0.5,
    max_electrons: int = 300,
    directory: str = '',
    run_dir: str = '',
    data_directory: str = '',
    fragment_dir: str = '',
    **kwargs,
):

    ncores = mp.cpu_count()
    
    traj_cores = 1 # max(int(cores/16),1) # int(config.get('trajectory_cores',cores))
    print(traj_cores, ncores, flush=True)

    traj_filename = f'tmp.{uuid.uuid4().hex}.traj'
    natoms = len(traj)
    write(traj_filename,traj)
    del traj

    generator = ((traj_filename, i, uncertainties[i], uncertainty_thresholds,
                    slab_config, supercell_size, cutoff, segment_type, max_volume_per_atom,
                    min_cluster_size, max_cluster_size, max_samples, molecule_vacuum, overlap_radius, max_electrons,
                    os.path.join(directory, run_dir), os.path.join(data_directory, fragment_dir)) 
                for i in range(natoms))
    
    if segment_type == 'embedding' or traj_cores==1:
        results = []
        for gen in generator:
            results.append(cluster_from_atoms(gen))
    else:
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
    traj_filename, i, uncertainty, uncertainty_thresholds = args[:4]

    atoms = read(traj_filename,index=i)
    max_uncertainty, min_uncertainty = uncertainty_thresholds
    
    process = psutil.Process(os.getpid())
    print(i,f'used {psutil.virtual_memory().used/1024**3} GB', f'total {psutil.virtual_memory().total/1024**3} GB', psutil.virtual_memory().percent, flush=True)
    segment = segment_atoms(atoms,*args[4:])

    uncertain_indices = np.where(np.logical_and(uncertainty>min_uncertainty,uncertainty<max_uncertainty))[0]
    uncertain_indices = uncertain_indices[np.flipud(np.argsort(uncertainty[uncertain_indices]))]


    if not isinstance(uncertain_indices,np.ndarray):
        uncertain_indices = np.array([uncertain_indices])

    # clusters, atom_indices = clusterstocalculate(atoms,uncertain_indices,n_components,component_list,max_cluster_size,vacuum_size)
    print('finding clusters to calculate', flush=True)
    clusters, atom_indices = segment.clusterstocalculate(uncertain_indices)

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

    # ncores = 1
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