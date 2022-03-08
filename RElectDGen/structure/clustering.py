import time
import numpy as np
from ase import neighborlist
from scipy import sparse
from sklearn.cluster import KMeans


class findclusters():
    def __init__(
        self,
        atoms,
        FragmentDB,
        cutoff=None,
        nlithium=4,
        natural = False,
        slab_config = None,
        fragment_type = 'iterative'
    ):  
        self.atoms = atoms
        self.FragmentDB = FragmentDB
        self.slab_config = slab_config
        self.fragment_type = fragment_type

        if cutoff is None:
            cutoff = np.array(neighborlist.natural_cutoffs(atoms))*1.3
            cutoff[atoms.get_atomic_numbers()==3]*=1.3
        elif isinstance(cutoff,float) or isinstance(cutoff,int):
            cutoff = [cutoff for _ in atoms]
        elif isinstance(cutoff,list):
            assert len(cutoff) == len(atoms)

        self.cutoff = cutoff

        nblist = neighborlist.NeighborList(cutoff,skin=0,self_interaction=False,bothways=True)
        nblist.update(atoms)

        matrix = nblist.get_connectivity_matrix()
        self.matrix = matrix

        n_components, component_list = sparse.csgraph.connected_components(matrix)
        
        if natural:
            #Separate molecules from lithium slab
            start_time = time.time()
            print('natural')
            clusters_to_check = [(atoms[component_list ==i].get_atomic_numbers()==3).sum()>nlithium for i in np.arange(n_components)]
            
            for cluster_index, check in enumerate(clusters_to_check):
                if check:
                    cluster_indices = np.argwhere(component_list==cluster_index).flatten()
                    self.separate_cluster(cluster_indices)

            time_separate = time.time()
            print('Time to separate ', time_separate-start_time)
            n_components, component_list = sparse.csgraph.connected_components(matrix)


            # Make sure they are smallest connected 
            clusters_to_check = [not self.FragmentDB.is_valid_cluster(atoms[component_list ==i]) for i in np.arange(n_components)]
            for cluster_index, check in enumerate(clusters_to_check):
                if check:
                    cluster_indices = np.argwhere(component_list==cluster_index).flatten()
                    self.split_molecules_fragments(cluster_indices)

            time_split = time.time()
            print('Time to fragment ', time_split- time_separate)

            n_components, component_list = sparse.csgraph.connected_components(matrix)

        self.n_components = n_components
        self.component_list = component_list

    def separate_cluster(self, cluster_indices):

        slab_indices, mixture_indices = segment_slab_mixture(self.atoms, cluster_indices, slab_config=self.slab_config)
        
        if len(slab_indices)>0 and len(mixture_indices)>0:

            separate_ind = mixture_indices if len(mixture_indices)<len(slab_indices) else slab_indices

            nblist = neighborlist.NeighborList(self.cutoff[separate_ind],skin=0,self_interaction=False,bothways=True)

            sub_cluster = self.atoms[separate_ind]
            nblist.update(sub_cluster)
            sub_matrix = nblist.get_connectivity_matrix()

            self.matrix[separate_ind] = 0
            self.matrix[:,separate_ind] = 0
            indices = np.array(list(sub_matrix.keys()))
            self.matrix[separate_ind[indices[:,0]],separate_ind[indices[:,1]]] = 1

    def split_molecules_fragments(self, cluster_indices):

        fc = fragment_cluster(self.atoms, self.FragmentDB, self.cutoff, self.fragment_type)
        fragments, cluster_indices = fc.fragment(cluster_indices)
        
        #Change matrix only if cluster was able to be decomposed into fragments
        if len(cluster_indices) == 0:
            for fragment in fragments:
                #Remove previous connectivity
                self.matrix[fragment] = 0
                self.matrix[:,fragment] = 0
                cluster = self.atoms[fragment]
                cluster_nblist = neighborlist.NeighborList(self.cutoff[fragment],skin=0,self_interaction=False,bothways=True)
                cluster_nblist.update(cluster)
                cluster_matrix = cluster_nblist.get_connectivity_matrix()

                #Add connectivity of the fragment
                for src, dst in cluster_matrix.keys():
                    self.matrix[fragment[src],fragment[dst]] = 1


def segment_slab_mixture(atoms, cluster_indices, slab_config=None):

        cluster = atoms[cluster_indices]

        lithium_ind = np.argwhere(cluster.get_atomic_numbers()==3).flatten()
        if len(lithium_ind)>5 and slab_config is not None:

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

class fragment_cluster():
    def __init__(
        self,
        atoms,
        FragmentDB,
        cutoff,
        type: str = 'iterative',
        
    ) -> None:

        self.type = type
        self.function = getattr(self,type)
        self.atoms=atoms
        self.FragmentDB = FragmentDB
        self.cutoff = cutoff

    def fragment(self, cluster_indices):

        fragments = self.function(cluster_indices)
        return fragments

    def iterative(self, cluster_indices):
        fragments_li, cluster_indices = self.fragment_lithium(cluster_indices)

        fragments_cf, cluster_indices = self.fragment_clusters_fragments(cluster_indices)

        fragments = fragments_li + fragments_cf
        
        if len(cluster_indices)>0:
            cluster = self.atoms[cluster_indices]

            max_iterations = len(cluster)

            cluster_matrix = self.get_adjacency_matrix(cluster_indices)

            #designate seed fragments
            potential_fragments = np.argwhere(cluster_matrix.sum(axis=0)==cluster_matrix.sum(axis=0).min())
            
            valid_fragments = []
            for i in range(max_iterations-1):
            
                potential_fragments = self.expand_fragments(cluster,cluster_indices, cluster_matrix, potential_fragments)

                if len(potential_fragments) == 0:
                    break
                else:
                    for frag in potential_fragments:
                        potential_cluster = cluster[frag]
                        if self.FragmentDB.is_valid_fragment(potential_cluster) or self.FragmentDB.is_valid_cluster(potential_cluster):
                            valid_fragments.append(cluster_indices[frag])

            # decide what fragments to keep
            
            valid_fragments = valid_fragments[::-1] #reverse order
            # print(valid_fragments)
            for i, start_fragment in enumerate(valid_fragments):
                test_indices = start_fragment
                add_indices = [i]
                for j, add_fragment in enumerate(valid_fragments[i+1:]):
                    if len(np.intersect1d(test_indices,add_fragment))==0:
                        test_indices = np.concatenate([test_indices,add_fragment])
                        add_indices.append(i+j+1)
                    
                    if len(test_indices) == len(cluster_indices):
                        fragments += [valid_fragments[ind] for ind in add_indices]
                        return fragments, np.array([],dtype=int)

        return fragments, cluster_indices

    def expand_fragments(self,cluster,cluster_indices,matrix,input_fragments):

        #expand input fragment
        expanded_fragments = []
        for frag in input_fragments:
            connections = np.argwhere(matrix[frag])[:,1]
            for con in connections:
                if con not in frag:
                    expanded_fragments.append(np.concatenate([frag,[con]]))
            

        if len(expanded_fragments)==0:
            return []
        else:
            expanded_fragments = np.array(expanded_fragments)
            expanded_fragments.sort(axis=1)
            expanded_fragments.sort(axis=0)
            
            #Remove duplicates
            mask = np.concatenate([[True],~np.all(expanded_fragments[1:]==expanded_fragments[:-1],axis=1)])
            output_fragments = expanded_fragments[mask]

            return output_fragments

    def fragment_lithium(self, cluster_indices):
        cluster = self.atoms[cluster_indices]
        lithium_indices = np.argwhere(cluster.get_atomic_numbers()==3).flatten()
            
        fragments = [np.array([ind]) for ind in cluster_indices[lithium_indices] if self.FragmentDB.is_valid_cluster(self.atoms[[ind]])]

        cluster_indices = np.setdiff1d(cluster_indices,np.array(fragments).flatten())

        return fragments, cluster_indices

    def fragment_clusters_fragments(self,cluster_indices):
        cluster = self.atoms[cluster_indices]
        # cutoff = np.array(neighborlist.natural_cutoffs(cluster))
        if self.FragmentDB.is_valid_cluster(cluster):
            fragments = [cluster_indices]
            cluster_indices = np.array([],dtype=int)
        elif self.FragmentDB.is_valid_fragment(cluster):
            fragments = [cluster_indices]
            cluster_indices = np.array([],dtype=int)
        else:
            fragments = []

        return fragments, cluster_indices

    def get_adjacency_matrix(self,cluster_indices):
        cluster = self.atoms[cluster_indices]
        cluster_nblist = neighborlist.NeighborList(self.cutoff[cluster_indices],skin=0,self_interaction=False,bothways=True)
        cluster_nblist.update(cluster)
        cluster_matrix = cluster_nblist.get_connectivity_matrix().asformat("array")

        return cluster_matrix

    def spectral(self, cluster_indices):
        
        fragments_li, cluster_indices = self.fragment_lithium(cluster_indices)

        fragments_cf, cluster_indices = self.fragment_clusters_fragments(cluster_indices)

        fragments = fragments_li + fragments_cf
        
        if len(cluster_indices)>0:
            cluster = self.atoms[cluster_indices]

            max_iterations = len(cluster)
            
            adjacency_matrix = self.get_adjacency_matrix(cluster_indices)
            degree_matrix = np.identity(len(cluster_indices))*adjacency_matrix.sum(axis=0)

            laplacian_matrix = degree_matrix-adjacency_matrix

            # eigenvalues and eigenvectors
            vals, vecs = np.linalg.eig(laplacian_matrix)

            # sort these based on the eigenvalues
            vecs = vecs[:,np.argsort(vals)]
            vals = vals[np.argsort(vals)]

            # kmeans on eigenvectors
            for n_clusters in range(2,len(cluster)):
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(vecs[:,1:n_clusters])
                fragments_f = []
                for j in range(n_clusters):
                    fragment = cluster[kmeans.labels_==j]
                    fragment_indices = cluster_indices[kmeans.labels_==j]
                    if (self.FragmentDB.is_valid_cluster(fragment) or 
                        self.FragmentDB.is_valid_fragment(fragment)):
                        fragments_f.append(fragment_indices)
                if len(fragments_f) == n_clusters:
                    fragments += fragments_f
                    return fragments, np.array([],dtype=int)
                elif len(fragments_f)>0:
                    fragments += [fragments_f[0]]
                    return fragments, np.setdiff1d(cluster_indices,fragments_f[0])

        return fragments, cluster_indices

    def spectral_iterative(self, cluster_indices):

        fragments_total = []
        for _ in range(len(cluster_indices)):
            fragments, cluster_indices = self.spectral(cluster_indices)
            fragments_total += fragments
            if len(cluster_indices) == 0:
                break

        return fragments_total, cluster_indices