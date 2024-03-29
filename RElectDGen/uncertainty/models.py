from functools import partial
import json
import sys
import os
import yaml
import time
import pickle
from ase import Atoms
import torch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as stats
import matplotlib.pyplot as plt
import multiprocessing as mp
from ase.io import Trajectory
from typing import Dict, List

from nequip.data import AtomicData, dataset_from_config, DataLoader, ASEDataset
from nequip.data.transforms import TypeMapper

from . import optimization_functions
from .optimization_functions import uncertainty_NN, uncertaintydistance_NN, uncertainty_ensemble_NN, train_NN, find_NLL_params, uncertainty_GPR, uncertainty_reg_NN, find_NLL_params_prefactor
from .utils import load_from_hdf5, save_to_hdf5

class uncertainty_base():
    def __init__(self, model, config, MLP_config):
        self.model = model
        self.config = config
        self.MLP_config = MLP_config
        self.r_max = MLP_config['r_max']
        self.latent_size = int(self.MLP_config['conv_to_output_hidden_irreps_out'].split('x')[0])

        self.chemical_symbol_to_type = MLP_config.get('chemical_symbol_to_type')
        self.transform = TypeMapper(chemical_symbol_to_type=self.chemical_symbol_to_type)

        self.self_interaction = self.MLP_config.get('dataset_extra_fixed_fields',{}).get('self_interaction',False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.uncertainty_dir = os.path.join(self.MLP_config['workdir'],self.config.get('uncertainty_dir', 'uncertainty'))
        os.makedirs(self.uncertainty_dir,exist_ok=True)

        self.parsed_data_filename = os.path.join(self.uncertainty_dir,self.config.get('parsed_data_filename','parsed_data.npz'))

        self.natoms = len(MLP_config['type_names'])
        self.kb = 8.6173e-5 #eV/K
        self.n_ensemble = config.get('n_uncertainty_ensembles',4)

    def transform_data_input(self,data):

        if isinstance(data, Atoms):
            data = AtomicData.from_ase(atoms=data,r_max=self.r_max, self_interaction=self.self_interaction)
        elif isinstance(data, AtomicData):
            pass
        elif isinstance(data,dict):
            return data
        else:
            raise ValueError('Data type not supported')

        data = self.transform(data)
        data.to(torch.device(self.device))
        data = AtomicData.to_AtomicDataDict(data)
        
        return data
    
    def load_parsed_data(self):

        if not os.path.isfile(self.parsed_data_filename):
            return False

        all_data = load_from_hdf5(self.parsed_data_filename)
        
        for rk in self.parse_keys:
            if rk not in all_data.keys():
                return False
            else:
                setattr(self,rk,all_data[rk].to(self.device))

        self.train_errors = torch.linalg.norm(self.train_component_errors,axis=-1).to(self.device)
        self.test_errors = torch.linalg.norm(self.test_component_errors,axis=-1).to(self.device)

        self.all_embeddings = torch.cat([self.train_embeddings,self.test_embeddings]).to(self.device)
        self.all_component_errors = torch.cat([self.train_component_errors,self.test_component_errors]).to(self.device)
        self.all_errors = torch.cat([self.train_errors,self.test_errors]).to(self.device)

        return True

    def calibrate(self, debug=False):
        pass

    def save_parsed_data(self):

        data = {}
        for key in self.parse_keys:
            data[key] = getattr(self,key)

        save_to_hdf5(self.parsed_data_filename,data)

    def parse_data(self):

        self.parse_keys = [
            'train_energies', 
            'train_embeddings', 
            'train_component_errors',
            'test_embeddings',
            'test_energies',
            'test_component_errors'
        ]

        success = self.load_parsed_data()

        if not success:

            dataset = dataset_from_config(self.MLP_config)

            train_embeddings = torch.empty((0,self.latent_size),device=self.device)
            train_component_errors = torch.empty((0),device=self.device)
            train_energies = torch.empty((0),device=self.device)
            for data in dataset[self.MLP_config.train_idcs]:
                out = self.model(self.transform_data_input(data))
                train_energies = torch.cat([train_energies, out['atomic_energy'].mean().detach().unsqueeze(0)])
                train_embeddings = torch.cat([train_embeddings,out['node_features'].detach()])
                
                error = out['forces'] - data.forces
                train_component_errors = torch.cat([train_component_errors,error.detach()])

            self.train_energies = train_energies
            self.train_embeddings = train_embeddings
            self.train_component_errors = train_component_errors
            self.train_errors = torch.linalg.norm(train_component_errors,axis=-1)

            test_embeddings = torch.empty((0,self.latent_size),device=self.device)
            test_component_errors = torch.empty((0),device=self.device)
            test_energies = torch.empty((0),device=self.device)
            for data in dataset[self.MLP_config.val_idcs]:
                out = self.model(self.transform_data_input(data))
                test_energies = torch.cat([test_energies, out['atomic_energy'].mean().detach().unsqueeze(0)])
                test_embeddings = torch.cat([test_embeddings,out['node_features'].detach()])
                
                error = torch.absolute(out['forces'] - data.forces)
                test_component_errors = torch.cat([test_component_errors,error.detach()])
                
            
            self.test_embeddings = test_embeddings
            self.test_energies = test_energies
            self.test_component_errors = test_component_errors
            self.test_errors = torch.linalg.norm(test_component_errors,axis=-1)

            self.all_embeddings = torch.cat([train_embeddings,test_embeddings])
            self.all_component_errors = torch.cat([self.train_component_errors,self.test_component_errors])
            self.all_errors = torch.cat([self.train_errors,self.test_errors])

            self.save_parsed_data()

    def parse_UQ_data(self):
        UQ_config = self.config.get('UQ_config')
        assert UQ_config is not None
        
        UQ_dataset = dataset_from_config(UQ_config)
        UQ_indices = (np.random.permutation(len(UQ_dataset))[:UQ_config.get('nsamples')]).tolist()
        UQ_config['indices'] = UQ_indices

        UQ_config_filename = os.path.join(self.uncertainty_dir,self.config.get('UQ_config_filename','UQ_config.yaml'))
        with open(UQ_config_filename,'w') as fl:
            yaml.dump(UQ_config,fl)

        UQ_embeddings = torch.empty((0,self.latent_size),device=self.device)
        UQ_component_errors = torch.empty((0),device=self.device)
        UQ_energies = torch.empty((0),device=self.device)
        for data in UQ_dataset[UQ_indices]:
            out = self.model(self.transform_data_input(data))
            UQ_energies = torch.cat([UQ_energies, out['atomic_energy'].mean().detach().unsqueeze(0)])
            UQ_embeddings = torch.cat([UQ_embeddings,out['node_features'].detach()])
            
            error = out['forces'] - data.forces
            UQ_component_errors = torch.cat([UQ_component_errors,error.detach()])

        self.UQ_embeddings = UQ_embeddings
        self.UQ_energies = UQ_energies
        self.UQ_component_errors = UQ_component_errors
        self.UQ_errors = torch.linalg.norm(UQ_component_errors,axis=-1)
        

    def adversarial_loss(self, data, T, distances='train'):

        out = self.model(self.transform_data_input(data))
        self.atom_embedding = out['node_features']

        if distances == 'train_val':
            energies = torch.cat([self.train_energies, self.test_energies])
        else:
            energies = self.train_energies

        emean = energies.mean()
        estd = max([energies.std(),1]) # Only allow contraction

        kT = self.kb * T
        Q = torch.exp(-(energies-emean)/estd/kT).sum()

        probability = 1/Q * torch.exp(-(out['atomic_energy'].mean()-emean)/estd/kT)
        
        out = self.predict_uncertainty(data)
        uncertainties = out['uncertainties'].to(self.device).sum(axis=-1)
        adv_loss = (probability * uncertainties).sum()

        return adv_loss

    def predict_uncertainty(self,data):
        self.atom_embedding = None
        self.atom_forces = None
        self.atoms_energy = None
        
        return data

    def predict_from_traj(self, traj, max=True, flat=False, batch_size=1):
        uncertainty = []
        atom_embeddings = []
        self.pred_forces = []
        self.pred_energies = []
        # data = [self.transform(atoms) for atoms in traj]
        # dataset = DataLoader(data, batch_size=batch_size)
        # for i, batch in enumerate(dataset):
        #     uncertainty.append(self.predict_uncertainty(batch))
        #     atom_embeddings.append(self.atom_embedding)
        ti = time.time()
        times = np.empty(len(traj))
        for i, atoms in enumerate(traj):
            out = self.predict_uncertainty(atoms)
            uncertainty.append(out['uncertainty'].detach())
            atom_embeddings.append(self.atom_embedding.detach())
            self.pred_forces.append(self.atom_forces.detach())
            self.pred_energies.append(self.atoms_energy.detach())
            tf = time.time()
            times[i] = tf-ti
            ti = tf

        print(f'Dataset Prediction Times:\nAverage Time: {times.mean()}, Max Time: {times.max()}, Min Time: {times.min()}')
        
        uncertainty = torch.cat(uncertainty).detach().cpu()
        atom_embeddings = torch.cat(atom_embeddings).detach().cpu()

        if max:
            atom_lengths = [len(atoms) for atoms in traj]
            start_inds = [0] + np.cumsum(atom_lengths[:-1]).tolist()
            end_inds = np.cumsum(atom_lengths).tolist()

            uncertainty_partition = [uncertainty[si:ei] for si, ei in zip(start_inds,end_inds)]
            embeddings = [atom_embeddings[si:ei] for si, ei in zip(start_inds,end_inds)]
            return torch.tensor([unc.max() for unc in uncertainty_partition]), embeddings
        elif flat:
            atom_lengths = [len(atoms) for atoms in traj]
            start_inds = [0] + np.cumsum(atom_lengths[:-1]).tolist()
            end_inds = np.cumsum(atom_lengths).tolist()

            uncertainty_partition = [uncertainty[si:ei] for si, ei in zip(start_inds,end_inds)]
            embeddings = [atom_embeddings[si:ei] for si, ei in zip(start_inds,end_inds)]
            return uncertainty_partition, embeddings
        else:
            self.pred_forces = torch.cat(self.pred_forces).cpu()
            self.pred_forces = self.pred_forces.reshape(len(traj),-1,3)
            self.pred_energies = torch.cat(self.pred_energies).cpu()
            uncertainty = uncertainty.reshape(len(traj),-1,2)
            return uncertainty, atom_embeddings.reshape(len(traj),-1,atom_embeddings.shape[-1])
        

class Nequip_unc_oracle(uncertainty_base):
    def __init__(self, model, config, MLP_config):
        super().__init__(model, config, MLP_config)

    def calibrate(self, debug=False):
        pass

    def adversarial_loss(self, data, T, distances='train'):
        pass

    def predict_uncertainty(self, data, atom_embedding=None, distances='train', extra_embeddings=None, type='full'):
        if isinstance(data,Atoms):
            true_forces = torch.tensor(data.get_forces(), device=self.device)
        elif isinstance(data,(AtomicData,dict)):
            true_forces = data['forces']
        else:
            raise TypeError(f'Data must be either Atoms or Atomic Data, but got {data.__class__}')
        
        data = self.transform_data_input(data)

        out = self.model(data)
        atom_embedding = out['node_features']
        self.atom_embedding = atom_embedding
        self.atom_forces = out['forces']
        self.atoms_energy = out['total_energy']
        
        uncertainties = torch.zeros(atom_embedding.shape[0],2, device=self.device)

        uncertainties[:,0] = torch.linalg.norm(true_forces-self.atom_forces,axis=-1)
        out['uncertainties'] = uncertainties
        return out

class Nequip_latent_distance(uncertainty_base):
    def __init__(self, model, config, MLP_config):
        super().__init__(model, config, MLP_config)

        func_name = config.get('params_func','optimize2params')
        self.params_func = getattr(optimization_functions,func_name)
        self.dist_name = config.get('UQ_distribution_type','norm')
        self.params_func = partial(self.params_func,dist=self.dist_name)
        self.parameter_length = 2 if func_name=='optimize2params' else self.latent_size+1
        params_file = config.get('params_file','uncertainty_params.pkl')
        self.params_file = os.path.join(self.uncertainty_dir,params_file)
        self.separate_unc = self.config.get('separate_unc',False)

    def load_parsed_data(self):

        if not os.path.isfile(self.parsed_data_filename):
            return False

        all_data = load_from_hdf5(self.parsed_data_filename)
        
        for rk in self.parse_keys:
            if rk not in all_data.keys():
                return False
            else:
                setattr(self,rk,all_data[rk])

        
        return True

    def parse_data(self):
        self.parse_keys = [
            'train_embeddings',
            'train_errors',
            'train_energies',
            'train_total_energy_errors',
            'test_embeddings',
            'test_energies',
            'test_errors',
            'test_total_energy_errors',
        ]
        success = self.load_parsed_data()

        if not success:
            dataset = dataset_from_config(self.MLP_config)

            train_embeddings = {}
            train_errors = {}
            train_energies = {}
            test_embeddings = {}
            test_errors = {}
            test_energies = {}
            train_total_energy_errors = torch.empty((0),device=self.device)
            test_total_energy_errors = torch.empty((0),device=self.device)

            for key in self.chemical_symbol_to_type:
                train_embeddings[key] = torch.empty((0,self.latent_size),device=self.device)
                train_errors[key] = torch.empty((0),device=self.device)
                train_energies[key] = torch.empty((0),device=self.device)
                test_embeddings[key] = torch.empty((0,self.latent_size),device=self.device)
                test_errors[key] = torch.empty((0),device=self.device)
                test_energies[key] = torch.empty((0),device=self.device)
        
            for data in dataset[self.MLP_config.train_idcs]:
                out = self.model(self.transform_data_input(data))
                error = torch.absolute(out['forces'] - data.forces)
                train_total_energy_errors = torch.cat([train_total_energy_errors,(data['total_energy'].squeeze()-out['total_energy'].detach().squeeze()).unsqueeze(0)])

                for key in self.MLP_config.get('chemical_symbol_to_type'):
                    mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                    train_embeddings[key] = torch.cat([train_embeddings[key],out['node_features'][mask].detach()])
                    train_energies[key] = torch.cat([train_energies[key], out['atomic_energy'][mask].detach()])
                    if self.separate_unc:
                        train_errors[key] = torch.cat([train_errors[key],error[mask].detach()])
                    else:
                        train_errors[key] = torch.cat([train_errors[key],error.mean(dim=1)[mask].detach()])

            self.train_embeddings = train_embeddings
            self.train_energies = train_energies
            self.train_errors = train_errors
            self.train_total_energy_errors = train_total_energy_errors

            for data in dataset[self.MLP_config.val_idcs]:
                out = self.model(self.transform_data_input(data))
                error = torch.absolute(out['forces'] - data.forces)
                test_total_energy_errors = torch.cat([test_total_energy_errors,(data['total_energy'].squeeze()-out['total_energy'].detach().squeeze()).unsqueeze(0)])

                for key in self.MLP_config.get('chemical_symbol_to_type'):
                    mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                    test_embeddings[key] = torch.cat([test_embeddings[key],out['node_features'][mask].detach()])
                    test_energies[key] = torch.cat([test_energies[key], out['atomic_energy'][mask].detach()])
                    if self.separate_unc:
                        test_errors[key] = torch.cat([test_errors[key],error[mask].detach()])
                    else:
                        test_errors[key] = torch.cat([test_errors[key],error.mean(dim=1)[mask].detach()])
            
            self.test_embeddings = test_embeddings
            self.test_energies = test_energies
            self.test_errors = test_errors
            self.test_total_energy_errors = test_total_energy_errors

            self.save_parsed_data()

    def calibrate(self, debug=False, print_params=True):
        
        fail = self.load_params(print_params)

        self.parse_data()

        if fail or debug:
            latent_distances = {}
            min_distances = {}
            min_vectors = {}
            params = {}
            for key in self.MLP_config.get('chemical_symbol_to_type'):
                latent_distances[key] = torch.cdist(self.train_embeddings[key],self.test_embeddings[key],p=2)
                inds = torch.argmin(latent_distances[key],axis=0)
                min_distances[key] = torch.tensor([latent_distances[key][ind,i] for i, ind in enumerate(inds)]).detach().cpu().numpy()

                min_vectors[key] = np.abs(torch.vstack([self.train_embeddings[key][ind]-self.test_embeddings[key][i] for i, ind in enumerate(inds)]).detach().cpu().numpy())

                params[key] = []
                for _ in range(self.n_ensemble):
                    params[key].append(self.params_func(self.test_errors[key].squeeze().detach().cpu().numpy(),min_vectors[key]))

            self.params = params
            self.save_params()
            self.copy_params_to_tensor(params)
            if debug:  
                self.min_distances = min_distances
                self.min_vectors = min_vectors

    def copy_params_to_tensor(self,params):
        self.params = {}
        for key, val in params.items():
            self.params[key] = []
            for i, p in enumerate(val):
                self.params[key].append(torch.tensor(p,device=self.device).abs())

    def load_params(self,print_params=True):
        fail = True

        if os.path.isfile(self.params_file):
            with open(self.params_file,'rb') as fl:
                params = pickle.load(fl)

            key_pass = []
            for key in self.MLP_config.get('chemical_symbol_to_type'):
                if len(params[key])==self.n_ensemble:
                    parameter_lengths = [len(par)==self.parameter_length for par in params[key]]
                    key_pass.append(np.all(parameter_lengths))

            fail = not np.all(key_pass)

        if not fail:
            self.copy_params_to_tensor(params)

        return fail
                        
    def save_params(self):
        if hasattr(self,'params'):
            with open(self.params_file,'wb') as fl:
                pickle.dump(self.params, fl)


    def adversarial_loss(self, data, T, distances='train'):

        data = self.transform_data_input(data)
        
        out = self.predict_uncertainty(data, distances=distances)

        self.uncertainties = out['uncertainties'].to(self.device)

        adv_loss = 0
        for key in self.chemical_symbol_to_type:
            if distances == 'train_val':
                energies = torch.cat([self.train_energies[key], self.test_energies[key]])
            else:
                energies = self.train_energies[key]
            
            emean = energies.mean()
            estd = max([energies.std(),1]) # Only allow contraction

            kT = self.kb * T
            Q = torch.exp(-(energies-emean)/estd/kT).sum()

            mask = data['atom_types'] == self.chemical_symbol_to_type[key]
            probability = 1/Q * torch.exp(-(out['atomic_energy'][mask]-emean)/estd/kT)
            
            adv_loss += (probability * self.uncertainties[mask.flatten()].sum(dim=-1)).sum()

        return adv_loss

    def predict_uncertainty(self, data_in, atom_embedding=None, distances='train', extra_embeddings=None, type='full'):
        
        data = self.transform_data_input(data_in)

        out = self.model(data)
        atom_embedding = out['node_features']
        self.atom_embedding = atom_embedding
        self.atom_forces = out['forces']
        self.atoms_energy = out['total_energy']
    
        atom_types = data['atom_types']

        embeddings = {}
        for key in self.MLP_config.get('chemical_symbol_to_type'):
            if distances == 'train_val':
                embeddings[key] = torch.cat([self.train_embeddings[key],self.test_embeddings[key]],device=self.device)
            elif distances == 'train':
                embeddings[key] = self.train_embeddings[key].to(self.device)
            if extra_embeddings is not None:
                tmp = extra_embeddings[key].to(device=self.device)
                embeddings[key] = torch.cat([embeddings[key],tmp])


        uncertainties = predict_distance_uncertainty(out,self.params,embeddings,self.MLP_config.get('chemical_symbol_to_type'),self.n_ensemble,self.device)
        out['uncertainties'] = uncertainties
        return out

    
    # def predict_uncertainty(self, data_in, atom_embedding=None, distances='train', extra_embeddings=None, type='full'):
        
    #     data = self.transform_data_input(data_in)

    #     out = self.model(data)
    #     atom_embedding = out['node_features']
    #     self.atom_embedding = atom_embedding
    #     self.atom_forces = out['forces']
    #     self.atoms_energy = out['total_energy']
    
    #     atom_types = data['atom_types']
    #     uncertainties = torch.zeros(atom_embedding.shape[0],2, device=self.device)

    #     # self.test_distances = {}
    #     self.min_vectors = {}
    #     for key in self.MLP_config.get('chemical_symbol_to_type'):
    #         if distances == 'train_val':
    #             embeddings = torch.cat([self.train_embeddings[key],self.test_embeddings[key]],device=self.device)
    #         elif distances == 'train':
    #             embeddings = self.train_embeddings[key]

    #         embeddings = embeddings.to(device=self.device)

    #         if extra_embeddings is not None:
    #             tmp = extra_embeddings[key].to(device=self.device)
    #             embeddings = torch.cat([embeddings,tmp])

    #         mask = (atom_types==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()

    #         if torch.any(mask):
                
    #             latent_force_distances = torch.cdist(embeddings,atom_embedding[mask],p=2)

    #             inds = torch.argmin(latent_force_distances,axis=0)
            
    #             # min_distance = torch.tensor([latent_force_distances[ind,i] for i, ind in enumerate(inds)])
    #             min_vectors = torch.vstack([embeddings[ind]-atom_embedding[mask][i] for i, ind in enumerate(inds)])

    #             # self.test_distances[key] = min_distance.detach().cpu().numpy()
    #             self.min_vectors[key] = min_vectors
            
    #             uncertainties[mask] = self.uncertainty_from_vector(min_vectors, key, type=type)
    #     out['uncertainties'] = uncertainties
    #     return out
    
    # def uncertainty_from_vector(self, vector:torch.Tensor, key, type='full'):
    #     if len(self.params[key]) == 2:
    #         distance = torch.linalg.norm(vector,axis=1).reshape(-1,1)
    #     else:
    #         distance = vector.abs()

        
    #     if not self.separate_unc:
    #         uncertainty_raw = torch.zeros(self.n_ensemble,distance.shape[0], device=self.device)
    #         for i in range(self.n_ensemble):
    #             sig_1 = self.params[key][i][0].type_as(distance)
    #             sig_2 = self.params[key][i][1:].type_as(distance)
                
    #             if type == 'full':
    #                 uncertainty = sig_1 + torch.sum(distance*sig_2,axis=1)
    #             elif type == 'std':
    #                 uncertainty = torch.sum(distance*sig_2,axis=1)

    #             uncertainty_raw[i] = uncertainty
    #         uncertainties_mean = torch.mean(uncertainty_raw,axis=0)
    #         if uncertainty_raw.shape[0] == 1:
    #             uncertainties_std = torch.zeros_like(uncertainties_mean)
    #         else:
    #             uncertainties_std = torch.std(uncertainty_raw,axis=0)

    #         uncertainty = torch.vstack([uncertainties_mean,uncertainties_std]).T
    #         uncertainty *= self.config.get('uncertainty_factor',1) ### for NLL to choose confidence level
    #     else:
    #         uncertainty_sep = torch.zeros(3,distance.shape[0],2, device=self.device)
    #         for j in range(3):
    #             uncertainty_raw = torch.zeros(self.n_ensemble,distance.shape[0], device=self.device)
    #             for i in range(self.n_ensemble):
    #                 sig_1 = torch.tensor(self.params[key][i][0]).abs().type_as(distance)
    #                 sig_2 = torch.tensor(self.params[key][i][1:]).abs().type_as(distance)
                    
    #                 if type == 'full':
    #                     uncertainty = sig_1 + torch.sum(distance*sig_2,axis=1)
    #                 elif type == 'std':
    #                     uncertainty = torch.sum(distance*sig_2,axis=1)

    #                 uncertainty_raw[i] = uncertainty
    #             uncertainties_mean = torch.mean(uncertainty_raw,axis=0)
    #             if uncertainty_raw.shape[0] == 1:
    #                 uncertainties_std = torch.zeros_like(uncertainties_mean)
    #             else:
    #                 uncertainties_std = torch.std(uncertainty_raw,axis=0)

    #             uncertainty = torch.vstack([uncertainties_mean,uncertainties_std]).T
    #             uncertainty *= self.config.get('uncertainty_factor',1) ### for NLL to choose confidence level
    #             uncertainty_sep[j] = uncertainty
            
    #         max_index = torch.max(uncertainty_sep.sum(dim=-1),dim=0).indices
    #         uncertainty = uncertainty_sep[max_index]
    #     return uncertainty

@torch.jit.script
def distance_uncertainty_from_distance(
        distance: torch.Tensor, 
        params: Dict[str,List[torch.Tensor]],
        chemical_symbol: str,
        n_ensemble: int,
        uncertainty_factor: int = 1,
    ):
    uncertainty_raw = torch.zeros(n_ensemble,distance.shape[0], device=distance.device)
    for i in range(n_ensemble):
        sig_1 = params[chemical_symbol][i][0].type_as(distance)
        sig_2 = params[chemical_symbol][i][1:].type_as(distance)
        
        uncertainty = sig_1 + torch.sum(distance*sig_2,dim=1)
        
        uncertainty_raw[i] = uncertainty
    uncertainties_mean = torch.mean(uncertainty_raw,dim=0)
    
    if n_ensemble == 1:
        uncertainties_std = torch.zeros_like(uncertainties_mean)
    else:
        uncertainties_std = torch.std(uncertainty_raw,dim=0)

    uncertainty = torch.vstack([uncertainties_mean,uncertainties_std]).T
    uncertainty *= uncertainty_factor ### for NLL to choose confidence level
    
    return uncertainty

def predict_distance_uncertainty(
        out: Dict[str,torch.Tensor],
        params: Dict[str,List[torch.Tensor]],
        embeddings: Dict[str,torch.Tensor],
        chemical_symbol_to_type: Dict[str, int],
        n_ensemble: int,
        device: str,
    ):
    uncertainties = torch.zeros(out['node_features'].shape[0],2, device=device)
    atom_embedding = out['node_features']
    # self.test_distances = {}
    # self.min_vectors = {}
    for chemical_symbol, atom_type in chemical_symbol_to_type.items():
        embeddingsi = embeddings[chemical_symbol]

        mask = (out['atom_types']==atom_type).flatten()

        if torch.any(mask):
            
            latent_force_distances = torch.cdist(embeddingsi,atom_embedding[mask],p=2.)

            inds = torch.argmin(latent_force_distances,dim=0)
        
            # min_distance = torch.tensor([latent_force_distances[ind,i] for i, ind in enumerate(inds)])
            # min_vectors = torch.vstack([embeddingsi[ind]-atom_embedding[mask][i] for i, ind in enumerate(inds)]).abs()
            min_vectors = (embeddingsi[inds]-atom_embedding[mask]).abs()

            # self.test_distances[key] = min_distance.detach().cpu().numpy()
            # min_vectors[chemical_symbol] = min_vectorsi
        
            uncertainties[mask] = distance_uncertainty_from_distance(min_vectors, params, chemical_symbol, n_ensemble)

    return uncertainties


class Nequip_error_NN(uncertainty_base):
    def __init__(self, model, config, MLP_config):
        super().__init__(model, config, MLP_config)

        self.hidden_dimensions = self.config.get('uncertainty_hidden_dimensions', [])
        self.unc_epochs = self.config.get('uncertainty_epochs', 2000)
        self.unc_data = self.config.get('uncertainty_data','all')

        self.state_dict_func = lambda n: os.path.join(self.uncertainty_dir, f'uncertainty_state_dict_{n}.pth')
        self.metrics_func = lambda n: os.path.join(self.uncertainty_dir, f'uncertainty_metrics_{n}.csv')
        self.separate_unc = self.config.get('separate_unc',False)

    def load_NNs(self):
        self.NNs = []
        train_indices = []
        for n in range(self.n_ensemble):
            state_dict_name = self.state_dict_func(n)
            if os.path.isfile(state_dict_name):
                NN = uncertainty_NN(self.latent_size, self.hidden_dimensions)
                try:
                    NN.load_state_dict(torch.load(state_dict_name))
                    self.NNs.append(NN)
                except:
                    print(f'Loading uncertainty {n} failed')
                    train_indices.append(n)
            else:
                train_indices.append(n)

        return train_indices

    def calibrate(self, debug = False):
        
        train_indices = self.load_NNs()
        self.parse_data()

        if len(train_indices)>0:
            
            for n in train_indices:    
                #train NN to fit energies
                NN = uncertainty_NN(self.latent_size, self.hidden_dimensions,epochs=self.unc_epochs)
                if self.unc_data == 'all':
                    NN.train(self.all_embeddings, self.all_errors)
                else:
                    NN.train(self.test_embeddings, self.test_errors)
                self.NNs.append(NN)
                torch.save(NN.get_state_dict(), self.state_dict_func(n))
                pd.DataFrame(NN.metrics).to_csv( self.metrics_func(n))

    def parse_data(self):

        dataset = dataset_from_config(self.MLP_config)

        train_embeddings = torch.empty((0,self.latent_size),device=self.device)
        train_errors = torch.empty((0),device=self.device)
        train_energies = torch.empty((0),device=self.device)
        for data in dataset[self.MLP_config.train_idcs]:
            out = self.model(self.transform_data_input(data))
            train_energies = torch.cat([train_energies, out['atomic_energy'].mean().detach().unsqueeze(0)])
            train_embeddings = torch.cat([train_embeddings,out['node_features'].detach()])
            
            error = torch.absolute(out['forces'] - data.forces)
            if not self.separate_unc:
                train_errors = torch.cat([train_errors,error.mean(dim=1).detach()])
            else:
                train_errors = torch.cat([train_errors,error.detach()])

        self.train_energies = train_energies
        self.train_embeddings = train_embeddings
        self.train_errors = train_errors

        test_embeddings = torch.empty((0,self.latent_size),device=self.device)
        test_errors = torch.empty((0),device=self.device)
        test_energies = torch.empty((0),device=self.device)
        for data in dataset[self.MLP_config.val_idcs]:
            out = self.model(self.transform_data_input(data))
            test_energies = torch.cat([test_energies, out['atomic_energy'].mean().detach().unsqueeze(0)])

            error = torch.absolute(out['forces'] - data.forces)
            test_embeddings = torch.cat([test_embeddings,out['node_features'].detach()])
            if not self.separate_unc:
                test_errors = torch.cat([test_errors,error.mean(dim=1).detach()])
            else:
                test_errors = torch.cat([test_errors,error.detach()])
        
        self.test_embeddings = test_embeddings
        self.test_errors = test_errors
        self.test_energies = test_energies

        self.all_embeddings = torch.cat([train_embeddings,test_embeddings])
        self.all_errors = torch.cat([train_errors,test_errors])

    def adversarial_loss(self, data, T, distances='train'):

        out = self.predict_uncertainty(data).to(self.device)
        uncertainties = out['uncertainties'].to(self.device).sum(axis=-1)
        self.atom_embedding = out['node_features']

        if distances == 'train_val':
            energies = torch.cat([self.train_energies, self.test_energies])
        else:
            energies = self.train_energies

        emean = energies.mean()
        estd = max([energies.std(),1]) # Only allow contraction

        kT = self.kb * T
        Q = torch.exp(-(energies-emean)/estd/kT).sum()

        probability = 1/Q * torch.exp(-(out['atomic_energy'].mean()-emean)/estd/kT)
        
        adv_loss = (probability * uncertainties).sum()

        return adv_loss

    def predict_uncertainty(self, data, atom_embedding=None, distances='train', extra_embeddings=None,type='full'):

        if atom_embedding is None:
            data = self.transform_data_input(data)
        
            out = self.model(data)
            atom_embedding = out['node_features']
            self.atom_embedding = atom_embedding
            self.atom_forces = out['forces']
            self.atoms_energy = out['total_energy']

        uncertainty_raw = torch.zeros(self.n_ensemble,atom_embedding.shape[0])
        for i, NN in enumerate(self.NNs):
            uncertainty_raw[i] = NN.predict(atom_embedding).squeeze()
        
        uncertainties_mean = torch.mean(uncertainty_raw,axis=0)
        if uncertainty_raw.shape[0] == 1:
            uncertainties_std = torch.zeros_like(uncertainties_mean)
        else:
            uncertainties_std = torch.std(uncertainty_raw,axis=0)

        if type == 'full':
            # uncertainty_ens = uncertainty_mean + uncertainty_std
            uncertainty = torch.vstack([uncertainties_mean,uncertainties_std]).T.to(self.device)
        elif type == 'mean':
            # uncertainty_ens = uncertainty_mean
            uncertainty = torch.vstack([torch.zeros_like(uncertainties_mean),uncertainties_std]).T.to(self.device)
        elif type == 'std':
            # uncertainty_ens = uncertainty_std
            uncertainty = torch.vstack([uncertainties_mean,torch.zeros_like(uncertainties_std)]).T.to(self.device)

        out['uncertainties'] = uncertainty
        return out

class Nequip_reg_error_NN(uncertainty_base):
    def __init__(self, model, config, MLP_config):
        super().__init__(model, config, MLP_config)

        self.hidden_dimensions = self.config.get('uncertainty_hidden_dimensions', [])
        self.unc_epochs = self.config.get('uncertainty_epochs', 2000)
        self.unc_data = self.config.get('uncertainty_data','all')

        self.state_dict_func = lambda n: os.path.join(self.uncertainty_dir, f'reg_error_NN_state_dict_{n}.pth')
        self.metrics_func = lambda n: os.path.join(self.uncertainty_dir, f'reg_error_NN_metrics_{n}.csv')
        self.separate_unc = self.config.get('separate_unc',False)

    def load_NNs(self):
        self.NNs = []
        train_indices = []
        for n in range(self.n_ensemble):
            state_dict_name = self.state_dict_func(n)
            if os.path.isfile(state_dict_name):
                NN = uncertainty_reg_NN(self.latent_size, self.hidden_dimensions)
                try:
                    NN.load_state_dict(torch.load(state_dict_name))
                    self.NNs.append(NN)
                except:
                    print(f'Loading uncertainty {n} failed')
                    train_indices.append(n)
            else:
                train_indices.append(n)

        return train_indices

    def calibrate(self, debug = False):
        
        train_indices = self.load_NNs()
        self.parse_data()

        if len(train_indices)>0:
            
            for n in train_indices:    
                #train NN to fit energies
                NN = uncertainty_reg_NN(self.latent_size, self.hidden_dimensions,epochs=self.unc_epochs)
                if self.unc_data == 'all':
                    NN.train(self.all_embeddings, self.all_component_errors)
                else:
                    NN.train(self.test_embeddings, self.test_component_errors)
                self.NNs.append(NN)
                torch.save(NN.get_state_dict(), self.state_dict_func(n))
                pd.DataFrame(NN.metrics).to_csv( self.metrics_func(n))

    def predict_uncertainty(self, data, atom_embedding=None, distances='train', extra_embeddings=None,type='full'):

        if atom_embedding is None:
            data = self.transform_data_input(data)
        
            out = self.model(data)
            atom_embedding = out['node_features']
            self.atom_embedding = atom_embedding
            self.atom_forces = out['forces']
            self.atoms_energy = out['total_energy']

        uncertainty_raw = torch.zeros(self.n_ensemble,atom_embedding.shape[0])
        for i, NN in enumerate(self.NNs):
            uncertainty_raw[i] = NN.predict(atom_embedding).squeeze()
        
        uncertainties_mean = torch.mean(uncertainty_raw,axis=0)
        if uncertainty_raw.shape[0] == 1:
            uncertainties_std = torch.zeros_like(uncertainties_mean)
        else:
            uncertainties_std = torch.std(uncertainty_raw,axis=0)

        if type == 'full':
            # uncertainty_ens = uncertainty_mean + uncertainty_std
            uncertainty = torch.vstack([uncertainties_mean,uncertainties_std]).T.to(self.device)
        elif type == 'mean':
            # uncertainty_ens = uncertainty_mean
            uncertainty = torch.vstack([torch.zeros_like(uncertainties_mean),uncertainties_std]).T.to(self.device)
        elif type == 'std':
            # uncertainty_ens = uncertainty_std
            uncertainty = torch.vstack([uncertainties_mean,torch.zeros_like(uncertainties_std)]).T.to(self.device)
        out['uncertainties'] = uncertainty
        return out


class Nequip_latent_distanceNN(uncertainty_base):
    def __init__(self, model, config, MLP_config):
        super().__init__(model, config, MLP_config)

        self.hidden_dimensions = self.config.get('uncertainty_hidden_dimensions', [])
        self.unc_epochs = self.config.get('uncertainty_epochs', 2000)

        self.state_dict_func = lambda n: os.path.join(self.uncertainty_dir, f'uncertainty_state_dict_{n}.pth')
        self.metrics_func = lambda n: os.path.join(self.uncertainty_dir, f'uncertainty_metrics_{n}.csv')
        
    def parse_data(self):

        dataset = dataset_from_config(self.MLP_config)

        train_embeddings = torch.empty((0,self.latent_size),device=self.device)
        train_errors = torch.empty((0),device=self.device)
        train_energies = torch.empty((0),device=self.device)
        for data in dataset[self.MLP_config.train_idcs]:
            out = self.model(self.transform_data_input(data))
            train_energies = torch.cat([train_energies, out['atomic_energy'].mean().detach().unsqueeze(0)])
            train_embeddings = torch.cat([train_embeddings,out['node_features'].detach()])
            
            error = torch.absolute(out['forces'] - data.forces)
            train_errors = torch.cat([train_errors,error.mean(dim=1)])

        self.train_energies = train_energies
        self.train_embeddings = train_embeddings
        self.train_errors = train_errors

        test_embeddings = torch.empty((0,self.latent_size),device=self.device)
        test_errors = torch.empty((0),device=self.device)
        test_energies = torch.empty((0),device=self.device)
        for data in dataset[self.MLP_config.val_idcs]:
            out = self.model(self.transform_data_input(data))
            test_energies = torch.cat([test_energies, out['atomic_energy'].mean().detach().unsqueeze(0)])

            error = torch.absolute(out['forces'] - data.forces)

            test_embeddings = torch.cat([test_embeddings,out['node_features'].detach()])
            test_errors = torch.cat([test_errors,error.mean(dim=1).detach()])
        
        self.test_embeddings = test_embeddings
        self.test_errors = test_errors
        self.test_energies = test_energies

    def load_NNs(self):
        self.NNs = []
        train_indices = []
        for n in range(self.n_ensemble):
            state_dict_name = self.state_dict_func(n)
            if os.path.isfile(state_dict_name):
                NN = uncertaintydistance_NN(self.latent_size, self.hidden_dimensions)
                try:
                    NN.load_state_dict(torch.load(state_dict_name))
                    self.NNs.append(NN)
                except:
                    print(f'Loading uncertainty {n} failed')
                    train_indices.append(n)
            else:
                train_indices.append(n)

        return train_indices

    def calibrate(self, debug=False):
        
        train_indices = self.load_NNs()

        self.parse_data()

        if len(train_indices)>0:
            latent_distances = torch.cdist(self.train_embeddings,self.test_embeddings,p=2)
            inds = torch.argmin(latent_distances,axis=0)
            min_distances = torch.tensor([latent_distances[ind,i] for i, ind in enumerate(inds)]).detach().cpu().numpy()

            min_vectors = np.abs(torch.vstack([self.train_embeddings[ind]-self.test_embeddings[i] for i, ind in enumerate(inds)]).detach().cpu().numpy())

            for n in train_indices:    
                #train NN to fit energies
                NN = uncertaintydistance_NN(self.latent_size, self.hidden_dimensions,epochs=self.unc_epochs)

                
                NN.train(min_vectors, self.test_errors)
                self.NNs.append(NN)
                torch.save(NN.get_state_dict(), self.state_dict_func(n))
                pd.DataFrame(NN.metrics).to_csv( self.metrics_func(n))
                        

    def adversarial_loss(self, data, T, distances='train'):

        data = self.transform_data_input(data)
        
        out = self.predict_uncertainty(data, distances=distances)
        atom_embedding = out['node_features']
        self.atom_embedding = atom_embedding

        self.uncertainties = out['uncertainties'].to(self.device)

        adv_loss = 0
        if distances == 'train_val':
            energies = torch.cat([self.train_energies, self.test_energies])
        else:
            energies = self.train_energies
        
        emean = energies.mean()
        estd = max([energies.std(),1]) # Only allow contraction

        kT = self.kb * T
        Q = torch.exp(-(energies-emean)/estd/kT).sum()

        # mask = data['atom_types'] == self.chemical_symbol_to_type
        probability = 1/Q * torch.exp(-(out['atomic_energy']-emean)/estd/kT)
        
        adv_loss += (probability * self.uncertainties.sum(dim=-1)).sum()

        return adv_loss

    def predict_uncertainty(self, data_in, atom_embedding=None, distances='train', extra_embeddings=None, type='full'):
        
        data = self.transform_data_input(data_in)

        if atom_embedding is None:
            out = self.model(data)
            atom_embedding = out['node_features']
            self.atom_embedding = atom_embedding
        else:
            atom_embedding = atom_embedding.to(device=torch.device(self.device))

        if distances == 'train_val':
            embeddings = torch.cat([self.train_embeddings,self.test_embeddings])
        elif distances == 'train':
            embeddings = self.train_embeddings
            
        atom_types = data['atom_types']
        uncertainties = torch.zeros(atom_embedding.shape[0],2, device=self.device)

        latent_distances = torch.cdist(embeddings,atom_embedding,p=2)
        inds = torch.argmin(latent_distances,axis=0)

        # min_vectors = np.abs(torch.vstack([self.train_embeddings[ind]-self.test_embeddings[i] for i, ind in enumerate(inds)]).detach().cpu().numpy())
        min_vectors = torch.vstack([embeddings[ind]-atom_embedding[i] for i, ind in enumerate(inds)]).abs()

        uncertainties = self.uncertainty_from_vector(min_vectors, type=type)

        # self.test_distances = {}
        # self.min_vectors = {}
        # for key in self.MLP_config.get('chemical_symbol_to_type'):
        #     if distances == 'train_val':
        #         embeddings = torch.cat([self.train_embeddings[key],self.test_embeddings[key]])
        #     elif distances == 'train':
        #         embeddings = self.train_embeddings[key]

        #     if extra_embeddings is not None:
        #         embeddings = torch.cat([embeddings,extra_embeddings[key]])

        #     embeddings.to(device=self.device)
        #     mask = (atom_types==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()

        #     if torch.any(mask):
                
        #         latent_force_distances = torch.cdist(embeddings,atom_embedding[mask],p=2)

        #         inds = torch.argmin(latent_force_distances,axis=0)
            
        #         min_distance = torch.tensor([latent_force_distances[ind,i] for i, ind in enumerate(inds)])
        #         min_vectors = torch.vstack([embeddings[ind]-atom_embedding[mask][i] for i, ind in enumerate(inds)])

        #         self.test_distances[key] = min_distance.detach().cpu().numpy()
        #         self.min_vectors[key] = min_vectors.detach().cpu().numpy()
            
        #         uncertainties[mask] = self.uncertainty_from_vector(min_vectors, key, type=type)
        out['uncertainties'] = uncertainties
        return out

    def uncertainty_from_vector(self, vector, type='full'):

        uncertainty_raw = torch.zeros(self.n_ensemble,vector.shape[0], device=self.device)
        for i, NN in enumerate(self.NNs):
            uncertainty_raw[i] = NN.predict(vector).squeeze()
        uncertainty_mean = torch.mean(uncertainty_raw,axis=0)
        uncertainty_std = torch.std(uncertainty_raw,axis=0)

        uncertainty = torch.vstack([uncertainty_mean,uncertainty_std]).T
        # uncertainty_ens = uncertainty_mean + uncertainty_std

        return uncertainty

class Nequip_ensemble_NN(uncertainty_base):
    def __init__(self, model, config, MLP_config):
        super().__init__(model, config, MLP_config)

        # self.nequip_model = model
        self.hidden_dimensions = self.config.get('uncertainty_hidden_dimensions', [])
        self.unc_epochs = self.config.get('uncertainty_epochs', 2000)
        self.unc_batch_size = self.config.get('uncertainty_batch_size', 100)
        self.optimization_function = config.get('optimization_function','uncertainty_ensemble_NN')
        self.natoms = len(MLP_config['type_names'])
        
        self.state_dict_func = lambda n: os.path.join(self.uncertainty_dir, f'uncertainty_state_dict_{n}.pth')
        self.metrics_func = lambda n: os.path.join(self.uncertainty_dir, f'uncertainty_metrics_{n}.csv')
        

    def load_NNs(self):
        self.NNs = []
        train_indices = []
        for n in range(self.n_ensemble):
            state_dict_name = self.state_dict_func(n)
            if os.path.isfile(state_dict_name):
                unc_func = getattr(optimization_functions,self.optimization_function)
                NN = unc_func(self.latent_size, self.natoms, self.hidden_dimensions)
                try:
                    NN.load_state_dict(torch.load(state_dict_name, map_location=self.device))
                    self.NNs.append(NN)
                except Exception as e:
                    print(e)
                    print(f'Loading uncertainty {n} failed', flush=True)
                    train_indices.append(n)
            else:
                train_indices.append(n)

        return train_indices

    def calibrate(self, debug = False):
        
        train_indices = self.load_NNs()
        self.parse_data()

        if len(train_indices)>0:
            NNs_trained = []
            ncores = self.config.get('train_NN_instances',1)
            uncertainty_training = self.config.get('uncertainty_training','energy')
            if uncertainty_training=='energy':
                train_energy_forces = self.train_energies
                validation_energy_forces = self.validation_energies
            elif uncertainty_training=='forces':
                train_energy_forces = self.train_forces
                validation_energy_forces = self.validation_forces
            
            ncores = min(ncores,len(train_indices))
            ncores = 1 #multiprocessing doesn't work yet I think its a slurm issue
            unc_func = getattr(optimization_functions,self.optimization_function)  
            if ncores == 1:
                for n in train_indices:
                    print('training ensemble network ', n, flush=True)  
                    NN = unc_func(self.latent_size, self.natoms, self.hidden_dimensions, epochs=self.unc_epochs,batch_size=self.unc_batch_size)
                    NN = train_NN((NN, uncertainty_training,self.train_embeddings,train_energy_forces,self.validation_embeddings,validation_energy_forces,None,None))
                    NNs_trained.append(NN)
            elif ncores >1:
            
                NNs_init = [unc_func(self.latent_size, self.natoms, self.hidden_dimensions, epochs=self.unc_epochs,batch_size=self.unc_batch_size) for n in train_indices]
                gen = ((NN,uncertainty_training,self.train_embeddings,train_energy_forces,self.validation_embeddings,validation_energy_forces,None,None) for NN in NNs_init)
                
                def produce(semaphore, generator):
                    for gen in generator:
                        semaphore.acquire()
                        yield gen

                semaphore = mp.Semaphore(ncores)
                with mp.Pool(ncores) as pool:
                    for NN in pool.imap_unordered(train_NN,produce(semaphore,gen)):
                        NNs_trained.append(NN)
                        semaphore.release()

            print('done training')
            print('Save NNs')
            for n, NN in zip(train_indices,NNs_trained):
                self.NNs.append(NN)
                print('Best loss ', NN.best_loss, flush=True)
                torch.save(NN.get_state_dict(), self.state_dict_func(n))
                pd.DataFrame(NN.metrics).to_csv(self.metrics_func(n))

    def fine_tune(self, embeddings, energies_or_forces):
        print('Fine Tuning Ensemble', flush=True)
        uncertainty_training = self.config.get('uncertainty_training','energy')
        if uncertainty_training=='energy':
            train_energy_forces = self.train_energies
            validation_energy_forces = self.validation_energies
        elif uncertainty_training=='forces':
            train_energy_forces = self.train_forces
            validation_energy_forces = self.validation_forces
        ncores = self.config.get('train_NN_instances',1)
        ncores = 1 #multiprocessing doesn't work yet
        if ncores == 1:
            for NN in self.NNs:
                train_NN((NN, uncertainty_training,self.train_embeddings,train_energy_forces,self.validation_embeddings,validation_energy_forces,embeddings,energies_or_forces))
        elif ncores >1:
            ncores = min(ncores,len(self.NNs))

            gen = ((NN,uncertainty_training,self.train_embeddings,train_energy_forces,self.validation_embeddings,validation_energy_forces,embeddings,energies_or_forces) for NN in self.NNs)
            NNs_tuned = []
            def produce(semaphore, generator):
                for gen in generator:
                    semaphore.acquire()
                    yield gen

            semaphore = mp.Semaphore(ncores)
            with mp.Pool(ncores) as pool:
                for NN in pool.imap_unordered(train_NN,produce(semaphore,gen)):
                    NNs_tuned.append(NN)
                    semaphore.release()

            self.NNs = NNs_tuned
        
    def parse_data(self):
        dataset = dataset_from_config(self.MLP_config)

        self.ML_train_indices = torch.tensor(self.MLP_config.train_idcs, dtype=int,device=self.device)
        self.UQ_train_indices = torch.empty((0), dtype= int,device=self.device)
        self.ML_validation_indices = torch.tensor(self.MLP_config.val_idcs, dtype=int,device=self.device)
        self.UQ_validation_indices = torch.empty((0),dtype= int,device=self.device)
        self.UQ_test_indices = torch.empty((0),dtype= int,device=self.device)

        self.train_dataset = dataset[self.MLP_config.train_idcs]
        self.validation_dataset = dataset[self.MLP_config.val_idcs]

        train_embeddings = {}
        train_energies = {}
        train_forces = {}
        train_indices = {}
        validation_embeddings = {}
        validation_energies = {}
        validation_forces = {}
        validation_indices = {}
        test_embeddings = {}
        test_energies = {}
        test_forces = {}
        test_indices = {}

        for key in self.chemical_symbol_to_type:
            train_embeddings[key] = torch.empty((0,self.latent_size+self.natoms),device=self.device)
            train_energies[key] = torch.empty((0),device=self.device)
            train_forces[key] = torch.empty((0),device=self.device)
            train_indices[key] = torch.empty(0,dtype=int).to(self.device)

            validation_embeddings[key] = torch.empty((0,self.latent_size+self.natoms),device=self.device)
            validation_energies[key] = torch.empty((0),device=self.device)
            validation_forces[key] = torch.empty((0),device=self.device)
            validation_indices[key] = torch.empty(0,dtype=int).to(self.device)

            test_embeddings[key] = torch.empty((0,self.latent_size+self.natoms),device=self.device)
            test_energies[key] = torch.empty((0),device=self.device)
            test_forces[key] = torch.empty((0),device=self.device)
            test_indices[key] = torch.empty(0,dtype=int).to(self.device)
        
        error_threshold=self.config.get('UQ_dataset_error', np.inf)
        for i, data in enumerate(self.train_dataset):
            out = self.model(self.transform_data_input(data))

            force_norm = data['forces'].norm(dim=1).unsqueeze(dim=1)
            force_lim = torch.max(force_norm,torch.ones_like(force_norm))
            perc_err = ((out['forces'].detach()-data['forces'])).abs()/force_lim
            
            if perc_err.max() < error_threshold:
                self.UQ_train_indices = torch.cat([self.UQ_train_indices, self.ML_train_indices[i].unsqueeze(dim=0)])
                for key in self.MLP_config.get('chemical_symbol_to_type'):
                    mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                    
                    atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze()[mask],num_classes=self.natoms).to(self.device)
                    NN_inputs = torch.hstack([out['node_features'][mask].detach(), atom_one_hot])
                    
                    train_embeddings[key] = torch.cat([train_embeddings[key],NN_inputs])
                    train_energies[key] = torch.cat([train_energies[key], out['atomic_energy'][mask].detach()])
                    train_forces[key] = torch.cat([train_forces[key], data['forces'][mask].detach().norm(dim=1).unsqueeze(1)])

                    # npoints = torch.tensor([train_indices[key][-1]+sum(mask) if i>0 else sum(mask)]).to(self.device)
                    # train_indices[key] = torch.cat([train_indices[key],npoints]).to(self.device)
            else:
                self.UQ_test_indices = torch.cat([self.UQ_test_indices, self.ML_train_indices[i].unsqueeze(dim=0)])
                for key in self.MLP_config.get('chemical_symbol_to_type'):
                    mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                    
                    atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze()[mask],num_classes=self.natoms).to(self.device)
                    NN_inputs = torch.hstack([out['node_features'][mask].detach(), atom_one_hot])
                    
                    test_embeddings[key] = torch.cat([test_embeddings[key],NN_inputs])
                    test_energies[key] = torch.cat([test_energies[key], out['atomic_energy'][mask].detach()])
                    test_forces[key] = torch.cat([test_forces[key], data['forces'][mask].detach().norm(dim=1).unsqueeze(1)])

                    # npoints = torch.tensor([test_indices[key][-1]+sum(mask) if i>0 else sum(mask)]).to(self.device)
                    # test_indices[key] = torch.cat([test_indices[key],npoints]).to(self.device)

        self.train_embeddings = train_embeddings
        self.train_energies = train_energies
        self.train_forces = train_forces
        self.train_indices = train_indices

        for i, data in enumerate(self.validation_dataset):
            out = self.model(self.transform_data_input(data))

            force_norm = data['forces'].norm(dim=1).unsqueeze(dim=1)
            force_lim = torch.max(force_norm,torch.ones_like(force_norm))
            perc_err = ((out['forces'].detach()-data['forces'])).abs()/force_lim
            
            if perc_err.max() < error_threshold:
                self.UQ_validation_indices = torch.cat([self.UQ_validation_indices, self.ML_validation_indices[i].unsqueeze(dim=0)])
                for key in self.MLP_config.get('chemical_symbol_to_type'):
                    mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()

                    atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze()[mask],num_classes=self.natoms).to(self.device)
                    NN_inputs = torch.hstack([out['node_features'][mask].detach(), atom_one_hot])

                    validation_embeddings[key] = torch.cat([validation_embeddings[key],NN_inputs])
                    validation_energies[key] = torch.cat([validation_energies[key], out['atomic_energy'][mask].detach()])
                    validation_forces[key] = torch.cat([validation_forces[key], data['forces'][mask].detach().norm(dim=1).unsqueeze(1)])
                    
                    # npoints = torch.tensor([validation_indices[key][-1]+sum(mask) if i>0 else sum(mask)]).to(self.device)
                    # validation_indices[key] = torch.cat([validation_indices[key],npoints]).to(self.device)
            else:
                self.UQ_test_indices = torch.cat([self.UQ_test_indices, self.ML_validation_indices[i].unsqueeze(dim=0)])
                for key in self.MLP_config.get('chemical_symbol_to_type'):
                    mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()

                    atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze()[mask],num_classes=self.natoms).to(self.device)
                    NN_inputs = torch.hstack([out['node_features'][mask].detach(), atom_one_hot])

                    test_embeddings[key] = torch.cat([test_embeddings[key],NN_inputs])
                    test_energies[key] = torch.cat([test_energies[key], out['atomic_energy'][mask].detach()])
                    test_forces[key] = torch.cat([test_forces[key], data['forces'][mask].detach().norm(dim=1).unsqueeze(1)])
                    
                    # npoints = torch.tensor([test_indices[key][-1]+sum(mask) if i>0 else sum(mask)]).to(self.device)
                    # test_indices[key] = torch.cat([test_indices[key],npoints]).to(self.device)

        self.validation_embeddings = validation_embeddings
        self.validation_energies = validation_energies
        self.validation_forces = validation_forces
        self.validation_indices = validation_indices

        self.test_embeddings = test_embeddings
        self.test_energies = test_energies
        self.test_forces = test_forces
        self.test_indices = test_indices

    def adversarial_loss(self, data, T, distances='train'):

        data = self.transform_data_input(data)
        
        out = self.predict_uncertainty(data, distances=distances)
        atom_embedding = out['node_features']
        self.atom_embedding = atom_embedding

        self.uncertainties = out['uncertainties'].to(self.device)

        adv_loss = 0
        for key in self.chemical_symbol_to_type:
            if distances == 'train_val':
                energies = torch.cat([self.train_energies[key], self.test_energies[key]])
            else:
                energies = self.train_energies[key]
            
            emean = energies.mean()
            estd = max([energies.std(),1]) # Only allow contraction

            kT = self.kb * T
            Q = torch.exp(-(energies-emean)/estd/kT).sum()

            mask = data['atom_types'] == self.chemical_symbol_to_type[key]
            probability = 1/Q * torch.exp(-(out['atomic_energy'][mask]-emean)/estd/kT)
            
            adv_loss += (probability * self.uncertainties[mask.flatten()].sum(dim=-1)).sum()

        return adv_loss

    def predict_uncertainty(self, data, atom_embedding=None, distances='train', extra_embeddings=None,type='full'):

        
        data = self.transform_data_input(data)
        out = self.model(data)
        self.atom_embedding = out['node_features']

        self.pred_atom_energies = torch.zeros((self.n_ensemble,out['atomic_energy'].shape[0])).to(self.device)
        for i, NN in enumerate(self.NNs):
            self.pred_atom_energies[i] = NN.predict(out).squeeze()
        
        uncertainty_training = self.config.get('uncertainty_training','energy')
        if uncertainty_training=='energy':
            uncertainties_mean = (self.pred_atom_energies.mean(dim=0)-out['atomic_energy'].squeeze()).abs()
            norm = torch.max(torch.ones_like(out['atomic_energy']),out['atomic_energy'].abs())
        elif uncertainty_training=='forces':
            uncertainties_mean = (self.pred_atom_energies.mean(dim=0)-out['forces'].norm(dim=1)).abs()
            norm = torch.ones_like(out['atomic_energy']) #torch.max(torch.ones_like(out['atomic_energy']),out['forces'].norm(dim=1).unsqueeze(1))
        
        # uncertainties_mean = (pred_atom_energies-out['atomic_energy'].squeeze().unsqueeze(0)).abs().max(dim=0).values
        # uncertainty_mean = (pred_atom_energies.sum(dim=1)-out['total_energy'].squeeze()).abs().max(dim=0).values
        # uncertainties_mean = torch.ones(pred_atom_energies.shape[1]).to(self.device)*uncertainty_mean
        
        uncertainties_std = self.pred_atom_energies.std(axis=0)#.sum(axis=-1)
        # uncertainty_std = pred_atom_energies.sum(dim=1).std(axis=0)#.sum(axis=-1)
        # uncertainties_std = torch.ones(pred_atom_energies.shape[1]).to(self.device)*uncertainty_std

        uncertainty = torch.vstack([uncertainties_mean,uncertainties_std]).T/norm

        # uncertainty *= self.config.get('uncertainty_factor',10)
        out['uncertainties'] = uncertainty
        return out

class Nequip_ensemble(uncertainty_base):
    def __init__(self, model, config, MLP_config):
        super().__init__(model, config, MLP_config[0])

        self.config = config
        self.calibration_type = self.config.get('UQ_calibration_type','linear')
        self.dist_name = config.get('UQ_distribution_type','norm')

        self.separate_unc = self.config.get('separate_unc',False)

        if self.calibration_type == 'power':
            self.calibration_polyorder = 1
        elif self.calibration_type == 'prefactor':
            self.calibration_polyorder = 0
        else:
            self.calibration_polyorder = self.config.get('calibration_polyorder',1)
        calibration_coeffs = {}
        for key in self.MLP_config.get('chemical_symbol_to_type'):    
            if self.separate_unc:
                calibration_coeffs[key] = np.zeros((self.calibration_polyorder+1,3))
                if self.calibration_polyorder>0:
                    calibration_coeffs[key][-2,:] = 1
                if self.calibration_type == 'prefactor':
                    calibration_coeffs[key] = np.ones((self.calibration_polyorder+1,3))
            else:
                calibration_coeffs[key] = np.zeros(self.calibration_polyorder+1)
                if self.calibration_polyorder>0:
                    calibration_coeffs[key][-2] = 1
                if self.calibration_type == 'prefactor':
                    calibration_coeffs[key] = np.ones((self.calibration_polyorder+1))
        self.calibration_coeffs = calibration_coeffs

        
        if self.separate_unc:
            self.calibration_coeffs_filename =  os.path.join(self.uncertainty_dir, self.config.get('uncertainty_calibration_filename', f'uncertainty_calibration_coeffs_sep.json'))
        else:
            self.calibration_coeffs_filename =  os.path.join(self.uncertainty_dir, self.config.get('uncertainty_calibration_filename', f'uncertainty_calibration_coeffs.json'))

    def calibrate(self, debug = False):
        
        if os.path.isfile(self.calibration_coeffs_filename):
            with open(self.calibration_coeffs_filename,'r') as fl:
                data = json.load(fl)
            calibration_coeffs = {}
            for key in self.MLP_config.get('chemical_symbol_to_type'):   
                calibration_coeffs[key] = np.array(data[key])
        else:
            self.parse_validation_data()

            #Calibration curves
            calibration_coeffs = {'ave_var': {},'max_var': {}}
            for key in self.MLP_config.get('chemical_symbol_to_type'):   
                # print(self.validation_err_pred[key].shape) 
                # print(self.validation_err_real[key].shape)
                # print(self.validation_err_pred[key].cpu())
                # print(self.validation_err_real[key].cpu())
                if self.calibration_type == 'power':
                    calibration_coeffs[key] = np.polyfit(np.log(self.validation_err_pred[key].cpu()),np.log(self.validation_err_real[key].cpu()),self.calibration_polyorder)
                elif self.calibration_type == 'NLL':
                    calibration_coeffs[key] = find_NLL_params(self.validation_err_real[key].cpu(),self.validation_err_pred[key].cpu(),self.calibration_polyorder,self.dist_name)
                elif self.calibration_type == 'prefactor':
                    calibration_coeffs[key] = find_NLL_params_prefactor(self.validation_err_real[key].cpu(),self.validation_err_pred[key].cpu(),self.calibration_polyorder,self.dist_name)
                else:
                    calibration_coeffs[key] = np.polyfit(self.validation_err_pred[key].cpu(),self.validation_err_real[key].cpu(),self.calibration_polyorder)
                try:
                    calibration_coeffs['ave_var'][key] = float((self.validation_err_pred[key].cpu()).mean())
                    calibration_coeffs['max_var'][key] = float((self.validation_err_pred[key].cpu()).max())
                except Exception as e:
                    print(self.validation_err_pred[key].cpu())
                    print(self.validation_err_pred[key].cpu().shape)
                    print(e)
            data = {}
            for key in self.MLP_config.get('chemical_symbol_to_type'):
                data[key] = calibration_coeffs[key].tolist()

            with open(self.calibration_coeffs_filename,'w') as fl:
                json.dump(data,fl)
        self.calibration_coeffs = calibration_coeffs

    def fine_tune(self, embeddings, energies_or_forces):
        pass
        
    def parse_data(self):
        dataset = dataset_from_config(self.MLP_config)

        self.ML_train_indices = torch.tensor(self.MLP_config.train_idcs, dtype=int,device=self.device)
        self.UQ_train_indices = torch.empty((0), dtype= int,device=self.device)
        self.ML_validation_indices = torch.tensor(self.MLP_config.val_idcs, dtype=int,device=self.device)
        self.UQ_validation_indices = torch.empty((0),dtype= int,device=self.device)
        self.UQ_test_indices = torch.empty((0),dtype= int,device=self.device)

        self.train_dataset = dataset[self.MLP_config.train_idcs]
        self.validation_dataset = dataset[self.MLP_config.val_idcs]

        train_embeddings = {}
        train_energies = {}
        train_forces = {}
        train_indices = {}
        validation_embeddings = {}
        validation_energies = {}
        validation_forces = {}
        validation_indices = {}
        test_embeddings = {}
        test_energies = {}
        test_forces = {}
        test_indices = {}
        validation_err_pred = {}
        validation_err_real = {}

        for key in self.chemical_symbol_to_type:
            train_embeddings[key] = torch.empty((0,self.latent_size+self.natoms),device=self.device)
            train_energies[key] = torch.empty((0),device=self.device)
            train_forces[key] = torch.empty((0),device=self.device)
            train_indices[key] = torch.empty(0,dtype=int).to(self.device)

            validation_embeddings[key] = torch.empty((0,self.latent_size+self.natoms),device=self.device)
            validation_energies[key] = torch.empty((0),device=self.device)
            validation_forces[key] = torch.empty((0),device=self.device)
            validation_indices[key] = torch.empty(0,dtype=int).to(self.device)

            test_embeddings[key] = torch.empty((0,self.latent_size+self.natoms),device=self.device)
            test_energies[key] = torch.empty((0),device=self.device)
            test_forces[key] = torch.empty((0),device=self.device)
            test_indices[key] = torch.empty(0,dtype=int).to(self.device)
            validation_err_pred[key] = torch.empty(0).to(self.device)
            validation_err_real[key] = torch.empty(0).to(self.device)
        
        error_threshold=self.config.get('UQ_dataset_error', np.inf)
        for i, data in enumerate(self.train_dataset):
            force_outputs = torch.empty(len(self.model),*data['pos'].shape,device=self.device)
            atom_energies = torch.empty(len(self.model),len(data['pos']),device=self.device)
            for ii, model in enumerate(self.model):
                out = model(self.transform_data_input(data))
                force_outputs[ii] = out['forces']
                atom_energies[ii] = out['atomic_energy'].squeeze()

            force_norm = data['forces'].norm(dim=1).unsqueeze(dim=1)
            force_lim = torch.max(force_norm,torch.ones_like(force_norm,device=self.device))
            perc_err = ((force_outputs.detach().mean(dim=0)-data['forces'])).abs().cpu()/force_lim.cpu()
            
            if perc_err.max() < error_threshold:
                self.UQ_train_indices = torch.cat([self.UQ_train_indices, self.ML_train_indices[i].unsqueeze(dim=0)])
                for key in self.MLP_config.get('chemical_symbol_to_type'):
                    mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                    
                    atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze()[mask],num_classes=self.natoms).to(self.device)
                    NN_inputs = torch.hstack([out['node_features'][mask].detach(), atom_one_hot])
                    
                    train_embeddings[key] = torch.cat([train_embeddings[key],NN_inputs])
                    train_energies[key] = torch.cat([train_energies[key], atom_energies.mean(dim=0)[mask].detach()])
                    train_forces[key] = torch.cat([train_forces[key], data['forces'][mask].detach().norm(dim=1).unsqueeze(1)])

                    # npoints = torch.tensor([train_indices[key][-1]+sum(mask) if i>0 else sum(mask)]).to(self.device)
                    # train_indices[key] = torch.cat([train_indices[key],npoints]).to(self.device)
            else:
                self.UQ_test_indices = torch.cat([self.UQ_test_indices, self.ML_train_indices[i].unsqueeze(dim=0)])
                for key in self.MLP_config.get('chemical_symbol_to_type'):
                    mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                    
                    atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze()[mask],num_classes=self.natoms).to(self.device)
                    NN_inputs = torch.hstack([out['node_features'][mask].detach(), atom_one_hot])
                    
                    test_embeddings[key] = torch.cat([test_embeddings[key],NN_inputs])
                    test_energies[key] = torch.cat([test_energies[key], atom_energies.mean(dim=0)[mask].detach()])
                    test_forces[key] = torch.cat([test_forces[key], data['forces'][mask].detach().norm(dim=1).unsqueeze(1)])

                    # npoints = torch.tensor([test_indices[key][-1]+sum(mask) if i>0 else sum(mask)]).to(self.device)
                    # test_indices[key] = torch.cat([test_indices[key],npoints]).to(self.device)

        self.train_embeddings = train_embeddings
        self.train_energies = train_energies
        self.train_forces = train_forces
        self.train_indices = train_indices

        for i, data in enumerate(self.validation_dataset):
            force_outputs = torch.empty(len(self.model),*data['pos'].shape,device=self.device)
            atom_energies = torch.empty(len(self.model),len(data['pos']),device=self.device)
            out = self.predict_uncertainty(data,type='full')
            pred_uncertainty = out['uncertainties'].detach().sum(axis=-1)
            
            force_norm = data['forces'].norm(dim=1).unsqueeze(dim=1)
            force_lim = torch.max(force_norm,torch.ones_like(force_norm,device=self.device))
            perc_err = ((out['forces'].detach()-data['forces'])).abs().cpu()/force_lim.cpu()
            force_error = ((out['forces'].detach()-data['forces']))
            # print(force_error,flush=True)
            for key in self.MLP_config.get('chemical_symbol_to_type'):
                mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                validation_err_real[key] = torch.cat([validation_err_real[key],force_error[mask]])
                validation_err_pred[key] = torch.cat([validation_err_pred[key],pred_uncertainty[mask]])
            
            if perc_err.max() < error_threshold:
                self.UQ_validation_indices = torch.cat([self.UQ_validation_indices, self.ML_validation_indices[i].unsqueeze(dim=0)])
                for key in self.MLP_config.get('chemical_symbol_to_type'):
                    mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()

                    atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze()[mask],num_classes=self.natoms).to(self.device)
                    NN_inputs = torch.hstack([out['node_features'][mask].detach(), atom_one_hot])

                    validation_embeddings[key] = torch.cat([validation_embeddings[key],NN_inputs])
                    validation_energies[key] = torch.cat([validation_energies[key], atom_energies.mean(dim=0)[mask].detach()])
                    validation_forces[key] = torch.cat([validation_forces[key], data['forces'][mask].detach().norm(dim=1).unsqueeze(1)])
                    
                    # npoints = torch.tensor([validation_indices[key][-1]+sum(mask) if i>0 else sum(mask)]).to(self.device)
                    # validation_indices[key] = torch.cat([validation_indices[key],npoints]).to(self.device)
            else:
                self.UQ_test_indices = torch.cat([self.UQ_test_indices, self.ML_validation_indices[i].unsqueeze(dim=0)])
                for key in self.MLP_config.get('chemical_symbol_to_type'):
                    mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()

                    atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze()[mask],num_classes=self.natoms).to(self.device)
                    NN_inputs = torch.hstack([out['node_features'][mask].detach(), atom_one_hot])

                    test_embeddings[key] = torch.cat([test_embeddings[key],NN_inputs])
                    test_energies[key] = torch.cat([test_energies[key], atom_energies.mean(dim=0)[mask].detach()])
                    test_forces[key] = torch.cat([test_forces[key], data['forces'][mask].detach().norm(dim=1).unsqueeze(1)])
                    
                    # npoints = torch.tensor([test_indices[key][-1]+sum(mask) if i>0 else sum(mask)]).to(self.device)
                    # test_indices[key] = torch.cat([test_indices[key],npoints]).to(self.device)
        
        self.validation_err_real = validation_err_real
        self.validation_err_pred = validation_err_pred
        
        self.validation_embeddings = validation_embeddings
        self.validation_energies = validation_energies
        self.validation_forces = validation_forces
        self.validation_indices = validation_indices

        self.test_embeddings = test_embeddings
        self.test_energies = test_energies
        self.test_forces = test_forces
        self.test_indices = test_indices

    def parse_validation_data(self):
        dataset = dataset_from_config(self.MLP_config)

        self.ML_validation_indices = torch.tensor(self.MLP_config.val_idcs, dtype=int,device=self.device)
        self.UQ_validation_indices = torch.empty((0),dtype= int,device=self.device)

        self.validation_dataset = dataset[self.MLP_config.val_idcs]

        validation_embeddings = {}
        validation_energies = {}
        validation_forces = {}
        validation_indices = {}
        validation_err_pred = {}
        validation_err_real = {}

        for key in self.chemical_symbol_to_type:

            validation_embeddings[key] = torch.empty((0,self.latent_size+self.natoms),device=self.device)
            validation_energies[key] = torch.empty((0),device=self.device)
            validation_forces[key] = torch.empty((0),device=self.device)
            validation_indices[key] = torch.empty(0,dtype=int).to(self.device)

            validation_err_pred[key] = torch.empty(0).to(self.device)
            validation_err_real[key] = torch.empty(0).to(self.device)
        
        error_threshold=self.config.get('UQ_dataset_error', np.inf)

        for i, data in enumerate(self.validation_dataset):
            force_outputs = torch.empty(len(self.model),*data['pos'].shape,device=self.device)
            atom_energies = torch.empty(len(self.model),len(data['pos']),device=self.device)
            
            out = self.predict_uncertainty(data,type='full')
            pred_uncertainty = out['uncertainties'].detach()
            
            force_norm = data['forces'].norm(dim=1).unsqueeze(dim=1)
            force_lim = torch.max(force_norm,torch.ones_like(force_norm,device=self.device))
            perc_err = ((out['forces'].detach()-data['forces'])).abs().cpu()/force_lim.cpu()
            force_error = ((out['forces'].detach()-data['forces']))
            if not self.separate_unc:
                force_error = force_error.norm(dim=1)
            
            
            # print(force_error,flush=True)
            for key in self.MLP_config.get('chemical_symbol_to_type'):
                mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                validation_err_real[key] = torch.cat([validation_err_real[key],force_error[mask]])
                validation_err_pred[key] = torch.cat([validation_err_pred[key],pred_uncertainty[mask]])
            
            # if perc_err.max() < error_threshold:
            self.UQ_validation_indices = torch.cat([self.UQ_validation_indices, self.ML_validation_indices[i].unsqueeze(dim=0)])
            for key in self.MLP_config.get('chemical_symbol_to_type'):
                mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()

                atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze()[mask],num_classes=self.natoms).to(self.device)
                NN_inputs = torch.hstack([out['node_features'][mask].detach(), atom_one_hot])

                validation_embeddings[key] = torch.cat([validation_embeddings[key],NN_inputs])
                validation_energies[key] = torch.cat([validation_energies[key], atom_energies.mean(dim=0)[mask].detach()])
                validation_forces[key] = torch.cat([validation_forces[key], data['forces'][mask].detach().norm(dim=1).unsqueeze(1)])
                
                # npoints = torch.tensor([validation_indices[key][-1]+sum(mask) if i>0 else sum(mask)]).to(self.device)
                # validation_indices[key] = torch.cat([validation_indices[key],npoints]).to(self.device)
            
        
        self.validation_err_real = validation_err_real
        self.validation_err_pred = validation_err_pred
        
        self.validation_embeddings = validation_embeddings
        self.validation_energies = validation_energies
        self.validation_forces = validation_forces
        self.validation_indices = validation_indices
    
    def get_train_energies(self):
        dataset = dataset_from_config(self.MLP_config)

        self.train_dataset = dataset[self.MLP_config.train_idcs]

        train_energies = {}
        test_energies = {}
        
        for key in self.chemical_symbol_to_type:
            train_energies[key] = torch.empty((0),device=self.device)
            test_energies[key] = torch.empty((0),device=self.device)
        for i, data in enumerate(self.train_dataset):
            out = self.model[0](self.transform_data_input(data))
            atom_energies = out['atomic_energy'].squeeze()

            for key in self.MLP_config.get('chemical_symbol_to_type'):
                mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
        
                train_energies[key] = torch.cat([train_energies[key], atom_energies[mask].detach()])
                  
        self.train_energies = train_energies
        self.test_energies = test_energies

    def apply_calibration(self, atom_types, raw):

        calibrated = torch.zeros_like(raw,device=self.device)
        for key in self.chemical_symbol_to_type:
            mask = (atom_types==self.chemical_symbol_to_type[key]).flatten()
            # print(self.calibration_coeffs[key])
            if self.calibration_type == 'power':
                coeffs = torch.tensor(self.calibration_coeffs[key],device=self.device)
                calibrated[mask] = torch.exp(coeffs[1]+torch.log(raw[mask])*coeffs[0])
            elif self.calibration_type == 'prefactor':
                coeffs = torch.tensor(self.calibration_coeffs[key],device=self.device,dtype=raw.dtype)
                calibrated[mask] = coeffs*raw[mask]
            else:
                for i, coeff in enumerate(self.calibration_coeffs[key][::-1]):
                    coeffs = torch.tensor(coeff,device=self.device)
                    calibrated[mask] += coeffs*raw[mask].pow(i)

        return calibrated

    def get_base_uncertainty(self):
        base_unc = 0.
        for key in self.chemical_symbol_to_type:
            base_unc = max([self.calibration_coeffs[key][-1],base_unc])
        return base_unc

    def adversarial_loss(self, data, T, distances='train'):

        data = self.transform_data_input(data)

        out = self.predict_uncertainty(data, distances=distances).to(self.device)
        self.uncertainties = out['uncertainties']
        
        adv_loss = 0
        for key in self.chemical_symbol_to_type:
            if distances == 'train_val':
                energies = torch.cat([self.train_energies[key], self.test_energies[key]])
            else:
                energies = self.train_energies[key]
            
            emean = energies.mean()
            estd = max([energies.std(),1]) # Only allow contraction

            kT = self.kb * T
            Q = torch.exp(-(energies-emean)/estd/kT).sum()

            mask = data['atom_types'] == self.chemical_symbol_to_type[key]
            probability = 1/Q * torch.exp(-(self.atom_energies[mask.squeeze()]-emean)/estd/kT)
            
            adv_loss += (probability * self.uncertainties[mask.flatten()].sum(dim=-1)).sum()

        return adv_loss

    def predict_uncertainty(self, data, atom_embedding=None, distances='train', extra_embeddings=None,type='single'):

        
        data = self.transform_data_input(data)
        force_outputs = torch.empty(len(self.model),*data['pos'].shape,device=self.device)
        atom_energies = torch.empty(len(self.model),len(data['pos']),device=self.device)
        total_energies = torch.empty(len(self.model),1,device=self.device)
        for i, model in enumerate(self.model):
            out = model(data)
            force_outputs[i] = out['forces']
            atom_energies[i] = out['atomic_energy'].squeeze()
            total_energies[i] = out['total_energy'].squeeze()

        self.atom_forces = force_outputs.mean(dim=0)
        self.atom_energies = atom_energies.mean(dim=0)
        self.atom_embedding = out['node_features']
        self.atoms_energy = total_energies.mean(dim=0)
        
        out['forces'] = self.atom_forces
        out['atomic_energy'] = self.atom_energies
        out['total_energy'] = self.atoms_energy
        
        if self.separate_unc:
            uncertainties_std = force_outputs.std(axis=0)
        else:
            uncertainties_std = force_outputs.std(axis=0).norm(dim=-1)
        
        uncertainty = torch.zeros((uncertainties_std.shape[0],2),device=self.device)

        # uncertainty = torch.transpose(torch.stack([uncertainties_mean,uncertainties_std]),0,1).to(self.device)

        if type =='full': # and self.separate_unc
            return self.apply_calibration(out['atom_types'],uncertainties_std)
        if self.separate_unc:
            uncertainty[:,1] = self.apply_calibration(out['atom_types'],uncertainties_std).max(dim=-1).values
        else:
            uncertainty[:,1] = self.apply_calibration(out['atom_types'],uncertainties_std)
        
        out['uncertainties'] = uncertainty
        return out

class Nequip_error_GPR(uncertainty_base):
    def __init__(self, model, config, MLP_config):
        super().__init__(model, config, MLP_config)

        self.ninducing_points = self.config.get('GPR_ninducing_points', 100)
        self.unc_epochs = self.config.get('uncertainty_epochs', 1000)
        self.learning_rate = self.config.get('GPR_learning_rate',0.01)
        self.log_transform = self.config.get('GPR_log_transform',True)
        self.inducing_points_initialization = self.config.get('GPR_inducing_points_initialization','random')

        self.separate_unc = self.config.get('separate_unc',False)

        self.state_dict_filename = os.path.join(self.uncertainty_dir, f'uncertainty_GPR_state_dict.pth')
        self.metrics_filename_fn = lambda key: os.path.join(self.uncertainty_dir, f'uncertainty_GPR_metrics_{key}.csv')

    def load_GPR(self):
        load = False
        if os.path.isfile(self.state_dict_filename):
            self.GPR = {}
            state_dicts = torch.load(self.state_dict_filename)
            for key in self.MLP_config.get('chemical_symbol_to_type'):
                self.GPR[key] = uncertainty_GPR(self.latent_size,self.ninducing_points,log_transform=self.log_transform)
                try:
                    self.GPR[key].load_state_dict(state_dicts[key])
                    load = True
                except:
                    pass
        
        return load

    def load_parsed_data(self):

        if not os.path.isfile(self.parsed_data_filename):
            return False

        all_data = load_from_hdf5(self.parsed_data_filename)
        
        for rk in self.parse_keys:
            if rk not in all_data.keys():
                return False
            else:
                setattr(self,rk,all_data[rk])
        
        return True

    def calibrate(self, debug = False):
        
        load = self.load_GPR()
        self.parse_data()

        if not load:
            self.GPR = {}
            state_dicts = {}
            UQ_dataset_type = self.config.get('train_UQ_dataset','all')
            for key in self.MLP_config.get('chemical_symbol_to_type'):
                print(f'Calibrating {key}')
                self.GPR[key] = uncertainty_GPR(self.latent_size,self.ninducing_points,self.unc_epochs,lr=self.learning_rate,log_transform=self.log_transform,inducing_points_initialization=self.inducing_points_initialization)
                if self.config.get('train_UQ_different_dataset',False):
                    self.parse_UQ_data()
                    self.GPR.train(self.UQ_embeddings, self.UQ_errors)
                elif UQ_dataset_type =='train':
                    self.GPR[key].train(self.train_embeddings[key], self.train_errors[key])
                elif UQ_dataset_type =='test':
                    self.GPR[key].train(self.test_embeddings[key], self.test_errors[key])
                elif UQ_dataset_type =='all':
                    self.GPR[key].train(self.all_embeddings[key], self.all_errors[key])
                state_dicts[key] = self.GPR[key].get_state_dict()
                pd.DataFrame(self.GPR[key].metrics).to_csv(self.metrics_filename_fn(key))
            torch.save(state_dicts,self.state_dict_filename)

    def parse_data(self):
        self.parse_keys = [
            'train_embeddings',
            'train_errors',
            'train_energies',
            'test_embeddings',
            'test_energies',
            'test_errors'
        ]
        success = self.load_parsed_data()

        if not success:
            dataset = dataset_from_config(self.MLP_config)

            train_embeddings = {}
            train_errors = {}
            train_energies = {}
            test_embeddings = {}
            test_errors = {}
            test_energies = {}

            for key in self.chemical_symbol_to_type:
                train_embeddings[key] = torch.empty((0,self.latent_size),device=self.device)
                train_errors[key] = torch.empty((0),device=self.device)
                train_energies[key] = torch.empty((0),device=self.device)
                test_embeddings[key] = torch.empty((0,self.latent_size),device=self.device)
                test_errors[key] = torch.empty((0),device=self.device)
                test_energies[key] = torch.empty((0),device=self.device)
        
            for data in dataset[self.MLP_config.train_idcs]:
                out = self.model(self.transform_data_input(data))
                error = torch.absolute(out['forces'] - data.forces)

                for key in self.MLP_config.get('chemical_symbol_to_type'):
                    mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                    train_embeddings[key] = torch.cat([train_embeddings[key],out['node_features'][mask].detach()])
                    train_energies[key] = torch.cat([train_energies[key], out['atomic_energy'][mask].detach()])
                    if self.separate_unc:
                        train_errors[key] = torch.cat([train_errors[key],error[mask].detach()])
                    else:
                        train_errors[key] = torch.cat([train_errors[key],error.mean(dim=1)[mask].detach()])

            self.train_embeddings = train_embeddings
            self.train_energies = train_energies
            self.train_errors = train_errors

            for data in dataset[self.MLP_config.val_idcs]:
                out = self.model(self.transform_data_input(data))
                
                error = torch.absolute(out['forces'] - data.forces)

                for key in self.MLP_config.get('chemical_symbol_to_type'):
                    mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                    test_embeddings[key] = torch.cat([test_embeddings[key],out['node_features'][mask].detach()])
                    test_energies[key] = torch.cat([test_energies[key], out['atomic_energy'][mask].detach()])
                    if self.separate_unc:
                        test_errors[key] = torch.cat([test_errors[key],error[mask].detach()])
                    else:
                        test_errors[key] = torch.cat([test_errors[key],error.mean(dim=1)[mask].detach()])
            
            self.test_embeddings = test_embeddings
            self.test_energies = test_energies
            self.test_errors = test_errors

            self.save_parsed_data()
        
        self.all_embeddings = {}
        self.all_energies = {}
        self.all_errors = {}
        for key in self.MLP_config.get('chemical_symbol_to_type'):
            self.all_embeddings[key] = torch.cat([self.train_embeddings[key],self.test_embeddings[key]])
            self.all_energies[key] = torch.cat([self.train_energies[key],self.test_energies[key]])
            self.all_errors[key] = torch.cat([self.train_errors[key],self.test_errors[key]])

    def adversarial_loss(self, data, T, distances='train'):

        out = self.predict_uncertainty(data['atom_types'])
        self.atom_embedding = out['node_features']

        if distances == 'train_val':
            energies = torch.cat([self.train_energies, self.test_energies])
        else:
            energies = self.train_energies

        emean = energies.mean()
        estd = max([energies.std(),1]) # Only allow contraction

        kT = self.kb * T
        Q = torch.exp(-(energies-emean)/estd/kT).sum()

        probability = 1/Q * torch.exp(-(out['atomic_energy'].mean()-emean)/estd/kT)
        
        uncertainties = out['uncertainties'].to(self.device).sum(axis=-1)

        adv_loss = (probability * uncertainties).sum()

        return adv_loss

    def predict_uncertainty(self, data, atom_embedding=None, distances='train', extra_embeddings=None,type='full'):

        if atom_embedding is None:
            data = self.transform_data_input(data)
        
            out = self.model(data)
            atom_embedding = out['node_features']
            self.atom_embedding = atom_embedding
            self.atom_forces = out['forces']
            self.atoms_energy = out['total_energy']
        
        atom_types = data['atom_types']
        uncertainties = torch.zeros(atom_embedding.shape[0],2, device=self.device)
        for key in self.MLP_config.get('chemical_symbol_to_type'):
            mask = (atom_types==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
            mean, upper_confidence = self.GPR[key].predict(atom_embedding[mask])
            
            if type == 'full':
                uncertainties[mask,0] = mean
                uncertainties[mask,1] = upper_confidence-mean
                # uncertainty_ens = uncertainty_mean + uncertainty_std
                # uncertainty = torch.vstack([mean,upper_confidence-mean]).T.to(self.device)
            elif type == 'std':
                # uncertainty_ens = uncertainty_mean
                # uncertainty = torch.vstack([torch.zeros_like(mean),upper_confidence-mean]).T.to(self.device)
                uncertainties[mask,1] = upper_confidence-mean
            elif type == 'mean':
                uncertainties[mask,0] = mean
                # uncertainty_ens = uncertainty_std
                # uncertainty = torch.vstack([mean,torch.zeros_like(mean)]).T.to(self.device)

        out['uncertainties'] = uncertainties
        return out

class Nequip_error_pos_NN(uncertainty_base):
    def __init__(self, model, config, MLP_config):
        super().__init__(model, config, MLP_config)

        self.uncertainty_config = self.config.get('uncertainty_config',{})
        self.uncertainty_config['r_max'] = MLP_config['r_max']
        self.uncertainty_config['chemical_symbol_to_type'] = MLP_config['chemical_symbol_to_type']

        self.unc_epochs = self.config.get('uncertainty_epochs', 1000)
        
        self.optimization_function = config.get('optimization_function','uncertainty_pos_NN')
        
        default_state_dict_func = lambda n: f'uncertainty_pos_NN_state_dict_{n}.pth'
        self.state_dict_func = self.uncertainty_config.get('state_dict_func',default_state_dict_func)
        if isinstance(self.state_dict_func,str):
            self.state_dict_func = eval(self.state_dict_func)
        default_metrics_func = lambda n: f'uncertainty_pos_NN_metrics_{n}.csv'
        self.metrics_func = self.uncertainty_config.get('metrics_func',default_metrics_func)
        if isinstance(self.metrics_func,str):
            self.metrics_func = eval(self.metrics_func)

    def load_NNs(self):
        self.NNs = []
        train_indices = []
        for n in range(self.n_ensemble):
            state_dict_name = os.path.join(
                self.uncertainty_dir,
                self.state_dict_func(n)
            )
            if os.path.isfile(state_dict_name):
                unc_func = getattr(optimization_functions,self.optimization_function)
                NN = unc_func(
                    self.uncertainty_config,
                    self.unc_epochs
                )
                try:
                    NN.load_state_dict(torch.load(state_dict_name, map_location=self.device))
                    self.NNs.append(NN)
                except Exception as e:
                    print(e)
                    print(f'Loading uncertainty {n} failed', flush=True)
                    train_indices.append(n)
            else:
                train_indices.append(n)

        return train_indices

    def calibrate(self, debug = False):
        
        train_indices = self.load_NNs()

        if len(train_indices)>0:
            self.parse_validation_data()
            NNs_trained = []
            unc_func = getattr(optimization_functions,self.optimization_function)  
            
            for n in train_indices:
                print('training ensemble network ', n, flush=True)  
                NN = unc_func(
                    self.uncertainty_config,
                    self.unc_epochs
                )
                NN.train(self.UQ_dataset)
                NNs_trained.append(NN)
            
            print('done training')
            print('Save NNs')
            for n, NN in zip(train_indices,NNs_trained):
                self.NNs.append(NN)
                print('Best loss ', NN.validation_loss, flush=True)
                state_dict_name = os.path.join(
                    self.uncertainty_dir,
                    self.state_dict_func(n)
                )
                torch.save(NN.get_state_dict(), state_dict_name)
                metrics_name = os.path.join(
                    self.uncertainty_dir,
                    self.metrics_func(n)
                )
                pd.DataFrame(NN.metrics).to_csv(metrics_name)

    def parse_validation_data(self):

        traj = Trajectory(self.MLP_config['dataset_file_name'])
        val_traj = []

        for ind in self.MLP_config.val_idcs:
            atoms = traj[ind]
            out = self.model(self.transform_data_input(atoms))

            errors = np.linalg.norm(out['forces'].detach() - atoms.get_forces(),axis=1)

            atoms.arrays['errors'] = errors
            val_traj.append(atoms)
        
        self.UQ_dataset = ASEDataset.from_atoms_list(
            val_traj,
            extra_fixed_fields={'r_max':self.MLP_config['r_max']},
            include_keys=['errors']
        )

    def adversarial_loss(self, data, T, distances='train'):

        out = self.predict_uncertainty(data)
        self.atom_embedding = out['node_features']

        if distances == 'train_val':
            energies = torch.cat([self.train_energies, self.test_energies])
        else:
            energies = self.train_energies

        emean = energies.mean()
        estd = max([energies.std(),1]) # Only allow contraction

        kT = self.kb * T
        Q = torch.exp(-(energies-emean)/estd/kT).sum()

        probability = 1/Q * torch.exp(-(out['atomic_energy'].mean()-emean)/estd/kT)
        
        uncertainties = out['uncertainties'].to(self.device).sum(axis=-1)

        adv_loss = (probability * uncertainties).sum()

        return adv_loss

    def predict_uncertainty(self, data, atom_embedding=None, distances='train', extra_embeddings=None,type='full'):

        if atom_embedding is None:
            data = self.transform_data_input(data)
        
            out = self.model(data)
            atom_embedding = out['node_features']
            self.atom_embedding = atom_embedding
            self.atom_forces = out['forces']
            self.atoms_energy = out['total_energy']

        uncertainty_raw = torch.zeros(self.n_ensemble,len(data['pos']), device=self.device)
        for i, NN in enumerate(self.NNs):
            uncertainty_raw[i] = NN.predict(data).squeeze()
        
        uncertainty_mean = torch.mean(uncertainty_raw,axis=0)
        if self.n_ensemble>1:
            uncertainty_std = torch.std(uncertainty_raw,axis=0)
        else:
            uncertainty_std = torch.zeros_like(uncertainty_mean)

        uncertainty = torch.vstack([uncertainty_mean,uncertainty_std]).T

        out['uncertainties'] = uncertainty
        return out

def f(x):
    return x*x