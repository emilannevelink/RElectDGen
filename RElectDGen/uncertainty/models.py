from logging import raiseExceptions
import sys
import os
import pickle
from ase import Atoms
import torch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as stats
import matplotlib.pyplot as plt

from nequip.data import AtomicData, dataset_from_config, DataLoader
from nequip.data.transforms import TypeMapper

from . import optimization_functions
from .optimization_functions import uncertainty_NN

class uncertainty_base():
    def __init__(self, model, config, MLP_config):
        self.model = model
        self.config = config
        self.MLP_config = MLP_config
        self.r_max = MLP_config.r_max
        self.latent_size = int(self.MLP_config['conv_to_output_hidden_irreps_out'].split('x')[0])

        self.chemical_symbol_to_type = MLP_config.get('chemical_symbol_to_type')
        self.transform = TypeMapper(chemical_symbol_to_type=self.chemical_symbol_to_type)

        self.self_interaction = self.MLP_config.get('dataset_extra_fixed_fields',{}).get('self_interaction',False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

        data.to(torch.device(self.device))
        data = self.transform(data)
        data = AtomicData.to_AtomicDataDict(data)
        
        return data


class Nequip_latent_distance(uncertainty_base):
    def __init__(self, model, config, MLP_config):
        super().__init__(model, config, MLP_config)

        func_name = config.get('params_func','optimize2params')
        self.params_func = getattr(optimization_functions,func_name)
        self.parameter_length = 2 if func_name=='optimize2params' else self.latent_size+1
        self.params_file = os.path.join(self.MLP_config['workdir'],'uncertainty_params.pkl')

    def parse_data(self):
        dataset = dataset_from_config(self.MLP_config)

        train_embeddings = {}
        train_energies = {}
        test_embeddings = {}
        test_errors = {}
        test_energies = {}

        for key in self.chemical_symbol_to_type:
            train_embeddings[key] = torch.empty((0,self.latent_size),device=self.device)
            train_energies[key] = torch.empty((0),device=self.device)
            test_embeddings[key] = torch.empty((0,self.latent_size),device=self.device)
            test_errors[key] = torch.empty((0),device=self.device)
            test_energies[key] = torch.empty((0),device=self.device)
    
        for data in dataset[self.MLP_config.train_idcs]:
            out = self.model(self.transform_data_input(data))

            for key in self.MLP_config.get('chemical_symbol_to_type'):
                mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                train_embeddings[key] = torch.cat([train_embeddings[key],out['node_features'][mask].detach()])
                train_energies[key] = torch.cat([train_energies[key], out['atomic_energy'][mask].detach()])

        self.train_embeddings = train_embeddings
        self.train_energies = train_energies

        for data in dataset[self.MLP_config.val_idcs]:
            out = self.model(self.transform_data_input(data))
            
            error = torch.absolute(out['forces'] - data.forces)

            for key in self.MLP_config.get('chemical_symbol_to_type'):
                mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                test_embeddings[key] = torch.cat([test_embeddings[key],out['node_features'][mask].detach()])
                test_errors[key] = torch.cat([test_errors[key],error.mean(dim=1)[mask].detach()])
                test_energies[key] = torch.cat([test_energies[key], out['atomic_energy'][mask].detach()])
        
        self.test_embeddings = test_embeddings
        self.test_energies = test_energies
        self.test_errors = test_errors

    def calibrate(self, debug=False):
        
        fail = self.load_params()

        self.parse_data()

        if fail:
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
                    params[key].append(self.params_func(self.test_errors[key].reshape(-1).detach().cpu().numpy(),min_vectors[key]))

            self.params = params
            self.save_params()
            if debug:  
                self.min_distances = min_distances
                self.min_vectors = min_vectors

    def load_params(self):
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
            self.params = params

        return fail
                        
    def save_params(self):
        if hasattr(self,'params'):
            with open(self.params_file,'wb') as fl:
                pickle.dump(self.params, fl)


    def adversarial_loss(self, data, T, distances='train_val'):

        data = self.transform_data_input(data)
        
        out = self.model(data)
        atom_embedding = out['node_features']
        self.atom_embedding = atom_embedding

        self.uncertainties = self.predict_uncertainty(data, self.atom_embedding, distances=distances).to(self.device)

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
            
            adv_loss += (probability * self.uncertainties[mask.flatten()]).sum()

        return adv_loss

    def predict_uncertainty(self, data_in, atom_embedding=None, distances='train_val', extra_embeddings=None, type='full'):
        
        data = self.transform_data_input(data_in)

        if atom_embedding is None:
            out = self.model(data)
            atom_embedding = out['node_features']
            self.atom_embedding = atom_embedding

        atom_types = data['atom_types']
        uncertainties = torch.zeros_like(atom_embedding[:,0])

        self.test_distances = {}
        self.min_vectors = {}
        for key in self.MLP_config.get('chemical_symbol_to_type'):
            if distances == 'train_val':
                embeddings = torch.cat([self.train_embeddings[key],self.test_embeddings[key]])
            elif distances == 'train':
                embeddings = self.train_embeddings[key]

            if extra_embeddings is not None:
                embeddings = torch.cat([embeddings,extra_embeddings[key]])

            mask = (atom_types==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()

            if torch.any(mask):
                latent_force_distances = torch.cdist(embeddings,atom_embedding[mask],p=2)

                inds = torch.argmin(latent_force_distances,axis=0)
            
                min_distance = torch.tensor([latent_force_distances[ind,i] for i, ind in enumerate(inds)])
                min_vectors = torch.vstack([embeddings[ind]-atom_embedding[mask][i] for i, ind in enumerate(inds)])

                self.test_distances[key] = min_distance.detach().cpu().numpy()
                self.min_vectors[key] = min_vectors.detach().cpu().numpy()
            
                uncertainties[mask] = self.uncertainty_from_vector(min_vectors, key, type=type)

        return uncertainties

    def uncertainty_from_vector(self,vector, key, type='full'):
        if len(self.params[key]) == 2:
            distance = torch.linalg.norm(vector,axis=1).reshape(-1,1)
        else:
            distance = torch.abs(vector)

        uncertainty_raw = torch.zeros(self.n_ensemble,distance.shape[0])
        for i in range(self.n_ensemble):
            sig_1 = torch.tensor(self.params[key][i][0]).abs().type_as(distance)
            sig_2 = torch.tensor(self.params[key][i][1:]).abs().type_as(distance)
            
            if type == 'full':
                uncertainty = sig_1 + torch.sum(distance*sig_2,axis=1)
            elif type == 'std':
                uncertainty = torch.sum(distance*sig_2,axis=1)

            uncertainty_raw[i] = uncertainty
        uncertainty_mean = torch.mean(uncertainty_raw,axis=0)
        uncertainty_std = torch.std(uncertainty_raw,axis=0)

        uncertainty_ens = uncertainty_mean + uncertainty_std

        return uncertainty_ens

    def predict_from_traj(self, traj, max=True, batch_size=1):
        uncertainty = []
        atom_embeddings = []
        # data = [self.transform(atoms) for atoms in traj]
        # dataset = DataLoader(data, batch_size=batch_size)
        # for i, batch in enumerate(dataset):
        #     uncertainty.append(self.predict_uncertainty(batch))
        #     atom_embeddings.append(self.atom_embedding)
        for atoms in traj:
            uncertainty.append(self.predict_uncertainty(atoms))
            atom_embeddings.append(self.atom_embedding)
        
        
        uncertainty = torch.cat(uncertainty).detach().cpu()
        atom_embeddings = torch.cat(atom_embeddings).detach().cpu()

        if max:
            atom_lengths = [len(atoms) for atoms in traj]
            start_inds = [0] + np.cumsum(atom_lengths[:-1]).tolist()
            end_inds = np.cumsum(atom_lengths).tolist()

            uncertainty_partition = [uncertainty[si:ei] for si, ei in zip(start_inds,end_inds)]
            embeddings = [atom_embeddings[si:ei] for si, ei in zip(start_inds,end_inds)]
            return torch.tensor([unc.max() for unc in uncertainty_partition]), embeddings
        else:
            uncertainty = uncertainty.reshape(len(traj),-1)
            return uncertainty, atom_embeddings.reshape(len(traj),-1,atom_embeddings.shape[-1])

    def plot_fit(self,filename=None):
        
        if not hasattr(self, 'test_errors'):
            self.parse_data()
            
        fig, ax = plt.subplots(1,3,figsize=(15,5))

        pred_errors = self.predict_uncertainty(0,self.train_embeddings,type='std').detach()
        min_err = min([self.train_errors.min()])#,pred_errors.min()])
        max_err = max([self.train_errors.max()])#,pred_errors.max()])
        ax[0].scatter(self.train_errors,pred_errors,alpha=0.5)
        ax[0].plot([min_err,max_err], [min_err,max_err],'k',linestyle='--')
        ax[0].set_xlabel('True Error')
        ax[0].set_ylabel('Predicted Error')
        ax[0].axis('square')

        print((pred_errors-self.train_errors).mean())
        print((pred_errors-self.train_errors).std())

        pred_errors = self.predict_uncertainty(0,self.test_embeddings,type='mean').detach()
        min_err = min([self.test_errors.min()])#,pred_errors.min()])
        max_err = max([self.test_errors.max()])#,pred_errors.max()])
        ax[1].scatter(self.test_errors,pred_errors,alpha=0.5)
        ax[1].plot([min_err,max_err], [min_err,max_err],'k',linestyle='--')
        ax[1].set_xlabel('True Error')
        ax[1].set_ylabel('Predicted Error')
        ax[1].axis('square')

        ax[2].axhline(0,color='k',linestyle='--')
        ax[2].scatter(np.arange(len(self.test_errors)),pred_errors-self.test_errors,alpha=0.5)
        ax[2].set_ylabel('Residual')

        print((pred_errors-self.test_errors).mean())
        print((pred_errors-self.test_errors).std())

        if filename is not None:
            plt.savefig(filename)

class Nequip_error_NN(uncertainty_base):
    def __init__(self, model, config, MLP_config):
        super().__init__(model, config, MLP_config)

        self.hidden_dimensions = self.config.get('uncertainty_hidden_dimensions', [])
        self.unc_epochs = self.config.get('uncertainty_epochs', 2000)

        uncertainty_dir = os.path.join(self.MLP_config['workdir'],self.config.get('uncertainty_dir', 'uncertainty'))
        os.makedirs(uncertainty_dir,exist_ok=True)
        self.state_dict_func = lambda n: os.path.join(uncertainty_dir, f'uncertainty_state_dict_{n}.pth')
        self.metrics_func = lambda n: os.path.join(uncertainty_dir, f'uncertainty_metrics_{n}.csv')

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
            train_embeddings = torch.cat([train_embeddings,out['node_features']])
            
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

            test_embeddings = torch.cat([test_embeddings,out['node_features']])
            test_errors = torch.cat([test_errors,error.mean(dim=1)])
        
        self.test_embeddings = test_embeddings
        self.test_errors = test_errors
        self.test_energies = test_energies

    def adversarial_loss(self, data, T, distances='train_val'):

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
        
        uncertainties = self.predict_uncertainty(data['atom_types'], self.atom_embedding).to(self.device)

        adv_loss = (probability * uncertainties).sum()

        return adv_loss

    def predict_uncertainty(self, data, atom_embedding=None, distances='train_val', extra_embeddings=None,type='full'):

        if atom_embedding is None:
            data = self.transform_data_input(data)
        
            out = self.model(data)
            atom_embedding = out['node_features']
            self.atom_embedding = atom_embedding

        uncertainty_raw = torch.zeros(self.n_ensemble,atom_embedding.shape[0])
        for i, NN in enumerate(self.NNs):
            uncertainty_raw[i] = NN.predict(atom_embedding).squeeze()
        
        uncertainty_mean = torch.mean(uncertainty_raw,axis=0)
        uncertainty_std = torch.std(uncertainty_raw,axis=0)

        if type == 'full':
            uncertainty_ens = uncertainty_mean + uncertainty_std
        elif type == 'mean':
            uncertainty_ens = uncertainty_mean
        elif type == 'std':
            uncertainty_ens = uncertainty_std

        return uncertainty_ens

    def predict_from_traj(self, traj, max=True, batch_size=1):
        uncertainty = []
        atom_embeddings = []
        # data = [self.transform(atoms) for atoms in traj]
        # dataset = DataLoader(data, batch_size=batch_size)
        # for i, batch in enumerate(dataset):
        #     uncertainty.append(self.predict_uncertainty(batch))
        #     atom_embeddings.append(self.atom_embedding)
        for atoms in traj:
            uncertainty.append(self.predict_uncertainty(atoms))
            atom_embeddings.append(self.atom_embedding)
        
        
        uncertainty = torch.cat(uncertainty).detach().cpu()
        atom_embeddings = torch.cat(atom_embeddings).detach().cpu()

        if max:
            atom_lengths = [len(atoms) for atoms in traj]
            start_inds = [0] + np.cumsum(atom_lengths[:-1]).tolist()
            end_inds = np.cumsum(atom_lengths).tolist()

            uncertainty_partition = [uncertainty[si:ei] for si, ei in zip(start_inds,end_inds)]
            embeddings = [atom_embeddings[si:ei] for si, ei in zip(start_inds,end_inds)]
            return torch.tensor([unc.max() for unc in uncertainty_partition]), embeddings
        else:
            uncertainty = uncertainty.reshape(len(traj),-1)
            return uncertainty, atom_embeddings.reshape(len(traj),-1,atom_embeddings.shape[-1])

    def plot_fit(self, filename=None):
            
        fig, ax = plt.subplots(1,3,figsize=(15,5))

        pred_errors = self.predict_uncertainty(0,self.train_embeddings,type='mean').detach()
        min_err = min([self.train_errors.min()])#,pred_errors.min()])
        max_err = max([self.train_errors.max()])#,pred_errors.max()])
        ax[0].scatter(self.train_errors,pred_errors,alpha=0.5)
        ax[0].plot([min_err,max_err], [min_err,max_err],'k',linestyle='--')
        ax[0].set_xlabel('True Error')
        ax[0].set_ylabel('Predicted Error')
        ax[0].axis('square')

        print((pred_errors-self.train_errors).mean())
        print((pred_errors-self.train_errors).std())

        pred_errors = self.predict_uncertainty(0,self.test_embeddings,type='mean').detach()
        min_err = min([self.test_errors.min()])#,pred_errors.min()])
        max_err = max([self.test_errors.max()])#,pred_errors.max()])
        ax[1].scatter(self.test_errors,pred_errors,alpha=0.5)
        ax[1].plot([min_err,max_err], [min_err,max_err],'k',linestyle='--')
        ax[1].set_xlabel('True Error')
        ax[1].set_ylabel('Predicted Error')
        ax[1].axis('square')

        ax[2].axhline(0,color='k',linestyle='--')
        ax[2].scatter(np.arange(len(self.test_errors)),pred_errors-self.test_errors,alpha=0.5)
        ax[2].set_ylabel('Residual')

        print((pred_errors-self.test_errors).mean())
        print((pred_errors-self.test_errors).std())

        if filename is not None:
            plt.savefig(filename)