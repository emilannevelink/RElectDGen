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
from .optimization_functions import uncertainty_NN, uncertaintydistance_NN, uncertainty_ensemble_NN

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

        data = self.transform(data)
        data.to(torch.device(self.device))
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
            for key in params:
                for i, p in enumerate(params[key]):
                    print(key, i, p, flush=True)

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
            
            adv_loss += (probability * self.uncertainties[mask.flatten()].sum(dim=-1)).sum()

        return adv_loss

    def predict_uncertainty(self, data_in, atom_embedding=None, distances='train_val', extra_embeddings=None, type='full'):
        
        data = self.transform_data_input(data_in)

        if atom_embedding is None:
            out = self.model(data)
            atom_embedding = out['node_features']
            self.atom_embedding = atom_embedding
        else:
            atom_embedding = atom_embedding.to(device=torch.device(self.device))

        atom_types = data['atom_types']
        uncertainties = torch.zeros(atom_embedding.shape[0],2, device=self.device)

        self.test_distances = {}
        self.min_vectors = {}
        for key in self.MLP_config.get('chemical_symbol_to_type'):
            if distances == 'train_val':
                embeddings = torch.cat([self.train_embeddings[key],self.test_embeddings[key]])
            elif distances == 'train':
                embeddings = self.train_embeddings[key]

            if extra_embeddings is not None:
                embeddings = torch.cat([embeddings,extra_embeddings[key]])

            embeddings.to(device=self.device)
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

    def uncertainty_from_vector(self, vector, key, type='full'):
        if len(self.params[key]) == 2:
            distance = torch.linalg.norm(vector,axis=1).reshape(-1,1)
        else:
            distance = torch.abs(vector)

        uncertainty_raw = torch.zeros(self.n_ensemble,distance.shape[0], device=self.device)
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

        uncertainty = torch.vstack([uncertainty_mean,uncertainty_std]).T
        # uncertainty_ens = uncertainty_mean + uncertainty_std

        return uncertainty

    def predict_from_traj(self, traj, max=True, batch_size=1):
        uncertainty = []
        atom_embeddings = []
        # data = [self.transform(atoms) for atoms in traj]
        # dataset = DataLoader(data, batch_size=batch_size)
        # for i, batch in enumerate(dataset):
        #     uncertainty.append(self.predict_uncertainty(batch))
        #     atom_embeddings.append(self.atom_embedding)
        for atoms in traj:
            uncertainty.append(self.predict_uncertainty(atoms).detach())
            atom_embeddings.append(self.atom_embedding.detach())
        
        uncertainty = torch.cat(uncertainty).cpu()
        atom_embeddings = torch.cat(atom_embeddings).cpu()

        if max:
            atom_lengths = [len(atoms) for atoms in traj]
            start_inds = [0] + np.cumsum(atom_lengths[:-1]).tolist()
            end_inds = np.cumsum(atom_lengths).tolist()

            uncertainty_partition = [uncertainty[si:ei] for si, ei in zip(start_inds,end_inds)]
            embeddings = [atom_embeddings[si:ei] for si, ei in zip(start_inds,end_inds)]
            return torch.vstack([unc[torch.argmax(unc.sum(dim=1))] for unc in uncertainty_partition]), embeddings
        else:
            uncertainty = uncertainty.reshape(len(traj),-1,2)
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

class Nequip_latent_distanceNN(uncertainty_base):
    def __init__(self, model, config, MLP_config):
        super().__init__(model, config, MLP_config)

        self.hidden_dimensions = self.config.get('uncertainty_hidden_dimensions', [])
        self.unc_epochs = self.config.get('uncertainty_epochs', 2000)

        uncertainty_dir = os.path.join(self.MLP_config['workdir'],self.config.get('uncertainty_dir', 'uncertainty'))
        os.makedirs(uncertainty_dir,exist_ok=True)
        self.state_dict_func = lambda n: os.path.join(uncertainty_dir, f'uncertainty_state_dict_{n}.pth')
        self.metrics_func = lambda n: os.path.join(uncertainty_dir, f'uncertainty_metrics_{n}.csv')
        
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
                        

    def adversarial_loss(self, data, T, distances='train_val'):

        data = self.transform_data_input(data)
        
        out = self.model(data)
        atom_embedding = out['node_features']
        self.atom_embedding = atom_embedding

        self.uncertainties = self.predict_uncertainty(data, self.atom_embedding, distances=distances).to(self.device)

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

    def predict_uncertainty(self, data_in, atom_embedding=None, distances='train_val', extra_embeddings=None, type='full'):
        
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

        return uncertainties

    def uncertainty_from_vector(self, vector, type='full'):

        uncertainty_raw = torch.zeros(self.n_ensemble,vector.shape[0], device=self.device)
        for i, NN in enumerate(self.NNs):
            uncertainty_raw[i] = NN.predict(vector).squeeze()
        uncertainty_mean = torch.mean(uncertainty_raw,axis=0)
        uncertainty_std = torch.std(uncertainty_raw,axis=0)

        uncertainty = torch.vstack([uncertainty_mean,uncertainty_std]).T
        # uncertainty_ens = uncertainty_mean + uncertainty_std

        return uncertainty

    def predict_from_traj(self, traj, max=True, batch_size=1):
        uncertainty = []
        atom_embeddings = []
        # data = [self.transform(atoms) for atoms in traj]
        # dataset = DataLoader(data, batch_size=batch_size)
        # for i, batch in enumerate(dataset):
        #     uncertainty.append(self.predict_uncertainty(batch))
        #     atom_embeddings.append(self.atom_embedding)
        for atoms in traj:
            uncertainty.append(self.predict_uncertainty(atoms).detach())
            atom_embeddings.append(self.atom_embedding.detach())
        
        uncertainty = torch.cat(uncertainty).cpu()
        atom_embeddings = torch.cat(atom_embeddings).cpu()

        if max:
            atom_lengths = [len(atoms) for atoms in traj]
            start_inds = [0] + np.cumsum(atom_lengths[:-1]).tolist()
            end_inds = np.cumsum(atom_lengths).tolist()

            uncertainty_partition = [uncertainty[si:ei] for si, ei in zip(start_inds,end_inds)]
            embeddings = [atom_embeddings[si:ei] for si, ei in zip(start_inds,end_inds)]
            return torch.vstack([unc[torch.argmax(unc.sum(dim=1))] for unc in uncertainty_partition]), embeddings
        else:
            uncertainty = uncertainty.reshape(len(traj),-1,2)
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

class Nequip_ensemble_NN(uncertainty_base):
    def __init__(self, model, config, MLP_config):
        super().__init__(model, config, MLP_config)

        # self.nequip_model = model
        self.hidden_dimensions = self.config.get('uncertainty_hidden_dimensions', [])
        self.unc_epochs = self.config.get('uncertainty_epochs', 2000)

        self.natoms = len(MLP_config['type_names'])
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
                NN = uncertainty_ensemble_NN(self.model, self.latent_size, self.natoms, self.hidden_dimensions)
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
            
            for n in train_indices:
                print('training ensemble network ', n, flush=True)    
                #train NN to fit energies
                NN = uncertainty_ensemble_NN(self.model, self.latent_size, self.natoms, self.hidden_dimensions, epochs=self.unc_epochs)
                # NN = uncertainty_ensemble_NN(self.model, self.latent_size, self.hidden_dimensions)
                uncertainty_training = self.config.get('uncertainty_training','energy')
                if uncertainty_training=='energy':
                    NN.train(self.train_embeddings, self.train_energies, self.validation_embeddings, self.validation_energies)
                elif uncertainty_training=='forces':
                    NN.train(self.train_embeddings, self.train_forces, self.validation_embeddings, self.validation_forces)
                else:
                    raise RuntimeError
                    
                print('Best loss ', NN.best_loss, flush=True)
                self.NNs.append(NN)
                torch.save(NN.get_state_dict(), self.state_dict_func(n))
                pd.DataFrame(NN.metrics).to_csv( self.metrics_func(n))

    def fine_tune(self, embeddings, energies_or_forces):
        print('Fine Tuning Ensemble', flush=True)
        for NN in self.NNs:
            uncertainty_training = self.config.get('uncertainty_training','energy')
            if uncertainty_training=='energy':
                NN.fine_tune(self.train_embeddings,self.train_energies,self.validation_embeddings,self.validation_energies,embeddings, energies_or_forces)
            elif uncertainty_training=='forces':
                NN.fine_tune(self.train_embeddings,self.train_forces,self.validation_embeddings,self.validation_forces,embeddings, energies_or_forces)
            else:
                raise RuntimeError
        
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
        
        for i, data in enumerate(self.train_dataset):
            out = self.model(self.transform_data_input(data))

            force_norm = data['forces'].norm(dim=1).unsqueeze(dim=1)
            force_lim = torch.max(force_norm,torch.ones_like(force_norm))
            perc_err = ((out['forces'].detach()-data['forces'])).abs()/force_lim
            
            if perc_err.max() < self.config.get('UQ_dataset_error', .5):
                self.UQ_validation_indices = torch.cat([self.UQ_validation_indices, self.ML_train_indices[i].unsqueeze(dim=0)])
                for key in self.MLP_config.get('chemical_symbol_to_type'):
                    mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                    
                    atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze()[mask],num_classes=self.natoms).to(self.device)
                    NN_inputs = torch.hstack([out['node_features'][mask].detach(), atom_one_hot])
                    
                    train_embeddings[key] = torch.cat([train_embeddings[key],NN_inputs])
                    train_energies[key] = torch.cat([train_energies[key], out['atomic_energy'][mask].detach()])
                    train_forces[key] = torch.cat([train_forces[key], out['forces'][mask].detach().norm(dim=1)])

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
                    test_forces[key] = torch.cat([test_forces[key], out['forces'][mask].detach().norm(dim=1)])

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
            
            if perc_err.max() < self.config.get('UQ_dataset_error', .5):
                self.UQ_validation_indices = torch.cat([self.UQ_validation_indices, self.ML_validation_indices[i].unsqueeze(dim=0)])
                for key in self.MLP_config.get('chemical_symbol_to_type'):
                    mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()

                    atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze()[mask],num_classes=self.natoms).to(self.device)
                    NN_inputs = torch.hstack([out['node_features'][mask].detach(), atom_one_hot])

                    validation_embeddings[key] = torch.cat([validation_embeddings[key],NN_inputs])
                    validation_energies[key] = torch.cat([validation_energies[key], out['atomic_energy'][mask].detach()])
                    validation_forces[key] = torch.cat([validation_forces[key], out['forces'][mask].detach().norm(dim=1)])
                    
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
                    test_forces[key] = torch.cat([test_forces[key], out['forces'][mask].detach().norm(dim=1)])
                    
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
            
            adv_loss += (probability * self.uncertainties[mask.flatten()].sum(dim=-1)).sum()

        return adv_loss

    def predict_uncertainty(self, data, atom_embedding=None, distances='train_val', extra_embeddings=None,type='full'):

        
        data = self.transform_data_input(data)
        out = self.model(data)
        self.atom_embedding = out['node_features']
        
        pred_atom_energies = torch.zeros((self.n_ensemble,out['atomic_energy'].shape[0])).to(self.device)
        for i, NN in enumerate(self.NNs):
            pred_atom_energies[i] = NN.predict(out).squeeze()
        
        uncertainties_mean = (pred_atom_energies.mean(dim=0)-out['atomic_energy'].squeeze()).abs()
        # uncertainties_mean = (pred_atom_energies-out['atomic_energy'].squeeze().unsqueeze(0)).abs().max(dim=0).values
        # uncertainty_mean = (pred_atom_energies.sum(dim=1)-out['total_energy'].squeeze()).abs().max(dim=0).values
        # uncertainties_mean = torch.ones(pred_atom_energies.shape[1]).to(self.device)*uncertainty_mean
        
        uncertainties_std = pred_atom_energies.std(axis=0)#.sum(axis=-1)
        # uncertainty_std = pred_atom_energies.sum(dim=1).std(axis=0)#.sum(axis=-1)
        # uncertainties_std = torch.ones(pred_atom_energies.shape[1]).to(self.device)*uncertainty_std

        uncertainty = torch.vstack([uncertainties_mean,uncertainties_std]).T/torch.max(torch.ones_like(out['atomic_energy']),out['atomic_energy'].abs())

        # uncertainty *= self.config.get('uncertainty_factor',10)
        return uncertainty

    def predict_from_traj(self, traj, max=True, batch_size=1):
        uncertainty = []
        atom_embeddings = []
        # data = [self.transform(atoms) for atoms in traj]
        # dataset = DataLoader(data, batch_size=batch_size)
        # for i, batch in enumerate(dataset):
        #     uncertainty.append(self.predict_uncertainty(batch))
        #     atom_embeddings.append(self.atom_embedding)
        for atoms in traj:
            uncertainty.append(self.predict_uncertainty(atoms).detach())
            atom_embeddings.append(self.atom_embedding.detach())
        
        
        uncertainty = torch.cat(uncertainty).cpu()
        atom_embeddings = torch.cat(atom_embeddings).cpu()

        if max:
            atom_lengths = [len(atoms) for atoms in traj]
            start_inds = [0] + np.cumsum(atom_lengths[:-1]).tolist()
            end_inds = np.cumsum(atom_lengths).tolist()

            uncertainty_partition = [uncertainty[si:ei] for si, ei in zip(start_inds,end_inds)]
            embeddings = [atom_embeddings[si:ei] for si, ei in zip(start_inds,end_inds)]
            
            return torch.vstack([unc[torch.argmax(unc.sum(dim=1))] for unc in uncertainty_partition]), embeddings
        else:
            uncertainty = uncertainty.reshape(len(traj),-1, 2)
            return uncertainty, atom_embeddings.reshape(len(traj),-1,atom_embeddings.shape[-1])

    def plot_fit(self, filename=None):
            
        if not hasattr(self, 'train_dataset'):
            self.parse_data()
            
        train_energy_real = torch.empty((0),device=self.device)
        train_energy_pred = torch.empty((0),device=self.device)
        train_max_force_real = torch.empty((0),device=self.device)
        train_max_force_pred = torch.empty((0),device=self.device)

        val_energy_real = torch.empty((0),device=self.device)
        val_energy_pred = torch.empty((0),device=self.device)
        val_max_force_real = torch.empty((0),device=self.device)
        val_max_force_pred = torch.empty((0),device=self.device)

        train_energy_err = torch.empty((0),device=self.device)
        train_energy_std = torch.empty((0),device=self.device)
        train_max_force_err = torch.empty((0),device=self.device)
        train_max_force_std = torch.empty((0),device=self.device)
        train_max_force_max_err = torch.empty((0),device=self.device)
        train_max_force_max_std = torch.empty((0),device=self.device)

        val_energy_err = torch.empty((0),device=self.device)
        val_energy_std = torch.empty((0),device=self.device)

        val_energy_err = torch.empty((0),device=self.device)
        val_energy_std = torch.empty((0),device=self.device)
        val_max_force_err = torch.empty((0),device=self.device)
        val_max_force_std = torch.empty((0),device=self.device)
        val_max_force_max_err = torch.empty((0),device=self.device)
        val_max_force_max_std = torch.empty((0),device=self.device)

        train_force_real = {}
        train_force_pred = {}
        train_force_unc_err = {}
        train_force_unc_std = {}
        for key in self.chemical_symbol_to_type:
            train_force_real[key] = torch.empty((0),device=self.device)
            train_force_pred[key] = torch.empty((0),device=self.device)
            train_force_unc_err[key] = torch.empty((0),device=self.device)
            train_force_unc_std[key] = torch.empty((0),device=self.device)
        for data in self.train_dataset:
            natoms = len(data['pos'])
            train_energy_real = torch.cat([train_energy_real,data['total_energy']/natoms])
            out = self.model(self.transform_data_input(data))
            unc = self.predict_uncertainty(data).detach()
            train_energy_pred = torch.cat([train_energy_pred,out['atomic_energy'].detach().mean().unsqueeze(dim=0)])


            train_energy_err = torch.cat([train_energy_err,unc[:,0].max().unsqueeze(dim=0)])
            train_energy_std = torch.cat([train_energy_std,unc[:,1].max().unsqueeze(dim=0)])

            ind = torch.argmax((data['forces'].detach()-out['forces'].detach()).norm(dim=1))
            train_max_force_real = torch.cat([train_max_force_real, data['forces'].detach()[ind].unsqueeze(dim=0)])
            train_max_force_pred = torch.cat([train_max_force_pred, out['forces'].detach()[ind].unsqueeze(dim=0)])
            train_max_force_err = torch.cat([train_max_force_err, unc[ind,0].unsqueeze(0)])
            train_max_force_std = torch.cat([train_max_force_std, unc[ind,1].unsqueeze(0)])
            ind_unc_max = torch.argmax(unc.sum(1))
            train_max_force_max_err = torch.cat([train_max_force_max_err, unc[ind_unc_max,0].unsqueeze(0)])
            train_max_force_max_std = torch.cat([train_max_force_max_std, unc[ind_unc_max,1].unsqueeze(0)])

            for key in self.chemical_symbol_to_type:
                mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                train_force_real[key] = torch.cat([train_force_real[key],data['forces'].detach()[mask]])
                train_force_pred[key] = torch.cat([train_force_pred[key],out['forces'].detach()[mask]])
                train_force_unc_err[key] = torch.cat([train_force_unc_err[key],unc[mask,0]])
                train_force_unc_std[key] = torch.cat([train_force_unc_std[key],unc[mask,1]])

        val_force_real = {}
        val_force_pred = {}
        val_force_unc_err = {}
        val_force_unc_std = {}
        for key in self.chemical_symbol_to_type:
            val_force_real[key] = torch.empty((0),device=self.device)
            val_force_pred[key] = torch.empty((0),device=self.device)
            val_force_unc_err[key] = torch.empty((0),device=self.device)
            val_force_unc_std[key] = torch.empty((0),device=self.device)
        for data in self.validation_dataset:
            natoms = len(data['pos'])
            val_energy_real = torch.cat([val_energy_real,data['total_energy']/natoms])
            out = self.model(self.transform_data_input(data))
            unc = self.predict_uncertainty(data).detach()
            val_energy_pred = torch.cat([val_energy_pred,out['atomic_energy'].detach().mean().unsqueeze(dim=0)])

            val_energy_err = torch.cat([val_energy_err,unc[:,0].max().unsqueeze(dim=0)])
            val_energy_std = torch.cat([val_energy_std,unc[:,1].max().unsqueeze(dim=0)])

            ind = torch.argmax((data['forces'].detach()-out['forces'].detach()).norm(dim=1))
            val_max_force_real = torch.cat([val_max_force_real, data['forces'].detach()[ind].unsqueeze(dim=0)])
            val_max_force_pred = torch.cat([val_max_force_pred, out['forces'].detach()[ind].unsqueeze(dim=0)])
            val_max_force_err = torch.cat([val_max_force_err, unc[ind,0].unsqueeze(0)])
            val_max_force_std = torch.cat([val_max_force_std, unc[ind,1].unsqueeze(0)])
            ind_unc_max = torch.argmax(unc.sum(1))
            val_max_force_max_err = torch.cat([val_max_force_max_err, unc[ind_unc_max,0].unsqueeze(0)])
            val_max_force_max_std = torch.cat([val_max_force_max_std, unc[ind_unc_max,1].unsqueeze(0)])

            for key in self.chemical_symbol_to_type:
                mask = (data['atom_types']==self.MLP_config.get('chemical_symbol_to_type')[key]).flatten()
                val_force_real[key] = torch.cat([val_force_real[key],data['forces'].detach()[mask]])
                val_force_pred[key] = torch.cat([val_force_pred[key],out['forces'].detach()[mask]])
                val_force_unc_err[key] = torch.cat([val_force_unc_err[key],unc[mask,0]])
                val_force_unc_std[key] = torch.cat([val_force_unc_std[key],unc[mask,1]])
            
        # print(train_real.shape, train_pred.shape) 
        # print(val_real.shape, val_pred.shape) 
        # print(train_unc_err.shape, val_unc_err.shape) 

        fig, ax = plt.subplots(4,5, figsize=(25,20))
        min_energy = np.inf
        max_energy = -np.inf

        min_energy = min([min_energy, train_energy_real.min(), train_energy_pred.min(), val_energy_real.min(), val_energy_pred.min()])
        max_energy = max([max_energy, train_energy_real.max(), train_energy_pred.max(), val_energy_real.max(),val_energy_pred.max()])

        ax[0,0].scatter(train_energy_real,train_energy_pred)
        ax[0,0].errorbar(train_energy_real,train_energy_pred, yerr = train_energy_err, fmt='o')
        
        ax[0,1].scatter(train_energy_real,train_energy_pred)
        ax[0,1].errorbar(train_energy_real,train_energy_pred, yerr = train_energy_std, fmt='o')

        ax[1,0].scatter(list(range(len(train_energy_real))),train_energy_real-train_energy_pred)
        ax[1,0].errorbar(list(range(len(train_energy_real))),train_energy_real-train_energy_pred, yerr = train_energy_err, fmt='o')
        
        # ax[1,1].scatter(list(range(len(train_energy_real))),train_energy_err)
        # ax[1,1].errorbar(list(range(len(train_energy_real))),train_energy_err, yerr = train_energy_std, fmt='o')
        ax[1,1].scatter(train_energy_real-train_energy_pred,train_energy_err)
        ax[1,1].errorbar(train_energy_real-train_energy_pred,train_energy_err, yerr = train_energy_std, fmt='o')
        
        ax[2,0].scatter(val_energy_real,val_energy_pred)
        ax[2,0].errorbar(val_energy_real,val_energy_pred, yerr = val_energy_err, fmt='o')
        
        ax[2,1].scatter(val_energy_real,val_energy_pred)
        ax[2,1].errorbar(val_energy_real,val_energy_pred, yerr = val_energy_std, fmt='o')

        ax[3,0].scatter(list(range(len(val_energy_real))),val_energy_real-val_energy_pred)
        ax[3,0].errorbar(list(range(len(val_energy_real))),val_energy_real-val_energy_pred, yerr = val_energy_err, fmt='o')
        
        # ax[3,1].scatter(list(range(len(val_energy_real))),val_energy_err)
        # ax[3,1].errorbar(list(range(len(val_energy_real))),val_energy_err, yerr = val_energy_std, fmt='o')
        ax[3,1].scatter(val_energy_real-val_energy_pred,val_energy_err)
        ax[3,1].errorbar(val_energy_real-val_energy_pred,val_energy_err, yerr = val_energy_std, fmt='o')
        
        ax[0,0].plot([min_energy,max_energy],[min_energy,max_energy],color='k',linestyle='--')
        ax[2,0].plot([min_energy,max_energy],[min_energy,max_energy],color='k',linestyle='--')
        ax[0,1].plot([min_energy,max_energy],[min_energy,max_energy],color='k',linestyle='--')
        ax[2,1].plot([min_energy,max_energy],[min_energy,max_energy],color='k',linestyle='--')

        ax[0,4].scatter(train_max_force_real.norm(dim=1),train_max_force_pred.norm(dim=1))
        ax[0,4].errorbar(train_max_force_real.norm(dim=1),train_max_force_pred.norm(dim=1), yerr=train_max_force_err+train_max_force_std, xerr=train_max_force_max_err+train_max_force_max_std, fmt='o')

        ax[1,4].scatter((train_max_force_real-train_max_force_pred).norm(dim=1),train_max_force_err)
        ax[1,4].errorbar((train_max_force_real-train_max_force_pred).norm(dim=1), train_max_force_err, yerr=train_max_force_std, xerr=train_max_force_max_err+train_max_force_max_std, fmt='o')

        ax[2,4].scatter(val_max_force_real.norm(dim=1),val_max_force_pred.norm(dim=1))
        ax[2,4].errorbar(val_max_force_real.norm(dim=1),val_max_force_pred.norm(dim=1), yerr=val_max_force_err+val_max_force_std, xerr=val_max_force_max_err+val_max_force_max_std, fmt='o')

        ax[3,4].scatter((val_max_force_real-val_max_force_pred).norm(dim=1),val_max_force_err)
        ax[3,4].errorbar((val_max_force_real-val_max_force_pred).norm(dim=1), val_max_force_err, yerr=val_max_force_std, xerr=val_max_force_max_err+val_max_force_max_std, fmt='o')

        min_force = np.inf
        max_force = -np.inf
        ntrain = nval = 0
        for key in self.chemical_symbol_to_type:
            # train_error = train_real[key]-train_pred[key]
            # train_distribution_err = train_error/train_unc_err[key]
            # train_distribution_std = train_error/train_unc_std[key]
            # train_distribution = train_error/(train_unc_err[key]+train_unc_std[key])
            # np.histogram(train_distribution.flatten())

            min_force = min([min_force, train_force_real[key].norm(dim=-1).min(), train_force_pred[key].norm(dim=-1).min(), val_force_real[key].norm(dim=-1).min(), val_force_pred[key].norm(dim=-1).min()])
            max_force = max([max_force, train_force_real[key].norm(dim=-1).max(), train_force_pred[key].norm(dim=-1).max(), val_force_real[key].norm(dim=-1).max(), val_force_pred[key].norm(dim=-1).max()])

            ax[0,2].scatter(train_force_real[key].norm(dim=-1),train_force_pred[key].norm(dim=-1))
            ax[0,2].errorbar(train_force_real[key].norm(dim=-1),train_force_pred[key].norm(dim=-1), yerr = train_force_unc_err[key], fmt='o')
            
            ax[0,3].scatter(train_force_real[key].norm(dim=-1),train_force_pred[key].norm(dim=-1))
            ax[0,3].errorbar(train_force_real[key].norm(dim=-1),train_force_pred[key].norm(dim=-1), yerr = train_force_unc_std[key], fmt='o')

            ax[1,2].scatter(range(ntrain,ntrain+len(train_force_real[key])),train_force_real[key].norm(dim=-1)-train_force_pred[key].norm(dim=-1))
            ax[1,2].errorbar(range(ntrain,ntrain+len(train_force_real[key])),train_force_real[key].norm(dim=-1)-train_force_pred[key].norm(dim=-1), yerr = train_force_unc_err[key], fmt='o')
            
            # ax[1,3].scatter(range(ntrain,ntrain+len(train_force_real[key])),train_force_unc_err[key])
            # ax[1,3].errorbar(range(ntrain,ntrain+len(train_force_real[key])),train_force_unc_err[key], yerr = train_force_unc_std[key], fmt='o')
            ax[1,3].scatter(train_force_real[key].norm(dim=-1)-train_force_pred[key].norm(dim=-1),train_force_unc_err[key])
            ax[1,3].errorbar(train_force_real[key].norm(dim=-1)-train_force_pred[key].norm(dim=-1),train_force_unc_err[key], yerr = train_force_unc_std[key], fmt='o')
            ntrain+=len(train_force_real[key])

            ax[2,2].scatter(val_force_real[key].norm(dim=-1),val_force_pred[key].norm(dim=-1))
            ax[2,2].errorbar(val_force_real[key].norm(dim=-1),val_force_pred[key].norm(dim=-1), yerr = val_force_unc_err[key], fmt='o')
            
            ax[2,3].scatter(val_force_real[key].norm(dim=-1),val_force_pred[key].norm(dim=-1))
            ax[2,3].errorbar(val_force_real[key].norm(dim=-1),val_force_pred[key].norm(dim=-1), yerr = val_force_unc_std[key], fmt='o')
            
            ax[3,2].scatter(range(nval,nval+len(val_force_real[key])),val_force_real[key].norm(dim=-1)-val_force_pred[key].norm(dim=-1))
            ax[3,2].errorbar(range(nval,nval+len(val_force_real[key])),val_force_real[key].norm(dim=-1)-val_force_pred[key].norm(dim=-1), yerr = val_force_unc_err[key], fmt='o')
            
            # ax[3,3].scatter(range(nval,nval+len(val_force_real[key])),val_force_unc_err[key])
            # ax[3,3].errorbar(range(nval,nval+len(val_force_real[key])),val_force_unc_err[key], yerr = val_force_unc_std[key], fmt='o')
            ax[3,3].scatter(val_force_real[key].norm(dim=-1)-val_force_pred[key].norm(dim=-1),val_force_unc_err[key])
            ax[3,3].errorbar(val_force_real[key].norm(dim=-1)-val_force_pred[key].norm(dim=-1),val_force_unc_err[key], yerr = val_force_unc_std[key], fmt='o')
            nval+=len(val_force_real[key])
        
        ax[0,2].plot([min_force,max_force],[min_force,max_force],color='k',linestyle='--')
        ax[2,2].plot([min_force,max_force],[min_force,max_force],color='k',linestyle='--')
        ax[0,3].plot([min_force,max_force],[min_force,max_force],color='k',linestyle='--')
        ax[2,3].plot([min_force,max_force],[min_force,max_force],color='k',linestyle='--')
        
        ax[0,4].plot([min_force,max_force],[min_force,max_force],color='k',linestyle='--')
        ax[2,4].plot([min_force,max_force],[min_force,max_force],color='k',linestyle='--')
        
        # ax[0,0].set_xscale('log')
        # ax[0,0].set_yscale('log')
        # ax[0,1].set_xscale('log')
        # ax[0,1].set_yscale('log')
        ax[0,2].set_xscale('log')
        ax[0,2].set_yscale('log')
        ax[0,3].set_xscale('log')
        ax[0,3].set_yscale('log')
        ax[0,4].set_xscale('log')
        ax[0,4].set_yscale('log')
        # ax[1,0].set_xscale('log')
        ax[1,0].set_yscale('log')
        # ax[1,1].set_xscale('log')
        ax[1,1].set_yscale('log')
        # ax[1,2].set_xscale('log')
        ax[1,2].set_yscale('log')
        # ax[1,3].set_xscale('log')
        ax[1,3].set_yscale('log')
        # ax[1,4].set_xscale('log')
        # ax[1,4].set_yscale('log')
        # ax[2,0].set_xscale('log')
        # ax[2,0].set_yscale('log')
        # ax[2,1].set_xscale('log')
        # ax[2,1].set_yscale('log')
        ax[2,2].set_xscale('log')
        ax[2,2].set_yscale('log')
        ax[2,3].set_xscale('log')
        ax[2,3].set_yscale('log')
        ax[2,4].set_xscale('log')
        ax[2,4].set_yscale('log')
        # ax[3,0].set_xscale('log')
        ax[3,0].set_yscale('log')
        # ax[3,1].set_xscale('log')
        ax[3,1].set_yscale('log')
        # ax[3,2].set_xscale('log')
        # ax[3,2].set_yscale('log')
        # ax[3,3].set_xscale('log')
        ax[3,3].set_yscale('log')
        ax[3,4].set_xscale('log')
        ax[3,4].set_yscale('log')

        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            plt.close()

