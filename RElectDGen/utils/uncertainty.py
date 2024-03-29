import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as stats
import matplotlib.pyplot as plt

from nequip.data import AtomicData, dataset_from_config, DataLoader
from nequip.data.transforms import TypeMapper

class latent_distance_uncertainty_Nequip():
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.r_max = config.r_max
        self.latent_size = int(self.config['conv_to_output_hidden_irreps_out'].split('x')[0])

        chemical_symbol_to_type = config.get('chemical_symbol_to_type')
        self.transform = TypeMapper(chemical_symbol_to_type=chemical_symbol_to_type)

        self.self_interaction = self.config.get('dataset_extra_fixed_fields',False).get('self_interaction',False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            

    def transform_data_input(self,data):
        data.to(torch.device(self.device))
        data = AtomicData.to_AtomicDataDict(data)
        # if torch.cuda.is_available():
        #     for key in data.keys():
        #         if isinstance(data[key],torch.Tensor):
        #             data[key].to(torch.device('cuda'))
        return data

    def calibrate(self,debug=False):
        #latent_size = self.model.final.tp.irreps_in1.dim #monolayer energy
        
        dataset = dataset_from_config(self.config)

        train_embeddings = {}
        test_embeddings = {}
        test_errors = {}
        for key in self.config.get('chemical_symbol_to_type'):
            train_embeddings[key] = torch.empty((0,self.latent_size),device=self.device)
            test_embeddings[key] = torch.empty((0,self.latent_size),device=self.device)
            test_errors[key] = torch.empty((0),device=self.device)
    
        for data in dataset[self.config.train_idcs]:
            out = self.model(self.transform_data_input(data))

            for key in self.config.get('chemical_symbol_to_type'):
                mask = (data['atom_types']==self.config.get('chemical_symbol_to_type')[key]).flatten()
                train_embeddings[key] = torch.cat([train_embeddings[key],out['node_features'][mask]])

        self.train_embeddings = train_embeddings
            

        for data in dataset[self.config.val_idcs]:
            out = self.model(self.transform_data_input(data))
            
            error = torch.absolute(out['forces'] - data.forces)

            for key in self.config.get('chemical_symbol_to_type'):
                mask = (data['atom_types']==self.config.get('chemical_symbol_to_type')[key]).flatten()
                test_embeddings[key] = torch.cat([test_embeddings[key],out['node_features'][mask]])
                test_errors[key] = torch.cat([test_errors[key],error.mean(dim=1)[mask]])
        
        self.test_embeddings = test_embeddings

        latent_distances = {}
        min_distances = {}
        params = {}
        for key in self.config.get('chemical_symbol_to_type'):
            latent_distances[key] = torch.cdist(train_embeddings[key],test_embeddings[key],p=2)
            inds = torch.argmin(latent_distances[key],axis=0)
            min_distances[key] = torch.tensor([latent_distances[key][ind,i] for i, ind in enumerate(inds)]).detach().cpu().numpy()

            params0 = (0.01,0.01)
            res = minimize(optimizeparams,params0,args=(test_errors[key].reshape(-1).detach().cpu().numpy(),min_distances[key]),method='Nelder-Mead')
            print(res,flush=True)
            params[key] = np.abs(res.x)

        self.params = params
        if debug:  
            self.min_distances = min_distances
            self.test_errors = test_errors

    def predict(self, data, distances='train_val'):
        out = self.model(self.transform_data_input(data))
        self.atom_embedding = out['node_features']
        
        uncertainties = torch.zeros(self.atom_embedding.shape[0])

        self.test_distances = {}
        for key in self.config.get('chemical_symbol_to_type'):
            if distances == 'train_val':
                embeddings = torch.cat([self.train_embeddings[key],self.test_embeddings[key]])
            elif distances == 'train':
                embeddings = self.train_embeddings[key]

            mask = (data['atom_types']==self.config.get('chemical_symbol_to_type')[key]).flatten()

            latent_force_distances = torch.cdist(embeddings,self.atom_embedding[mask],p=2)
        
            inds = torch.argmin(latent_force_distances,axis=0)
            min_distance = torch.tensor([latent_force_distances[ind,i] for i, ind in enumerate(inds)])

            self.test_distances[key] = min_distance.detach().cpu().numpy()
        
            uncertainties[mask] = self.params[key][0]+min_distance*self.params[key][1]

        return uncertainties

    def predict_from_traj(self, traj, max=True, batch_size=2):
        uncertainty = []
        atom_embeddings = []
        data = [self.transform(AtomicData.from_ase(atoms=atoms,r_max=self.r_max, self_interaction=self.self_interaction)) for atoms in traj]
        dataset = DataLoader(data, batch_size=batch_size)
        for i, batch in enumerate(dataset):
            uncertainty.append(self.predict(batch))
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

    def plot_fit(self):
        
        fig = plt.figure()
        ax = plt.gca()
        for key in self.config.get('chemical_symbol_to_type'):
            ax.scatter(self.min_distances[key],self.test_errors[key].reshape(-1),alpha=0.2)

            sigabs = self.params[key]
            d_fit = np.linspace(0,self.d_force_test.max())
            error_fit = sigabs[0] + sigabs[1]*d_fit
            ax.plot(d_fit,error_fit,label=key)

        ax.legend()
        ax.set_xlabel('Embedding Distance')
        ax.set_ylabel('Force Error')

class latent_distance_uncertainty_Nequip_adversarial():
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.r_max = config.r_max
        self.latent_size = int(self.config['conv_to_output_hidden_irreps_out'].split('x')[0])

        chemical_symbol_to_type = config.get('chemical_symbol_to_type')
        self.transform = TypeMapper(chemical_symbol_to_type=chemical_symbol_to_type)

        self.self_interaction = self.config.get('dataset_extra_fixed_fields',False).get('self_interaction',False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.kb = 8.6173e-5 #eV/K
        self.params_func = getattr(sys.modules[__name__],config.get('params_func','optimize2params'))

        self.n_ensemble = config.get('n_uncertainty_ensembles',4)
            

    def transform_data_input(self,data):
        data.to(torch.device(self.device))
        data = AtomicData.to_AtomicDataDict(data)
        
        return data

    def calibrate(self,debug=False):
        
        dataset = dataset_from_config(self.config)

        train_embeddings = {}
        test_embeddings = {}
        test_errors = {}
        train_energies = torch.empty((0),device=self.device)
        for key in self.config.get('chemical_symbol_to_type'):
            train_embeddings[key] = torch.empty((0,self.latent_size),device=self.device)
            test_embeddings[key] = torch.empty((0,self.latent_size),device=self.device)
            test_errors[key] = torch.empty((0),device=self.device)
    
        for data in dataset[self.config.train_idcs]:
            out = self.model(self.transform_data_input(data))
            train_energies = torch.cat([train_energies, out['total_energy']])

            for key in self.config.get('chemical_symbol_to_type'):
                mask = (data['atom_types']==self.config.get('chemical_symbol_to_type')[key]).flatten()
                train_embeddings[key] = torch.cat([train_embeddings[key],out['node_features'][mask]])

        self.train_embeddings = train_embeddings
        self.train_energies = train_energies
            
        test_energies = torch.empty((0),device=self.device)
        for data in dataset[self.config.val_idcs]:
            out = self.model(self.transform_data_input(data))
            test_energies = torch.cat([test_energies, out['total_energy']])

            error = torch.absolute(out['forces'] - data.forces)

            for key in self.config.get('chemical_symbol_to_type'):
                mask = (data['atom_types']==self.config.get('chemical_symbol_to_type')[key]).flatten()
                test_embeddings[key] = torch.cat([test_embeddings[key],out['node_features'][mask]])
                test_errors[key] = torch.cat([test_errors[key],error.mean(dim=1)[mask]])
        
        self.test_embeddings = test_embeddings
        self.test_energies = test_energies

        latent_distances = {}
        min_distances = {}
        min_vectors = {}
        params = {}
        for key in self.config.get('chemical_symbol_to_type'):
            latent_distances[key] = torch.cdist(train_embeddings[key],test_embeddings[key],p=2)
            inds = torch.argmin(latent_distances[key],axis=0)
            min_distances[key] = torch.tensor([latent_distances[key][ind,i] for i, ind in enumerate(inds)]).detach().cpu().numpy()
            min_vectors[key] = np.abs(torch.vstack([train_embeddings[key][ind]-test_embeddings[key][i] for i, ind in enumerate(inds)]).detach().cpu().numpy())

            params[key] = []
            for _ in range(self.n_ensemble):
                params[key].append(self.params_func(test_errors[key].reshape(-1).detach().cpu().numpy(),min_vectors[key]))
            # params0 = [0.01]*(self.latent_size+1)
            # res = minimize(optimizeparams,params0,args=(test_errors[key].reshape(-1).detach().cpu().numpy(),min_distances[key]),method='Nelder-Mead')
            # print(res,flush=True)
            # params[key] = np.abs(res.x)

        self.params = params
        if debug:  
            self.min_distances = min_distances
            self.min_vectors = min_vectors
            self.test_errors = test_errors

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

        probability = 1/Q * torch.exp(-(out['total_energy']-emean)/estd/kT)
        
        uncertainties = self.predict_uncertainty(data['atom_types'], self.atom_embedding, distances=distances).to(self.device)

        adv_loss = (probability * uncertainties).sum()

        return adv_loss

    def predict_uncertainty(self, atom_types, embedding, distances='train_val', extra_embeddings=None,type='full'):

        uncertainties = torch.zeros_like(embedding[:,0])

        self.test_distances = {}
        self.min_vectors = {}
        for key in self.config.get('chemical_symbol_to_type'):
            if distances == 'train_val':
                embeddings = torch.cat([self.train_embeddings[key],self.test_embeddings[key]])
            elif distances == 'train':
                embeddings = self.train_embeddings[key]

            if extra_embeddings is not None:
                embeddings = torch.cat([embeddings,extra_embeddings[key]])

            mask = (atom_types==self.config.get('chemical_symbol_to_type')[key]).flatten()

            latent_force_distances = torch.cdist(embeddings,embedding[mask],p=2)

            inds = torch.argmin(latent_force_distances,axis=0)
            min_distance = torch.tensor([latent_force_distances[ind,i] for i, ind in enumerate(inds)])
            min_vectors = torch.vstack([embeddings[ind]-embedding[mask][i] for i, ind in enumerate(inds)])

            self.test_distances[key] = min_distance.detach().cpu().numpy()
            self.min_vectors[key] = min_vectors.detach().cpu().numpy()
        
            # uncertainties[mask] = self.params[key][0]+min_distance*self.params[key][1]
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
            elif type == 'distance':
                uncertainty = torch.sum(distance*sig_2,axis=1)

            uncertainty_raw[i] = uncertainty
        uncertainty_mean = torch.mean(uncertainty_raw,axis=0)
        uncertainty_std = torch.std(uncertainty_raw,axis=0)

        uncertainty_ens = uncertainty_mean + uncertainty_std

        return uncertainty_ens

    def predict_from_traj(self, traj, max=True, batch_size=2):
        uncertainty = []
        atom_embeddings = []
        data = [self.transform(AtomicData.from_ase(atoms=atoms,r_max=self.r_max, self_interaction=self.self_interaction)) for atoms in traj]
        dataset = DataLoader(data, batch_size=batch_size)
        for i, batch in enumerate(dataset):
            uncertainty.append(self.predict(batch))
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
        
        n = len(self.config.get('chemical_symbol_to_type'))
        fig, ax = plt.subplots(1,n, figsize=(5*n,5))
        if n == 1:
            ax = [ax]
        max_x = 0
        max_y = 0
        for i, key in enumerate(self.config.get('chemical_symbol_to_type')):
            max_x = max([max_x, self.min_distances[key].max()])
            max_y = max([max_y, self.test_errors[key].reshape(-1).detach().numpy().max()])
        for i, key in enumerate(self.config.get('chemical_symbol_to_type')):
            ax[i].scatter(self.min_distances[key],self.test_errors[key].reshape(-1).detach(),alpha=0.2)

            sigabs = self.params[key][0]
            d_fit = np.linspace(0,max_x)
            if len(sigabs) == 2:
                error_fit = sigabs[0] + sigabs[1]*d_fit
            else:
                error_fit = sigabs[0] + max(sigabs[1:])*d_fit
            ax[i].plot(d_fit,error_fit,label=key)

            ax[i].legend()
            ax[i].set_xlabel('Embedding Distance')
            ax[i].set_ylabel('Force Error')
            ax[i].set_ylim((0,1.1*max_y))
            
        if filename is not None:
            plt.savefig(filename)

class latent_distance_uncertainty_Nequip_adversarialNN():
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.r_max = config.r_max
        self.latent_size = int(self.config['conv_to_output_hidden_irreps_out'].split('x')[0])

        chemical_symbol_to_type = config.get('chemical_symbol_to_type')
        self.transform = TypeMapper(chemical_symbol_to_type=chemical_symbol_to_type)

        self.self_interaction = self.config.get('dataset_extra_fixed_fields',False).get('self_interaction',False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.kb = 8.6173e-5 #eV/K
        self.params_func = getattr(sys.modules[__name__],config.get('params_func','optimize2params'))

        self.n_ensemble = config.get('n_uncertainty_ensembles',4)
            

    def transform_data_input(self,data):
        data.to(torch.device(self.device))
        data = AtomicData.to_AtomicDataDict(data)
        
        return data

    def calibrate(self,debug=False):
        
        
        uncertainty_dir = os.path.join(self.config['workdir'],self.config.get('uncertainty_dir', 'uncertainty'))
        os.makedirs(uncertainty_dir,exist_ok=True)
        state_dict_func = lambda n: os.path.join(uncertainty_dir, f'uncertainty_state_dict_{n}.pth')
        metrics_func = lambda n: os.path.join(uncertainty_dir, f'uncertainty_metrics_{n}.csv')

        hidden_dimensions = self.config.get('uncertainty_hidden_dimensions', [])
        unc_epochs = self.config.get('uncertainty_epochs', 2000)

        self.NNs = []
        train_indices = []
        for n in range(self.n_ensemble):
            state_dict_name = state_dict_func(n)
            if os.path.isfile(state_dict_name):
                NN = uncertainty_NN(self.latent_size, hidden_dimensions)
                try:
                    NN.load_state_dict(torch.load(state_dict_name))
                    self.NNs.append(NN)
                except:
                    print(f'Loading uncertainty {n} failed')
                    train_indices.append(n)
            else:
                train_indices.append(n)

        if len(train_indices)>0:
            self.parse_data()

            for n in train_indices:
                
                #train NN to fit energies
                NN = uncertainty_NN(self.latent_size, hidden_dimensions,epochs=unc_epochs)
                NN.train(self.test_embeddings, self.test_errors)
                self.NNs.append(NN)
                torch.save(NN.get_state_dict(),state_dict_func(n))
                pd.DataFrame(NN.metrics).to_csv(metrics_func(n))

    
    def parse_data(self):

        dataset = dataset_from_config(self.config)

        train_embeddings = torch.empty((0,self.latent_size),device=self.device)
        train_errors = torch.empty((0),device=self.device)
        train_energies = torch.empty((0),device=self.device)
        for data in dataset[self.config.train_idcs]:
            out = self.model(self.transform_data_input(data))
            train_energies = torch.cat([train_energies, out['total_energy']])
            train_embeddings = torch.cat([train_embeddings,out['node_features']])
            
            error = torch.absolute(out['forces'] - data.forces)
            train_errors = torch.cat([train_errors,error.mean(dim=1)])

        self.train_energies = train_energies
        self.train_embeddings = train_embeddings
        self.train_errors = train_errors

        test_embeddings = torch.empty((0,self.latent_size),device=self.device)
        test_errors = torch.empty((0),device=self.device)
        test_energies = torch.empty((0),device=self.device)
        for data in dataset[self.config.val_idcs]:
            out = self.model(self.transform_data_input(data))
            test_energies = torch.cat([test_energies, out['total_energy']])

            error = torch.absolute(out['forces'] - data.forces)

            test_embeddings = torch.cat([test_embeddings,out['node_features']])
            test_errors = torch.cat([test_errors,error.mean(dim=1)])
        
        self.test_embeddings = test_embeddings
        self.test_errors = test_errors
        self.test_energies = test_energies


    def adversarial_loss(self, data, T, distances='train_val'):
        if not hasattr(self, 'train_energies'):
            self.parse_data()

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

        probability = 1/Q * torch.exp(-(out['total_energy']-emean)/estd/kT)
        
        uncertainties = self.predict_uncertainty(data['atom_types'], self.atom_embedding).to(self.device)

        adv_loss = (probability * uncertainties).sum()

        return adv_loss

    def predict_uncertainty(self, atom_types, embedding, distances='train_val', extra_embeddings=None,type='full'):

        uncertainty_raw = torch.zeros(self.n_ensemble,embedding.shape[0])
        for i, NN in enumerate(self.NNs):
            uncertainty_raw[i] = NN.predict(embedding).squeeze()
        
        uncertainty_mean = torch.mean(uncertainty_raw,axis=0)
        uncertainty_std = torch.std(uncertainty_raw,axis=0)

        if type == 'full':
            uncertainty_ens = uncertainty_mean + uncertainty_std
        elif type == 'mean':
            uncertainty_ens = uncertainty_mean
        elif type == 'std':
            uncertainty_ens = uncertainty_std

        return uncertainty_ens


    def predict_from_traj(self, traj, max=True, batch_size=2):
        uncertainty = []
        atom_embeddings = []
        data = [self.transform(AtomicData.from_ase(atoms=atoms,r_max=self.r_max, self_interaction=self.self_interaction)) for atoms in traj]
        dataset = DataLoader(data, batch_size=batch_size)
        for i, batch in enumerate(dataset):
            uncertainty.append(self.predict(batch))
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

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class unc_Dataset(Dataset):
    def __init__(self, x, y) -> None:
        super(Dataset, self).__init__()
        assert len(x)==len(y)
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class embed_input(torch.nn.Module):
    def __init__(self, input_dim, num_basis=8, trainable=True):
        super().__init__()

        self.input_dim = input_dim
        self.trainable = trainable
        self.num_basis = num_basis

        stds = torch.ones((input_dim,num_basis))
        means = torch.full(size=stds.size(),fill_value=-(num_basis+1)/2.)+stds.cumsum(dim=1)


        if self.trainable:
            self.means = torch.nn.Parameter(means)
            self.stds = torch.nn.Parameter(stds)
        else:
            self.register_buffer("means", means)
            self.register_buffer("stds", stds)

    def forward(self, x):
        x = (x[..., None] - self.means[None,...]) / self.stds[None,...]
        x = x.square().mul(-0.5).exp() / self.stds  # sqrt(2 * pi)
        x = x.reshape(x.shape[0],-1)
        return x

    def rewrite(self, x):
        mean = x.mean(dim=0)
        std = x.std(dim=0)

        stds = torch.outer(std,torch.ones(self.num_basis))
        mean_shifts = torch.outer(std,torch.linspace(-(self.num_basis-1)/2.,(self.num_basis-1)/2.,self.num_basis))
        means = torch.outer(mean,torch.ones(self.num_basis)) + mean_shifts

        self.means = torch.nn.Parameter(means)
        self.stds = torch.nn.Parameter(stds)

class rescale_input(torch.nn.Module):
    def __init__(self, x: torch.Tensor=None, input_dim = 1, trainable = True):
        # super(torch.nn.Module, self).__init__()
        super(rescale_input, self).__init__()

        self.trainable = trainable
        if x is not None:
            # self.mean = torch.tensor(x.mean(dim=0), requires_grad=trainable)
            # self.std = torch.tensor(x.std(dim=0), requires_grad=trainable)
            self.mean = torch.nn.Parameter(x.mean(dim=0))
            self.std = torch.nn.Parameter(x.std(dim=0))
        else:
            self.mean = torch.nn.Parameter(torch.zeros((input_dim)))
            self.std = torch.nn.Parameter(torch.ones((input_dim)))

    def forward(self, x):
        x = (x-self.mean)/self.std
        return x

    def rewrite(self, x):

        self.mean = torch.nn.Parameter(x.mean(dim=0))
        self.std = torch.nn.Parameter(x.std(dim=0))


class uncertainty_NN():
    def __init__(self, 
    input_dim, 
    hidden_dimensions=[], 
    act=torch.nn.ReLU, 
    train_percent = 0.8,
    epochs=1000, 
    lr = 0.001, 
    momentum=0.9,
    patience= None,
    min_lr = None) -> None:
        self.train_percent = train_percent
        if patience is None:
            patience = epochs/10
        if min_lr is None:
            self.min_lr = lr/100
        
        # layers = [rescale_input(input_dim=input_dim)]
        layers = [embed_input(input_dim=input_dim)]
        if isinstance(layers[-1],embed_input):
            input_dim = layers[-1].input_dim*layers[-1].num_basis
        
        if len(hidden_dimensions)==0:
            layers.append(torch.nn.Linear(input_dim, 1))
        else:
            # layers = []
            for i, hd in enumerate(hidden_dimensions):
                if i == 0:
                    layers.append(torch.nn.Linear(input_dim, hd))
                else:
                    layers.append(torch.nn.Linear(hidden_dimensions[i-1], hd))
                layers.append(act())

            layers.append(torch.nn.Linear(hidden_dimensions[-1], 1))
        
        self.model = torch.nn.Sequential(*layers)
        # self.model = Network(input_dim, hidden_dimensions, act)
        print('Trainable parameters:', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.epochs = epochs
        self.loss = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.lr_scheduler = LRScheduler(self.optim, patience, self.min_lr)

    def train(self, x, y):
        x = torch.tensor(x) #Break computational graph for training
        y = torch.tensor(y)

        y = torch.log(y)

        if isinstance(self.model[0], rescale_input) or isinstance(self.model[0], embed_input) :
            self.model[0].rewrite(x)

        n_train = int(len(x)*self.train_percent)
        rand_ind = torch.randperm(len(x))
        train_ind = rand_ind[:n_train]
        training_data = unc_Dataset(x[train_ind],y[train_ind])
        val_ind = rand_ind[n_train:]
        validation_data = unc_Dataset(x[val_ind],y[val_ind])

        train_dataloader = DataLoader(training_data, batch_size=1000, shuffle=True)
        validation_dataloader = DataLoader(validation_data, batch_size=1000, shuffle=True)

        metrics = {
            'lr': [],
            'train_loss': [],
            'validation_loss': [],
        }
        for n in range(self.epochs):
            running_loss = 0
            for i, data in enumerate(train_dataloader):
                inputs, outputs = data
                self.model.train()
                self.model.zero_grad()
                pred = self.model(inputs)

                loss = self.loss(pred,outputs.unsqueeze(1))

                # self.optim.zero_grad()
                loss.backward()

                self.optim.step()
                running_loss += loss.item()*len(inputs)
            
            train_loss = running_loss/n_train
            running_loss = 0
            for i, data in enumerate(validation_dataloader):
                inputs, outputs = data
                self.model.eval()
                pred = self.model(inputs)

                loss = self.loss(pred,outputs.unsqueeze(1))

                running_loss += loss.item()*len(inputs)
            validation_loss = running_loss/len(val_ind)
            
            self.lr_scheduler(validation_loss)
            
            metrics['lr'].append(self.optim.param_groups[0]['lr'])
            metrics['train_loss'].append(train_loss)
            metrics['validation_loss'].append(validation_loss)

            if self.optim.param_groups[0]['lr'] == self.min_lr:
                break
        
        self.metrics = metrics
        self.train_loss = train_loss
        self.validation_loss = validation_loss

    def predict(self,x):
        self.model.eval()
        pred = torch.exp(self.model(x))

        return pred

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)





def optimize2params(test_errors, min_vectors):

    min_distances = np.linalg.norm(min_vectors,axis=1).reshape(-1,1)
    params0 = np.random.rand(2)
    res = minimize(optimizeparams,params0,args=(test_errors,min_distances),method='Nelder-Mead')
    print(res,flush=True)
    params = np.abs(res.x)

    return params

def optimizevecparams(test_errors, min_vectors):

    min_vectors = np.abs(min_vectors)
    params0 = np.random.rand(min_vectors.shape[1]+1) # [0.01]*(min_vectors.shape[1]+1)
    res = minimize(optimizeparams,params0,args=(test_errors,min_vectors),method='Nelder-Mead', options={'maxiter':1000000})
    print(res,flush=True)
    params = np.abs(res.x)

    return params

def optimizeparams(params,eps_d,d):
    sig_1, sig_2 = params[0],params[1:]
    
    sd = np.abs(sig_1) + (d*np.abs(sig_2)).sum(axis=1)
    
    negLL = -np.sum( stats.norm.logpdf(eps_d, loc=0, scale=sd) )
    
    return negLL    

   
