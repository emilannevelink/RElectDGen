import torch
import numpy as np
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

    def calibrate(self):
        #latent_size = self.model.final.tp.irreps_in1.dim #monolayer energy
        
        dataset = dataset_from_config(self.config)

        train_force_latent_distances = torch.empty((0,self.latent_size),device=self.device)

        train_embeddings = {}
        test_embeddings = {}
        test_errors = {}
        for key in self.config.get('chemical_symbol_to_type'):
            train_embeddings[key] = torch.empty((0,self.latent_size),device=self.device)
            test_embeddings[key] = torch.empty((0,self.latent_size),device=self.device)
            test_errors[key] = torch.empty((0),device=self.device)
    
        for data in dataset[self.config.train_idcs]:
            out = self.model(self.transform_data_input(data))
            
            train_force_latent_distances = torch.cat([train_force_latent_distances,out['node_features']])

            for key in self.config.get('chemical_symbol_to_type'):
                mask = data['atom_types']==self.config.get('chemical_symbol_to_type')[key]
                train_embeddings[key] = torch.cat([train_embeddings[key],out['node_features'][mask]])

        self.train_embeddings = train_embeddings

        self.train_force_latent_distances = train_force_latent_distances

        test_force_latent_distances = torch.empty((0,self.latent_size),device=self.device)
        test_force_errors = torch.empty((0,3),device=self.device)

        for data in dataset[self.config.val_idcs]:
            out = self.model(self.transform_data_input(data))
            test_force_latent_distances = torch.cat([test_force_latent_distances,out['node_features']]) 
            
            error = torch.absolute(out['forces'] - data.forces)
            test_force_errors = torch.cat([test_force_errors,error])

            for key in self.config.get('chemical_symbol_to_type'):
                mask = data['atom_types']==self.config.get('chemical_symbol_to_type')[key]
                test_embeddings[key] = torch.cat([test_embeddings[key],out['node_features'][mask]])
                test_errors[key] = torch.cat([test_errors[key],error.mean(dim=1)[mask]])
        self.test_embeddings = test_embeddings

        self.test_force_errors = test_force_errors.mean(dim=1).detach().cpu().numpy()
        
        self.val_latent_distances = test_force_latent_distances
        self.train_val_latent_distances = torch.cat([train_force_latent_distances,test_force_latent_distances])

        self.latent_force_distances = torch.cdist(train_force_latent_distances,test_force_latent_distances,p=2)
        
        inds = torch.argmin(self.latent_force_distances,axis=0)
        self.d_force_test = torch.tensor([self.latent_force_distances[ind,i] for i, ind in enumerate(inds)]).detach().cpu().numpy()

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

            mask = data['atom_types']==self.config.get('chemical_symbol_to_type')[key]

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
        ax.scatter(self.d_force_test,self.test_force_errors.reshape(-1),alpha=0.2)

        sigabs = np.abs(self.sigmas)
        d_fit = np.linspace(0,self.d_force_test.max())
        error_fit = sigabs[0] + sigabs[1]*d_fit
        ax.plot(d_fit,error_fit,'r')

        ax.set_title(f'Sigmas: {sigabs[0]}, {sigabs[1]}')
        ax.set_xlabel('Embedding Distance')
        ax.set_ylabel('Force Error')

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

def optimizeparams(params,eps_d,d):
    sig_1, sig_2 = params[0],params[1]
    
    sd = np.abs(sig_1) + d*np.abs(sig_2)
    
    negLL = -np.sum( stats.norm.logpdf(eps_d, loc=0, scale=sd) )
    
    return negLL    
