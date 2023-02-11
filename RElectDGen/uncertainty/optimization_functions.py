import torch
import numpy as np
from scipy.optimize import minimize
from scipy import stats

from torch.utils.data import Dataset, DataLoader
from nequip.data import AtomicData
import copy

def find_NLL_params(errors,raw_uncertainties,polyorder=1):
    errors = np.array(errors)
    raw_uncertainties = np.array(raw_uncertainties)
    def loss(coeffs):
        uncertainties = np.poly1d(coeffs)(raw_uncertainties)
        return npNLL(errors,uncertainties)

    coeffs0 = np.ones(polyorder+1)
    bounds = [(0,None)]*len(coeffs0)
    res = minimize(loss,coeffs0,bounds=bounds,method='Nelder-Mead')
    print(res,flush=True)
    coeffs = res.x

    return coeffs

def NLL(errors,uncertainties):
    return (torch.pow(errors/uncertainties,2) + torch.log(uncertainties)).mean()

def npNLL(errors,uncertainties):
    return (np.power(errors/uncertainties,2) + np.power(uncertainties)).mean()

def optimize2params(test_errors, min_vectors):

    min_distances = np.linalg.norm(min_vectors,axis=1).reshape(-1,1)
    params0 = np.random.rand(2)
    bounds = [(0,None)]*len(params0)
    res = minimize(optimizeparams,params0,args=(test_errors,min_distances),bounds=bounds,method='Nelder-Mead')
    print(res,flush=True)
    params = np.abs(res.x)

    return params

def optimizevecparams(test_errors, min_vectors):

    min_vectors = np.abs(min_vectors)
    params0 = np.random.rand(min_vectors.shape[1]+1) # [0.01]*(min_vectors.shape[1]+1)
    bounds = [(0,None)]*len(params0)
    res = minimize(optimizeparams,params0,args=(test_errors,min_vectors),bounds=bounds,method='Nelder-Mead', options={'maxiter':1000000})
    print(res,flush=True)
    params = np.abs(res.x)

    return params

def optimizeparams(params,error_d,d):
    sig_1, sig_2 = params[0],params[1:]
    
    sd = np.abs(sig_1) + (d*np.abs(sig_2)).sum(axis=1)
    
    # negLL = -np.sum( stats.norm.logpdf(error_d, loc=0, scale=sd) )
    # negLL = NLL(error_d,sd)
    negLL = (np.power(error_d/sd,2) + np.log(sd)).mean()
    
    return negLL    


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
        self.best_model = copy.deepcopy(self.model)
        # self.model = Network(input_dim, hidden_dimensions, act)
        print('Trainable parameters:', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.epochs = epochs
        # self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.NLLLoss()
        self.loss = NLL
        self.optim = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.lr_scheduler = LRScheduler(self.optim, patience, self.min_lr)

    def train(self, x, y):
        # x = torch.tensor(x).to(self.device) #Break computational graph for training
        x = x.clone().detach()
        # y = torch.tensor(y).to(self.device)
        y = y.clone().detach()

        # y = torch.log(y)

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
                inputs, errors = data
                self.model.train()
                self.model.zero_grad()
                unc = torch.exp(self.model(inputs))

                loss = self.loss(errors.unsqueeze(1),unc)

                # self.optim.zero_grad()
                loss.backward()

                self.optim.step()
                running_loss += loss.item()*len(inputs)
            
            train_loss = running_loss/n_train
            running_loss = 0
            for i, data in enumerate(validation_dataloader):
                inputs, errors = data
                self.model.eval()
                unc = torch.exp(self.model(inputs))

                loss = self.loss(errors.unsqueeze(1),unc)

                running_loss += loss.item()*len(inputs)
            validation_loss = running_loss/len(val_ind)
            
            self.lr_scheduler(validation_loss)
            
            metrics['lr'].append(self.optim.param_groups[0]['lr'])
            metrics['train_loss'].append(train_loss)
            metrics['validation_loss'].append(validation_loss)

            if np.argmin(metrics['validation_loss']) == n:
                self.best_model = copy.deepcopy(self.model)

            if self.optim.param_groups[0]['lr'] == self.min_lr:
                break
        
        self.metrics = metrics
        self.train_loss = train_loss
        self.validation_loss = validation_loss

    def predict(self,x):
        self.model.eval()
        unc = torch.exp(self.best_model(x)) ### this dramatically increases performance and ensures uncertainties are positive
        # unc = torch.abs(self.model(x))

        return unc

    def get_state_dict(self):
        return self.best_model.state_dict()

    def load_state_dict(self, state_dict):
        self.best_model.load_state_dict(state_dict)

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(ard_num_dims=2,num_mixtures=4)
        # self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
class uncertainty_GPR():
    def __init__(self, 
    input_dim, 
    ninducing_points,
    epochs=1000, 
    train_percent = 0.8,
    lr = 0.01, 
    momentum=0.9,
    patience= None,
    min_lr = None) -> None:
        
        if patience is None:
            patience = epochs/10
        if min_lr is None:
            min_lr = lr/100
        
        self.patience = patience
        self.min_lr = min_lr
        self.train_percent = train_percent
        self.ninducing_points = ninducing_points
        self.epochs = epochs
        self.input_dim = input_dim

        inducing_points = torch.rand((self.ninducing_points,input_dim))#*(xmax-xmin)-xmean)*1.5
        self.model = GPModel(inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.best_model = copy.deepcopy(self.model)
        self.best_likelihood = copy.deepcopy(self.likelihood)
        

    def train(self, x, y):
        # x = torch.tensor(x).to(self.device) #Break computational graph for training
        x = x.clone().detach()
        # y = torch.tensor(y).to(self.device)
        y = torch.log(y.clone().detach())

        xmax = x.max(axis=0).values
        xmin = x.min(axis=0).values
        xmean = x.mean(axis=0)

        inducing_points = (torch.rand((self.ninducing_points,self.input_dim))*(xmax-xmin)-xmean)*1.5
        self.model = GPModel(inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
        
        print('Trainable parameters:', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        

        n_train = int(len(x)*self.train_percent)
        rand_ind = torch.randperm(len(x))
        train_ind = rand_ind[:n_train]
        training_data = unc_Dataset(x[train_ind],y[train_ind])
        val_ind = rand_ind[n_train:]
        validation_data = unc_Dataset(x[val_ind],y[val_ind])

        train_dataloader = DataLoader(training_data, batch_size=1000, shuffle=True)
        validation_dataloader = DataLoader(validation_data, batch_size=1000, shuffle=False)

        metrics = {
            'lr': [],
            'train_loss': [],
            'validation_loss': [],
        }

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=n_train)

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.01)
        self.lr_scheduler = LRScheduler(optimizer, self.patience, self.min_lr)
        self.model.train()
        self.likelihood.train()
        for n in range(self.epochs):
            running_loss = 0
            for i, data in enumerate(train_dataloader):
                inputs, errors = data
                self.model.train()
                self.likelihood.train()
                
                optimizer.zero_grad()
                
                unc = self.model(inputs)

                loss = -mll(unc, errors)

                # self.optim.zero_grad()
                loss.backward()

                optimizer.step()
                running_loss += loss*len(inputs)
            
            train_loss = running_loss/n_train
            running_loss = 0
            for i, data in enumerate(validation_dataloader):
                inputs, errors = data
                self.model.eval()
                self.likelihood.eval()
                unc = self.model(inputs)

                loss = -mll(unc, errors)

                running_loss += loss*len(inputs)
            validation_loss = running_loss/len(val_ind)
            
            self.lr_scheduler(validation_loss)
            
            metrics['lr'].append(optimizer.param_groups[0]['lr'])
            metrics['train_loss'].append(train_loss.detach())
            metrics['validation_loss'].append(validation_loss.detach())

            if np.argmin(metrics['validation_loss']) == n:
                self.best_model.load_state_dict(self.model.state_dict())
                self.best_likelihood.load_state_dict(self.likelihood.state_dict())

            if optimizer.param_groups[0]['lr'] == self.min_lr:
                break
        
        self.metrics = metrics
        self.train_loss = train_loss
        self.validation_loss = validation_loss

    def predict(self,x):
        self.best_model.eval()
        # unc = torch.exp(self.best_model(x)) ### this dramatically increases performance and ensures uncertainties are positive
        # unc = torch.abs(self.model(x))
        observed_pred = self.best_likelihood(self.best_model(x))
        lower, upper = observed_pred.confidence_region()

        return torch.exp(observed_pred.mean), torch.exp(upper)

    def get_state_dict(self):
        return {
            'model_dict': self.best_model.state_dict(),
            'likelihood_dict': self.best_likelihood.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.best_model.load_state_dict(state_dict['model_dict'])
        self.best_likelihood.load_state_dict(state_dict['likelihood_dict'])


class uncertaintydistance_NN():
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

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train_percent = train_percent
        if patience is None:
            patience = epochs/10
        if min_lr is None:
            self.min_lr = lr/100
        
        # layers = [rescale_input(input_dim=input_dim)]
        layers = []
        # if isinstance(layers[-1],embed_input):
        #     input_dim = layers[-1].input_dim*layers[-1].num_basis
        
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
        
        self.model = torch.nn.Sequential(*layers).to(self.device)
        # self.model = Network(input_dim, hidden_dimensions, act)
        print('Trainable parameters:', sum(p.numel() for p in self.model.parameters() if p.requires_grad), flush=True)
        self.epochs = epochs
        self.prob = lambda value, std: torch.max(torch.hstack([torch.exp(-value.pow(2)/2/std.pow(2))/std/(2*np.pi)**0.5,1e-10*torch.ones_like(value)]),dim=-1).values
        self.loss = lambda error, pred_std: -torch.sum( torch.log( self.prob(error, pred_std)) )
        self.optim = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.lr_scheduler = LRScheduler(self.optim, patience, self.min_lr)
        
        

    def train(self, x, y):
        # x = torch.tensor(x).to(self.device) #Break computational graph for training
        x = x.clone().detach()
        # y = torch.tensor(y).to(self.device)
        y = y.clone().detach()

        # y = torch.log(y)

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

                # loss = self.loss(pred,outputs.unsqueeze(1))
                loss = self.loss(outputs.unsqueeze(1), pred.abs())
                # loss = self.loss(pred.squeeze(),outputs)

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

                loss = self.loss(outputs.unsqueeze(1), pred.abs())

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
        x = torch.tensor(x)
        self.model.eval()
        pred = self.model(x)
        pred = torch.abs(pred)
        # pred = torch.exp(pred)

        return pred

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

class uncertainty_ensemble_NN():
    def __init__(self,
    input_dim,
    natoms,
    hidden_dimensions=[], 
    act=torch.nn.ReLU, 
    batch_size=500,
    epochs=2000, 
    lr = 0.001, 
    momentum=0.9,
    patience= None,
    early_stopping_patience = None,
    min_lr = None,
    fine_tune_train_dataset_percentage = 0.5) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.nequip_model = nequip_model #.train()
        # self.nequip_model.train()
        self.natoms = natoms
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.initial_lr = lr
        self.fine_tune_train_dataset_percentage = fine_tune_train_dataset_percentage
        
        if patience is None:
            patience = epochs/10
        if early_stopping_patience is None:
            early_stopping_patience = int(patience * 2.5)
        if min_lr is None:
            self.min_lr = lr/1e3

        self.patience = patience
        self.early_stopping_patience = early_stopping_patience
        
        layers = []
        if len(hidden_dimensions)==0:
            layers.append(torch.nn.Linear(input_dim+natoms, 1))
        else:
            # layers = []
            for i, hd in enumerate(hidden_dimensions):
                if i == 0:
                    layers.append(torch.nn.Linear(input_dim+natoms, hd))
                else:
                    layers.append(torch.nn.Linear(hidden_dimensions[i-1], hd))
                layers.append(act())

            layers.append(torch.nn.Linear(hidden_dimensions[-1], 1))
        
        self.model = torch.nn.Sequential(*layers).to(self.device)
        # self.model = Network(input_dim, hidden_dimensions, act)
        print('Trainable parameters:', sum(p.numel() for p in self.model.parameters() if p.requires_grad),flush=True)
        self.epochs = epochs
        self.loss = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.lr_scheduler = LRScheduler(self.optim, patience, self.min_lr)

    def make_dataloader(self, latents, energies):
        all_latents = torch.empty(0,self.input_dim+self.natoms, device=self.device)
        all_energies = torch.empty(0, device=self.device)
        for key in latents:
            all_latents = torch.cat([all_latents, latents[key]])
            all_energies = torch.cat([all_energies, energies[key]])

        data = unc_Dataset(all_latents,all_energies)
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        return dataloader

    def train(self,
        train_latents,
        train_energies,
        validation_latents,
        validation_energies
    ):


        train_dataloader = self.make_dataloader(train_latents, train_energies)
        validation_dataloader = self.make_dataloader(validation_latents, validation_energies)
        
        # train_latents = torch.empty(0,self.input_dim).to(self.device)
        # train_latent_sensitivities = torch.empty(0,3,self.input_dim).to(self.device)
        # train2_latent_sensitivities = torch.empty(0,self.input_dim,3).to(self.device)
        # train3_latent_sensitivities = []
        # train_indices = torch.empty(0,dtype=int).to(self.device)
        # for i, data in enumerate(self.train_dataset):
        #     data['pos'].requires_grad = True
        #     out = self.nequip_model(self.transform_data_input(data))

        #     tmp_latent_sensititivies = torch.zeros(*data['pos'].shape,self.input_dim)
        #     tmp2_latent_sensititivies = torch.zeros(self.input_dim,*data['pos'].shape)
        #     for j, feat in enumerate(out['node_features']):
        #         for k, dd in enumerate(feat):
        #             latent_senstivities = torch.autograd.grad(
        #                 dd, data['pos'],  retain_graph=True
        #             )[0]
        #             tmp2_latent_sensititivies[k,:,:] = latent_senstivities

        #         train3_latent_sensitivities.append(tmp2_latent_sensititivies.unsqueeze(0))

        #     tmp2_latent_sensititivies = torch.zeros(self.input_dim,3)
        #     for j, feat in enumerate(out['node_features'].T):
        #         latent_senstivities = torch.autograd.grad(
        #             feat, data['pos'], grad_outputs=torch.ones(len(data['pos'])), retain_graph=True
        #         )[0]
        #         tmp_latent_sensititivies[:,:,j] = latent_senstivities

        #         latent_senstivities = torch.autograd.grad(
        #             feat, data['pos'], grad_outputs=torch.ones(len(data['pos'])), retain_graph=True
        #         )[0]
        #         tmp2_latent_sensititivies[j,:] = latent_senstivities.sum(dim=0)


        #     train_latents = torch.cat([train_latents, out['node_features']],dim=0).to(self.device)
        #     train_latent_sensitivities = torch.cat([train_latent_sensitivities, tmp_latent_sensititivies],dim=0).to(self.device)
        #     train2_latent_sensitivities = torch.cat([train2_latent_sensitivities, tmp2_latent_sensititivies[None,:,:]],dim=0).to(self.device)
        #     npoints = torch.tensor([train_indices[-1]+len(data['pos']) if i>0 else len(data['pos'])]).to(self.device)
        #     train_indices = torch.cat([train_indices,npoints]).to(self.device)

        # validation_latents = torch.empty(0,self.input_dim).to(self.device)
        # validation_latent_sensitivities = torch.empty(0,3,self.input_dim).to(self.device)
        # validation_indices = torch.empty(0,dtype=int).to(self.device)
        # for i, data in enumerate(self.validation_dataset):
        #     data['pos'].requires_grad = True
        #     out = self.nequip_model(self.transform_data_input(data))

        #     tmp_latent_sensititivies = torch.zeros(*data['pos'].shape,self.input_dim)
        #     # tmp2_latent_sensititivies = torch.zeros(*data['pos'].shape,self.input_dim)
        #     for j, feat in enumerate(out['node_features'].T):
        #         latent_senstivities = torch.autograd.grad(
        #             feat, data['pos'], grad_outputs=torch.ones(len(data['pos'])), retain_graph=True
        #         )[0]
        #         tmp_latent_sensititivies[:,:,j] = latent_senstivities
        #         # for k, dd in enumerate(feat):
        #         #     latent_senstivities = torch.autograd.grad(
        #         #         dd, data['pos'],  retain_graph=True
        #         #     )[0]
        #         #     tmp2_latent_sensititivies[:,:,j] += latent_senstivities


        #     validation_latents = torch.cat([validation_latents, out['node_features']],dim=0).to(self.device)
        #     validation_latent_sensitivities = torch.cat([validation_latent_sensitivities, tmp_latent_sensititivies],dim=0).to(self.device)
        #     npoints = torch.tensor([validation_indices[-1]+len(data['pos']) if i>0 else len(data['pos'])]).to(self.device)
        #     validation_indices = torch.cat([validation_indices,npoints]).to(self.device)

        self.best_model = copy.deepcopy(self.model)
        self.best_loss = torch.tensor(np.inf)

        metrics = {
            'lr': [],
            'train_loss': [],
            'validation_loss': [],
        }
        for n in range(self.epochs):
            running_loss = 0
            for i, data in enumerate(train_dataloader):
                latents, energies = data
                self.model.train()
                self.model.zero_grad()

                # atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze(),num_classes=self.natoms)

                # data['pos'].requires_grad = True
                # out = self.nequip_model(self.transform_data_input(data))

                # NN_inputs = torch.hstack([out['node_features'], atom_one_hot])
                # pred = self.model(NN_inputs).sum() #pred atomic energies and sum to get total energies
                # pred_forces = torch.autograd.grad(
                #     pred, data['pos'], retain_graph=True
                # )[0]
                # ind_start = int(train_indices[i-1] if i>0 else 0)
                # ind_final = int(train_indices[i])
                # latents = train_latents[ind_start:ind_final]
                # NN_inputs = torch.hstack([latents, atom_one_hot])
                # pred = self.model(NN_inputs).sum() #pred atomic energies and sum to get total energies

                # dl_dx = train_latent_sensitivities[ind_start:ind_final]
                # dE_dl = torch.autograd.grad(
                #     pred, latents, retain_graph=True
                # )[0]

                # dl_dx2 = torch.vstack(train3_latent_sensitivities[ind_start:ind_final])

                # dE_dx4 = torch.einsum('ijkl,ij->kl', dl_dx2, dE_dl)
                # dE_dx3 = torch.einsum('ik,kj->ij',dE_dl,train2_latent_sensitivities[i])
                # dE_dx2 = torch.einsum('ik,jk->ij',dE_dl,dl_dx.sum(dim=0))
                # dE_dx = torch.einsum('ik,ijk->ij',dE_dl,dl_dx)

                # loss = (self.energy_factor * self.loss(pred,data['total_energy']) + 
                #     self.force_factor * self.loss(pred_forces, data['forces']))
                # latents = torch.empty(0,self.input_dim+self.natoms, device=self.device)
                # energies = torch.empty(0, device=self.device)
                # for key in train_latents:
                #     ind_start = int(train_indices[key][i-1] if i>0 else 0)
                #     ind_final = int(train_indices[key][i])
                #     latents = torch.cat([latents, train_latents[key][ind_start:ind_final]])
                #     energies = torch.cat([energies, train_energies[key][ind_start:ind_final]])
                
                # NN_inputs = torch.hstack([latents, atom_one_hot])
                pred = self.model(latents) #pred atomic energies and sum to get total energies
                loss = self.loss(pred,energies)

                # self.optim.zero_grad()
                loss.backward()

                self.optim.step()
                running_loss += loss.item()
            
            train_loss = running_loss/(i+1)
            running_loss = 0
            for i, data in enumerate(validation_dataloader):
                latents, energies = data
                # if 'batch' in data:
                #     n_inputs = max(data['batch'])+1
                # else:
                #     n_inputs = 1

                # data['pos'].requires_grad = True
                # out = self.nequip_model(self.transform_data_input(data))

                # atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze(),num_classes=self.natoms)

                # NN_inputs = torch.hstack([out['node_features'], atom_one_hot])
                # pred = self.model(NN_inputs).sum() #pred atomic energies and sum to get total energies
                # pred_forces = torch.autograd.grad(
                #     pred, data['pos'], retain_graph=True
                # )[0]
                
                # loss = (self.energy_factor * self.loss(pred,data['total_energy']) + 
                #         self.force_factor * self.loss(pred_forces, data['forces']))
                # latents = torch.empty(0,self.input_dim+self.natoms, device=self.device)
                # energies = torch.empty(0, device=self.device)
                # for key in validation_latents:
                #     ind_start = int(validation_indices[key][i-1] if i>0 else 0)
                #     ind_final = int(validation_indices[key][i])
                #     latents = torch.cat([latents, validation_latents[key][ind_start:ind_final]])
                #     energies = torch.cat([energies, validation_energies[key][ind_start:ind_final]])

                pred = self.model(latents) #pred atomic energies and sum to get total energies
                loss = self.loss(pred,energies)

                running_loss += loss.item()
            validation_loss = running_loss/(i+1)
            
            if validation_loss < self.best_loss:
                self.best_epoch = n
                self.best_loss = validation_loss
                self.best_model = copy.deepcopy(self.model)
            elif n - self.best_epoch > self.early_stopping_patience:
                print(f'Validation loss hasnt improved in {self.early_stopping_patience} epochs stopping optimization at epoch {n}')
                break
            
            self.lr_scheduler(validation_loss)
            
            metrics['lr'].append(self.optim.param_groups[0]['lr'])
            metrics['train_loss'].append(train_loss)
            metrics['validation_loss'].append(validation_loss)

            if self.optim.param_groups[0]['lr'] == self.min_lr:
                print('Reached minimum learning rate')
                break
        
        self.metrics = metrics
        self.train_loss = train_loss
        self.validation_loss = validation_loss
        self.model = self.best_model

    def predict(self, data, nequip_model=None):
        if 'node_features' in data:
            out = data
        elif nequip_model is not None:
            if not data['pos'].requires_grad:
                data['pos'].requires_grad = True
            out = nequip_model(self.transform_data_input(data))
        else:
            ValueError
        atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze(),num_classes=self.natoms)

        NN_inputs = torch.hstack([out['node_features'], atom_one_hot])
        pred = self.model(NN_inputs)#.sum() #pred atomic energies and sum to get total energies

        return pred #, pred_forces

    def fine_tune(self,
        train_latents,
        train_energies,
        validation_latents,
        validation_energies,
        fine_tune_latents, 
        fine_tune_energies, 
        fine_tune_epochs=None,
    ):
        
        if fine_tune_epochs is None:
            fine_tune_epochs = int(self.epochs/10)

        self.fine_tune_early_stopping_patience = self.early_stopping_patience*fine_tune_epochs/self.epochs

        initial_lr = max(self.initial_lr/10, self.min_lr*10)
        optim = torch.optim.Adam(self.model.parameters(), lr = initial_lr)
        lr_scheduler = LRScheduler(optim, self.patience/10, self.min_lr)

        train_dataloader = self.make_dataloader(train_latents, train_energies)
        validation_dataloader = self.make_dataloader(validation_latents, validation_energies)
        fine_tune_dataloader = self.make_dataloader(fine_tune_latents, fine_tune_energies)

        i_train_max = self.fine_tune_train_dataset_percentage*len(train_dataloader)

        self.best_model = copy.deepcopy(self.model)
        self.best_loss = torch.tensor(np.inf)

        fine_tune_metrics = {
            'lr': [],
            'fine_tune_loss': [],
            'train_loss': [],
            'validation_loss': [],
        }
        for n in range(fine_tune_epochs):
            running_loss = 0
            for i, data in enumerate(fine_tune_dataloader):
                if i > i_train_max:
                    break
                latents, energies = data
                self.model.train()
                self.model.zero_grad()

                pred = self.model(latents) #pred atomic energies and sum to get total energies
                loss = self.loss(pred,energies)

                loss.backward()

                optim.step()
                running_loss += loss.item()
            
            fine_tune_loss = running_loss/(i+1)
            running_loss = 0
            for i, data in enumerate(train_dataloader):
                latents, energies = data
                self.model.train()
                self.model.zero_grad()

                pred = self.model(latents) #pred atomic energies and sum to get total energies
                loss = self.loss(pred,energies)

                loss.backward()

                optim.step()
                running_loss += loss.item()
            
            train_loss = running_loss/(i+1)
            running_loss = 0
            for i, data in enumerate(validation_dataloader):
                latents, energies = data
                
                pred = self.model(latents) #pred atomic energies and sum to get total energies
                loss = self.loss(pred,energies)

                running_loss += loss.item()
            validation_loss = running_loss/(i+1)
            
            if validation_loss < self.best_loss:
                self.best_epoch = n
                self.best_loss = validation_loss
                self.best_model = copy.deepcopy(self.model)
            elif n - self.best_epoch > self.fine_tune_early_stopping_patience:
                print(f'Validation loss hasnt improved in {self.fine_tune_early_stopping_patience} epochs stopping optimization at epoch {n}')
                break
            lr_scheduler(validation_loss)
            
            fine_tune_metrics['lr'].append(optim.param_groups[0]['lr'])
            fine_tune_metrics['fine_tune_loss'].append(fine_tune_loss)
            fine_tune_metrics['train_loss'].append(train_loss)
            fine_tune_metrics['validation_loss'].append(validation_loss)

            if optim.param_groups[0]['lr'] == self.min_lr:
                break
        
        self.fine_tune_metrics = fine_tune_metrics
        self.train_loss = train_loss
        self.validation_loss = validation_loss
        self.model = self.best_model

    def transform_data_input(self, data):
        # assert len(data['total_energy']) == 1
        data = AtomicData.to_AtomicDataDict(data)
        
        return data
    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

class uncertainty_ensemble_NN_local():
    def __init__(self,
    input_dim,
    natoms,
    hidden_dimensions=[], 
    act=torch.nn.ReLU, 
    batch_size=100,
    epochs=2000, 
    lr = 0.001, 
    momentum=0.9,
    patience= None,
    early_stopping_patience = None,
    min_lr = None,
    fine_tune_train_dataset_percentage = 0.5) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.nequip_model = nequip_model #.train()
        # self.nequip_model.train()
        self.natoms = natoms
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.initial_lr = lr
        self.fine_tune_train_dataset_percentage = fine_tune_train_dataset_percentage
        
        if patience is None:
            patience = epochs/10
        if early_stopping_patience is None:
            early_stopping_patience = int(patience * 2.5)
        if min_lr is None:
            self.min_lr = lr/1e3

        self.patience = patience
        self.early_stopping_patience = early_stopping_patience
        
        self.models = []
        self.optims = []
        self.lr_scheduler = []
        for i in range(natoms):
            layers = []
            if len(hidden_dimensions)==0:
                layers.append(torch.nn.Linear(input_dim+natoms, 1))
            else:
                # layers = []
                for i, hd in enumerate(hidden_dimensions):
                    if i == 0:
                        layers.append(torch.nn.Linear(input_dim+natoms, hd))
                    else:
                        layers.append(torch.nn.Linear(hidden_dimensions[i-1], hd))
                    layers.append(act())

                layers.append(torch.nn.Linear(hidden_dimensions[-1], 1))
            
            self.models.append(torch.nn.Sequential(*layers).to(self.device))
            self.optims.append(torch.optim.Adam(self.models[-1].parameters(), lr = lr))
            self.lr_scheduler.append(LRScheduler(self.optims[-1], patience, self.min_lr))
        
        print('Trainable parameters:', sum(p.numel() for p in self.models[-1].parameters() if p.requires_grad),flush=True)
        self.epochs = epochs
        self.loss = torch.nn.MSELoss()
        

    def make_dataloader(self, latents, energies, i):
        key = list(latents.keys())[i]
        data = unc_Dataset(latents[key],energies[key])
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        return dataloader

    def train(self,
        train_latents,
        train_energies,
        validation_latents,
        validation_energies
    ):
        metrics = {}
        self.best_models = []
        self.best_loss = []
        for ii in range(self.natoms):

            train_dataloader = self.make_dataloader(train_latents, train_energies,ii)
            validation_dataloader = self.make_dataloader(validation_latents, validation_energies, ii)
        
            self.best_models.append(copy.deepcopy(self.models[ii]))
            self.best_loss.append(torch.tensor(np.inf))

            metrics[f'lr_{ii}'] = np.full((self.epochs),np.nan)
            metrics[f'train_loss_{ii}'] = np.full((self.epochs),np.nan)
            metrics[f'validation_loss_{ii}'] = np.full((self.epochs),np.nan)
            
            for n in range(self.epochs):
                running_loss = 0
                for i, data in enumerate(train_dataloader):
                    latents, energies = data
                    self.models[ii].train()
                    self.models[ii].zero_grad()

                    pred = self.models[ii](latents) #pred atomic energies and sum to get total energies
                    loss = self.loss(pred,energies)

                    # self.optim.zero_grad()
                    loss.backward()

                    self.optims[ii].step()
                    running_loss += loss.item()
                
                train_loss = running_loss/(i+1)
                running_loss = 0
                for i, data in enumerate(validation_dataloader):
                    latents, energies = data
                    
                    pred = self.models[ii](latents) #pred atomic energies and sum to get total energies
                    loss = self.loss(pred,energies)

                    running_loss += loss.item()
                validation_loss = running_loss/(i+1)
                
                if validation_loss < self.best_loss[ii]:
                    self.best_epoch = n
                    self.best_loss[ii] = validation_loss
                    self.best_models[ii] = copy.deepcopy(self.models[ii])
                elif n - self.best_epoch > self.early_stopping_patience:
                    print(f'Validation loss hasnt improved in {self.early_stopping_patience} epochs stopping optimization at epoch {n}')
                    break
                
                self.lr_scheduler[ii](validation_loss)
                
                metrics[f'lr_{ii}'][n] = float(self.optims[ii].param_groups[0]['lr'])
                metrics[f'train_loss_{ii}'][n] = float(train_loss)
                metrics[f'validation_loss_{ii}'][n] = float(validation_loss)

                if self.optims[ii].param_groups[0]['lr'] == self.min_lr:
                    print('Reached minimum learning rate')
                    break
        
        self.metrics = metrics
        self.train_loss = train_loss
        self.validation_loss = validation_loss
        self.models = self.best_models

    def predict(self, data, nequip_model=None):
        if 'node_features' in data:
            out = data
        elif nequip_model is not None:
            if not data['pos'].requires_grad:
                data['pos'].requires_grad = True
            out = nequip_model(self.transform_data_input(data))
        else:
            ValueError
        atom_one_hot = torch.nn.functional.one_hot(data['atom_types'].squeeze(),num_classes=self.natoms)

        pred = torch.empty_like(out['atomic_energy'])
        for ii in range(self.natoms):
            mask = (data['atom_types']==ii).squeeze()
            NN_inputs = torch.hstack([out['node_features'], atom_one_hot])[mask]
            pred[mask] = self.models[ii](NN_inputs)

        return pred #, pred_forces

    def fine_tune(self,
        train_latents,
        train_energies,
        validation_latents,
        validation_energies,
        fine_tune_latents, 
        fine_tune_energies, 
        fine_tune_epochs=None,
    ):
        
        if fine_tune_epochs is None:
            fine_tune_epochs = int(self.epochs/10)

        self.fine_tune_early_stopping_patience = self.early_stopping_patience*fine_tune_epochs/self.epochs

        initial_lr = max(self.initial_lr/10, self.min_lr*10)
        
        fine_tune_metrics = {}
        self.best_models = []
        if not hasattr(self,'best_loss'):
            self.best_loss = []
        for ii in range(self.natoms):
            optim = torch.optim.Adam(self.models[ii].parameters(), lr = initial_lr)
            lr_scheduler = LRScheduler(optim, self.patience/10, self.min_lr)

            train_dataloader = self.make_dataloader(train_latents, train_energies,ii)
            validation_dataloader = self.make_dataloader(validation_latents, validation_energies,ii)
            fine_tune_dataloader = self.make_dataloader(fine_tune_latents, fine_tune_energies,ii)

            i_train_max = self.fine_tune_train_dataset_percentage*len(train_dataloader)

            self.best_models.append(copy.deepcopy(self.models[ii]))
            self.best_epoch = 0
            if len(self.best_loss)<self.natoms:
                self.best_loss.append(torch.tensor(np.inf))

            fine_tune_metrics[f'lr_{ii}'] = np.full((self.epochs),np.nan)
            fine_tune_metrics[f'fine_tune_loss_{ii}'] = np.full((self.epochs),np.nan)
            fine_tune_metrics[f'train_loss_{ii}'] = np.full((self.epochs),np.nan)
            fine_tune_metrics[f'validation_loss_{ii}'] = np.full((self.epochs),np.nan)

            for n in range(fine_tune_epochs):
                running_loss = 0
                for i, data in enumerate(fine_tune_dataloader):
                    if i > i_train_max:
                        break
                    latents, energies = data
                    self.models[ii].train()
                    self.models[ii].zero_grad()

                    pred = self.models[ii](latents) #pred atomic energies and sum to get total energies
                    loss = self.loss(pred,energies)

                    loss.backward()

                    optim.step()
                    running_loss += loss.item()
                
                fine_tune_loss = running_loss/(i+1)
                running_loss = 0
                for i, data in enumerate(train_dataloader):
                    latents, energies = data
                    self.models[ii].train()
                    self.models[ii].zero_grad()

                    pred = self.models[ii](latents) #pred atomic energies and sum to get total energies
                    loss = self.loss(pred,energies)

                    loss.backward()

                    optim.step()
                    running_loss += loss.item()
                
                train_loss = running_loss/(i+1)
                running_loss = 0
                for i, data in enumerate(validation_dataloader):
                    latents, energies = data
                    
                    pred = self.models[ii](latents) #pred atomic energies and sum to get total energies
                    loss = self.loss(pred,energies)

                    running_loss += loss.item()
                validation_loss = running_loss/(i+1)
                
                if validation_loss < self.best_loss[ii]:
                    self.best_epoch = n
                    self.best_loss[ii] = validation_loss
                    self.best_models[ii] = copy.deepcopy(self.models[ii])
                elif n - self.best_epoch > self.fine_tune_early_stopping_patience:
                    print(f'Validation loss hasnt improved in {self.fine_tune_early_stopping_patience} epochs stopping optimization at epoch {n}')
                    break
                lr_scheduler(validation_loss)
                
                fine_tune_metrics[f'lr_{ii}'][n] = float(optim.param_groups[0]['lr'])
                fine_tune_metrics[f'fine_tune_loss_{ii}'] = float(fine_tune_loss)
                fine_tune_metrics[f'train_loss_{ii}'] = float(train_loss)
                fine_tune_metrics[f'validation_loss_{ii}'] = float(validation_loss)

                if optim.param_groups[0]['lr'] == self.min_lr:
                    break
            
        self.fine_tune_metrics = fine_tune_metrics
        self.train_loss = train_loss
        self.validation_loss = validation_loss
        self.models = self.best_models

    def transform_data_input(self, data):
        # assert len(data['total_energy']) == 1
        data = AtomicData.to_AtomicDataDict(data)
        
        return data
    def get_state_dict(self):
        state_dict = {}
        for i in range(self.natoms):
            state_dict[str(i)] = self.models[i].state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for i in range(self.natoms):
            self.models[i].load_state_dict(state_dict[str(i)])


def train_NN(args):
    NN,uncertainty_training,train_embeddings,train_energy_forces,validation_embeddings,validation_energy_forces,other_embeddings,other_energy_forces = args
    print(uncertainty_training)
    if uncertainty_training=='energy':
        if other_embeddings is None:
            NN.train(train_embeddings, train_energy_forces, validation_embeddings, validation_energy_forces)
        else:
            NN.fine_tune(train_embeddings, train_energy_forces, validation_embeddings, validation_energy_forces,other_embeddings,other_energy_forces)
    elif uncertainty_training=='forces':
        if other_embeddings is None:
            NN.train(train_embeddings, train_energy_forces, validation_embeddings, validation_energy_forces)
        else:
            NN.fine_tune(train_embeddings, train_energy_forces, validation_embeddings, validation_energy_forces,other_embeddings,other_energy_forces)
    else:
        raise RuntimeError
    return NN

# def train_NN(args):
#     NN,uncertainty_training,train_embeddings,train_energy_forces,validation_embeddings,validation_energy_forces,other_embeddings,other_energy_forces = args
#     print(NN)