import torch
import numpy as np
from scipy.optimize import minimize
from scipy import stats

from torch.utils.data import Dataset, DataLoader

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
        print('Trainable parameters:', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.epochs = epochs
        self.prob = lambda value, std: torch.max(torch.hstack([torch.exp(-value.pow(2)/2/std.pow(2))/std/(2*np.pi)**0.5,1e-10*torch.ones_like(value)]),dim=-1).values
        self.loss = lambda error, pred_std: -torch.sum( torch.log( self.prob(error, pred_std)) )
        self.optim = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.lr_scheduler = LRScheduler(self.optim, patience, self.min_lr)
        
        

    def train(self, x, y):
        x = torch.tensor(x).to(self.device) #Break computational graph for training
        y = torch.tensor(y).to(self.device)

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
