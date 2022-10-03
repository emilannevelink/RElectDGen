import argparse, subprocess
from genericpath import isdir
from ase.io.trajectory import Trajectory
import numpy as np
import os
import yaml
import time
import gc
import shutil

import torch
from nequip.model import model_from_config
from nequip.data.transforms import TypeMapper
from nequip.utils import Config
from nequip.data import dataset_from_config

from ..uncertainty import models as uncertainty_models

from RElectDGen.calculate.calculator import nn_from_results
from RElectDGen.utils.save import check_NN_parameters, check_nan_parameters
from RElectDGen.utils.logging import get_mae_from_results, write_to_tmp_dict, get_dataset_sizes

from memory_profiler import profile

def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    parser.add_argument('--MLP_config_file', dest='MLP_config',
                        help='Nequip configuration file', type=str)
    args = parser.parse_args(argsin)

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    # with open(args.MLP_config,'r') as fl:
    #     MLP_config_new = yaml.load(fl,yaml.FullLoader)

    MLP_config_new = Config.from_file(args.MLP_config)

    return config, MLP_config_new, args.MLP_config


def use_previous_model(MLP_config_new, nmodels):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ### Check for previous model
    train = False
    models = []
    MLP_configs = []
    for i in range(nmodels):
        root = f'results_{i}'
        try:
            calc_nn, model_load, MLP_config = nn_from_results(root=root)
            train = not check_nan_parameters(model_load) #check to make sure no parameters are nan
            if not train:
                print(f'Loaded {i} model successfully')
        except (FileNotFoundError, OSError, ValueError, UnboundLocalError):
            print('No previous results',flush=True)
            train = True
            MLP_config = {}
            model_load = 0

        try:
            chemical_symbol_to_type = MLP_config_new.get('chemical_symbol_to_type')
            transform = TypeMapper(chemical_symbol_to_type=chemical_symbol_to_type)
            MLP_config_new['type_names'] = transform.type_names
            MLP_config_new['num_types'] = transform.num_types
            dataset = dataset_from_config(MLP_config_new)
            model = model_from_config(
                config=MLP_config_new, initialize=True, dataset=dataset
            )
            model.load_state_dict(model_load.state_dict())
            model.eval()
            model.to(torch.device(device))
            # if MLP_config.compile_model:
            import e3nn
            model = e3nn.util.jit.compile(model)
            print('compiled model', flush=True)
            torch._C._jit_set_bailout_depth(MLP_config.get("_jit_bailout_depth",2))
            torch._C._jit_set_profiling_executor(False)
            
            del model_load, calc_nn, transform
        except Exception as e:
            print(e)
            print('previous model is not the same as state dict', flush=True)
            train = True
            model = 0
            MLP_config = {}
        
        models.append(model)
        MLP_configs.append(MLP_config)

    return train, models, MLP_configs

@profile
def main(args=None):
    
    start_time = time.time()
    config, MLP_config_new, MLP_config_filename = parse_command_line(args)
    torch._C._jit_set_bailout_depth(MLP_config_new.get("_jit_bailout_depth",2))
    tmp_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('tmp_file','tmp.json'))
    
    n_ensemble = config.get('n_uncertainty_ensembles',4)
    
    train, models, MLP_configs = use_previous_model(MLP_config_new, n_ensemble)
    MLP_config = MLP_configs[0]

    if config.get('force_retrain', False):
        train = True

    train_directory = config['train_directory']
    if train_directory[-1] == '/':
        train_directory = train_directory[:-1]

    if not train:
        #Check to make sure the previously trained network had training and validation losses that were 'close' to each other
        
        for i in range(n_ensemble):
            root = train_directory + f'_{i}'
            prev_mae_dict = get_mae_from_results(root,index=i)
            print(prev_mae_dict[f'best_validation_loss_{i}'],prev_mae_dict[f'best_training_loss_{i}'], flush=True)
            if not np.isclose(prev_mae_dict[f'best_validation_loss_{i}'],prev_mae_dict[f'best_training_loss_{i}'],rtol=10):
                train = True
                print(f'Previous train and validation losses are too far apart for network {i}', flush=True) 
            else:
                print(f'Previous train and validation losses are close enough for network {i}', flush=True) 

    
    traj = Trajectory(MLP_config_new['dataset_file_name'])
    uncertainty_dict = {}
    load = False
    reset_train_indices = True
    if not train:
        last_dataset_size, current_dataset_size = get_dataset_sizes(config, tmp_filename)
        
        if max(max(MLP_config.get('train_idcs')),max(MLP_config.get('val_idcs'))) > len(traj):
            train = True

        elif last_dataset_size == current_dataset_size:
            print('Dataset size hasnt changed', flush=True)
            train = False
            reset_train_indices = False
        else:
            gc.collect()
            ### Calibrate Uncertainty Quantification
            UQ_func = getattr(uncertainty_models,config.get('uncertainty_function', 'Nequip_latent_distance'))

            UQ = UQ_func(models, config, MLP_configs)
            UQ.calibrate()
            
            maximum_uncertain_datapoints = config.get('retrain_uncertainty_percent',0.01)*len(traj)
            nuncertain_data = 0
            batch_size = 100
            nbatches = int(len(traj)/batch_size+1)
            indices = np.random.permutation(np.arange(len(traj),dtype=int))
            uncertainty_sum = torch.zeros(len(traj))
            for i in range(nbatches):
                uncertainty, embedding = UQ.predict_from_traj(traj[indices[batch_size*i:batch_size*(i+1)]])

                uncertainty_sum[indices[batch_size*i:batch_size*(i+1)]] = uncertainty.sum(dim=1)
                nuncertain_data += len(np.argwhere(uncertainty_sum.numpy()>config.get('UQ_min_uncertainty')).flatten())
                if nuncertain_data>maximum_uncertain_datapoints:
                    break

            mask = uncertainty_sum>0
            uncertainty_dict['dataset_uncertainty_mean'] = float(uncertainty_sum[mask].mean())
            uncertainty_dict['dataset_uncertainty_std'] = float(uncertainty_sum[mask].std())
            
            uncertain_data = np.argwhere(uncertainty_sum.numpy()>config.get('UQ_min_uncertainty')).flatten()

            if nuncertain_data>maximum_uncertain_datapoints:
                print(f'Sampled {sum(mask)} datapoints.')
                print(f'First uncertain datapoint {uncertain_data.min()}, of {len(uncertain_data)} uncertain point from {len(traj)} data points',flush=True)
                train = True

                for i, conf in enumerate(MLP_configs):
                    if MLP_config_new.get('load_previous') and check_NN_parameters(MLP_config_new, conf):
                        print('Load previous', flush = True)
                        load = True
                        reset_train_indices = False

            else:
                print('No uncertain points, reusing neural network')

            del UQ, uncertainty, embedding, uncertain_data

    gc.collect()

    if train:
        if reset_train_indices:
            current_train_idcs = set([])
            current_val_idcs = set([])
        else:
            current_train_idcs = set(np.array(MLP_config.get('train_idcs')))
            current_val_idcs = set(np.array(MLP_config.get('val_idcs')))

        n_train_add = MLP_config_new.get('n_train') - len(current_train_idcs)
        n_val_add = MLP_config_new.get('n_val') - len(current_val_idcs)

        all_indices = set(torch.arange(0,len(traj),1).numpy())
        remaining_idcs = torch.tensor(list(all_indices.difference(current_train_idcs).difference(current_val_idcs)))

        ind_select = torch.randperm(len(remaining_idcs))
        if reset_train_indices:
            train_idcs = remaining_idcs[ind_select[:n_train_add]]
            val_idcs = remaining_idcs[ind_select[n_train_add:n_train_add+n_val_add]]
        else:
            train_idcs = torch.cat([MLP_config.get('train_idcs'), remaining_idcs[ind_select[:n_train_add]]])
            val_idcs = torch.cat([MLP_config.get('val_idcs'), remaining_idcs[ind_select[n_train_add:n_train_add+n_val_add]]])
        MLP_config_new['train_idcs'] = train_idcs
        MLP_config_new['val_idcs'] = val_idcs
        
        for i, conf in enumerate(MLP_configs):
            MLP_config_new['root'] = train_directory + f'_{i}'
            if load:
                MLP_config_new['workdir_load'] = conf['workdir']

            if os.path.isdir(MLP_config_new['root']):
                for dir in os.listdir(MLP_config_new['root']):
                    if 'processed_dataset' in dir:
                        delete_directory = os.path.join(
                            MLP_config_new['root'],
                            dir
                        )
                        shutil.rmtree(delete_directory)

            tmp_MLP_filename = f'tmp_MLP_{i}.yaml'
            with open(tmp_MLP_filename, "w+") as fp:
                yaml.dump(dict(MLP_config_new), fp)

    logging_dict = {
        **uncertainty_dict,
        'train': train,
        'load': load,
    }

    write_to_tmp_dict(tmp_filename,logging_dict)

if __name__ == '__main__':
    main()