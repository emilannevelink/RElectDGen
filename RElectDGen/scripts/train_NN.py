import argparse, subprocess
from ase.io.trajectory import Trajectory
import numpy as np
import os
import yaml
import time
import gc

import torch
from nequip.model import model_from_config
from nequip.data.transforms import TypeMapper
from nequip.utils import Config
from nequip.data import dataset_from_config

from ..uncertainty import models as uncertainty_models

from RElectDGen.calculate._MLIP import nn_from_results
from RElectDGen.utils.save import check_NN_parameters
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


def use_previous_model(MLP_config_new):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ### Check for previous model
    try:
        calc_nn, model_load, MLP_config = nn_from_results()
        train = False
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
        train = False
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

    return train, model, MLP_config

@profile
def main(args=None):
    
    start_time = time.time()
    config, MLP_config_new, MLP_config_filename = parse_command_line(args)
    torch._C._jit_set_bailout_depth(MLP_config_new.get("_jit_bailout_depth",2))
    tmp_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('tmp_file','tmp.json'))
    train, model, MLP_config = use_previous_model(MLP_config_new)

    if config.get('force_retrain', False):
        train = True

    if not train:
        #Check to make sure the previously trained network had training and validation losses that were 'close' to each other
        prev_mae_dict = get_mae_from_results()
        print(prev_mae_dict['best_validation_loss'],prev_mae_dict['best_training_loss'], flush=True)
        if not np.isclose(prev_mae_dict['best_validation_loss'],prev_mae_dict['best_training_loss'],rtol=10):
            train = True
            print('Previous train and validation losses are too far apart', flush=True) 
        else:
            print('Previous train and validation losses are close enough', flush=True) 

    uncertainty_dict = {}
    commands = ['nequip-train', MLP_config_filename]
    if not train:
        
        traj = Trajectory(MLP_config['dataset_file_name'])
        last_dataset_size, current_dataset_size = get_dataset_sizes(config, tmp_filename)
        if max(max(MLP_config.get('train_idcs')),max(MLP_config.get('val_idcs'))) > len(traj):
            train = True     
        elif last_dataset_size == current_dataset_size:
            print('Dataset size hasnt changed', flush=True)
            train = False
        else:
            gc.collect()
            ### Calibrate Uncertainty Quantification
            UQ_func = getattr(uncertainty_models,config.get('uncertainty_function', 'Nequip_latent_distance'))

            UQ = UQ_func(model, config, MLP_config)
            UQ.calibrate()
            
            uncertainty, embedding = UQ.predict_from_traj(traj)

            uncertainty_sum = uncertainty.sum(dim=1)
            # print(uncertainty)
            # print(uncertainty_sum, flush = True)
            uncertainty_dict['dataset_uncertainty_mean'] = float(uncertainty_sum.mean())
            uncertainty_dict['dataset_uncertainty_std'] = float(uncertainty_sum.std())
            
            uncertain_data = np.argwhere(uncertainty_sum.numpy()>config.get('UQ_min_uncertainty')).flatten()

            if len(uncertain_data)>config.get('retrain_uncertainty_percent',0.01)*len(traj):
                print(f'First uncertain datapoint {uncertain_data.min()}, of {len(uncertain_data)} uncertain point from {len(traj)} data points',flush=True)
                train = True

                if MLP_config_new.get('load_previous') and check_NN_parameters(MLP_config_new, MLP_config):
                    MLP_config_new['workdir_load'] = MLP_config['workdir']

                    with open(MLP_config_filename, "w+") as fp:
                        yaml.dump(dict(MLP_config_new), fp)

                    print('Load previous', flush = True)
                    commands = ['nequip-train-load', MLP_config_filename]

            else:
                print('No uncertain points, reusing neural network')

            del UQ, uncertainty, embedding, uncertain_data, MLP_config, MLP_config_new

    gc.collect()
    if train:
        print('Training NN ... ', flush=True)
            
        process = subprocess.run(commands,capture_output=True)
        # print(process)
        [print(line) for line in process.stderr.split(b'\n')]

        # print(process.stderr.decode('ascii'))

    mae_dict = get_mae_from_results()

    logging_dict = {
        **mae_dict,
        **uncertainty_dict,
        'train': train,
        'load': 'load' in commands[0],
        'train_time': time.time()-start_time
    }

    write_to_tmp_dict(tmp_filename,logging_dict)

if __name__ == '__main__':
    main()