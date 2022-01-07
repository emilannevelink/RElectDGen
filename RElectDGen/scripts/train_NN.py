import argparse, subprocess
from ase.io.trajectory import Trajectory
import numpy as np
import os
import yaml
import time

import torch
from torch.nn import L1Loss
from nequip.data import AtomicData, AtomicDataDict
from nequip.model import model_from_config
from nequip.data.transforms import TypeMapper

from ..utils.uncertainty import latent_distance_uncertainty_Nequip

from ..calculate.calculator import nn_from_results
from ..utils.save import check_NN_parameters
from ..utils.logging import get_mae_from_results, write_to_tmp_dict, UQ_params_to_dict

def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    parser.add_argument('--MLP_config_file', dest='MLP_config',
                        help='Nequip configuration file', type=str)
    args = parser.parse_args(argsin)

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    with open(args.MLP_config,'r') as fl:
        MLP_config_new = yaml.load(fl,yaml.FullLoader)

    return config, MLP_config_new, args.MLP_config

def main(args=None):
    start_time = time.time()
    config, MLP_config_new, MLP_config_filename = parse_command_line(args)
    torch._C._jit_set_bailout_depth(MLP_config_new.get("_jit_bailout_depth",2))

    ### Check for previous model
    try:
        calc_nn, model_load, MLP_config = nn_from_results()
        train = False
    except (FileNotFoundError, OSError, ValueError):
        print('No previous results',flush=True)
        train = True

    try:
        chemical_symbol_to_type = MLP_config_new.get('chemical_symbol_to_type')
        transform = TypeMapper(chemical_symbol_to_type=chemical_symbol_to_type)
        MLP_config_new['type_names'] = transform.type_names
        MLP_config_new['num_types'] = transform.num_types
        model = model_from_config(
            config=MLP_config_new, initialize=False, # dataset=dataset
        )
        model.load_state_dict(model_load.state_dict())
        train = False
    except:
        print('previous model is not the same as state dict', flush=True)
        train = True

    if config.get('force_retrain', False):
        train = True

    commands = ['nequip-train', MLP_config_filename]
    UQ_dict = {}
    if not train:
        
        traj = Trajectory(MLP_config['dataset_file_name'])
        if max(MLP_config.get('train_idcs').max(),MLP_config.get('val_idcs').max()) > len(traj):
            train = True     
        else:
            UQ = latent_distance_uncertainty_Nequip(model, MLP_config)
            UQ.calibrate()
            UQ_dict = UQ_params_to_dict(UQ.params,'train')
            
            uncertainty, embedding = UQ.predict_from_traj(traj)
            
            uncertain_data = np.argwhere(np.array(uncertainty.detach())>config.get('UQ_min_uncertainty')).flatten()

            if len(uncertain_data)>0:
                print(f'First uncertaint datapoint {uncertain_data.min()}, of {len(uncertain_data)} uncertain point from {len(traj)} data points',flush=True)
                train = True

                if MLP_config_new.get('load_previous') and check_NN_parameters(MLP_config_new, MLP_config):
                    MLP_config_new['workdir_load'] = MLP_config['workdir']

                    with open(MLP_config_filename, "w+") as fp:
                        yaml.dump(dict(MLP_config_new), fp)

                    print('Load previous', flush = True)
                    commands = ['nequip-train-load', MLP_config_filename]

            else:
                print('No uncertain points, reusing neural network')

    if train:
        print('Training NN ... ', flush=True)
            
        process = subprocess.run(commands,capture_output=True)
        # print(process)
        [print(line) for line in process.stderr.split(b'\n')]

        # print(process.stderr.decode('ascii'))

    mae_dict = get_mae_from_results()

    logging_dict = {
        **mae_dict,
        'train': train,
        'load': 'load' in commands[0],
        **UQ_dict,
        'train_time': time.time()-start_time
    }

    tmp_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('tmp_file','tmp.json'))
    write_to_tmp_dict(tmp_filename,logging_dict)

if __name__ == '__main__':
    main()