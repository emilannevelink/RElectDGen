import argparse, subprocess
from ase.io.trajectory import Trajectory
import numpy as np
import os
import shutil
import yaml
import time
import json

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
    parser.add_argument('--array_index', dest='array_index',
                        help='active_learning_loop', type=int)
    args = parser.parse_args(argsin)

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    # with open(args.MLP_config,'r') as fl:
    #     MLP_config_new = yaml.load(fl,yaml.FullLoader)

    MLP_config_new = Config.from_file(args.MLP_config)

    return config, MLP_config_new, args.MLP_config, args.array_index

def remove_processed(results_directory):

    for root, dirs, files in os.walk(results_directory):
        for name in dirs:
            if 'processed' in name:
                shutil.rmtree(os.path.join(root,name))

@profile
def main(args=None):
    
    start_time = time.time()
    config, MLP_config, MLP_config_filename, array_index = parse_command_line(args)

    remove_processed(MLP_config.get('root'))
    
    # tmp_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('tmp_file','tmp.json'))

    train_directory = config['train_directory']
    if train_directory[-1] == '/':
        train_directory = train_directory[:-1]

    # if os.path.isfile(tmp_filename):
    #     with open(tmp_filename,'r') as fl:
    #         logging_dict = json.load(fl)
    # else:
    #     logging_dict = {}
    
    base_dir = os.path.dirname(MLP_config_filename)
    base_file = os.path.basename(MLP_config_filename)
    tmp_MLP_config = os.path.join(
            base_dir,
            'tmp',
            base_file.split('.yaml')[0] + f'_{array_index}.yaml'
        )
    # if logging_dict.get('load',False):
    #     commands = ['nequip-train-load', MLP_config_filename]
    # else:
    commands = ['nequip-train', tmp_MLP_config]

    # if logging_dict.get('train',True):
    print('Training NN ... ', flush=True)
    print(commands, flush=True)
        
    process = subprocess.run(commands,capture_output=True)
    # print(process)
    [print(line) for line in process.stderr.split(b'\n')]

    # print(process.stderr.decode('ascii'))

    uncertainty_function = config.get('uncertainty_function')
    if uncertainty_function in ['Nequip_ensemble']:
        n_ensemble = config.get('n_uncertainty_ensembles',4)
    else:
        n_ensemble = 1
    
    if n_ensemble>1:
        root = train_directory + f'_{array_index}'
    else:
        root = train_directory

    mae_dict = get_mae_from_results(root,index=array_index, template=MLP_config['run_name'])

    logging_dict = {
        **mae_dict,
        f'train_time_{array_index}': time.time()-start_time
    }

    ###TODO: this leads to a race condition... add some check functionality to check
    # write_to_tmp_dict(tmp_filename,logging_dict)
    os.remove(tmp_MLP_config)

if __name__ == '__main__':
    main()