import argparse, subprocess
from ase.io.trajectory import Trajectory
import numpy as np

from torch.nn import L1Loss
from nequip.data import AtomicData, AtomicDataDict
from nequip.model import model_from_config
from nequip.data.transforms import TypeMapper

from e3nn_networks.utils.train_val import latent_distance_uncertainty_Nequip

from RElectDGen.calculate.calculator import nn_from_results
from RElectDGen.utils.save import check_NN_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', dest='config',
                    help='active_learning configuration file', type=str)
parser.add_argument('--MLP_config_file', dest='MLP_config',
                    help='Nequip configuration file', type=str)
args = parser.parse_args()

import yaml
with open(args.config,'r') as fl:
    config = yaml.load(fl,yaml.FullLoader)

with open(args.MLP_config,'r') as fl:
    MLP_config_new = yaml.load(fl,yaml.FullLoader)

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

commands = ['nequip-train', args.MLP_config]
if not train:
    
    traj = Trajectory(MLP_config['dataset_file_name'])
    if max(MLP_config.get('train_idcs').max(),MLP_config.get('val_idcs').max()) > len(traj):
        train = True     
    else:
        UQ = latent_distance_uncertainty_Nequip(model, MLP_config)
        UQ.calibrate()

        
        uncertainty = UQ.predict_from_traj(traj)
        
        uncertain_data = np.argwhere(np.array(uncertainty.detach())>config.get('UQ_min_uncertainty')).flatten()

        if len(uncertain_data)>0:
            print(f'First uncertaint datapoint {uncertain_data.min()}, of {len(uncertain_data)} uncertain point from {len(traj)} data points',flush=True)
            train = True

            if MLP_config_new.get('load_previous') and check_NN_parameters(MLP_config_new, MLP_config):
                MLP_config_new['workdir_load'] = MLP_config['workdir']

                with open(args.MLP_config, "w+") as fp:
                    yaml.dump(dict(MLP_config_new), fp)

                print('Load previous', flush = True)
                commands = ['nequip-train-load', args.MLP_config]

        else:
            print('No uncertain points, reusing neural network')

if train:
    print('Training NN ... ', flush=True)
        
    process = subprocess.run(commands,capture_output=True)
    print(process)
    print(process.stderr)
    print(process.stderr.decode('ascii'))
