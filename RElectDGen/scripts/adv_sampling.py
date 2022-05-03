import os
import argparse

from ase.io.trajectory import Trajectory
from ase.io import read

import yaml
import torch

from RElectDGen.sampling.adv_sampling import adv_sampling


def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    parser.add_argument('--MLP_config_file', dest='MLP_config',
                        help='Nequip configuration file', type=str)
    parser.add_argument('--loop_learning_count', dest='loop_learning_count', default=1,
                        help='active_learning_loop', type=int)
    args = parser.parse_args(argsin)

    
    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    return config, args.config, args.loop_learning_count

def main(args=None):
    print('Starting timer', flush=True)
    
    config, filename_config, loop_learning_count = parse_command_line(args)

    traj = read(config.get('dataset_file_name'), index=':')
    max_samples = int(min([0.1*len(traj), config.get('max_samples')]))
    n_adversarial_samples = int(config.get('n_adversarial_samples',2*max_samples))
    
    traj_indices = torch.randperm(len(traj))[:2*n_adversarial_samples].numpy()
    traj_adv = [traj[i] for i in traj_indices]
    
    adv_uncertain, _, config = adv_sampling(config, traj_adv, loop_learning_count)
    
    if len(adv_uncertain)>0:
        active_learning_configs = os.path.join(config.get('data_directory'),config.get('active_learning_configs'))
        traj_write = Trajectory(active_learning_configs,mode='w')
        [traj_write.write(atoms) for atoms in adv_uncertain]

    else:
        print('No uncertain data points')
    

    with open(filename_config,'w') as fl:
        yaml.dump(config, fl)

if __name__ == "__main__":
    main()