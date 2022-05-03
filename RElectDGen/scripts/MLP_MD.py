import os
import argparse


from ase.io.trajectory import Trajectory
import yaml
from RElectDGen.sampling.md_sampling import MD_sampling


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

    config, filename_config, loop_learning_count = parse_command_line(args)
    
    MD_uncertain, _, config = MD_sampling(config, loop_learning_count)
    
    if len(MD_uncertain)>0:
        
        active_learning_configs = os.path.join(config.get('data_directory'),config.get('active_learning_configs'))
        traj_write = Trajectory(active_learning_configs,mode='w')
        [traj_write.write(atoms) for atoms in MD_uncertain]

    else:
        print('No uncertain data points')
    
    with open(filename_config,'w') as fl:
        yaml.dump(config, fl)

if __name__ == "__main__":
    main()