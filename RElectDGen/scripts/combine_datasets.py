import argparse
import yaml, os
from ase.io import read, Trajectory
from ..utils.save import update_config_trainval
from ..utils.data import reduce_traj

def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    parser.add_argument('--MLP_config_file', dest='MLP_config',
                        help='Nequip configuration file', type=str)
    args = parser.parse_args(argsin)

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    return config, args.MLP_config

def main(args=None):

    config, filename_MLP_config = parse_command_line(args)
    trajectory_file = os.path.join(config.get('data_directory'),config.get('trajectory_file'))
    combined_trajectory = os.path.join(config.get('data_directory'),config.get('combined_trajectory'))

    
    traj = []
    for filename in config.get('pretraining_data',[]):
        traj += read(os.path.join(config.get('data_directory'),filename), index=':')

        print(len(traj))

    try:
        traj += read(trajectory_file,index=':')
    except:
        print('Trajectory file couldnt be appended', flush=True)
        print(trajectory_file, flush=True)

    traj_reduced = reduce_traj(traj)

    print(len(traj), len(traj_reduced),flush=True)
    
    print(combined_trajectory,flush=True)
    writer = Trajectory(combined_trajectory,'w')
    for atoms in traj_reduced:
        writer.write(atoms)

    update_config_trainval(config,filename_MLP_config)

if __name__ == "__main__":
    main()
