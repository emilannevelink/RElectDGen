import argparse
import yaml, os
from ase.io import read, Trajectory
from ..utils.save import update_config_trainval
from ..utils.data import reduce_traj_finite, reduce_traj_isolated, reduce_traj_free_H

from RElectDGen.utils.logging import write_to_tmp_dict

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
    trajectory_file = os.path.join(config.get('data_directory'),config.get('trajectory_file',''))
    adversarial_trajectory = os.path.join(config.get('data_directory'),config.get('adv_trajectory_file',''))
    combined_trajectory = os.path.join(config.get('data_directory'),config.get('combined_trajectory'))

    
    traj = []
    for filename in config.get('pretraining_data',[]):
        try:
            add_traj = read(os.path.join(config.get('data_directory'),filename), index=':')
            traj += add_traj

            print(len(add_traj), len(traj), filename)
        except:
            print('Trajectory file couldnt be appended', flush=True)
            print(filename, flush=True)

    pretraining_size = len(traj)
    try:
        add_traj = read(trajectory_file,index=':')
        traj += add_traj
        print(len(add_traj), len(traj))
    except:
        print('Trajectory file couldnt be appended', flush=True)
        print(trajectory_file, flush=True)

    try:
        add_traj = read(adversarial_trajectory,index=':')
        traj += add_traj
        print(len(add_traj), len(traj))
    except:
        print('Adversarial trajectory file couldnt be appended', flush=True)
        print(adversarial_trajectory, flush=True)

    reduce_traj_type = config.get('reduce_traj_type','finite_isolated')
    combined_size = len(traj)
    if 'finite' in reduce_traj_type:
        traj_reduced = reduce_traj_finite(traj)
        print('Reduce finite ', len(traj), len(traj_reduced),flush=True)
    else:
        traj_reduced = traj

    with open(filename_MLP_config,'r') as fl:
        MLP_config = yaml.load(fl,yaml.FullLoader)
    
    if 'isolated' in reduce_traj_type:
        _, traj_reduced = reduce_traj_isolated(traj_reduced,MLP_config.get('r_max'))
        print('Reduce isolated ', len(traj), len(traj_reduced),flush=True)
    
    if 'free_H' in reduce_traj_type:
        _, traj_reduced = reduce_traj_free_H(traj_reduced)
        print('Reduce free Hydrogen ', len(traj), len(traj_reduced),flush=True)
    reduced_size = len(traj_reduced)

    print(combined_trajectory,flush=True)
    writer = Trajectory(combined_trajectory,'w')
    for atoms in traj_reduced:
        writer.write(atoms)

    update_config_trainval(config,filename_MLP_config)

    logging_dict = {
        'dataset_size': combined_size,
        'pretraining_dataset_size': pretraining_size,
        'reduced_dataset_size': reduced_size
    }

    tmp_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('tmp_file','tmp.json'))
    write_to_tmp_dict(tmp_filename,logging_dict)

if __name__ == "__main__":
    main()
