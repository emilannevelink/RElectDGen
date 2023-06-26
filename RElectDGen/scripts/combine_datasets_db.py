import argparse
import yaml, os
from ase.io import read, Trajectory
from ase.db import connect
from ..utils.save import update_config_trainval
from ..utils.data import reduce_trajectory

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

    with open(filename_MLP_config,'r') as fl:
        MLP_config = yaml.load(fl,yaml.FullLoader)

    return config, MLP_config, args.MLP_config

def main(args=None):

    config, MLP_config, filename_MLP_config = parse_command_line(args)

    ase_db_file = os.path.join(
        config.get('data_directory'),
        config.get('ase_db_filename')
    )
    adversarial_trajectory = os.path.join(config.get('data_directory'),config.get('adv_trajectory_file',''))
    combined_trajectory = os.path.join(config.get('data_directory'),config.get('combined_trajectory'))

    
    traj = []
    for filename in config.get('pretraining_data',[]):
        try:
            if filename.endswith('traj'):
                add_traj = read(os.path.join(config.get('data_directory'),filename), index=':')
            elif filename.endswith('db'):
                db = connect(os.path.join(config.get('data_directory'),filename))
                add_traj = []
                for row in db.select(success=True):
                    add_traj.append(row.toatoms())
            traj += add_traj

            print(len(add_traj), len(traj), filename)
        except:
            print('Trajectory file couldnt be appended', flush=True)
            print(filename, flush=True)

    pretraining_size = len(traj)
    if os.path.isfile(ase_db_file):
        try:
            db = connect(ase_db_file)
            add_traj = []
            for row in db.select(success=True):
                add_traj.append(row.toatoms())
            traj += add_traj
            print(len(add_traj), len(traj))
        except:
            print('ASE DB could not be appended', flush=True)
            print(ase_db_file, flush=True)
    
    combined_size = len(traj)
    traj_reduced = reduce_trajectory(traj,config,MLP_config)
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
