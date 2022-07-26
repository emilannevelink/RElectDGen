import os, argparse

from ase.io.ulm import InvalidULMFileError
from ase.io.trajectory import Trajectory

from ase.parallel import world
from nequip.utils import config

import numpy as np

from RElectDGen.utils.logging import write_to_tmp_dict

def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    parser.add_argument('--MLP_config_file', dest='MLP_config',
                        help='Nequip configuration file', type=str)
    parser.add_argument('--loop_learning_count', dest='loop_learning_count', default=0,
                        help='active_learning_loop', type=int)
    args = parser.parse_args()
    import yaml

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    return config

def main(args=None):

    config = parse_command_line(args)
    # hdf5_file = os.path.join(config.get('data_directory'),config.get('hdf5_file'))
    trajectory_file = os.path.join(config.get('data_directory'),config.get('trajectory_file'))
    adversarial_trajectory = os.path.join(config.get('data_directory'),config.get('adv_trajectory_file'))
    active_learning_calc = os.path.join(config.get('data_directory'),config.get('active_learning_calc'))
    active_learning_configs = os.path.join(config.get('data_directory'),config.get('active_learning_configs'))

    calc_inds_uncertain = config['calc_inds_uncertain']
    n_MD_uncertain = config['n_MD_uncertain']

    if os.path.isfile(active_learning_configs):
        traj_calc = Trajectory(active_learning_configs)

        traj_active = []
        calc_ind = []
        for i in range(len(traj_calc)):
            calc_file = active_learning_calc+f'.{i}'
            try:
                traj_active += list(Trajectory(calc_file))
                calc_ind.append(i)
                print(f'retrieved {i} from array', flush=True)
            except (InvalidULMFileError, FileNotFoundError):
                print(f'failed to retrieve {i} from array', flush=True)
        
        print(f'Retrived {len(traj_active)} calculations from array', flush=True)
        try:
            calculation_times = []
            for atoms in traj_active:
                print(atoms)
                print(atoms.info)
                calculation_times.append(atoms.info['calculation_time'])
            
            print(calculation_times)
            print(np.mean(calculation_times))
            logging_dict = {
                'calculation_times_mean': float(np.mean(calculation_times)),
                'calculation_times_max': float(np.max(calculation_times)),
                'calculation_times_min': float(np.min(calculation_times))
            }
            tmp_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('tmp_file','tmp.json'))
            write_to_tmp_dict(tmp_filename,logging_dict)
        except Exception as e:
            print(e)
            print('error')

        traj_writer = Trajectory(trajectory_file,mode='a')
        [traj_writer.write(atoms) for (ind, atoms) in zip(calc_ind,traj_active) if calc_inds_uncertain[ind]<n_MD_uncertain]

        if sum(np.array(calc_inds_uncertain)>=n_MD_uncertain)>0:
            traj_writer = Trajectory(adversarial_trajectory,mode='a')
            [traj_writer.write(atoms) for (ind, atoms) in zip(calc_ind,traj_active) if calc_inds_uncertain[ind]>=n_MD_uncertain]

        if world.rank == 0:
            os.remove(active_learning_configs)

        # if world.rank == 0:
        #     print(len(traj_active),flush=True)
        #     if len(traj_active)>0:
        #         data_active = extract_traj_data(traj_active)
        #         write_learning_data(hdf5_file,data_active,iteration=args.loop_learning_count)
        #     else:
        #         print('Not writing hdf5 data',len(traj_active),flush=True)

if __name__ == "__main__":
    main()