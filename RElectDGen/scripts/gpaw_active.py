import os, argparse

from ase.io.ulm import InvalidULMFileError
from ase.io.trajectory import Trajectory

from ase.parallel import world
import numpy as np

from RElectDGen.calculate.recalculate import recalculate_traj_energies
from RElectDGen.calculate.calculator import oracle_from_config
from RElectDGen.utils.logging import write_to_tmp_dict

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


# calc_oracle = oracle_from_config(config)

trajectory_file = os.path.join(config.get('data_directory'),config.get('trajectory_file'))
active_learning_calc = os.path.join(config.get('data_directory'),config.get('active_learning_calc'))
active_learning_configs = os.path.join(config.get('data_directory'),config.get('active_learning_configs'))

if os.path.isfile(active_learning_configs):
    # try:
    traj_calc = Trajectory(active_learning_configs)
    
    writer = Trajectory(active_learning_calc,'w')
    recalculate_traj_energies(traj_calc, config=config, writer=writer)#,rewrite_pbc=True)
    try:
        traj_active = Trajectory(active_learning_calc)
    except InvalidULMFileError:
        print('Invalid ULM error')
        traj_active = []

    traj_writer = Trajectory(trajectory_file,mode='a')
    [traj_writer.write(atoms) for atoms in traj_active]
    
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

    if world.rank == 0:
        os.remove(active_learning_configs)

    # if world.rank == 0:
    #     print(len(traj_active),flush=True)
    #     if len(traj_active)>0:
    #         data_active = extract_traj_data(traj_active)
    #         write_learning_data(config.get('hdf5_file'),data_active,iteration=args.loop_learning_count)
    #     else:
    #         print('Not writing hdf5 data',len(traj_active),flush=True)
