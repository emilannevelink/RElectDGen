import os, argparse

from ase.io.ulm import InvalidULMFileError
from ase.io.trajectory import Trajectory

from ase.parallel import world

from RElectDGen.calculate.recalculate import recalculate_traj_energies
from RElectDGen.calculate.calculator import oracle_from_config



parser = argparse.ArgumentParser()
parser.add_argument('--config_file', dest='config',
                    help='active_learning configuration file', type=str)
parser.add_argument('--MLP_config_file', dest='MLP_config',
                    help='Nequip configuration file', type=str)
parser.add_argument('--loop_learning_count', dest='loop_learning_count', default=0,
                    help='active_learning_loop', type=int)
parser.add_argument('--array_index', dest='array_index',
                    help='active_learning_loop', type=int)
args = parser.parse_args()
import yaml

with open(args.config,'r') as fl:
    config = yaml.load(fl,yaml.FullLoader)


calc_oracle = oracle_from_config(config)
active_learning_configs = os.path.join(config.get('data_directory'),config.get('active_learning_configs'))
active_learning_calc = os.path.join(config.get('data_directory'),config.get('active_learning_calc'))

if os.path.isfile(active_learning_configs):
    # try:
    traj_calc = Trajectory(active_learning_configs)
    
    calc_file = active_learning_calc+f'.{args.array_index}'

    if world.rank == 0:
        print('Active Learning Array',flush=True)
        print(f'File: {calc_file}',flush=True)
        
    if len(traj_calc)>args.array_index:
        writer = Trajectory(calc_file,'w')

        atoms = traj_calc[args.array_index]
        recalculate_traj_energies([atoms], config=config, writer=writer)#,rewrite_pbc=True)

    else:
        if os.path.isfile(calc_file):
            os.remove(calc_file)
 