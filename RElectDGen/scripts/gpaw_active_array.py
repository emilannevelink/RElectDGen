import os, argparse
import yaml

from ase.io.ulm import InvalidULMFileError
from ase.io.trajectory import Trajectory

from ase.parallel import world

from RElectDGen.calculate.recalculate import recalculate_traj_energies
from RElectDGen.calculate.calculator import oracle_from_config
from RElectDGen.structure.utils import extend_cell


def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    parser.add_argument('--MLP_config_file', dest='MLP_config',
                        help='Nequip configuration file', type=str)
    parser.add_argument('--loop_learning_count', dest='loop_learning_count', default=0,
                        help='active_learning_loop', type=int)
    parser.add_argument('--array_index', dest='array_index',
                        help='active_learning_loop', type=int)
    args = parser.parse_args(argsin)
    

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    return config, args.array_index

def main(args=None):

    config, array_index = parse_command_line(args)
    
    # calc_oracle = oracle_from_config(config)
    active_learning_configs = os.path.join(config.get('data_directory'),config.get('active_learning_configs'))
    active_learning_calc = os.path.join(config.get('data_directory'),config.get('active_learning_calc'))

    if os.path.isfile(active_learning_configs):
        # try:
        traj_calc = Trajectory(active_learning_configs)
        
        calc_file = active_learning_calc+f'.{array_index}'

        if world.rank == 0:
            print('Active Learning Array',flush=True)
            print(f'File: {calc_file}',flush=True)
            
        if len(traj_calc)>array_index:
            writer = Trajectory(calc_file,'w')

            atoms = traj_calc[array_index]

            atoms = extend_cell(atoms,config)

            recalculate_traj_energies([atoms], config=config, writer=writer)#,rewrite_pbc=True)
            traj = Trajectory(calc_file)
            if world.rank == 0:
                for atoms in traj:
                    print(atoms)
                    print(atoms.info)

        else:
            if world.rank == 0:
                print('array index larger than number of clusters to calculate', flush=True)
                if os.path.isfile(calc_file):
                    os.remove(calc_file)
    
if __name__ == '__main__':
    main()