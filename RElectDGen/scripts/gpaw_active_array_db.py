import sys
import os, argparse
import yaml

from ase.db import connect
from ase.parallel import world

from RElectDGen.calculate.recalculate import calculate_atoms
from RElectDGen.structure.utils import extend_cell
from RElectDGen.statistics.energy_range import check_energy_range


def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='config_file', type=str,
                        help='active_learning configuration file')
    parser.add_argument('MLP_config', metavar='MLP_config', type=str,
                        help='Nequip configuration file')
    parser.add_argument('array_index', metavar='array_index', type=int,
                        help='Active Learning Index')
    # parser.add_argument('--config_file', dest='config',
    #                     help='active_learning configuration file', type=str)
    # parser.add_argument('--MLP_config_file', dest='MLP_config',
    #                     help='Nequip configuration file', type=str)
    # parser.add_argument('--array_index', dest='array_index',
    #                     help='active_learning_loop', type=int)
    args = parser.parse_args(argsin)
    
    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    return config, args.array_index

def main(args=None):

    config, array_index = parse_command_line(args)
    
    ### get atoms objects
    db_filename = os.path.join(
        config.get('data_directory'),
        config.get('ase_db_filename')
    )
    assert os.path.isfile(db_filename)
    db = connect(db_filename)

    ## This isn't perfect as if the calculation time is short compared to initializing
    ## or, all the calculations aren't started at the same time, the number of db.select(calc=False)
    ## are different between different array indexes
    active_learning_index = config.get('active_learning_index')
    for i, row in enumerate(db.select(active_learning_index=active_learning_index)):
        if i == array_index:
            atoms = row.toatoms()
            id = row['id']
            if world.rank == 0:
                print('Got atoms',atoms)
            break
    
    if 'atoms' not in locals():
        if world.rank == 0:
            print(f'Array index {array_index} too large, exiting')
        sys.exit()

    if world.rank == 0:
        print(i, array_index, id)
    
    try:
        atoms = extend_cell(atoms,config)
    except Exception as e:
        if world.rank == 0:
            print(e)
    atoms, success = calculate_atoms(atoms, config.get('oracle_config'),data_directory=config.get('data_directory'))
    
    if success:
        MD_kwargs = config.get('MLP_md_kwargs')
        if 'temperature' in MD_kwargs:
            temperature = MD_kwargs['temperature']
        elif 'temperature_range' in MD_kwargs:
            temperature = max(MD_kwargs['temperature_range'])
        else:
            raise KeyError('Temperature not in MD kwargs')
        temperature *= config.get('energy_range_factor',2)
        success = check_energy_range(db,row,temperature)
    db.update(id,atoms,calc=True,success=success)
    
if __name__ == '__main__':
    main()