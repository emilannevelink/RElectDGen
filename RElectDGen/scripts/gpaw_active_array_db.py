import sys
import os, argparse
import yaml

from ase.db import connect
from ase.parallel import world

from RElectDGen.calculate.recalculate import calculate_atoms
from RElectDGen.structure.utils import extend_cell


def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    parser.add_argument('--MLP_config_file', dest='MLP_config',
                        help='Nequip configuration file', type=str)
    parser.add_argument('--array_index', dest='array_index',
                        help='active_learning_loop', type=int)
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

    
    for i, row in enumerate(db.select(calc=False)):
        if i == array_index:
            atoms = row.toatoms()
            id = row['id']
            break
    
    if 'atoms' not in locals():
        sys.exit()

    if world.rank == 0:
        print(i, array_index, id)
    
    atoms, success = calculate_atoms(atoms, config.get('dft_config'))
    
    db.update(id,atoms,calc=True,success=success)
    
if __name__ == '__main__':
    main()