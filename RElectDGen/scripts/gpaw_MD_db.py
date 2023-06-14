import os, argparse
from ase.io import trajectory

from ase.io.trajectory import Trajectory
from ase.io import read, write
from RElectDGen.sampling.md import md_from_atoms
from ase.md import MDLogger
from ase import units


from RElectDGen.structure.build import get_initial_structure

from ase.parallel import world
from ase.db import connect
import yaml

def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='config_file', type=str,
                        help='active_learning configuration file')
    parser.add_argument('MLP_config', metavar='MLP_config', type=str,
                        help='Nequip configuration file')
    # parser.add_argument('--config_file', dest='config',
    #                     help='active_learning configuration file', type=str)
    # parser.add_argument('--MLP_config_file', dest='MLP_config',
    #                     help='Nequip configuration file', type=str)
    args = parser.parse_args()

    
    # if world.rank==0:
    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)
    
    return config


def check_dft_steps(config):
    dft_md_kwargs = config.get('dft_md_kwargs')
    trajectory_file = os.path.join(
        config.get('data_directory'),
        dft_md_kwargs.get('trajectory_file')
    )
    initial_MD_steps = dft_md_kwargs.get('steps')
    
    if os.path.isfile(trajectory_file):
        if world.rank == 0:
            print('traj file')
        try:
            traj = Trajectory(trajectory_file)
            initial_MD_steps-=len(traj)
        except:
            if world.rank == 0:
                print('Loading Trajectory fialed, re-initializing',flush=True)

    return initial_MD_steps


def main(args=None):

    config = parse_command_line(args)

    supercell = get_initial_structure(config)

    config['cell'] = supercell.get_cell()

    data_directory = config.get('data_directory')
    dft_md_kwargs = config.get('dft_md_kwargs')
    trajectory_file = os.path.join(
        data_directory,
        dft_md_kwargs.get('trajectory_file')
    )
    # data_file = os.path.join(config.get('data_directory'),config.get('hdf5_file'))
    
    print(trajectory_file)

    initial_MD_steps = check_dft_steps(config)

    if initial_MD_steps > 0:
        if world.rank == 0:
            print(f'Running GPAW, initial MD steps {initial_MD_steps}', flush =True)
        
        try:
            if os.path.isfile(trajectory_file):
                traj = Trajectory(trajectory_file)
                if len(traj)>0:
                    supercell = Trajectory(trajectory_file)[-1]
        except:
            pass

        from RElectDGen.calculate._dft import oracle_from_config
        calc_oracle = oracle_from_config(config.get('dft_config'), atoms=supercell,data_directory=data_directory)
        supercell.calc = calc_oracle    
        # MaxwellBoltzmannDistribution(supercell, temperature_K=config.get('GPAW_MD_temperature'))
        # ZeroRotation(supercell)
        # Stationary(supercell)

        # md_func, md_kwargs = md_func_from_config(config, prefix='GPAW')

        traj, stable = md_from_atoms(supercell,**dft_md_kwargs,delete_tmp=False)

        # GPAW_MD_dump_file = os.path.join(config.get('data_directory'),config.get('GPAW_MD_dump_file'))
        # md_kwargs['logfile'] = GPAW_MD_dump_file
        
        # dyn = md_func(supercell, **md_kwargs)
        
        # # if os.path.isfile(GPAW_MD_dump_file):
        # #     os.remove(GPAW_MD_dump_file)
        # # dyn.attach(MDLogger(dyn,supercell,GPAW_MD_dump_file,mode='w'),interval=1)
        # traj = Trajectory(trajectory_file, 'a', supercell)
        # dyn.attach(traj.write, interval=1)
        # dyn.run(initial_MD_steps)
        # traj.close()
        # traj = Trajectory(trajectory_file)

        # add traj to db
        db_filename = os.path.join(
            config.get('data_directory'),
            config.get('ase_db_filename')
        )
        db = connect(db_filename)
        for atoms in traj:
            db.write(atoms,md_stable=0,calc=True,success=True)



if __name__ == "__main__":
    main()