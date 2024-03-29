import os, argparse
from ase.io import trajectory

from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, ZeroRotation, Stationary
from RElectDGen.utils.md_utils import md_func_from_config
from ase.md import MDLogger
from ase import units


from RElectDGen.structure.build import get_initial_structure

from ase.parallel import world
import yaml

def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    parser.add_argument('--MLP_config_file', dest='MLP_config',
                        help='Nequip configuration file', type=str)
    args = parser.parse_args()

    
    # if world.rank==0:
    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)
    
    return config


def get_initial_MD_steps(config):
    trajectory_file = os.path.join(config.get('data_directory'),config.get('trajectory_file'))
    initial_MD_steps = config.get('GPAW_MD_steps')
    
    if os.path.isfile(trajectory_file):
        if world.rank == 0:
            print('traj file')
        try:
            traj = Trajectory(trajectory_file)
            # print('traj file load')
            # data_write = extract_traj_data(traj)
            # write_learning_data(data_file, data_write, append=False)
        
            # supercell = traj[-1]
            initial_MD_steps-=len(traj)
        except:
            if world.rank == 0:
                print('Loading Trajectory fialed, re-initializing',flush=True)

    return initial_MD_steps

def main(args=None):

    config = parse_command_line(args)

    supercell = get_initial_structure(config)

    config['cell'] = supercell.get_cell()

    trajectory_file = os.path.join(config.get('data_directory'),config.get('trajectory_file'))
    # data_file = os.path.join(config.get('data_directory'),config.get('hdf5_file'))
    
    print(trajectory_file)

    initial_MD_steps = get_initial_MD_steps(config)

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

        from RElectDGen.calculate._MLIP import oracle_from_config
        calc_oracle = oracle_from_config(config, atoms=supercell)
        supercell.calc = calc_oracle    
        MaxwellBoltzmannDistribution(supercell, temperature_K=config.get('GPAW_MD_temperature'))
        ZeroRotation(supercell)
        Stationary(supercell)
        md_func, md_kwargs = md_func_from_config(config, prefix='GPAW')
        GPAW_MD_dump_file = os.path.join(config.get('data_directory'),config.get('GPAW_MD_dump_file'))
        md_kwargs['logfile'] = GPAW_MD_dump_file
        
        dyn = md_func(supercell, **md_kwargs)
        
        # if os.path.isfile(GPAW_MD_dump_file):
        #     os.remove(GPAW_MD_dump_file)
        # dyn.attach(MDLogger(dyn,supercell,GPAW_MD_dump_file,mode='w'),interval=1)
        traj = Trajectory(trajectory_file, 'a', supercell)
        dyn.attach(traj.write, interval=1)
        dyn.run(initial_MD_steps)
        traj.close()
        traj = Trajectory(trajectory_file)

if __name__ == "__main__":
    main()