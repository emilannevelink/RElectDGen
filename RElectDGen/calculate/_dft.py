import os
import numpy as np
from ase.parallel import world

def oracle_from_config(config,atoms=None):

    calculator_type = config.get('calculator_type', 'gpaw')
    if calculator_type == 'gpaw':
        from gpaw import GPAW, PW, FermiDirac, restart
        from gpaw.eigensolvers import Davidson
        from gpaw.poisson import PoissonSolver
        from gpaw.xc.vdw import VDWFunctional

        if 'vdW' in config.get('xc'):
            xc = VDWFunctional(config.get('xc'))
        else:
            xc=config.get('xc')

        if atoms is not None:
            config['cell'] = atoms.cell
            config['pbc'] = atoms.pbc
            if sum(atoms.pbc)==0:
                atoms.center()
            charge = atoms.get_initial_charges().sum().round()
        else:
            charge = 0
        
        if world.rank == 0:
            print('Charge: ', charge)

        if config.get('kxl') is not None and config.get('cell') is not None:
            if len(np.shape(config.get('cell'))) == 2:
                kpts = np.ceil((config.get('kxl')/np.linalg.norm(config.get('cell'),axis=1))).astype(int)
            elif len(np.shape(config.get('cell'))) == 1:
                kpts = np.ceil((config.get('kxl')/np.array(config.get('cell')))).astype(int)

            kpts = [kpts[i] if bool else 1 for i, bool in enumerate(config.get('pbc'))]
        else:
            kpts = config.get('kpoints')

        GPAW_dump_file = os.path.join(config.get('data_directory'),config.get('GPAW_dump_file'))
        calculator = GPAW(
                    xc=xc,
                    kpts=(kpts),
                    h = config.get('grid_spacing'),
                    occupations={'name': config.get('occupation'), 'width': config.get('occupation_width')},
                    eigensolver=Davidson(config.get('Davidson_steps')),
                    poissonsolver=PoissonSolver(),#eps=config.get('Poisson_tol')),
                    symmetry={'point_group': False},
                    # maxiter=333,
                    txt=GPAW_dump_file)

        calculator.set(charge=charge)

        if not config.get('pbc')[-1] and np.sum(config.get('pbc'))==2:
            calculator.set(poissonsolver={'dipolelayer': 'xy'})

    elif calculator_type == 'equation':
        from .ase_equation import FunctionCalculator

        function = eval(config.get('equation'))
        calculator = FunctionCalculator(
            function,
            config.get('cutoff',2)
        )

    return calculator
