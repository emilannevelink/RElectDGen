import os
import torch
from nequip.utils import Config
from nequip.ase.nequip_calculator import NequIPCalculator
from nequip.data.transforms import TypeMapper
from nequip.model import model_from_config
import numpy as np
from ase.parallel import world

from RElectDGen.utils.save import get_results_dir

def oracle_from_config(config,atoms=None):

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

    return calculator

def nn_from_results():
    
    train_directory = get_results_dir()
    
    file_config = train_directory + "/config_final.yaml"
    MLP_config = Config.from_file(file_config)

    chemical_symbol_to_type = MLP_config.get('chemical_symbol_to_type')
    transform = TypeMapper(chemical_symbol_to_type=chemical_symbol_to_type)

    model_path = train_directory + "/best_model.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_state_dict = torch.load(model_path, map_location=torch.device(device))
    model = model_from_config(
            config=MLP_config, initialize=False, # dataset=dataset
        )
    model.load_state_dict(model_state_dict)
    # if MLP_config.compile_model:
    import e3nn
    model = e3nn.util.jit.compile(model)
    print('compiled model', flush=True)
    torch._C._jit_set_bailout_depth(MLP_config.get("_jit_bailout_depth",2))
    torch._C._jit_set_profiling_executor(False)
    # model = torch.jit.script(model)
    # model = torch.jit.freeze(model)
    # model = torch.jit.optimize_for_inference(model)
    
    calc_nn = NequIPCalculator(model=model, r_max=MLP_config.r_max,device=device, transform=transform)

    return calc_nn, model, MLP_config