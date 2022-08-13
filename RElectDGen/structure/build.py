import os

import ase
from ase.io import read, write
from ase.build import bulk, add_adsorbate, stack
import pandas as pd
import numpy as np
from ase.constraints import FixAtoms, FixBondLengths
import uuid

from RElectDGen.utils.io import add_to_trajectory


def structure_from_config(config):
    config = config.copy()
    constraints = []
    # Load initial structure if it exists
    if config.get('initial_structure') is not None:
        return read(config.get('initial_structure'))
        
    if config.get('slab_direction') is not None and config.get('mixture') is not None:
        ### Generate Slab
        vacuum = config.get('vacuum')
        config['vacuum'] = config.get('adsorbate_height',1)
        supercell = create_slab(config)

        slab_area = np.prod(supercell.cell.diagonal()[:2])
        molecule_volume = np.prod(config.get('cell'))

        print('Cell z dimension changed from ' + str(config.get('cell')[2]) + f' to {molecule_volume/slab_area}')
        cell_diag = [*supercell.cell.diagonal()[:2]-config.get('molecule_separation')/2,molecule_volume/slab_area]
        config['cell'] = np.identity(3)*cell_diag

        packmolpath = os.path.join(config.get('directory'),config.get('path_to_packmol_inputs'))
        molecules = createstructurefrompackmol(config.get('molecule'),
            config.get('nmolecules'),
            config.get('cell'),
            tolerance=config.get('molecule_separation'),
            packmolpath=packmolpath)

        cell_diag = [*supercell.cell.diagonal()[:2],molecule_volume/slab_area]
        molecules.set_cell(np.identity(3)*cell_diag)
        atomspath = os.path.join(config.get('data_directory'),config.get('atomspath','data/molecules/atoms'))
        molecule_charges = getchargesforpackmol(config.get('molecule'),
                                                config.get('nmolecules'),
                                                atomspath=atomspath)
        
        
        supercell.set_initial_charges([0]*len(supercell))
        molecules.set_initial_charges(molecule_charges)

        supercell = stack(supercell,molecules)

        supercell.center(vacuum = vacuum,axis=2)
        return supercell

    
    if config.get('crystal_a0') is not None:
        if config.get('slab_direction') is not None:

            supercell = create_slab(config)

        elif config.get('crystal_structure') is not None: #Generate Bulk cell
            a0 = config.get('crystal_a0')
            if isinstance(a0, float):
                a = b = c = a0
            supercell = bulk(config.get('element'), config.get('crystal_structure'), a=a, b=b, c=c, orthorhombic=True)
            supercell = supercell.repeat(config.get('supercell_size',[1,1,1]))

        return supercell

    if config.get('mixture') is not None:
        assert isinstance(config.get('molecule'), list)
        assert isinstance(config.get('nmolecules'), list)

        
        if isinstance(config.get('cell'), list):
            if isinstance(config.get('cell')[0], list):
                cell = np.array(config.get('cell'))
            else:
                cell = np.array([
                    [config.get('cell')[0],0,0],
                    [0,config.get('cell')[1],0],
                    [0,0,config.get('cell')[2]]])
        elif isinstance(config.get('cell'), float):
            cell = np.array([
                [config.get('cell'),0,0],
                [0,config.get('cell'),0],
                [0,0,config.get('cell')]])

        packmolpath = os.path.join(config.get('directory'),config.get('path_to_packmol_inputs'))
        
        supercell = createstructurefrompackmol(config.get('molecule'),
            config.get('nmolecules'),
            cell,
            tolerance=config.get('molecule_separation'),
            packmolpath=packmolpath)

        supercell.center(vacuum=config.get('vacuum'))

        atomspath = os.path.join(config.get('data_directory'),config.get('atomspath','data/molecules/atoms'))
        molecule_charges = getchargesforpackmol(config.get('molecule'),
                                                config.get('nmolecules'),
                                                atomspath=atomspath)

        supercell.set_initial_charges(molecule_charges)

        # molecules = [load_molecule(molecule,config.get('directory'))[0] for molecule in config.get('molecule')]

        # cell = np.zeros((3,3))
        # for mol in molecules:
        #     mol.center(vacuum = config.get('molecule_separation')/2.)
        #     cell_max = np.maximum(mol.cell.array,cell)
        # cell_max = np.identity(3)*cell_max.max()

        # np.random.seed(config.get('structure_seed',0))
        # for z in range(len(molecules)):
        #     for y in range(len(molecules)):
        #         for x, ind in enumerate(np.random.permutation(len(molecules))):
        #             if x == 0:
        #                 row = molecules[ind].copy()
        #                 row.set_cell(cell_max)
        #                 row.center()
        #                 if config.get('rotate',True):
        #                     row.rotate(np.random.uniform(0,360),'x',center='COM')
        #                     row.rotate(np.random.uniform(0,360),'y',center='COM')
        #                     row.rotate(np.random.uniform(0,360),'z',center='COM')
        #             else:
        #                 mol = molecules[ind].copy()
        #                 mol.set_cell(cell_max)
        #                 mol.center()
        #                 if config.get('rotate',True):
        #                     mol.rotate(np.random.uniform(0,360),'x',center='COM')
        #                     mol.rotate(np.random.uniform(0,360),'y',center='COM')
        #                     mol.rotate(np.random.uniform(0,360),'z',center='COM')
        #                 row = stack(row, mol, axis=0)
                    
        #         if y == 0:
        #             plane = row.copy()
        #         else:
        #             plane = stack(plane, row, axis=1)
                
        #     if z == 0:
        #         supercell = plane.copy()
        #     else:
        #         supercell = stack(supercell, plane, axis=2)
            
        supercell.pbc=config.get('pbc')
        return supercell

    if config.get('molecule') is not None:

        if isinstance(config.get('molecule'),str):
            ### Get 3D Molecule Structure
            mol, conformer_id = load_molecule(config.get('molecule'),config.get('directory'))
            if 'supercell' in locals():
                add_adsorbate(supercell, mol, config.get('adsorbate_height'),position=(supercell.cell[0,0]/2,supercell.cell[1,1]/2))
            else:
                supercell = mol
        elif isinstance(config.get('molecule'), list):
            for molecule in config.get('molecule'):
                mol, conformer_id = load_molecule(molecule,config.get('directory'))
                if 'supercell' in locals():
                    supercell.center(vacuum = config.get('molecule_separation')/2.)
                    mol.center(vacuum = config.get('molecule_separation')/2.)
                    cell_max = np.maximum(mol.cell.array,supercell.cell.array)
                    supercell.set_cell(cell_max)
                    supercell.center()
                    mol.set_cell(cell_max)
                    mol.center()
                    supercell = stack(supercell,mol)
                else:
                    supercell = mol
            
            #Transform to cubic cell
            cell_cubic = np.identity(3)*np.diag(supercell.cell.array).max()
            supercell.set_cell(cell_cubic)



    #Apply the constraints to the cell
    # supercell.set_constraint(constraints)
    supercell.set_pbc(config.get('pbc'))

    return supercell


def load_molecule(name, directory=None):
	if directory is not None:
		path = directory+'data/molecules/'
	else:
		path = 'data/molecules/'
	
	fl_energy = path+'energies/molecule_energy.csv'
	df = pd.read_csv(fl_energy)
	mask = df['molecule'].values == name
	potential_energies = df['potential energy'][mask]
	index = potential_energies.index[potential_energies.argmin()]
	conformer_id, charge = df[['conformer_id','charge']].loc[index]
	fl_xyz = path + f'coordinates/{name}_{conformer_id}_{charge}.xyz'
	mol = read(fl_xyz)
	return mol, conformer_id


def create_slab(config):
    constraints = []
    if config.get('slab_direction') == '110':
        supercell = ase.build.bcc110(config.get('element'), (config.get('supercell_size')), config.get('crystal_a0'),orthogonal=True,vacuum=config.get('vacuum'))
    elif config.get('slab_direction') == '100':
        supercell = ase.build.bcc100(config.get('element'), (config.get('supercell_size')), config.get('crystal_a0'),vacuum=config.get('vacuum'))

    #Constrain the bottom of the slab to be fixed
    indices = range(int(len(supercell.positions)*2/config.get('supercell_size')[2]))
    constraints.append(FixAtoms(indices = indices))

    supercell.set_constraint(constraints)

    if config.get('zperiodic', False):
        supercell.cell[2] *= config.get('supercell_size')[2]/(config.get('supercell_size')[2]-1)

    return supercell

filenames = {
    'EC': 'ethylenecarbonate.xyz',
    'DMC': 'dimethylcarbonate.xyz',
    'PF6': 'hexafluorophosphate.xyz',
    'Li+': 'li_plus.xyz',
    'FEMC': '1-fluoroethylmethylcarbonate.xyz',
    'HFE': '1,1,2,2-tetrafluoroethyl2,2,2-trifluoroethylether.xyz',
    'FSI':  'bis(fluorosulfonyl)imide.xyz',
    'TFSI': 'bis(trifluoromethanesulfon)imide.xyz',
    'DME': 'dimethoxyethane.xyz',
    'DOL': 'dioxolane.xyz',
    'FEC': 'fluoroethylenecarbonate.xyz',
    'nitrate': 'nitrate.xyz',
    'PC': 'propylenecarbonate.xyz',
    'VC': 'vinylenecarbonate.xyz',
}

def createstructurefrompackmol(
    molecules: list,
    nmolecules: list,
    cell: np.ndarray,
    tolerance: float = 2.0,
    filetype: str = 'xyz',
    packmolpath: str = '',
):
    tmp_input = 'input.' + str(uuid.uuid4()) 
    tmp_output = 'output.' + str(uuid.uuid4())  + '.xyz'

    #create packmol input file
    packmol_txt = (
        f"tolerance {tolerance}\n" +
        f"filetype {filetype}\n" +
        f"output {tmp_output}\n")

    for i, (molecule, nmolecule) in enumerate(zip(molecules,nmolecules)):
        filename = os.path.join(packmolpath,filenames[molecule])
        packmol_txt += (
            f"structure {filename}\n"
            f"  number {nmolecule}\n"
            f"  inside box 0.0 0.0 0.0 {cell[0,0]} {cell[1,1]} {cell[2,2]}\n"
            "end structure\n\n"
        )

    with open(tmp_input, 'w') as fl:
        fl.write(packmol_txt)

    command_line = f"packmol < {tmp_input}"
    os.system(command_line)
    
    atoms = read(tmp_output)
    atoms.set_cell(cell)

    os.remove(tmp_input)
    os.remove(tmp_output)

    return atoms

def getchargesforpackmol(
    molecules: list,
    nmolecules: list,
    atomspath: str = '',
):

    charges = []

    for i, (molecule, nmolecule) in enumerate(zip(molecules,nmolecules)):
        filename = os.path.join(atomspath,filenames[molecule].split('.xyz')[0]+'.json')
        atoms = read(filename)
        
        charges += list(atoms.get_initial_charges())*nmolecule

    return charges

def get_initial_structure(config):

    structure_file = os.path.join(config.get('data_directory'),config.get('structure_file'))
    if os.path.isfile(structure_file):
        supercell = read(structure_file)
    else:
        supercell = structure_from_config(config)
        write(structure_file,supercell)
        if config.get('initial_structures_file') is not None:
            initial_structures_filename = os.path.join(
                config.get('data_directory'),
                config.get('initial_structures_file')
            )
            add_to_trajectory(supercell,initial_structures_filename)

    return supercell