from ase.units import kB
from ase.parallel import world

def check_energy_range(db,atoms,temperature):
    row_id = atoms.info.get('start_row_id')
    if row_id is None:
        if world.rank == 0:
            print('start row id not in atoms.info')
        return True
    atoms_start = db[row_id].toatoms()
    energy_start = atoms_start.get_potential_energy()
    energy_end = atoms.get_potential_energy()

    if (energy_end-energy_start)/len(atoms) < kB*temperature:
        return True
    
    return False
