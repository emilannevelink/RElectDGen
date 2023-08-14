from ase.units import kB
from ase.parallel import world

def check_energy_range(db,row,temperature):
    if not 'start_row_id' in row:
        if world.rank == 0:
            print('start row id not in row')
        return True
    
    atoms = row.toatoms()
    row_id = row.get('start_row_id')
        
    atoms_start = db[row_id].toatoms()
    energy_start = atoms_start.get_potential_energy()
    energy_end = atoms.get_potential_energy()

    if (energy_end-energy_start)/len(atoms) < kB*temperature:
        return True
    if world.rank == 0:
        print('Energy not close enough')
    return False
