from ase.units import kB

def check_energy_range(db,atoms,temperature):
    row_id = atoms.info['start_row_id']
    atoms_start = db[row_id].toatoms()
    energy_start = atoms_start.get_potential_energy()
    energy_end = atoms.get_potential_energy()

    if (energy_end-energy_start)/len(atoms) < kB*temperature:
        return True
    
    return False
