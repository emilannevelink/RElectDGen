from ase.io import Trajectory

def add_to_trajectory(atoms, file):
    writer = Trajectory(file,'a')
    writer.write(atoms)