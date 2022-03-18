from typing import Union, Optional, Callable, Dict
import warnings
import torch
import numpy as np

import ase.data
from ase.calculators.calculator import Calculator, all_changes
import ase

class FunctionCalculator(Calculator):

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        function,
        cutoff = 2,
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.function = function
        self.cutoff = cutoff

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Calculate properties.

        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        first_index, second_index, shifts = ase.neighborlist.primitive_neighbor_list(
            "ijS",
            atoms.pbc,
            atoms.cell,
            atoms.positions,
            cutoff=float(self.cutoff),
            self_interaction=False,  # we want edges from atom to itself in different periodic images!
            use_scaled_positions=False,
        )


        positions = torch.tensor(atoms.positions, requires_grad=True)
        pair_distances = positions[first_index] - positions[second_index] + torch.tensor(np.dot(shifts,atoms.cell))
        # pair_distances = torch.tensor(D,requires_grad=True)
        pair_distances = torch.abs(pair_distances)
        energy = self.function(pair_distances).sum()/2.
        forces = -torch.autograd.grad(energy,positions)[0]

        # store results
        self.results = {
            "energy": energy.detach().numpy(),
            "forces": forces.detach().numpy(),
        }
