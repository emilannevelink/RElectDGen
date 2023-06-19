import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from nequip.data import AtomicDataDict
from ase.calculators.singlepoint import SinglePointCalculator
from nequip.ase.nequip_calculator import NequIPCalculator

from RElectDGen.uncertainty.models import uncertainty_base
from RElectDGen.uncertainty.io import load_UQ

def load_unc_calc(config, MLP_config):

    UQ = load_UQ(config,MLP_config)
    UQ.calibrate()

    unc_calc = UncCalculator(UQ)

    return UQ, unc_calc

class UncCalculator(NequIPCalculator): # so that it passes through nequip
    """NequIP ASE Calculator.

    .. warning::

        If you are running MD with custom species, please make sure to set the correct masses for ASE.

    """

    implemented_properties = ["energy", "energies", "forces", "stress", "free_energy", "magmoms"] 

    def __init__(
        self,
        uq_module: uncertainty_base,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.uq_module = uq_module
        
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A

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
        # print('Uncertainty Calculator')
        # print(atoms)
        # predict + extract data
        out = self.uq_module.predict_uncertainty(atoms)
        # print(out)
        self.results = {}
        atoms.info['uncertainties'] = out['uncertainties'].sum(axis=-1).detach().squeeze(-1).cpu().numpy() #doesn't save in results
        # only store results the model actually computed to avoid KeyErrors
        if AtomicDataDict.TOTAL_ENERGY_KEY in out:
            self.results["energy"] = self.energy_units_to_eV * (
                out[AtomicDataDict.TOTAL_ENERGY_KEY]
                .detach()
                .cpu()
                .numpy()
                .reshape(tuple())
            )
            # "force consistant" energy
            self.results["free_energy"] = self.results["energy"]
        if AtomicDataDict.PER_ATOM_ENERGY_KEY in out:
            self.results["energies"] = self.energy_units_to_eV * (
                out[AtomicDataDict.PER_ATOM_ENERGY_KEY]
                .detach()
                .squeeze(-1)
                .cpu()
                .numpy()
            )
        if AtomicDataDict.FORCE_KEY in out:
            # force has units eng / len:
            self.results["forces"] = (
                self.energy_units_to_eV / self.length_units_to_A
            ) * out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
        if AtomicDataDict.STRESS_KEY in out:
            stress = out[AtomicDataDict.STRESS_KEY].detach().cpu().numpy()
            stress = stress.reshape(3, 3) * (
                self.energy_units_to_eV / self.length_units_to_A**3
            )
            # ase wants voigt format
            stress_voigt = full_3x3_to_voigt_6_stress(stress)
            self.results["stress"] = stress_voigt

class FakeUncCalculator(NequIPCalculator): # so that it passes through nequip
    """NequIP ASE Calculator.

    .. warning::

        If you are running MD with custom species, please make sure to set the correct masses for ASE.

    """

    implemented_properties = ["energy", "energies", "forces", "stress", "free_energy", "magmoms"]

    def __init__(
        self,
        seed: int = 0,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.seed = seed
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A

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
        # print('Uncertainty Calculator')
        # print(atoms)
        self.seed+=1
        np.random.seed(self.seed)
        # predict + extract data
        self.results = {
            'energy': np.random.rand(1),
            'forces': np.random.rand(*atoms.get_positions().shape),
        }
        atoms.info['uncertainties'] = np.random.rand(atoms.get_positions().shape[0])
        print(atoms.info['uncertainties'])
