from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
# version_dict = {}
# with open(Path(__file__).parents[0] / "nequip/_version.py") as fp:
#     exec(fp.read(), version_dict)
# version = version_dict["__version__"]
# del version_dict

setup(
    name="RElectDGen",
    description="Private extension for running e3nn networks and nequip models",
    author="Emil Annevelink",
    python_requires=">=3.6",
    packages=find_packages(include=["RElectDGen", "RElectDGen.*"]),
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
            "REDGEN-start = RElectDGen.scripts.start_active_learning:main",
            "REDGEN-generate-shell = RElectDGen.scripts.generate_shell:main",
            "REDGEN-combine-datasets = RElectDGen.scripts.combine_datasets:main",
            "REDGEN-gpaw-MD = RElectDGen.scripts.gpaw_MD:main",
            "REDGEN-train-NN = RElectDGen.scripts.train_NN:main",
            "REDGEN-MLP-MD = RElectDGen.scripts.MLP_MD:main",
            "REDGEN-MLP-MD-run = RElectDGen.scripts.MD_run:main",
            "REDGEN-sample-adv = RElectDGen.scripts.adv_sampling:main",
            "REDGEN-md-adv = RElectDGen.scripts.MD_adv:main",
            "REDGEN-gpaw-active = RElectDGen.scripts.gpaw_active:main",
            "REDGEN-gpaw-active-array = RElectDGen.scripts.gpaw_active_array:main",
            "REDGEN-gpaw-summary = RElectDGen.scripts.gpaw_summary_array:main",
            "REDGEN-log = RElectDGen.scripts.write_to_log:main",
            "REDGEN-restart = RElectDGen.scripts.restart:main",
        ]
    },
    install_requires=[
        "numpy",
        "ase",
        "h5py",
        "uuid",
        "pandas",
        "torch>=1.8",
        "e3nn>=0.3.3",
        "pyyaml",
        "packaging",
        "memory_profiler",
    ],
    zip_safe=True,
)