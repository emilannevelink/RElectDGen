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
        ]
    },
    install_requires=[
        "numpy",
        "ase",
        "h5py",
        "uuid",
        "pandas"
        "torch>=1.8",
        "e3nn>=0.3.3",
        "pyyaml",
        "packaging",
    ],
    zip_safe=True,
)