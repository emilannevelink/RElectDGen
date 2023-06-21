import numpy as np
from ase.io import read

from nequip.data import dataset_from_config

from RElectDGen.uncertainty.models import uncertainty_base
from RElectDGen.uncertainty import models as uncertainty_models
from RElectDGen.calculate._MLIP import nn_from_results, nns_from_results

def load_UQ(config,MLP_config)->uncertainty_base:
    train_directory = config.get('train_directory','results')
    if train_directory[-1] == '/':
        train_directory = train_directory[:-1]
    
    template = MLP_config.get('run_name')
    uncertainty_function = config.get('uncertainty_function', 'Nequip_latent_distance')
    print(uncertainty_function,flush=True)
    ### Setup NN ASE calculator
    if uncertainty_function in ['Nequip_ensemble']:
        n_ensemble = config.get('n_uncertainty_ensembles',4)
    else:
        n_ensemble = 1
    
    calc_nn, model, MLP_config = nns_from_results(train_directory,n_ensemble,template)
    if n_ensemble == 1:
        model = model[0]
        MLP_config = MLP_config[0]

    ### Calibrate Uncertainty Quantification
    UQ_func = getattr(uncertainty_models,uncertainty_function)

    # UQ = UQ_func(model, config, MLP_config)
    UQ = UQ_func(model, config, MLP_config)
    return UQ
    
def get_dataset_uncertainties(UQ: uncertainty_base):

    dataset = dataset_from_config(UQ.MLP_config)

    dataset_train_uncertainties = {}
    dataset_val_uncertainties = {}
    for symbol in UQ.MLP_config.get('chemical_symbol_to_type'):
        dataset_train_uncertainties[symbol] = np.empty(0)
        dataset_val_uncertainties[symbol] = np.empty(0)

    train_idcs = np.array(UQ.MLP_config.get('train_idcs'))
    for tind in train_idcs:
        data = dataset[tind]
        out = UQ.predict_uncertainty(data)
        for symbol in UQ.MLP_config.get('chemical_symbol_to_type'):
            mask = (data['atom_types']==UQ.MLP_config.get('chemical_symbol_to_type')[symbol]).flatten()
            dataset_train_uncertainties[symbol] = np.concatenate([
                dataset_train_uncertainties[symbol],
                out['uncertainties'].sum(axis=-1)[mask].detach().cpu().numpy()
            ])

    val_idcs = np.array(UQ.MLP_config.get('val_idcs'))
    for vind in val_idcs:
        data = dataset[vind]
        out = UQ.predict_uncertainty(data)
        for symbol in UQ.MLP_config.get('chemical_symbol_to_type'):
            mask = (data['atom_types']==UQ.MLP_config.get('chemical_symbol_to_type')[symbol]).flatten()
            dataset_val_uncertainties[symbol] = np.concatenate([
                dataset_val_uncertainties[symbol],
                out['uncertainties'].sum(axis=-1)[mask].detach().cpu().numpy()
            ])

    return dataset_train_uncertainties, dataset_val_uncertainties