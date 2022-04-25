import os
import json
import pandas as pd
import numpy as np
from .save import get_results_dir

def write_to_tmp_dict(filename,dict):
    
    if os.path.isfile(filename):
        with open(filename,'r') as fl:
            old_dict = json.load(fl)
    else:
        old_dict = {}

    new_dict = {**old_dict,**dict}

    with open(filename,'w') as fl:
        json.dump(new_dict,fl)


def get_mae_from_results():

    results_dir = get_results_dir()

    results_metrics = os.path.join(results_dir,'metrics_epoch.csv')
    data = pd.read_csv(results_metrics)

    best_ind = np.argmin(data[' validation_loss'])

    mae_dict = {
        'training_f_mae': data[' training_loss_f'][best_ind],
        'training_e_mae': data[' training_e_mae'][best_ind],
        'validation_f_mae': data[' validation_loss_f'][best_ind],
        'validation_e_mae': data[' validation_e_mae'][best_ind],
    }

    return mae_dict

def UQ_params_to_dict(params, prefix: str = ''):
    if not prefix.endswith('_'):
        prefix += '_'
    out_dict = {}
    for key in params:
        if isinstance(params[key],list):
            out_dict[prefix+key+'_base'] = float(params[key][0][0])
            out_dict[prefix+key+'_linear'] = float(params[key][0][1])
        else:
            out_dict[prefix+key+'_base'] = float(params[key][0])
            out_dict[prefix+key+'_linear'] = float(params[key][1])
    
    return out_dict