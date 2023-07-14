import h5py
from scipy.stats.stats import KstestResult
from RElectDGen.statistics.cutoffs import FakeFitResult

def save_cutoffs_distribution_info(filename, cutoff_dist_all,index=None):
    with h5py.File(filename,'a') as hf:
        save_data(hf,index,cutoff_dist_all)

def save_data(hf:h5py.Group, key:str, data):
    if key in hf:
        del hf[key]
    if isinstance(data, dict):
        if key is not None:
            hf = hf.create_group(key)
        for key, val in data.items():
            save_data(hf,key,val)
    elif isinstance(data,KstestResult):
        save_data(hf,key,data._asdict())
    elif isinstance(data,FakeFitResult):
        data_dict = {'pvalue': data.pvalue}
        save_data(hf,key,data_dict)
    else:
        hf.create_dataset(key,data=data)

def load_cutoffs_distribution_info(filename,index=None):

    with h5py.File(filename,'r') as hf:
        data = read_data(hf,index)
    return data

def read_data(hf:h5py.Group, key:str = None):
    if key is not None:
        hf = hf[key]
    if isinstance(hf,h5py.Group):
        data = {}
        for k in hf.keys():
            data[k] = read_data(hf,k)
    elif isinstance(hf,h5py.Dataset):
        data = hf[()]
    if isinstance(data,bytes):
        data = data.decode()
    return data