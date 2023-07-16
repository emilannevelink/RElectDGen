import h5py
import torch

def load_from_hdf5(filename):

    all_data = {}
    with h5py.File(filename) as fl:
        all_data = load_all_keys(fl)

    return all_data

def load_all_keys(hdf5_file):

    all_data = {}
    for key in hdf5_file.keys():
        if isinstance(hdf5_file[key],h5py.Dataset):
            all_data[key] = torch.tensor(hdf5_file[key][()],dtype=torch.float)
        else:
            all_data[key] = load_all_keys(hdf5_file[key])

    return all_data

def save_to_hdf5(filename,data):
    with h5py.File(filename, 'a') as hf:
        write_all_keys(hf,data)

def write_all_keys(hf,data):
    for key in data.keys():
        if key in hf:
            del hf[key]
        if isinstance(data[key], dict):
            hf.create_group(key)
            write_all_keys(hf[key],data[key])
        else:
            if isinstance(data[key],list):
                hf.create_dataset(key,data=data[key])
            else:
                hf.create_dataset(key,data=data[key].tolist())