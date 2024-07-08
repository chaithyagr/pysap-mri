import nibabel as nib
import numpy as np
import pickle as pkl
from mrinufft.io.siemens import read_siemens_rawdat


def load_input_data(obs_file):
    if obs_file[:-4] == ".nii":
        image = nib.load(obs_file).get_fdata(dtype=np.complex64)
    elif obs_file[:-4] == ".pkl":
        image = pkl.load(open(obs_file, "rb"))['recon']
    elif obs_file[:-4] == ".dat":
        raw_data, data_header = read_siemens_rawdat(obs_file)    