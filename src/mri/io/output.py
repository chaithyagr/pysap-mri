from scipy.io import savemat
import pickle as pkl
import nibabel as nib
import numpy as np
from pydicom.dataset import Dataset, FileDataset
from pydicom import dcmwrite
import pydicom
import datetime


def create_dicom_from_matrix(matrix, filename, slice_index=1):
    # Create a new dataset (DICOM object)
    ds = Dataset()

    # Populate the required DICOM metadata (minimal example)
    ds.PatientName = "Test^Patient"
    ds.PatientID = "123456"
    ds.Modality = "MR"
    ds.StudyInstanceUID = "1.2.3.4.5.6.7.8.9"
    ds.SeriesInstanceUID = "1.2.3.4.5.6.7.8.9.1"
    ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9.1.{}".format(slice_index)
    ds.SOPClassUID = pydicom.uid.MRImageStorage
    ds.InstanceNumber = slice_index

    # Set slice location and spacing if available
    ds.SliceLocation = float(slice_index)  # Optional: for multi-slice datasets

    # Add the Pixel Data (must be bytes)
    ds.Rows, ds.Columns = matrix.shape
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # 1 for signed integers
    ds.PixelData = matrix.tobytes()

    # Optional: Add additional DICOM metadata, like image orientation, pixel spacing, etc.
    ds.ImagePositionPatient = [0.0, 0.0, float(slice_index)]  # Example for slice position

    # Add date and time
    dt = datetime.datetime.now()
    ds.StudyDate = dt.strftime('%Y%m%d')
    ds.StudyTime = dt.strftime('%H%M%S')
    # Set DICOM file transfer syntax attributes
    ds.is_little_endian = True  # Standard for most DICOM files
    ds.is_implicit_VR = True    # True for implicit VR, False for explicit VR

    # Save the DICOM file
    dcmwrite(filename, ds)

def save_data(filename, recon, header=None):
    """Save reconstructed data to a file.

    Parameters
    ----------
    filename : str
        Path to the output file.
    recon : array_like
        Reconstructed data to be saved.
    header : object, optional
        Header information for the file format (default: None).

    """
    extension = filename.split('.')[-1].lower()
    save_dict = {'recon': recon, 'header': header}
    if extension == 'pkl':
        with open(filename, 'wb') as f:
            pkl.dump(save_dict, f)
    elif extension == 'mat':
        savemat(filename, save_dict)
    elif extension == 'nii':
        orient = np.eye(4)
        if header is not None and 'orientation' in header:
            orient = header['orientation']
        recon = np.abs(recon)
        recon /= np.max(recon)
        img = nib.Nifti1Image(recon, orient)
        nib.save(img, filename)
    elif extension == 'dcm':
        if header is not None and 'slice_num' in header:
            create_dicom_from_matrix(recon, filename, header['slice_num'])
        else:
            raise ValueError("DICOM save is reserved for 2D images.")
    else:
        raise ValueError("Unsupported file extension.")
