import numpy as np
import nilearn
from get_3d_peaks import get_3d_peaks


def get_sub_dict(X, Y, Z, sample_size=108):
    '''
        TO-FIX: add sample_size and space as arguments
        Build sub dictionnary of a study using the
        nimare structure

        Args:
            X (list): Store the X coordinates
            Y (list): Store the Y coordinates
            Z (list): Store the Z coordinates

        Returns:
            (dict): Dictionary storing the coordinates for a
                single study using the Nimare structure.
    '''
    return {
        'contrasts': {
            '0': {
                'coords':
                    {
                        'x': X,
                        'y': Y,
                        'z': Z,
                        'space': 'MNI'
                    },
                'sample_sizes': sample_size
            }
        }
    }


def get_activations(filepath, gray_mask=None, threshold=1.96):
    '''
        Given the path of a Nifti1Image, retrieve the xyz activation
        coordinates.

        Inputs:
        --------
            filepath (string):
                Path to a nibabel.Nifti1Image from which to extract coordinates
            threshold (float):
                Same as in the extract function

        Returns:
        --------
            (tuple): Size 3 tuple of lists storing respectively the X, Y and
                Z coordinates
    '''
    X, Y, Z = [], [], []

    try:
        img = nilearn.image.load_img(filepath)
    except ValueError:  # File path not found
        print(f'File {filepath} not found. Ignored.')
        return None

    if np.isnan(img.get_fdata()).any():
        print(f'File {filepath} contains Nan. Ignored.')
        return None

    peaks = get_3d_peaks(img, mask=gray_mask, threshold=threshold)
    if peaks is None:
        return None

    for peak in peaks:
        X.append(peak['pos'][0])
        Y.append(peak['pos'][1])
        Z.append(peak['pos'][2])

    del peaks
    return X, Y, Z
