"""
This is a hacked version of nipy.labs.statistical_mapping
to address some bugs in that code
"""


import nibabel
import numpy as np
from nipy.algorithms.graph.graph import wgraph_from_3d_grid
from nipy.algorithms.graph.field import field_from_graph_and_data
from nipy.io.nibcompat import get_affine


def get_3d_peaks(image, mask=None, threshold=0., nn=18, order_th=0):
    """
    returns all the peaks of image that are with the mask
    and above the provided threshold

    Parameters
    ----------
    image, (3d) test image
    mask=None, (3d) mask image
        By default no masking is performed
    threshold=0., float, threshold value above which peaks are considered
    nn=18, int, number of neighbours of the topological spatial model
    order_th=0, int, threshold on topological order to validate the peaks

    Returns
    -------
    peaks, a list of dictionaries, where each dict has the fields:
    vals, map value at the peak
    order, topological order of the peak
    ijk, array of shape (1,3) grid coordinate of the peak
    pos, array of shape (n_maxima,3) mm coordinates (mapped by affine)
        of the peaks
    """
    # Masking
    shape = image.shape
    if mask is not None:
        data = image.get_data() * mask.get_data()
        xyz = np.array(np.where(data > threshold)).T
        data = data[data > threshold]
    else:
        data = image.get_data().ravel()
        xyz = np.reshape(np.indices(shape), (3, np.prod(shape))).T
    affine = get_affine(image)

    if not (data > threshold).any():
        print('no suprathreshold voxels found')
        return None

    # Extract local maxima and connex components above some threshold
    ff = field_from_graph_and_data(wgraph_from_3d_grid(xyz, k=18), data)
    maxima, order = ff.get_local_maxima(th=threshold)

    # retain only the maxima greater than the specified order
    maxima = maxima[order > order_th]
    order = order[order > order_th]

    n_maxima = len(maxima)
    if n_maxima == 0:
        # should not occur ?
        return None

    # reorder the maxima to have decreasing peak value
    vals = data[maxima]
    idx = np.argsort(- vals)
    maxima = maxima[idx]
    order = order[idx]

    vals = data[maxima]
    ijk = xyz[maxima]
    pos = np.dot(np.hstack((ijk, np.ones((n_maxima, 1)))), affine.T)[:, :3]
    peaks = [{'val': vals[k], 'order': order[k], 'ijk': ijk[k], 'pos': pos[k]}
             for k in range(n_maxima)]

    return peaks


if __name__ == '__main__':
    # testing code
    # this assumes one is using the narps docker image
    mask = nibabel.load(
        '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
    image = nibabel.load(
        '/data/output/zstat/4953_08MQ/hypo1_thresh.nii.gz')
    print(image.shape)
    print(mask.shape)
    p = get_3d_peaks(image, mask)
