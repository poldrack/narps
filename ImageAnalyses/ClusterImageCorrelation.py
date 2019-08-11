"""
Compute correlations between cluster maps
as well as their correlation with the original
Tom et al. maps
"""

import os
import glob
import numpy
import pandas
import nilearn.input_data
import seaborn
import matplotlib.pyplot as plt


def cluster_image_correlation(basedir):
    # load cluster maps
    cluster_img_dir = os.path.join(
        basedir,
        'output/cluster_maps')
    images = glob.glob(os.path.join(
        cluster_img_dir, '*'
    ))
    images.sort()
    labels = [os.path.basename(i).split('_mean')[0] for i in images]

    mask_img = os.path.join(
        os.environ['FSLDIR'],
        'data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
    )
    masker = nilearn.input_data.NiftiMasker(mask_img=mask_img)
    data = masker.fit_transform(images)

    # compute correlation between all maps
    cc = numpy.corrcoef(data)
    cc_df = pandas.DataFrame(cc, index=labels, columns=labels)

    # make heatmap
    plt.figure(figsize=(18, 14))
    seaborn.heatmap(cc_df, annot=True)
    plt.tight_layout()
    plt.savefig(os.path.join(basedir, 'figures/cluster_correlation.pdf'))

    # load original Tom et al. maps and compare to each cluster map
    tom_files = glob.glob(os.path.join(
        basedir,
        'orig/TomEtAl/zstat*'))
    tom_files.sort()
    tom_data = masker.fit_transform(tom_files)
    tom_corr = numpy.zeros((data.shape[0], tom_data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(tom_data.shape[0]):
            tom_corr[i, j] = numpy.corrcoef(data[i, :], tom_data[j, :])[0, 1]

    tom_corr_df = pandas.DataFrame(tom_corr,
                                   columns=['Task', 'Gain', 'Loss'],
                                   index=labels)
    plt.figure(figsize=(6, 12))
    seaborn.heatmap(tom_corr_df, annot=True)
    plt.xlabel('Tom et al. contrasts')
    plt.ylabel('NARPS contrasts')
    plt.tight_layout()
    plt.savefig(os.path.join(basedir, 'figures/tom_correlation.pdf'))

    # compare consensus results to Tom et al.
    consensus_files = glob.glob(os.path.join(
        basedir,
        'output/consensus_analysis/hypo[1,5]_t.nii.gz'))
    consensus_files.sort()
    consensus_data = masker.fit_transform(consensus_files)

    consensus_tom_corr = numpy.zeros((
        consensus_data.shape[0], tom_data.shape[0]))
    for i in range(consensus_data.shape[0]):
        for j in range(tom_data.shape[0]):
            consensus_tom_corr[i, j] = numpy.corrcoef(
                consensus_data[i, :], tom_data[j, :])[0, 1]


if __name__ == "__main__":

    basedir = os.environ['NARPS_BASEDIR']
    cluster_image_correlation(basedir)
