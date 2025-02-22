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


def cluster_image_correlation(basedir,
                              targetdir,
                              zstat_desc):
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

    # load target maps and compare to each cluster map
    target_files = glob.glob(os.path.join(
        basedir, 'orig/%s/zstat*' % targetdir))
    target_files.sort()
    target_data = masker.fit_transform(target_files)
    target_corr = numpy.zeros((data.shape[0], target_data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(target_data.shape[0]):
            target_corr[i, j] = numpy.corrcoef(
                data[i, :], target_data[j, :])[0, 1]

    target_corr_df = pandas.DataFrame(target_corr,
                                      columns=zstat_desc,
                                      index=labels)
    plt.figure(figsize=(6, 12))
    seaborn.heatmap(target_corr_df, annot=True)
    plt.xlabel('%s contrasts' % targetdir)
    plt.ylabel('NARPS contrasts')
    plt.tight_layout()
    plt.savefig(os.path.join(
        basedir,
        'figures/%s_correlation.pdf' % targetdir))

    target_corr_df.to_csv(os.path.join(
        basedir,
        'metadata/cluster_corr_%s.csv' % targetdir
    ))


if __name__ == "__main__":

    basedir = os.environ['NARPS_BASEDIR']
    cluster_image_correlation(
        basedir,
        'TomEtAl',
        ['Task', 'Gain', 'Loss'])
    cluster_image_correlation(
        basedir,
        'NARPS_mean',
        ['Task'])
