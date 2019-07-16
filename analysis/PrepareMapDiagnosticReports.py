#!/usr/bin/env python
# coding: utf-8
"""
generate individual reports for data prepared using narps.py
"""

import os
import glob
import warnings
import matplotlib.pyplot as plt
import numpy
import nilearn.input_data

from narps import Narps
from utils import get_masked_data

cut_coords = [-24, -10, 4, 18, 32, 52, 64]
bins = numpy.linspace(-5, 5)

# instantiate main Narps class, which loads data
# to rerun everything, set overwrite to True
if 'NARPS_BASEDIR' in os.environ:
    basedir = os.environ['NARPS_BASEDIR']
else:
    basedir = '/data'

narps = Narps(basedir)
# narps.load_data()

output_dir = narps.dirs.dirs['output']
mask_img = narps.dirs.MNI_mask


def create_map_overlays(narps, overwrite=True,
                        hypnums=[1, 2, 5, 6, 7, 8, 9]):
    """
    Make report showing all orig maps with threshold overlays
    This report includes all maps for which data were available,
    including those that were excluded
    """
    figdir = os.path.join(narps.dirs.dirs['figures'], 'orig_map_overlays')

    if not os.path.exists(figdir):
        os.mkdir(figdir)

    for hyp in hypnums:
        outfile = os.path.join(
            figdir,
            'hyp%d_orig_map_overlays.pdf' % hyp)

        if not os.path.exists(outfile) or overwrite:
            print('making map overlay figure for hyp', hyp)

            # find all maps
            hmaps = glob.glob(os.path.join(
                narps.dirs.dirs['orig'],
                '*_*'))

            collection_ids = [os.path.basename(i) for i in hmaps]
            collection_ids.sort()

            fig, ax = plt.subplots(
                len(collection_ids), 2,
                figsize=(len(collection_ids), 140),
                gridspec_kw={'width_ratios': [2, 1]})
            ctr = 0

            for collection_id in collection_ids:
                teamID = collection_id.split('_')[1]
                unthresh_img = os.path.join(
                    output_dir,
                    'orig/%s/hypo%d_unthresh.nii.gz' % (collection_id, hyp))
                thresh_img = os.path.join(
                    output_dir,
                    'thresh_mask_orig/%s/hypo%d_thresh.nii.gz' % (
                        collection_id, hyp))

                if not (os.path.exists(thresh_img) or
                        os.path.exists(unthresh_img)):
                    print('skipping', teamID)
                    continue

                if teamID not in narps.complete_image_sets:
                    imagetitle = '%s (excluded)' % teamID
                else:
                    imagetitle = teamID

                display = nilearn.plotting.plot_stat_map(
                    unthresh_img,
                    display_mode="z",
                    colorbar=True, title=imagetitle,
                    cut_coords=cut_coords,
                    axes=ax[ctr, 0], cmap='gray')

                # ignore levels warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    display.add_contours(
                        thresh_img, filled=False,
                        alpha=0.7, levels=[0.5],
                        colors='b')

                masker = nilearn.input_data.NiftiMasker(mask_img=thresh_img)
                maskdata = masker.fit_transform(unthresh_img)
                if numpy.sum(maskdata) > 0:  # check for empty mask
                    _ = ax[ctr, 1].hist(maskdata, bins=bins)
                ctr += 1
            plt.savefig(outfile)
            plt.close(fig)


def create_unthresh_histograms(narps, overwrite=True,
                               hypnums=[1, 2, 5, 6, 7, 8, 9]):
    """
`   Create histograms for in-mask values in unthresholded images
    These are only created for the images that were successfully
    registered and rectified.
    """
    figdir = os.path.join(
        narps.dirs.dirs['figures'],
        'unthresh_histograms')

    if not os.path.exists(figdir):
        os.mkdir(figdir)

    for hyp in hypnums:
        outfile = os.path.join(
            figdir,
            'hyp%d_unthresh_histogram.pdf' % hyp)

        if not os.path.exists(outfile) or overwrite:
            print('making figure for hyp', hyp)
            unthresh_data, labels = get_masked_data(
                hyp, mask_img, output_dir,
                imgtype='unthresh', dataset='rectified')

            fig, ax = plt.subplots(
                int(numpy.ceil(len(labels)/3)), 3,
                figsize=(16, 50))

            # make three columns - these are row and column counters
            ctr_x = 0
            ctr_y = 0

            for i, l in enumerate(labels):
                ax[ctr_x, ctr_y].hist(unthresh_data[i, :], 100)
                ax[ctr_x, ctr_y].set_title(l)
                ctr_y += 1
                if ctr_y > 2:
                    ctr_y = 0
                    ctr_x += 1
            plt.tight_layout()
            plt.savefig(outfile)
