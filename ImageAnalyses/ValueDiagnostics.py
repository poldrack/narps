#!/usr/bin/env python
# coding: utf-8
"""
generate individual reports for data prepared using narps.py
"""

import os
import matplotlib.pyplot as plt
import numpy
import nilearn.input_data
import pandas
import nibabel

from narps import Narps, hypnums # noqa
from utils import log_to_file


def compare_thresh_unthresh_values(
        narps, collectionID, logfile,
        unthresh_dataset='zstat',
        thresh_dataset='resampled',
        verbose=True,
        error_thresh=.05,
        create_histogram=False):
    """examine unthresh values within thresholded map voxels
    to check direction of maps
    if more than error_thresh percent of voxels are
    in opposite direction, then flag a problem
    - we allow a few to bleed over due to interpolation"""

    diagnostic_data = pandas.DataFrame({
        'collectionID': collectionID,
        'hyp': hypnums,
        'rectify': False,
        'problem': False,
        'n_thresh_vox': numpy.nan,
        'min_unthresh': numpy.nan,
        'max_unthresh': numpy.nan,
        'p_pos_unthresh': numpy.nan,
        'p_neg_unthresh': numpy.nan})

    teamdir_unthresh = os.path.join(
        narps.dirs.dirs[unthresh_dataset],
        collectionID
    )
    teamdir_thresh = os.path.join(
        narps.dirs.dirs[thresh_dataset],
        collectionID
    )

    if not os.path.exists(teamdir_unthresh):
        print('no %s for %s' % (unthresh_dataset, collectionID))
        return(None)
    if not os.path.exists(teamdir_thresh):
        print('no %s for %s' % (thresh_dataset, collectionID))
        return(None)

    masker = nilearn.input_data.NiftiMasker(
        mask_img=narps.dirs.MNI_mask)

    for hyp in hypnums:
        threshfile = os.path.join(
            teamdir_thresh, 'hypo%d_thresh.nii.gz' % hyp)
        if not os.path.exists(threshfile):
            print('no thresh hyp %d for %s' % (hyp, collectionID))
            continue
        threshdata = masker.fit_transform(threshfile)
        n_thresh_vox = numpy.sum(threshdata > 0)
        diagnostic_data.loc[
            diagnostic_data.hyp == hyp,
            'n_thresh_vox'] = n_thresh_vox

        if n_thresh_vox == 0:
            log_to_file(
                logfile,
                'WARN: %s %d - empty mask' % (
                    collectionID, hyp
                ))
            continue

        unthreshfile = os.path.join(
            teamdir_thresh, 'hypo%d_unthresh.nii.gz' % hyp)
        if not os.path.exists(unthreshfile):
            print('no unthresh hyp %d for %s' % (hyp, collectionID))
            continue
        unthreshdata = masker.fit_transform(unthreshfile)
        inmask_unthreshdata = unthreshdata[threshdata > 0]
        min_val = numpy.min(inmask_unthreshdata)
        max_val = numpy.max(inmask_unthreshdata)
        if max_val < 0:  # need to rectify
            diagnostic_data.loc[
                diagnostic_data.hyp == hyp,
                'rectify'] = True
        diagnostic_data.loc[
            diagnostic_data.hyp == hyp,
            'min_unthresh'] = min_val
        diagnostic_data.loc[
            diagnostic_data.hyp == hyp,
            'max_unthresh'] = max_val
        p_pos_unthresh = numpy.mean(inmask_unthreshdata > 0)
        p_neg_unthresh = numpy.mean(inmask_unthreshdata < 0)
        diagnostic_data.loc[
            diagnostic_data.hyp == hyp,
            'p_pos_unthresh'] = p_pos_unthresh
        diagnostic_data.loc[
            diagnostic_data.hyp == hyp,
            'p_neg_unthresh'] = p_neg_unthresh
        min_p_direction = numpy.min([p_pos_unthresh, p_neg_unthresh])
        if min_p_direction > error_thresh:
            log_to_file(
                logfile,
                'WARN: %s hyp%d invalid in-mask values (%f, %f)' % (
                    collectionID, hyp, p_neg_unthresh, p_pos_unthresh
                ))
            diagnostic_data.loc[
                diagnostic_data.hyp == hyp,
                'problem'] = True
            if create_histogram:
                # also load their orig thresh map and create a histogram
                orig_threshfile = os.path.join(
                    narps.dirs.dirs['orig'],
                    collectionID,
                    'hypo%d_thresh.nii.gz' % hyp)
                threshdata = nibabel.load(orig_threshfile).get_data()
                threshdata = threshdata[numpy.abs(threshdata) > 1e-6]
                plt.hist(threshdata, bins=50)
                plt.savefig(
                    os.path.join(
                        narps.dirs.dirs['diagnostics'],
                        'thresh_hist_%s_%d.pdf' % (
                            collectionID, hyp
                        )
                    )
                )
                plt.close()

    return(diagnostic_data)
