#!/usr/bin/env python
# coding: utf-8
"""
compute image diagnostics for rectification
"""

import os
import matplotlib.pyplot as plt
import numpy
import pandas
import nibabel

from utils import log_to_file, get_map_metadata


def compare_thresh_unthresh_values(
        dirs, collectionID, logfile,
        unthresh_dataset='zstat',
        thresh_dataset='zstat',
        verbose=True,
        error_thresh=.05,
        create_histogram=False,
        map_metadata_file=None):
    """examine unthresh values within thresholded map voxels
    to check direction of maps
    if more than error_thresh percent of voxels are
    in opposite direction, then flag a problem
    - we allow a few to bleed over due to interpolation"""
    verbose = True
    hyps = [i for i in range(1, 10)]
    teamID = collectionID.split('_')[1]
    diagnostic_data = pandas.DataFrame({
        'collectionID': collectionID,
        'teamID': teamID,
        'hyp': hyps,
        'autorectify': False,
        'problem': numpy.nan,
        'reverse_contrast': False,
        'n_thresh_vox': numpy.nan,
        'min_unthresh': numpy.nan,
        'max_unthresh': numpy.nan,
        'p_pos_unthresh': numpy.nan,
        'p_neg_unthresh': numpy.nan})

    teamdir_unthresh = os.path.join(
        dirs.dirs[unthresh_dataset],
        collectionID
    )
    teamdir_thresh = os.path.join(
        dirs.dirs[thresh_dataset],
        collectionID
    )
    print('using %s for thresh and %s for unthresh' %
          (thresh_dataset, unthresh_dataset))
    if not os.path.exists(teamdir_unthresh):
        print('no unthresh %s for %s' % (unthresh_dataset, collectionID))
        print(teamdir_unthresh)
        return(None)
    if not os.path.exists(teamdir_thresh):
        print('no thresh %s for %s' % (thresh_dataset, collectionID))
        print(teamdir_unthresh)
        return(None)

    for hyp in hyps:
        autorectify = False
        threshfile = os.path.join(
            teamdir_thresh, 'hypo%d_thresh.nii.gz' % hyp)
        if not os.path.exists(threshfile):
            print('no thresh hyp %d for %s' % (hyp, collectionID))
            continue
        threshdata = nibabel.load(threshfile).get_data().flatten()
        threshdata = numpy.nan_to_num(threshdata)
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

        unthreshfile = os.path.join(
            teamdir_thresh, 'hypo%d_unthresh.nii.gz' % hyp)
        if not os.path.exists(unthreshfile):
            print('no unthresh hyp %d for %s' % (hyp, collectionID))
            continue
        unthreshdata = nibabel.load(unthreshfile).get_data().flatten()
        unthreshdata = numpy.nan_to_num(unthreshdata)
        if not unthreshdata.shape == threshdata.shape:
            log_to_file(
                logfile,
                'ERROR: thresh/unthresh size mismatch for %s hyp%d' %
                (collectionID, hyp))
            continue
        if numpy.sum(threshdata > 0) > 0:
            inmask_unthreshdata = unthreshdata[threshdata > 0]
            min_val = numpy.min(inmask_unthreshdata)
            max_val = numpy.max(inmask_unthreshdata)
            p_pos_unthresh = numpy.mean(inmask_unthreshdata > 0)
            p_neg_unthresh = numpy.mean(inmask_unthreshdata < 0)
        else:
            min_val = 0
            max_val = 0
            p_pos_unthresh = 0
            p_neg_unthresh = 0

        if max_val < 0:  # need to rectify
            autorectify = True
            if verbose:
                print('autorectify:', teamID, hyp)
            diagnostic_data.loc[
                diagnostic_data.hyp == hyp,
                'autorectify'] = True
        diagnostic_data.loc[
            diagnostic_data.hyp == hyp,
            'min_unthresh'] = min_val
        diagnostic_data.loc[
            diagnostic_data.hyp == hyp,
            'max_unthresh'] = max_val
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
                    dirs.dirs['orig'],
                    collectionID,
                    'hypo%d_thresh.nii.gz' % hyp)
                threshdata = nibabel.load(orig_threshfile).get_data()
                threshdata = threshdata[numpy.abs(threshdata) > 1e-6]
                plt.hist(threshdata, bins=50)
                plt.savefig(
                    os.path.join(
                        dirs.dirs['diagnostics'],
                        'thresh_hist_%s_%d.pdf' % (
                            collectionID, hyp
                        )
                    )
                )
                plt.close()

        # also get info from metadata file about direction
        # of contrasts
        if map_metadata_file is None:
            map_metadata_file = os.path.join(
                dirs.dirs['orig'],
                'narps_neurovault_images_details_responses_corrected.csv')
        map_metadata = get_map_metadata(map_metadata_file)

        reverse_contrast = False

        if hyp in [5, 6]:
            mdstring = map_metadata.query(
                'teamID == "%s"' % teamID
                )['hyp%d_direction' % hyp].iloc[0]
            reverse_contrast = mdstring.split()[0] == 'Negative'
            if verbose:
                print('manual rectify:', teamID, hyp)
        elif hyp == 9 and teamID in ['R7D1']:
            # manual fix for one team with reversed maps
            reverse_contrast = True
        diagnostic_data.loc[
            diagnostic_data.hyp == hyp,
            'reverse_contrast'] = reverse_contrast
        if reverse_contrast != autorectify:
            log_to_file(
                logfile,
                'WARN: %s %d rectification mismatch' %
                (collectionID, hyp))

    return(diagnostic_data)
