#!/usr/bin/env python
# coding: utf-8
"""
generate individual reports for data prepared using narps.py
"""

import os
import argparse
import glob
import warnings
import matplotlib.pyplot as plt
import numpy
import nilearn.input_data
import pandas
import nibabel

from narps import Narps
from utils import get_masked_data, log_to_file

cut_coords = [-24, -10, 4, 18, 32, 52, 64]
bins = numpy.linspace(-5, 5)

hypnums = [i for i in range(1, 10)]


def create_team_reports(narps, logfile):
    diagnostic_results = None
    origdir = narps.dirs.dirs['orig']
    teamdirs = glob.glob(os.path.join(
        origdir, '*'))
    collectionIDs = [
        os.path.basename(i) for i in teamdirs
        if os.path.isdir(i)]
    for collectionID in collectionIDs:
        diagdata = create_team_report(
            narps,
            collectionID,
            logfile)
        print(diagdata)
        if diagnostic_results is None:
            diagnostic_results = diagdata
        else:
            diagnostic_results = pandas.concat(
                [diagnostic_results, diagdata]
            )

    return(diagnostic_results)


def create_team_report(narps, collectionID, logfile,
                       unthresh_dataset='zstat',
                       thresh_dataset='resampled',
                       verbose=True,
                       mean_thresh=1,
                       std_thresh=2,
                       voxelmap_thresh=0.9):
    diagnostic_data = pandas.DataFrame({
        'collectionID': collectionID,
        'hyp': hypnums,
        'meanZ': numpy.nan,
        'stdZ': numpy.nan,
        'p_map_voxels': numpy.nan})

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

    # get all maps
    unthresh_maps = glob.glob(os.path.join(
        teamdir_unthresh, 'hypo*_unthresh.nii.gz'))
    unthresh_maps.sort()
    nmaps = len(unthresh_maps)
    if not nmaps == 9:
        print('found %d maps for %s' % (nmaps, collectionID))
        print('WARNING: hypothesis numbers may not be correct!')

    # load data and check them for deviations from Z distribution
    check_values = True
    if check_values:
        masker = nilearn.input_data.NiftiMasker(
            mask_img=narps.dirs.MNI_mask)

        unthresh_data = masker.fit_transform(
            unthresh_maps)

        for hyp in hypnums:
            # check zstat to make sure it looks like a zstat
            hypdata_unthresh = unthresh_data[hyp-1, :]
            hypdata_mean = numpy.mean(hypdata_unthresh)
            hypdata_std = numpy.std(hypdata_unthresh)
            diagnostic_data.loc[
                diagnostic_data.hyp == hyp,
                'meanZ'] = hypdata_mean
            diagnostic_data.loc[
                diagnostic_data.hyp == hyp,
                'stdZ'] = hypdata_std

            if numpy.abs(hypdata_mean) > mean_thresh or \
                    hypdata_std > std_thresh or \
                    hypdata_std < 1./std_thresh:
                log_to_file(
                    logfile,
                    'FLAG: %s %d %f %f' % (
                        collectionID, hyp, hypdata_mean, hypdata_std))

    # create overlays for all hypotheses
    # first for thresh
    make_thresh_overlays = False
    if make_thresh_overlays:
        fig, ax = plt.subplots(
            len(hypnums), 1,
            figsize=(8, 11))
        ctr = 0
        for hyp in hypnums:
            threshfile = os.path.join(
                teamdir_thresh, 'hypo%d_thresh.nii.gz' % hyp)
            threshdata = nibabel.load(threshfile).get_data()
            diagnostic_data.loc[
                diagnostic_data.hyp == hyp,
                'n_over_thresh'] = numpy.sum(threshdata > 1e-6)
            if not os.path.exists(threshfile):
                print('missing file for ', collectionID, hyp)
                continue
            _ = nilearn.plotting.plot_stat_map(
                threshfile,
                display_mode="z",
                cmap='Reds',
                colorbar=False,
                title='%s: hyp %d' % (collectionID, hyp),
                cut_coords=cut_coords,
                axes=ax[ctr])
            ctr += 1
        outfile = os.path.join(
            narps.dirs.dirs['diagnostics'],
            '%s_thresh_overlay.pdf' % collectionID)
        if verbose:
            print('saving', outfile)
        plt.savefig(outfile)
        plt.close()

    # then for unthresh (looking for nonzero)
    fig, ax = plt.subplots(
        len(hypnums), 1,
        figsize=(8, 11))
    ctr = 0
    for hyp in hypnums:
        unthreshfile = os.path.join(
            teamdir_thresh, 'hypo%d_unthresh.nii.gz' % hyp)
        if not os.path.exists(unthreshfile):
            print('missing file for ', collectionID, hyp)
            continue
        voxelmap = nibabel.load(
            os.path.join(
                narps.dirs.dirs['output'],
                'unthresh_concat_%s/hypo%d_voxelmap.nii.gz' %
                (unthresh_dataset, hyp))
        )
        voxelmap_good = voxelmap.get_data() > voxelmap_thresh
        unthreshimg = nibabel.load(unthreshfile)
        unthreshdata = numpy.abs(unthreshimg.get_data())
        maskimg = nibabel.Nifti1Image((unthreshdata > 1e-6).astype('int'),
                                      affine=unthreshimg.affine)
        diagnostic_data.loc[
            diagnostic_data.hyp == hyp,
            'p_map_voxels'] = numpy.mean(maskimg.get_data()[voxelmap_good])

        nilearn.plotting.plot_stat_map(
            maskimg,
            display_mode="z",
            cmap='jet',
            threshold=1e-6,
            colorbar=False,
            title='%s: hyp %d' % (collectionID, hyp),
            cut_coords=cut_coords,
            axes=ax[ctr])
        ctr += 1
    outfile = os.path.join(
        narps.dirs.dirs['diagnostics'],
        '%s_unthresh_overlay.pdf' % collectionID)
    if verbose:
        print('saving', outfile)
    plt.savefig(outfile)
    plt.close()
    return(diagnostic_data)


def create_map_overlays(narps, overwrite=True):
    """
    Make report showing all orig maps with threshold overlays
    This report includes all maps for which data were available,
    including those that were excluded
    """
    figdir = os.path.join(
        narps.dirs.dirs['figures'],
        'team_diagnostics')

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
                    narps.dirs.dirs['orig'],
                    '%s/hypo%d_unthresh.nii.gz' % (collection_id, hyp))
                thresh_img = os.path.join(
                    narps.dirs.dirs['thresh_mask_orig'],
                    '%s/hypo%d_thresh.nii.gz' % (
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


def create_unthresh_histograms(narps, overwrite=True):
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
                hyp, narps.dirs.MNI_mask, narps.dirs.dirs['output'],
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
            plt.close(fig)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Make diagnostic reports for each team')
    parser.add_argument('-b', '--basedir',
                        help='base directory')
    parser.add_argument('-t', '--test',
                        action='store_true',
                        help='use testing mode (no processing)')
    args = parser.parse_args()

    # set up base directory
    if args.basedir is not None:
        basedir = args.basedir
    elif 'NARPS_BASEDIR' in os.environ:
        basedir = os.environ['NARPS_BASEDIR']
        print("using basedir specified in NARPS_BASEDIR")
    else:
        basedir = '/data'
        print("using default basedir:", basedir)

    narps = Narps(basedir)
    narps.dirs.get_output_dir('diagnostics',
                              base='figures')

    logfile = os.path.join(
        narps.dirs.dirs['logs'],
        'diagnostics.log')

    if not args.test:

        log_to_file(
            logfile,
            'Running diagnostics',
            flush=True)
        diagnostic_results = create_team_reports(narps, logfile)
        diagnostic_results.to_csv(os.path.join(
            narps.dirs.dirs['metadata'],
            'image_diagnostics.csv'
        ))
