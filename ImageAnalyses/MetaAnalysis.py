#!/usr/bin/env python
# coding: utf-8
"""
Run ALE meta-analysis based on code from JB Poline
"""


import argparse
import pickle
import pandas
import nibabel
import nimare
import os
import nilearn.image
import nilearn.input_data
import nilearn.plotting
import nilearn.masking
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

from narps import Narps, hypnums, hypotheses
from narps import NarpsDirs # noqa, flake8 issue
from ALEextract import get_activations, get_sub_dict


# use binarized/thresholded maps and z maps to create
# thresholded z maps
def get_thresholded_Z_maps(narps, verbose=False, overwrite=False):
    for teamID in narps.complete_image_sets['thresh']:
        if 'zstat' not in narps.teams[teamID].images['thresh']:
            narps.teams[teamID].images['thresh']['zstat'] = {}

        for hyp in range(1, 10):
            if hyp not in narps.teams[teamID].images['unthresh']['zstat']:
                # fill missing data with nan
                if verbose:
                    print('no unthresh zstat present for', teamID, hyp)
                continue
            if hyp not in narps.teams[teamID].images['thresh']['resampled']:
                # fill missing data with nan
                if verbose:
                    print('no thresh resampled present for', teamID, hyp)
                continue
            zfile = narps.teams[teamID].images['unthresh']['zstat'][hyp]
            if not os.path.exists(zfile):
                if verbose:
                    print('no image present:', zfile)
                continue
            threshfile = narps.teams[teamID].images['thresh']['resampled'][hyp]
            if not os.path.exists(threshfile):
                if verbose:
                    print('no image present:', threshfile)
                continue
            narps.teams[teamID].images['thresh']['zstat'][hyp] = threshfile

            outfile = zfile.replace('unthresh', 'thresh')
            assert outfile != zfile

            if not os.path.exists(outfile) or overwrite:
                if verbose:
                    print('creating thresholded Z for', teamID, hyp)
                unthresh_img = nibabel.load(zfile)
                unthresh_data = unthresh_img.get_data()

                thresh_img = nibabel.load(threshfile)
                thresh_data = thresh_img.get_data()

                thresh_z_data = thresh_data * unthresh_data

                thresh_z_img = nibabel.Nifti1Image(
                    thresh_z_data,
                    affine=thresh_img.affine)
                thresh_z_img.to_filename(outfile)
            else:
                if verbose:
                    print('using existing thresholded Z for', teamID, hyp)

    return(narps)


def extract_peak_coordinates(narps, hyp,
                             overwrite=False, threshold=0.,
                             verbose=False):
    '''
    def extract(dir_path, filename, threshold=0., load=True, save_dir=None):
    # threshold_type='percentile',
        Extracts coordinates from the data and put it in a
        dictionnary using Nimare structure.

        Inputs
        -----------------
        dir_path : str
            Path to the folder containing the studies folders
        filename : str
            Name of the image to look for inside each study folder
        threshold : float
            Threshold (in z scale) zero values below threshold in input images
            # used to be: between 0-1 percentile
        load : bool
                    If True, try to load a previously dumped dict if any.
                    If dump fails or False, compute a new one.

        Returns:
        -----------------
        (dict): Dictionnary storing the coordinates using the Nimare
                structure.
    '''

    gray_mask = nibabel.load(narps.dirs.MNI_mask)  # no masking

    dictfile = os.path.join(
        narps.dirs.dirs['ALE'],
        'hyp%d_peak_coords.pkl' % hyp
    )
    # Loading previously computed dict if any
    if os.path.exists(dictfile) and not overwrite:
        with open(dictfile, 'rb') as f:
            ds_dict = pickle.load(f)
        return(ds_dict)

    res = []

    # get activations for all teams with a thresh zstat

    for teamID in narps.complete_image_sets['thresh']:
        if 'zstat' not in narps.teams[teamID].images['thresh']:
            continue
        if hyp not in narps.teams[teamID].images['thresh']['zstat']:
            continue
        thresh_zfile = narps.teams[teamID].images['thresh']['zstat'][hyp]
        if verbose:
            print('get_activations:', thresh_zfile, threshold)
        XYZ = get_activations(thresh_zfile, gray_mask, threshold)
        if XYZ is not None:
            res.append(get_sub_dict(XYZ[0], XYZ[1], XYZ[2]))
        else:
            res.append({})

    # Removing potential None values
    res = list(filter(None, res))

    # Merging all dictionaries
    ds_dict = {k: v for k, v in enumerate(res)}

    # Dumping
    with open(dictfile, 'wb') as f:
        pickle.dump(ds_dict, f)

    return(ds_dict)


def run_ALE(ds_dict, hyp, overwrite=False):
    outfile = os.path.join(
            narps.dirs.dirs['ALE'],
            'ALE_results_hyp%d.pkl' % hyp
        )
    if os.path.exists(outfile) and not overwrite:
        print('using saved meta-analysis results')
        with open(outfile, 'rb') as f:
            res = pickle.load(f)
        return(res)
    ds = nimare.dataset.Dataset(ds_dict)
    ALE = nimare.meta.cbma.ale.ALE()
    res = ALE.fit(ds)
    with open(outfile, 'wb') as f:
        pickle.dump(res, f)
    return(res)


def save_results(hyp, res, narps, fdr_thresh=0.05):

    # saving results images
    images = {}
    for i in ['ale', 'p', 'z']:
        images[i] = res.get_map(i)
        save_file_img = os.path.join(
            narps.dirs.dirs['ALE'],
            'img_%s_hyp%d.nii.gz' % (i, hyp))
        nibabel.save(images[i], save_file_img)

    # -------------------------- threshold with fdr --------------- #
    masker = nilearn.input_data.NiftiMasker(
        mask_img=narps.dirs.MNI_mask
    )
    pvals = masker.fit_transform(images['p'])
    fdr_results = multipletests(pvals[0, :], 0.05, 'fdr_tsbh')
    images['fdr_oneminusp'] = masker.inverse_transform(1 - fdr_results[1])
    images['fdr_oneminusp'].to_filename(os.path.join(
            narps.dirs.dirs['ALE'],
            'hypo%d_fdr_oneminusp.nii.gz' % hyp))
    images['fdr_thresholded'] = masker.inverse_transform(
        (fdr_results[1] < fdr_thresh).astype('int'))
    images['fdr_thresholded'].to_filename(os.path.join(
            narps.dirs.dirs['ALE'],
            'hypo%d_fdr_thresholded.nii.gz' % hyp))

    return(images)


def make_figures(narps, hyp, images, fdr_thresh=0.05):
    cut_coords = [-24, -10, 4, 18, 32, 52, 64]
    for i in ['ale', 'p', 'z', 'fdr_oneminusp']:
        nilearn.plotting.plot_stat_map(images[i], title=i)
        outfile = os.path.join(
            narps.dirs.dirs['figures'],
            'hyp%d_ALE_%s.png' % (hyp, i)
        )
        plt.savefig(outfile)
        plt.close()

    nilearn.plotting.plot_stat_map(
        images['fdr_oneminusp'],
        threshold=1 - fdr_thresh,
        title='H%d meta-analysis (FDR < %0.2f' % (hyp, fdr_thresh),
        display_mode="z",
        colorbar=True,
        cmap='jet',
        cut_coords=cut_coords,
        annotate=False)

    outfile = os.path.join(
        narps.dirs.dirs['figures'],
        'hyp%d_ALE_fdr_thresh_%0.2f.png' % (hyp, fdr_thresh)
    )
    plt.savefig(outfile)
    plt.close()
    return(None)


def make_combined_figure(narps, thresh=0.95):

    fig, ax = plt.subplots(7, 1, figsize=(12, 24))
    cut_coords = [-24, -10, 4, 18, 32, 52, 64]

    for i, hyp in enumerate(hypnums):
        pmap = os.path.join(
            narps.dirs.dirs['ALE'],
            'hypo%d_fdr_thresholded.nii.gz' % hyp)
        tmap = os.path.join(
            narps.dirs.dirs['ALE'],
            'img_z_hyp%d.nii.gz' % hyp)
        pimg = nibabel.load(pmap)
        timg = nibabel.load(tmap)
        pdata = pimg.get_fdata()
        tdata = timg.get_fdata()
        threshdata = (pdata > thresh)*tdata
        threshimg = nibabel.Nifti1Image(threshdata, affine=timg.affine)
        nilearn.plotting.plot_stat_map(
            threshimg,
            threshold=0.1,
            display_mode="z",
            colorbar=True,
            title='hyp %d:' % hyp+hypotheses[hyp],
            vmax=8,
            cmap='jet',
            cut_coords=cut_coords,
            axes=ax[i])

    plt.savefig(os.path.join(
        narps.dirs.dirs['figures'],
        'ALE_map.pdf'))
    plt.close(fig)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Run ALE meta-analysis')
    parser.add_argument('-b', '--basedir',
                        help='base directory')
    parser.add_argument('-t', '--test',
                        action='store_true',
                        help='use testing mode (no processing)')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='verbose mode')
    parser.add_argument('-o', '--overwrite',
                        action='store_true',
                        help='overwrite existing results')
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

    # setup main class
    narps = Narps(basedir)
    narps.load_data()
    output_dir = narps.dirs.get_output_dir('ALE')

    # Load full metadata and put into narps structure
    narps.metadata = pandas.read_csv(
        os.path.join(narps.dirs.dirs['metadata'], 'all_metadata.csv'))

    if not args.test:
        # create thresholded versions of Z maps
        narps = get_thresholded_Z_maps(
            narps,
            verbose=args.verbose,
            overwrite=args.overwrite)

        # extract peak coordinates
        for hyp in range(1, 10):
            print('Hypothesis', hyp)
            ds_dict = extract_peak_coordinates(
                narps,
                hyp,
                overwrite=args.overwrite,
                verbose=args.verbose)

            # Performing ALE
            res = run_ALE(ds_dict, hyp, overwrite=args.overwrite)

            images = save_results(hyp, res, narps)

            make_figures(narps, hyp, images)

        # make a figure with all hypotheses
        make_combined_figure(narps)
