#!/usr/bin/env python
# coding: utf-8
"""
functions to create simulated data
"""

import os
import shutil
import glob

import numpy
import nibabel
from utils import log_to_file, randn_from_shape


# functions for simulated data
def setup_simulated_data(
        narps,
        verbose=False,
        overwrite=False):
    """create directories for simulated data"""

    # consensus analysis must exist
    assert os.path.exists(narps.dirs.dirs['consensus'])

    basedir = narps.basedir + '_simulated'
    if verbose:
        print("writing files to new directory:", basedir)
    if not os.path.exists(os.path.join(basedir, 'logs')):
        os.makedirs(os.path.join(basedir, 'logs'))

    log_to_file(os.path.join(basedir, 'logs/simulated_data.log'),
                'Creating simulated dataset', flush=True)
    # copy data from orig/templates
    origdir = narps.dirs.dirs['orig']
    new_origdir = os.path.join(basedir, 'orig')
    templatedir = narps.dirs.dirs['templates']
    if verbose:
        print('using basedir:', basedir)
    if os.path.exists(basedir) and overwrite:
        shutil.rmtree(basedir)
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    if not os.path.exists(new_origdir) or overwrite:
        if verbose:
            print('copying template data to new basedir')
        shutil.copytree(
            templatedir,
            os.path.join(basedir, 'templates'))
        if not os.path.exists(new_origdir):
            os.mkdir(new_origdir)
        # copy metadata files from orig
        for f in glob.glob(os.path.join(
                origdir, '*.*')):
            if os.path.isfile(f):
                if verbose:
                    print('copying', f, 'to', new_origdir)
                shutil.copy(f, new_origdir)
    else:
        print('using existing new basedir')

    return(basedir)


def make_orig_images(basedir,
                     teamCollectionID,
                     consensus_dir,
                     noise_sd=.5,
                     thresh=2.,
                     noise_team=None,
                     flip_team=None,
                     rectify_status=None,
                     verbose=True):
    """
    for a particular team and hypothesis,
    generate the images
    NOTE: flip_teams are teams whose data will be
    flipped but will not be rectified
    - included to simulate flipping errors
    We also flip those teams/hyps that will be rectified
    """
    if rectify_status is None:
        rectify_status = {i: False for i in range(1, 10)}
    if verbose:
        print(teamCollectionID, rectify_status)
    if noise_team is None:
        noise_team = False
    if flip_team is None:
        flip_team = False

    if verbose:
        print('make_orig_images:', teamCollectionID)

    teamdir = os.path.join(basedir, 'orig/%s' % teamCollectionID)
    if not os.path.exists(teamdir):
        os.mkdir(teamdir)
    if verbose:
        print('teamdir', teamdir)

    for hyp in range(1, 10):
        # deal with missing hypotheses in consensus
        if hyp in [3, 4]:
            hyp_orig = hyp - 2
        else:
            hyp_orig = hyp

        outfile = {'thresh': os.path.join(
            teamdir, 'hypo%d_thresh.nii.gz' % hyp),
            'unthresh': os.path.join(
            teamdir, 'hypo%d_unthresh.nii.gz' % hyp)}

        # get t image from consensus map
        baseimgfile = os.path.join(
            consensus_dir,
            'hypo%d_t.nii.gz' % hyp_orig)
        if verbose:
            print("baseimg:", baseimgfile)
        assert os.path.exists(baseimgfile)

        # load template data
        baseimg = nibabel.load(baseimgfile)
        baseimgdata = baseimg.get_data()
        newimgdata = baseimgdata.copy()

        if noise_team:
            # threshold to get in-mask voxels
            baseimgvox = numpy.abs(baseimgdata) > 0
            # fill in-mask voxels with N(0,std) Gaussian noise
            # where std is based on the original image
            newimgdata = randn_from_shape(
                baseimgdata.shape)*numpy.std(baseimgdata)
            # clear out-of-mask voxels
            newimgdata = newimgdata * baseimgvox
        else:
            newimgdata = newimgdata + randn_from_shape(
                newimgdata.shape)*noise_sd

        flip_sign = False
        if flip_team:
            flip_sign = True
            if verbose:
                print('flipping (flip_sign)', teamCollectionID, hyp)
        if rectify_status[hyp]:
            if verbose:
                print('flipping (reverse_contrast)', teamCollectionID, hyp)
            flip_sign = True

        if flip_sign:
            # flip sign of data
            newimgdata = newimgdata * -1

        # save image
        newimg = nibabel.Nifti1Image(
            newimgdata,
            affine=baseimg.affine)
        if verbose:
            print('saving', outfile['unthresh'])
        newimg.to_filename(outfile['unthresh'])
        # for rectified images, make sure the thresh map
        # will match the re-rectified unthresh map
        if rectify_status[hyp]:
            threshdata = (newimgdata < -1*thresh).astype('int')
        else:
            threshdata = (newimgdata > thresh).astype('int')
        newimg = nibabel.Nifti1Image(
            threshdata,
            affine=baseimg.affine)
        if verbose:
            print('saving', outfile['thresh'])
        newimg.to_filename(outfile['thresh'])


def make_orig_image_sets(narps, basedir, verbose=False,
                         n_teams=24,
                         n_flip_teams=0,
                         n_noise_teams=0,
                         n_highvar_teams=0,
                         rectify_maps=True,
                         testing=True):
    """for each team in orig,
    make a set of orig_simulated images
    rectify_maps: if true, then flip the original
    maps (which will be rectified again in the
    main analysis)
    """
    if testing:
        teams = ['0JO0']
        n_teams = 1
    else:
        teams = narps.complete_image_sets['unthresh'][:n_teams]
        teams.sort()

    if verbose:
        print('found %d team dirs' % len(teams))
        print(teams)

    # arbitrarily assign some teams to be flipped or noise
    logfile = os.path.join(basedir, 'logs/simulated_data.log')
    if n_flip_teams > 0:
        flip_teams = [i for i in teams[0:n_flip_teams]]
        log_to_file(logfile,
                    'flipped teams: %s' % ' '.join(flip_teams))
    else:
        flip_teams = []

    if n_noise_teams > 0:
        noise_teams = teams[
            (n_flip_teams + 1):(n_flip_teams + n_noise_teams)]
        log_to_file(logfile,
                    'noise teams: %s' % ' '.join(noise_teams))
    else:
        noise_teams = []

    if n_highvar_teams > 0:
        highvar_teams = teams[
            (n_flip_teams + n_noise_teams + 1):(
                n_flip_teams + n_noise_teams + n_highvar_teams)]
        log_to_file(logfile,
                    'high variance teams: %s' % ' '.join(highvar_teams))
    else:
        highvar_teams = []

    for teamID in teams:
        if rectify_maps:
            rectify_status = narps.teams[teamID].rectify
        else:
            rectify_status = None
        print(rectify_status)
        teamCollectionID = '%s_%s' % (
            narps.teams[teamID].NV_collection_id, teamID
        )
        if verbose:
            print('making simulated data for',
                  teamID, teamCollectionID)
        make_orig_images(
            basedir,
            teamCollectionID,
            narps.dirs.dirs['consensus'],
            noise_team=teamID in noise_teams,
            flip_team=teamID in flip_teams,
            noise_sd=0.5 + 2*(teamID in highvar_teams),
            rectify_status=rectify_status,
            verbose=verbose)
