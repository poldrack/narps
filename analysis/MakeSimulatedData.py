#!/usr/bin/env python
# coding: utf-8
"""
generate simulated data for NARPS analysis
to validate image processing

- use maps from consensus analysis as basis for maps
- generate maps for each team using these plus noise
- flip some maps
- inject complete noise in some maps
- vary smoothness of added noise

original downloaded data are moved to orig_basis
and simulated data go into orig
"""

import os
import glob
import shutil
import pandas
import numpy
import nibabel
from narps import Narps, hypnums
from narps import NarpsDirs # noqa, flake8 issue
from utils import get_map_metadata

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
            print('copying orig data to new basedir')
        shutil.copytree(
            origdir,
            os.path.join(basedir, 'orig_basis'))
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
                     noise_sd=.1,
                     noise_team=None,
                     flip_team=None,
                     rectify_status=[],
                     verbose=False):
    """
    for a particular team and hypothesis,
    generate the images
    NOTE: flip_teams are teams whose data will be
    flipped but will not be rectified
    - included to simulate flipping errors
    We also flip those teams/hyps that will be rectified
    """

    if noise_team is None:
        noise_team = False
    if flip_team is None:
        flip_team = False


    if verbose:
        print('make_orig_images:', teamCollectionID)

    teamdir = os.path.join(basedir,'orig/%s' % teamCollectionID)
    if not os.path.exists(teamdir):
        os.mkdir(teamdir)
    if verbose:
        print('teamdir', teamdir)

    for hyp in hypnums:
        outfile = {'thresh': os.path.join(
            teamdir, 'hypo%d_thresh.nii.gz' % hyp),
            'unthresh': os.path.join(
            teamdir, 'hypo%d_unthresh.nii.gz' % hyp)}

        # get t image from consensus map
        baseimgfile = os.path.join(
            consensus_dir,
            'hypo%d_t.nii.gz' % hyp)
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
        if hyp in rectify_status:
            flip_sign = True
        
        if flip_sign:
            if verbose:
                print('flipping', teamCollectionID, hyp)
            # flip sign of data
            newimgdata = newimgdata * -1

        # save image
        newimg = nibabel.Nifti1Image(
            newimgdata,
            affine=baseimg.affine)
        if verbose:
            print('saving', outfile['unthresh'])
        newimg.to_filename(outfile['unthresh'])

def get_teams_to_rectify(narps):
    map_metadata_file = os.path.join(
        narps.dirs.dirs['orig'],
        'narps_neurovault_images_details.csv')
    map_metadata = get_map_metadata(map_metadata_file)
    rectify_status = {}
    # manual fix for one
    rectify_status['R7D1'] = [5,6]
    for teamID in narps.teams:
        for hyp in [5, 6]:
            mdstring = map_metadata.query(
                'teamID == "%s"' % teamID
                )['hyp%d_direction' % hyp].iloc[0]
            rectify = mdstring.split()[0] == 'Negative'
            if rectify:
                if teamID not in rectify_status:
                    rectify_status[teamID] = []
                rectify_status[teamID].append(hyp)
    return(rectify_status)

def make_orig_image_sets(narps, basedir, verbose=True):
    """for each team in orig,
    make a set of orig_simulated images
    """
    dirlist = [os.path.basename(i) 
        for i in glob.glob(os.path.join(narps.basedir, 'orig/*_*')) if 
        os.path.isdir(i)]
    if verbose:
        print('found %d team dirs' % len(dirlist))

    # for teams that are to be rectified, we should
    # also flip their data
    rectify_status = get_teams_to_rectify(narps)

    # arbitrarily assign some teams to be flipped or noise
    flip_dirs = dirlist[0:4]
    noise_dirs = dirlist[5:8]
    for teamCollectionID in dirlist:
            teamID = teamCollectionID.split('_')[1]
            if not teamID in rectify_status:
                rectify_status[teamID] = []
            make_orig_images(
                basedir,
                teamCollectionID,
                narps.dirs.dirs['consensus'],
                noise_team=teamCollectionID in noise_dirs,
                flip_team=teamCollectionID in flip_dirs,
                rectify_status=rectify_status[teamID],
                verbose=True)


if __name__ == "__main__":
    # team data (from neurovault) should be in
    # # <basedir>/orig
    # some data need to be renamed before using -
    # see rename.sh in individual dirs

    # set an environment variable called NARPS_BASEDIR
    # with location of base directory
    if 'NARPS_BASEDIR' in os.environ:
        basedir = os.environ['NARPS_BASEDIR']
    else:
        basedir = '/data'

    # setup main class
    narps = Narps(basedir)
    narps.load_data()

    # Load full metadata and put into narps structure
    narps.metadata = pandas.read_csv(
        os.path.join(narps.dirs.dirs['metadata'], 'all_metadata.csv'))

    basedir = setup_simulated_data(narps, verbose=True)

    narps = make_orig_image_sets(narps, basedir)
