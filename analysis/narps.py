"""
This is the main class for the NARPS analysis
There are three classes defined here:
Narps: this is a class that wraps the entire dataset
NarpsTeam: this class is instantiated for each team
NarpsDirs: This class contains info about all
of the directories that are needed for this and
subsequent analyses

The code under the main loop at the bottom
runs all of the image preprocessing that is
needed for subsequent analyses

"""

import argparse
import numpy
import pandas
import nibabel
import json
import os
import sys
import time
import glob
import datetime
import nilearn.image
import nilearn.input_data
import nilearn.plotting
import shutil
import warnings
import pickle
from nipype.interfaces.fsl.model import SmoothEstimate
import wget
import tarfile
from urllib.error import HTTPError
import hashlib
import inspect
from utils import get_metadata, TtoZ, get_map_metadata,\
    log_to_file, stringify_dict, randn_from_shape

# set up data url
# this is necessary for now because the data are still private
# once the data are public we can share the info.json file

if 'DATA_URL' in os.environ:
    DATA_URL = os.environ['DATA_URL']
    print('reading data URL from environment')
elif os.path.exists('info.json'):
    info = json.load(open('info.json'))
    DATA_URL = info['DATA_URL']
    print('reading data URL from info.json')
else:
    DATA_URL = None
    print('info.json no present - data downloading will be disabled')

# Hypotheses:
#
# Parametric effect of gain:
#
# 1. Positive effect in ventromedial PFC - equal indifference group
# 2. Positive effect in ventromedial PFC - equal range group
# 3. Positive effect in ventral striatum - equal indifference group
# 4. Positive effect in ventral striatum - equal range group
#
# Parametric effect of loss:
# - 5: Negative effect in VMPFC - equal indifference group
# - 6: Negative effect in VMPFC - equal range group
# - 7: Positive effect in amygdala - equal indifference group
# - 8: Positive effect in amygdala - equal range group
#
# Equal range vs. equal indifference:
#
# - 9: Greater positive response to losses in amygdala for equal range
# condition vs. equal indifference condition.

hypotheses = {1: '+gain: equal indiff',
              2: '+gain: equal range',
              3: '+gain: equal indiff',
              4: '+gain: equal range',
              5: '-loss: equal indiff',
              6: '-loss: equal range',
              7: '+loss: equal indiff',
              8: '+loss: equal range',
              9: '+loss:ER>EI'}

hypnums = [1, 2, 5, 6, 7, 8, 9]


# separate class to store base directories,
# since we need them in multiple places

class NarpsDirs(object):
    """
    class defining directories for project
    """
    def __init__(self, basedir, data_url=None,
                 force_download=False):

        # set up a dictionary to contain all of the
        # directories
        self.dirs = {}

        # check to make sure home of basedir exists
        assert os.path.exists(os.path.dirname(basedir))
        self.dirs['base'] = basedir
        if not os.path.exists(basedir):
            os.mkdir(basedir)
        self.force_download = force_download
        if data_url is None:
            self.data_url = DATA_URL

        dirs_to_add = ['output', 'templates', 'metadata',
                       'cached', 'figures', 'logs', 'orig']
        for d in dirs_to_add:
            self.dirs[d] = os.path.join(self.dirs['base'], d)

        # autogenerate all of the directories
        # except for the orig dir
        dirs_to_make = list(set(dirs_to_add).difference(['orig']))
        for d in dirs_to_make:
            if not os.path.exists(self.dirs[d]):
                os.mkdir(self.dirs[d])

        self.logfile = os.path.join(self.dirs['logs'], 'narps.txt')
        log_to_file(self.logfile, 'Running Narps main class', flush=True)

        output_dirs = ['resampled', 'rectified', 'zstat',
                       'thresh_mask_orig', 'concat_thresh']

        for o in output_dirs:
            self.dirs[o] = os.path.join(self.dirs['output'], o)
            if not os.path.exists(self.dirs[o]):
                os.mkdir(self.dirs[o])

        # if raw data don't exist, download them
        if self.force_download and os.path.exists(self.dirs['orig']):
            shutil.rmtree(self.dirs['orig'])
        if not os.path.exists(self.dirs['orig']):
            self.get_orig_data()
        assert os.path.exists(self.dirs['orig'])

        # make sure the necessary templates are present
        # these should be downloaded with the raw data
        self.MNI_mask = os.path.join(self.dirs['templates'],
                                     'MNI152_T1_2mm_brain_mask.nii.gz')
        assert os.path.exists(self.MNI_mask)

        self.MNI_template = os.path.join(self.dirs['templates'],
                                         'MNI152_T1_2mm.nii.gz')
        assert os.path.exists(self.MNI_template)

        self.full_mask_img = os.path.join(self.dirs['templates'],
                                          'MNI152_all_voxels.nii.gz')

    def get_orig_data(self):
        """
        download original data from repository
        """
        log_to_file(self.logfile, '\n\nget_orig_data')
        log_to_file(self.logfile, 'DATA_URL: %s' % DATA_URL)
        MAX_TRIES = 5

        if self.data_url is None:
            print('no URL for original data, cannot download')
            print('should be specified in info.json')
            return

        print('orig data do not exist, downloading...')
        output_directory = self.dirs['base']
        no_dl = True
        ntries = 0
        # try several times in case of http error
        while no_dl:
            try:
                filename = wget.download(self.data_url, out=output_directory)
                no_dl = False
            except HTTPError:
                ntries += 1
                time.sleep(1)  # wait a second
            if ntries > MAX_TRIES:
                raise Exception('Problem downloading original data')

        # save a hash of the tarball for data integrity
        filehash = hashlib.md5(open(filename, 'rb').read()).hexdigest()
        log_to_file(self.logfile, 'hash of tar file: %s' % filehash)
        tarfile_obj = tarfile.open(filename)
        tarfile_obj.extractall(path=self.dirs['base'])
        os.remove(filename)


class NarpsTeam(object):
    """
    class defining team information
    """
    def __init__(self, teamID, NV_collection_id, dirs, verbose=False):
        self.dirs = dirs
        self.teamID = teamID
        self.NV_collection_id = NV_collection_id
        self.datadir_label = '%s_%s' % (NV_collection_id, teamID)
        # directory for the original maps
        self.input_dir = os.path.join(self.dirs.dirs['orig'],
                                      '%s_%s' % (NV_collection_id, teamID))
        if not os.path.exists(self.input_dir):
            print("Warning: Input dir (%s) does not exist" % self.input_dir)

        self.verbose = verbose
        self.image_json = None
        self.jsonfile = None
        self.has_all_images = None
        self.logs = {}

        # create image directory structure
        output_dirs = {'thresh': ['orig', 'resampled', 'thresh_mask_orig'],
                       'unthresh': ['orig', 'resampled', 'rectified', 'zstat']}
        self.images = {}
        for imgtype in ['thresh', 'unthresh']:
            self.images[imgtype] = {}
            for o in output_dirs[imgtype]:
                self.images[imgtype][o] = {}
        self.n_mask_vox = {}
        self.n_nan_inmask_values = {}
        self.n_zero_inmask_values = {}
        self.has_resampled = None
        self.has_binarized_masks = None

        # populate the image data structure
        self.get_orig_images()

    def get_orig_images(self):
        """
        find orig images
        """
        self.has_all_images = True
        for hyp in hypotheses:
            for imgtype in self.images:
                imgfile = os.path.join(
                    self.input_dir,
                    'hypo%d_%s.nii.gz' % (hyp, imgtype))
                if os.path.exists(imgfile):
                    self.images[imgtype]['orig'][hyp] = imgfile
                else:
                    self.images[imgtype]['orig'][hyp] = None
                    self.has_all_images = False

    def create_binarized_thresh_masks(self, thresh=1e-6,
                                      overwrite=False,
                                      replace_na=True):
        """
        create binarized version of thresholded maps
        """
        self.has_binarized_masks = True
        if self.verbose:
            print('creating binarized masks for', self.teamID)

        for hyp in self.images['thresh']['orig']:
            img = self.images['thresh']['orig'][hyp]
            self.images['thresh']['thresh_mask_orig'][hyp] = os.path.join(
                    self.dirs.dirs['thresh_mask_orig'],
                    self.datadir_label,
                    os.path.basename(img))
            if not os.path.exists(os.path.dirname(
                    self.images['thresh']['thresh_mask_orig'][hyp])):
                os.mkdir(os.path.dirname(
                    self.images['thresh']['thresh_mask_orig'][hyp]))
            if overwrite or not os.path.exists(
                    self.images['thresh']['thresh_mask_orig'][hyp]):

                # load the image and threshold/binarize it
                threshimg = nibabel.load(img)
                threshdata = threshimg.get_data()
                # some images use nan instead of zero for the non-excursion
                # voxels, so we need to replace with zeros
                if replace_na:
                    threshdata = numpy.nan_to_num(threshdata)
                threshdata_bin = numpy.zeros(threshdata.shape)
                # use a small number instead of zero to address
                # numeric issues
                threshdata_bin[numpy.abs(threshdata) > thresh] = 1
                # save back to a nifti image with same geometry
                # as original
                bin_img = nibabel.Nifti1Image(threshdata_bin,
                                              affine=threshimg.affine)
                bin_img.to_filename(
                    self.images['thresh']['thresh_mask_orig'][hyp])
            else:
                # if it already exists, just use existing
                bin_img = nibabel.load(
                    self.images['thresh']['thresh_mask_orig'][hyp])
                if self.verbose:
                    print('using existing binary mask for', self.teamID)
                threshdata_bin = bin_img.get_data()
            # get number of suprathreshold voxels
            self.n_mask_vox[hyp] = numpy.sum(threshdata_bin)

    def get_resampled_images(self, overwrite=False, replace_na=False):
        """
        resample images into common space using nilearn
        """
        self.has_resampled = True
        # use linear interpolation for binarized maps, then threshold at 0.5
        # this avoids empty voxels that can occur with NN interpolation
        interp_type = {'thresh': 'linear', 'unthresh': 'continuous'}
        data_dirname = {'thresh': 'thresh_mask_orig',
                        'unthresh': 'orig'}

        for hyp in hypotheses:
            for imgtype in self.images:
                infile = os.path.join(
                    self.dirs.dirs[data_dirname[imgtype]],
                    self.datadir_label,
                    'hypo%d_%s.nii.gz' % (hyp, imgtype))
                outfile = os.path.join(
                    self.dirs.dirs['resampled'],
                    self.datadir_label,
                    'hypo%d_%s.nii.gz' % (hyp, imgtype))
                self.images[imgtype]['resampled'][hyp] = outfile
                if not os.path.exists(os.path.dirname(outfile)):
                    os.mkdir(os.path.dirname(outfile))
                if not os.path.exists(outfile) or overwrite:
                    if self.verbose:
                        print("resampling", infile)

                    # create resampled file

                    # ignore nilearn warnings
                    # these occur on some of the unthresholded images
                    # that contains NaN values
                    # we probably don't want to set those to zero
                    # because those would enter into interpolation
                    # and then would be treated as real zeros later
                    # rather than "missing data" which is the usual
                    # intention
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        resampled = nilearn.image.resample_to_img(
                            infile,
                            self.dirs.MNI_template,
                            interpolation=interp_type[imgtype])

                    if imgtype == 'thresh':
                        resampled = nilearn.image.math_img(
                            'img>0.5',
                            img=resampled)

                    resampled.to_filename(outfile)

                else:
                    if self.verbose:
                        print('using existing resampled image for',
                              self.teamID)


class Narps(object):
    """
    main class for NARPS analysis
    """
    def __init__(self, basedir, metadata_file=None,
                 verbose=False, overwrite=False):
        self.basedir = basedir
        self.dirs = NarpsDirs(basedir)
        self.verbose = verbose
        self.teams = {}
        self.overwrite = overwrite
        self.started_at = datetime.datetime.now()

        # create the full mask image if it doesn't already exist
        if not os.path.exists(self.dirs.full_mask_img):
            print('making full image mask')
            self.mk_full_mask_img(self.dirs)
        assert os.path.exists(self.dirs.full_mask_img)

        # get input dirs for orig data
        self.image_jsons = None
        self.input_dirs = self.get_input_dirs(self.dirs)

        # check images for each team
        self.complete_image_sets = None
        self.get_orig_images(self.dirs)
        log_to_file(
            self.dirs.logfile,
            'found %d teams with complete original datasets' % len(
                self.complete_image_sets))

        # set up metadata
        if metadata_file is None:
            self.metadata_file = os.path.join(
                self.dirs.dirs['orig'],
                'analysis_pipelines_SW.xlsx')
        else:
            self.metadata_file = metadata_file

        self.metadata = get_metadata(self.metadata_file)

        self.hypothesis_metadata = pandas.DataFrame(
            columns=['teamID', 'hyp', 'n_na', 'n_zero'])

        self.all_maps = {'thresh': {'resampled': None},
                         'unthresh': {'resampled': None}}
        self.rectified_list = []

    def mk_full_mask_img(self, dirs):
        """
        create a mask image with ones in all voxels
        """
        # make full image mask (all voxels)
        mi = nibabel.load(self.dirs.MNI_mask)
        d = mi.get_data()
        d = numpy.ones(d.shape)
        full_mask = nibabel.Nifti1Image(d, affine=mi.affine)
        full_mask.to_filename(self.dirs.full_mask_img)

    def get_input_dirs(self, dirs, verbose=True, load_json=True):
        """
        get orig dirs
        - assumes that images.json is present for each valid dir
        """

        input_files = glob.glob(
            os.path.join(dirs.dirs['orig'], '*/hypo1_thresh.nii.gz'))
        input_dirs = [os.path.dirname(i) for i in input_files]

        log_to_file(
            self.dirs.logfile,
            'found %d input directories' % len(input_dirs))
        for i in input_dirs:
            collection_id = os.path.basename(i)
            NV_collection_id, teamID = collection_id.split('_')
            if teamID not in self.teams:
                self.teams[teamID] = NarpsTeam(
                    teamID, NV_collection_id, dirs, verbose=self.verbose)
                if os.path.exists(os.path.join(i, 'images.json')):
                    self.teams[teamID].jsonfile = os.path.join(
                        i, 'images.json')
                    with open(self.teams[teamID].jsonfile) as f:
                        self.teams[teamID].image_json = json.load(f)

    def get_orig_images(self, dirs):
        """
        load orig images
        """
        self.complete_image_sets = []
        for teamID in self.teams:
            self.teams[teamID].get_orig_images()
            if self.teams[teamID].has_all_images:
                self.complete_image_sets.append(teamID)
        # sort the teams - this is the order that will be used
        self.complete_image_sets.sort()

    def get_binarized_thresh_masks(self):
        """
        create binarized thresholded maps for each team
        """
        log_to_file(
            self.dirs.logfile, '\n\n%s' %
            sys._getframe().f_code.co_name)
        for teamID in self.complete_image_sets:
            self.teams[teamID].create_binarized_thresh_masks()

    def get_resampled_images(self, overwrite=None):
        """
        resample all images into FSL MNI space
        """
        log_to_file(
            self.dirs.logfile, '\n\n%s' %
            sys._getframe().f_code.co_name)
        if overwrite is None:
            overwrite = self.overwrite
        for teamID in self.complete_image_sets:
            self.teams[teamID].get_resampled_images()

    def check_image_values(self, overwrite=None):
        """
        get # of nonzero and NA voxels for each image
        """
        log_to_file(
            self.dirs.logfile, '\n\n%s' %
            sys._getframe().f_code.co_name)
        if overwrite is None:
            overwrite = self.overwrite
        image_metadata_file = os.path.join(
            self.dirs.dirs['metadata'], 'image_metadata_df.csv')
        if os.path.exists(image_metadata_file) and not overwrite:
            print('using cached image metdata')
            image_metadata_df = pandas.read_csv(image_metadata_file)
            return(image_metadata_df)
        # otherwise load from scractch
        image_metadata = []
        masker = nilearn.input_data.NiftiMasker(mask_img=self.dirs.MNI_mask)
        for teamID in self.complete_image_sets:
            for hyp in self.teams[teamID].images['thresh']['resampled']:
                threshfile = self.teams[teamID].images[
                    'thresh']['resampled'][hyp]
                threshdata = masker.fit_transform(threshfile)
                image_metadata.append(
                    [teamID, hyp, numpy.sum(numpy.isnan(threshdata)),
                     numpy.sum(threshdata == 0.0)])

        image_metadata_df = pandas.DataFrame(
            image_metadata, columns=['teamID', 'hyp', 'n_na', 'n_nonzero'])

        image_metadata_df.to_csv(image_metadata_file)
        return(image_metadata_df)

    def create_concat_images(self, datatype='resampled',
                             imgtypes=None,
                             overwrite=None):
        """
        create images concatenated across teams
        ordered by self.complete_image_sets
        """
        log_to_file(
            self.dirs.logfile, '\n\n%s' %
            sys._getframe().f_code.co_name)
        func_args = inspect.getargvalues(
            inspect.currentframe()).locals
        log_to_file(
            self.dirs.logfile,
            stringify_dict(func_args))

        if imgtypes is None:
            imgtypes = ['thresh', 'unthresh']
        if overwrite is None:
            overwrite = self.overwrite
        for imgtype in imgtypes:
            self.dirs.dirs['concat_%s' % imgtype] = os.path.join(
                self.dirs.dirs['output'],
                '%s_concat_%s' % (imgtype, datatype))
            for hyp in range(1, 10):
                outfile = os.path.join(
                    self.dirs.dirs['concat_%s' % imgtype],
                    'hypo%d.nii.gz' % hyp)
                if not os.path.exists(os.path.dirname(outfile)):
                    os.mkdir(os.path.dirname(outfile))
                if not os.path.exists(outfile) or overwrite:
                    if self.verbose:
                        print('%s - hypo %d: creating concat file' % (
                            imgtype, hyp))
                    concat_teams = [
                        teamID for teamID in self.complete_image_sets
                        if os.path.exists(
                            self.teams[teamID].images[imgtype][datatype][hyp])]
                    self.all_maps[imgtype][datatype] = [
                        self.teams[teamID].images[imgtype][datatype][hyp]
                        for teamID in concat_teams]

                    # use nilearn NiftiMasker to load data
                    # and save to a new file
                    masker = nilearn.input_data.NiftiMasker(
                        mask_img=self.dirs.MNI_mask)
                    concat_data = masker.fit_transform(
                        self.all_maps[imgtype][datatype])
                    concat_img = masker.inverse_transform(concat_data)
                    concat_img.to_filename(outfile)
                else:
                    if self.verbose:
                        print('%s - hypo %d: using existing file' % (
                            imgtype, hyp))
        return(self.all_maps)

    def create_thresh_overlap_images(self, datatype='resampled',
                                     overwrite=None, thresh=1e-5):
        """
        create overlap maps for thresholded images
        """
        log_to_file(
            self.dirs.logfile, '\n\n%s' %
            sys._getframe().f_code.co_name)
        func_args = inspect.getargvalues(
            inspect.currentframe()).locals
        log_to_file(
            self.dirs.logfile,
            stringify_dict(func_args))

        imgtype = 'thresh'
        if overwrite is None:
            overwrite = self.overwrite
        self.dirs.dirs['overlap_binarized_thresh'] = os.path.join(
            self.dirs.dirs['output'], 'overlap_binarized_thresh')
        for hyp in range(1, 10):
            outfile = os.path.join(
                self.dirs.dirs['overlap_binarized_thresh'],
                'hypo%d.nii.gz' % hyp)
            if not os.path.exists(os.path.dirname(outfile)):
                os.mkdir(os.path.dirname(outfile))
            if not os.path.exists(outfile) or overwrite:
                if self.verbose:
                    print('%s - hypo %d: creating overlap file' % (
                        imgtype, hyp))
                concat_file = os.path.join(
                    self.dirs.dirs['concat_thresh'],
                    'hypo%d.nii.gz' % hyp)
                concat_img = nibabel.load(concat_file)
                concat_data = concat_img.get_data()
                concat_data = (concat_data > thresh).astype('float')
                concat_mean = numpy.mean(concat_data, 3)
                concat_mean_img = nibabel.Nifti1Image(concat_mean,
                                                      affine=concat_img.affine)
                concat_mean_img.to_filename(outfile)

            else:
                if self.verbose:
                    print('%s - hypo %d: using existing file' % (
                        imgtype, hyp))

    def create_rectified_images(self, map_metadata_file=None,
                                overwrite=None):
        """
        create rectified images
        - contrasts 5 and 6 were negative contrasts
        some teams uploaded images where negative values
        provided evidence in favor of the contrast
        using metadata provided by teams, we identify these
        images and flip their valence so that all maps
        present positive evidence for each contrast
        """
        log_to_file(
            self.dirs.logfile, '\n\n%s' %
            sys._getframe().f_code.co_name)
        func_args = inspect.getargvalues(
            inspect.currentframe()).locals
        log_to_file(
            self.dirs.logfile,
            stringify_dict(func_args))
        if map_metadata_file is None:
            map_metadata_file = os.path.join(
                self.dirs.dirs['orig'],
                'narps_neurovault_images_details.csv')
        map_metadata = get_map_metadata(map_metadata_file)
        if overwrite is None:
            overwrite = self.overwrite
        for teamID in self.complete_image_sets:
            for hyp in range(1, 10):
                if hyp in [5, 6]:
                    mdstring = map_metadata.query(
                        'teamID == "%s"' % teamID
                        )['hyp%d_direction' % hyp].iloc[0]
                    rectify = mdstring.split()[0] == 'Negative'
                elif hyp == 9:
                    # manual fix for one team with reversed maps
                    if teamID in ['R7D1']:
                        mdstring = map_metadata.query(
                            'teamID == "%s"' % teamID)[
                                'hyp%d_direction' % hyp].iloc[0]
                        rectify = True
                else:  # just copy the other hypotheses directly
                    rectify = False

                # load data from unthresh map within
                # positive voxels of thresholded mask
                unthresh_file = self.teams[
                    teamID].images['unthresh']['resampled'][hyp]

                self.teams[
                    teamID].images[
                        'unthresh']['rectified'][hyp] = os.path.join(
                            self.dirs.dirs['rectified'],
                            self.teams[teamID].datadir_label,
                            'hypo%d_unthresh.nii.gz' % hyp)

                if not os.path.exists(
                    os.path.dirname(
                        self.teams[
                            teamID].images['unthresh']['rectified'][hyp])):
                    os.mkdir(os.path.dirname(
                            self.teams[teamID].images[
                                'unthresh']['rectified'][hyp]))

                if overwrite or not os.path.exists(
                        self.teams[
                            teamID].images['unthresh']['rectified'][hyp]):
                    # if values were flipped for negative contrasts
                    if rectify:
                        print('rectifying hyp', hyp, 'for', teamID)
                        print(mdstring)
                        print('')
                        img = nibabel.load(unthresh_file)
                        img_rectified = nilearn.image.math_img(
                            'img*-1', img=img)
                        img_rectified.to_filename(
                            self.teams[
                                teamID].images['unthresh']['rectified'][hyp])
                        self.rectified_list.append((teamID, hyp))
                    else:  # just copy original
                        shutil.copy(
                            unthresh_file,
                            self.teams[
                                teamID].images['unthresh']['rectified'][hyp])
        # write list of rectified teams to disk
        if len(self.rectified_list) > 0:
            with open(os.path.join(self.dirs.dirs['metadata'],
                                   'rectified_images_list.txt'), 'w') as f:
                for l in self.rectified_list:
                    f.write('%s\t%s\n' % (l[0], l[1]))

    def compute_image_stats(self, datatype='zstat', overwrite=None):
        """
        compute std and range on statistical images
        """
        log_to_file(
            self.dirs.logfile, '\n\n%s' %
            sys._getframe().f_code.co_name)
        func_args = inspect.getargvalues(
            inspect.currentframe()).locals
        log_to_file(
            self.dirs.logfile,
            stringify_dict(func_args))

        if overwrite is None:
            overwrite = self.overwrite
        for hyp in range(1, 10):

            unthresh_file = os.path.join(
                self.dirs.dirs['output'],
                'unthresh_concat_%s/hypo%d.nii.gz' % (datatype, hyp))

            range_outfile = os.path.join(
                self.dirs.dirs['output'],
                'unthresh_range_%s/hypo%d.nii.gz' % (datatype, hyp))
            if not os.path.exists(os.path.join(
                self.dirs.dirs['output'],
                    'unthresh_range_%s' % datatype)):
                os.mkdir(os.path.join(
                    self.dirs.dirs['output'],
                    'unthresh_range_%s' % datatype))

            std_outfile = os.path.join(
                self.dirs.dirs['output'],
                'unthresh_std_%s/hypo%d.nii.gz' % (datatype, hyp))
            if not os.path.exists(os.path.join(
                    self.dirs.dirs['output'],
                    'unthresh_std_%s' % datatype)):
                os.mkdir(os.path.join(
                    self.dirs.dirs['output'],
                    'unthresh_std_%s' % datatype))

            if not os.path.exists(range_outfile) \
                    or not os.path.exists(std_outfile) \
                    or overwrite:
                unthresh_img = nibabel.load(unthresh_file)
                unthresh_data = unthresh_img.get_data()
                concat_data = numpy.nan_to_num(unthresh_data)
                datarange = numpy.max(concat_data, axis=3) \
                    - numpy.min(concat_data, axis=3)
                range_img = nibabel.Nifti1Image(
                    datarange,
                    affine=unthresh_img.affine)
                range_img.to_filename(range_outfile)
                datastd = numpy.std(concat_data, axis=3)
                std_img = nibabel.Nifti1Image(
                    datastd,
                    affine=unthresh_img.affine)
                std_img.to_filename(std_outfile)

    def convert_to_zscores(self, map_metadata_file=None, overwrite=None):
        """
        convert rectified images to z scores
        - unthresholded images could be either t or z images
        - if they are already z then just copy
        - use metadata supplied by teams to determine image type
        """
        log_to_file(
            self.dirs.logfile, '\n\n%s' %
            sys._getframe().f_code.co_name)
        func_args = inspect.getargvalues(
            inspect.currentframe()).locals
        log_to_file(
            self.dirs.logfile,
            stringify_dict(func_args))

        if overwrite is None:
            overwrite = self.overwrite
        if map_metadata_file is None:
            map_metadata_file = os.path.join(
                self.dirs.dirs['orig'],
                'narps_neurovault_images_details.csv')
        unthresh_stat_type = get_map_metadata(map_metadata_file)
        metadata = get_metadata(self.metadata_file)

        n_participants = metadata[['n_participants', 'NV_collection_string']]

        n_participants.index = metadata.teamID

        unthresh_stat_type = unthresh_stat_type.merge(
            n_participants, left_index=True, right_index=True)

        for teamID in self.complete_image_sets:
            if teamID not in unthresh_stat_type.index:
                print('no map metadata for', teamID)
                continue
            # this is a bit of a kludge
            # since some contrasts include all subjects
            # but others only include some
            # we don't have the number of participants in each
            # group so we just use the entire number
            n = unthresh_stat_type.loc[teamID, 'n_participants']

            for hyp in range(1, 10):
                infile = self.teams[
                    teamID].images['unthresh']['rectified'][hyp]
                if not os.path.exists(infile):
                    print('skipping', infile)
                    continue
                self.teams[
                    teamID].images['unthresh']['zstat'][hyp] = os.path.join(
                        self.dirs.dirs['zstat'],
                        self.teams[teamID].datadir_label,
                        'hypo%d_unthresh.nii.gz' % hyp)
                if not overwrite and os.path.exists(
                        self.teams[teamID].images['unthresh']['zstat'][hyp]):
                    continue

                if unthresh_stat_type.loc[
                        teamID, 'unthresh_type'].lower() == 't':
                    if not os.path.exists(
                            os.path.dirname(
                                self.teams[
                                    teamID].images['unthresh']['zstat'][hyp])):
                        os.mkdir(os.path.dirname(
                            self.teams[
                                teamID].images['unthresh']['zstat'][hyp]))
                    print("converting %s (hyp %d) to z - %d participants" % (
                        teamID, hyp, n))
                    TtoZ(infile,
                         self.teams[teamID].images['unthresh']['zstat'][hyp],
                         n-1)
                elif unthresh_stat_type.loc[teamID, 'unthresh_type'] == 'z':
                    if not os.path.exists(os.path.dirname(
                            self.teams[
                                teamID].images['unthresh']['zstat'][hyp])):
                        os.mkdir(os.path.dirname(
                            self.teams[
                                teamID].images['unthresh']['zstat'][hyp]))
                    if not os.path.exists(
                            self.teams[
                                teamID].images['unthresh']['zstat'][hyp]):
                        print('copying', teamID)
                        shutil.copy(
                            infile,
                            os.path.dirname(
                                self.teams[
                                    teamID].images['unthresh']['zstat'][hyp]))
                else:
                    # if it's not T or Z then we skip it as it's not usable
                    print('skipping %s - other data type' % teamID)

    def estimate_smoothness(self, overwrite=None, imgtype='zstat'):
        """
        estimate smoothness of Z maps using FSL's smoothness estimation
        """
        log_to_file(
            self.dirs.logfile, '\n\n%s' %
            sys._getframe().f_code.co_name)
        func_args = inspect.getargvalues(
            inspect.currentframe()).locals
        log_to_file(
            self.dirs.logfile,
            stringify_dict(func_args))

        if overwrite is None:
            overwrite = self.overwrite
        output_file = os.path.join(self.dirs.dirs['metadata'],
                                   'smoothness_est.csv')
        if os.path.exists(output_file) and not overwrite:
            if self.verbose:
                print('using existing smoothness file')
            smoothness_df = pandas.read_csv(output_file)
            return(smoothness_df)

        # use nipype's interface to the FSL smoothest command
        est = SmoothEstimate()
        smoothness = []
        for teamID in self.complete_image_sets:
            for hyp in range(1, 10):
                if hyp not in self.teams[teamID].images['unthresh'][imgtype]:
                    # fill missing data with nan
                    print('no zstat present for', teamID, hyp)
                    smoothness.append([teamID, hyp, numpy.nan,
                                       numpy.nan, numpy.nan])
                    continue
                infile = self.teams[teamID].images['unthresh'][imgtype][hyp]
                if not os.path.exists(infile):
                    print('no image present:', infile)
                    continue
                else:
                    if self.verbose:
                        print('estimating smoothness for hyp', hyp)

                    est.inputs.zstat_file = infile
                    est.inputs.mask_file = self.dirs.MNI_mask
                    est.terminal_output = 'file_split'
                    smoothest_output = est.run()
                    smoothness.append([teamID, hyp,
                                       smoothest_output.outputs.dlh,
                                       smoothest_output.outputs.volume,
                                       smoothest_output.outputs.resels])
                    self.teams[teamID].logs['smoothest'] = (
                        smoothest_output.runtime.stdout,
                        smoothest_output.runtime.stderr)

        smoothness_df = pandas.DataFrame(
            smoothness,
            columns=['teamID', 'hyp', 'dhl', 'volume', 'resels'])
        smoothness_df.to_csv(output_file)
        return(smoothness_df)

    def write_data(self, save_data=True, outfile=None):
        """
        serialize important info and save to file
        """
        info = {}
        info['started_at'] = self.started_at
        info['save_time'] = datetime.datetime.now()
        info['dirs'] = self.dirs
        info['teamlist'] = self.complete_image_sets
        info['teams'] = {}

        for teamID in self.complete_image_sets:
            info['teams'][teamID] = {
                'images': self.teams[teamID].images,
                'image_json': self.teams[teamID].image_json,
                'input_dir': self.teams[teamID].input_dir,
                'NV_collection_id': self.teams[teamID].NV_collection_id,
                'jsonfile': self.teams[teamID].jsonfile}
        if save_data:
            if not os.path.exists(self.dirs.dirs['cached']):
                os.mkdir(self.dirs.dirs['cached'])
            if outfile is None:
                outfile = os.path.join(self.dirs.dirs['cached'],
                                       'narps_prepare_maps.pkl')
            with open(outfile, 'wb') as f:
                pickle.dump(info, f)
        return(info)

    def load_data(self, infile=None):
        """
        load data from pickle
        """
        if not infile:
            infile = os.path.join(self.dirs.dirs['cached'],
                                  'narps_prepare_maps.pkl')
        assert os.path.exists(infile)

        with open(infile, 'rb') as f:
            info = pickle.load(f)

        self.dirs = info['dirs']
        self.complete_image_sets = info['teamlist']
        for teamID in self.complete_image_sets:
            self.teams[teamID] = NarpsTeam(
                teamID,
                info['teams'][teamID]['NV_collection_id'],
                info['dirs'],
                verbose=self.verbose)
            self.teams[teamID].jsonfile = info[
                'teams'][teamID]['jsonfile']
            self.teams[teamID].images = info[
                'teams'][teamID]['images']
            self.teams[teamID].image_json = info[
                'teams'][teamID]['image_json']
            self.teams[teamID].input_dir = info[
                'teams'][teamID]['input_dir']


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
                     verbose=False):
    """
    for a particular team and hypothesis,
    generate the images
    NOTE: flip_teams are teams whose data will be
    flipped but will not be rectified
    - included to simulate flipping errors
    We also flip those teams/hyps that will be rectified
    """
    if rectify_status is None:
        rectify_status = []
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
        threshdata = (newimgdata > thresh).astype('int')
        newimg = nibabel.Nifti1Image(
            threshdata,
            affine=baseimg.affine)
        if verbose:
            print('saving', outfile['thresh'])
        newimg.to_filename(outfile['thresh'])


def get_teams_to_rectify(narps):
    map_metadata_file = os.path.join(
        narps.dirs.dirs['orig'],
        'narps_neurovault_images_details.csv')
    map_metadata = get_map_metadata(map_metadata_file)
    rectify_status = {}
    # manual fix for one
    rectify_status['R7D1'] = [5, 6]
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


def make_orig_image_sets(narps, basedir, verbose=False):
    """for each team in orig,
    make a set of orig_simulated images
    """
    dirlist = [os.path.basename(i)
               for i in glob.glob(os.path.join(
                   narps.basedir,
                   'orig/*_*')) if os.path.isdir(i)]
    if verbose:
        print('found %d team dirs' % len(dirlist))

    # for teams that are to be rectified, we should
    # also flip their data
    rectify_status = get_teams_to_rectify(narps)

    # arbitrarily assign some teams to be flipped or noise
    logfile = os.path.join(basedir, 'logs/simulated_data.log')
    flip_dirs = dirlist[0:4]
    flip_teams = [i.split('_')[1] for i in flip_dirs]
    log_to_file(logfile,
                'flipped teams: %s' % ' '.join(flip_teams))
    noise_dirs = dirlist[5:9]
    noise_teams = [i.split('_')[1] for i in noise_dirs]
    log_to_file(logfile,
                'noise teams: %s' % ' '.join(noise_teams))
    highvar_dirs = dirlist[9:21]
    highvar_teams = [i.split('_')[1] for i in highvar_dirs]
    log_to_file(logfile,
                'high variance teams: %s' % ' '.join(highvar_teams))

    for teamCollectionID in dirlist:
        teamID = teamCollectionID.split('_')[1]
        if teamID not in rectify_status:
            rectify_status[teamID] = []
        make_orig_images(
            basedir,
            teamCollectionID,
            narps.dirs.dirs['consensus'],
            noise_team=teamCollectionID in noise_dirs,
            flip_team=teamCollectionID in flip_dirs,
            noise_sd=0.5 + 2*(teamCollectionID in highvar_dirs),
            rectify_status=rectify_status[teamID],
            verbose=verbose)


if __name__ == "__main__":
    # team data (from neurovault) should be in
    # # <basedir>/orig
    # some data need to be renamed before using -
    # see rename.sh in individual dirs

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Process NARPS data')
    parser.add_argument('-u', '--dataurl',
                        help='URL to download data')
    parser.add_argument('-b', '--basedir',
                        help='base directory')
    parser.add_argument('-s', '--simulate',
                        action='store_true',
                        help='use simulated data')
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

    # set up simulation if specified
    if args.simulate:
        print('using simulated data')

        # load main class from real analysis
        narps_orig = Narps(basedir, overwrite=False)

        # create simulated data
        # setup main class from original data
        narps = Narps(basedir)
        narps.load_data()

        # Load full metadata and put into narps structure
        narps.metadata = pandas.read_csv(
            os.path.join(narps.dirs.dirs['metadata'], 'all_metadata.csv'))

        basedir = setup_simulated_data(narps, verbose=False)

        make_orig_image_sets(narps, basedir)

        # doublecheck basedir name
        assert basedir.find('_simulated') > -1

    # setup main class

    narps = Narps(basedir)

    assert len(narps.complete_image_sets) > 0

    if args.test:
        print('testing mode, exiting after setup')

    else:  # run all modules

        print('getting binarized/thresholded orig maps')
        narps.get_binarized_thresh_masks()

        print("getting resampled images...")
        narps.get_resampled_images()

        print("creating concatenated thresholded images...")
        narps.create_concat_images(datatype='resampled',
                                   imgtypes=['thresh'])

        print("checking image values...")
        image_metadata_df = narps.check_image_values()

        print("creating rectified images...")
        narps.create_rectified_images()

        print('Creating overlap images for thresholded maps...')
        narps.create_thresh_overlap_images()

        print('converting to z-scores')
        narps.convert_to_zscores()

        print("creating concatenated zstat images...")
        narps.create_concat_images(datatype='zstat',
                                   imgtypes=['unthresh'])

        print("computing image stats...")
        narps.compute_image_stats()

        print('estimating image smoothness')
        smoothness_df = narps.estimate_smoothness()

        # save directory structure
        narps.write_data()
