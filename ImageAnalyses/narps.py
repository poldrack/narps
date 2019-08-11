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
    log_to_file, stringify_dict
from ValueDiagnostics import compare_thresh_unthresh_values

# # set up data url - COMMENTING NOW, WILL REMOVE
# # this is necessary for now because the data are still private
# # once the data are public we can share the info.json file


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

# one team had thresholded maps that
# had 0 for exceedence and 1 for null
# so we flip those
FLIP_THRESH_MAPS = {'27SS': [2, 5, 7]}


# separate class to store base directories,
# since we need them in multiple places
class NarpsDirs(object):
    """
    class defining directories for project
    """
    def __init__(self, basedir, dataurl=None,
                 force_download=False, testing=False):

        # set up a dictionary to contain all of the
        # directories
        self.dirs = {}
        self.testing = testing

        # check to make sure home of basedir exists
        assert os.path.exists(os.path.dirname(basedir))
        self.dirs['base'] = basedir
        if not os.path.exists(basedir):
            os.mkdir(basedir)
        self.force_download = force_download
        self.data_url = dataurl

        dirs_to_add = ['output', 'metadata', 'templates',
                       'cached', 'figures', 'logs', 'orig',
                       'image_diagnostics']
        for d in dirs_to_add:
            self.dirs[d] = os.path.join(self.dirs['base'], d)

        self.dirs['fsl_templates'] = os.path.join(
            os.environ['FSLDIR'],
            'data/standard')

        # autogenerate all of the directories
        # except for the orig dir
        for d in dirs_to_add:
            if d != 'orig' and not os.path.exists(self.dirs[d]):
                os.mkdir(self.dirs[d])

        self.logfile = os.path.join(self.dirs['logs'], 'narps.txt')
        if not self.testing:
            log_to_file(
                self.logfile,
                'Running Narps main class',
                flush=True)

        output_dirs = ['resampled', 'rectified', 'zstat',
                       'thresh_mask_orig']

        for o in output_dirs:
            self.get_output_dir(o)

        # if raw data don't exist, download them
        if self.force_download and os.path.exists(self.dirs['orig']):
            shutil.rmtree(self.dirs['orig'])
        if not os.path.exists(self.dirs['orig']):
            self.get_orig_data()
        assert os.path.exists(self.dirs['orig'])

        # make sure the necessary templates are present
        # these should be downloaded with the raw data
        self.MNI_mask = os.path.join(self.dirs['fsl_templates'],
                                     'MNI152_T1_2mm_brain_mask.nii.gz')
        assert os.path.exists(self.MNI_mask)

        self.MNI_template = os.path.join(self.dirs['fsl_templates'],
                                         'MNI152_T1_2mm.nii.gz')
        assert os.path.exists(self.MNI_template)

        self.full_mask_img = os.path.join(self.dirs['templates'],
                                          'MNI152_all_voxels.nii.gz')

    def get_output_dir(self, dirID, base='output'):
        """get the directory path for a particular ID. if it doesn't
        exist then create it and save to the dirs list
        dir names always match the dir ID exactly
        """
        if dirID in self.dirs:
            return(self.dirs[dirID])
        else:
            self.dirs[dirID] = os.path.join(
                self.dirs[base],
                dirID
            )
            if not os.path.exists(self.dirs[dirID]):
                os.mkdir(self.dirs[dirID])
        return(self.dirs[dirID])

    def get_orig_data(self):
        """
        download original data from repository
        """
        log_to_file(
            self.logfile,
            'get_orig_data',
            headspace=2)
        log_to_file(self.logfile, 'DATA_URL: %s' % self.data_url)
        MAX_TRIES = 5

        if self.data_url is None:
            raise Exception('no URL for original data, cannot download')

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
        self.n_nan_inmask_values = {}
        self.n_zero_inmask_values = {}
        self.has_resampled = None
        self.has_binarized_masks = None

        # populate the image data structure
        self.get_orig_images()

        # check whether image needs to be rectified
        logfile = os.path.join(
            self.dirs.dirs['logs'],
            'image_diagnostics.log')
        collection_string = '%s_%s' % (self.NV_collection_id, self.teamID)
        self.image_diagnostics_file = os.path.join(
            self.dirs.dirs['image_diagnostics'],
            '%s.csv' % collection_string
        )
        if not os.path.exists(self.image_diagnostics_file):
            self.image_diagnostics = compare_thresh_unthresh_values(
                dirs, collection_string, logfile)
            self.image_diagnostics.to_csv(self.image_diagnostics_file)
        else:
            self.image_diagnostics = pandas.read_csv(
                self.image_diagnostics_file)
        # create a dict with the rectified values
        # use answers from spreadsheet
        self.rectify = {}
        for i in self.image_diagnostics.index:
            self.rectify[
                self.image_diagnostics.loc[
                    i, 'hyp']] = self.image_diagnostics.loc[
                        i, 'reverse_contrast']
        # manual fixes to rectify status per spreadsheet answers for hyp 9
        if self.teamID in ['R7D1', '46CD']:
            self.rectify[9] = True

    def get_orig_images(self):
        """
        find orig images
        """
        self.has_all_images = {
            'thresh': True,
            'unthresh': True}
        for hyp in hypotheses:
            for imgtype in self.images:
                imgfile = os.path.join(
                    self.input_dir,
                    'hypo%d_%s.nii.gz' % (hyp, imgtype))
                if os.path.exists(imgfile):
                    self.images[imgtype]['orig'][hyp] = imgfile
                else:
                    self.images[imgtype]['orig'][hyp] = None
                    self.has_all_images[imgtype] = False

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
            maskimg = os.path.join(
                    self.dirs.dirs['thresh_mask_orig'],
                    self.datadir_label,
                    os.path.basename(img))
            self.images['thresh']['thresh_mask_orig'][hyp] = maskimg
            if not os.path.exists(os.path.dirname(
                    maskimg)):
                os.mkdir(os.path.dirname(maskimg))
            if overwrite or not os.path.exists(maskimg):
                # load the image and threshold/binarize it
                threshimg = nibabel.load(img)
                threshdata = threshimg.get_data()
                # some images use nan instead of zero for the non-excursion
                # voxels, so we need to replace with zeros
                if replace_na:
                    threshdata = numpy.nan_to_num(threshdata)
                threshdata_bin = numpy.zeros(threshdata.shape)
                # fix teams with maps where 1 is null and zero is supra
                if self.teamID in FLIP_THRESH_MAPS:
                    if hyp in FLIP_THRESH_MAPS[self.teamID]:
                        threshdata_bin = -1 * (threshdata_bin - 1)
                # if the team reported using a negative contrast,
                # then we use the negative direction, otherwise
                # use the positive direction.
                # we use a small number instead of zero to address
                # numeric issues
                if self.rectify[hyp]:
                    # use negative
                    threshdata_bin[threshdata < -1*thresh] = 1
                else:
                    # use positive
                    threshdata_bin[threshdata > thresh] = 1

                # save back to a nifti image with same geometry
                # as original
                bin_img = nibabel.Nifti1Image(threshdata_bin,
                                              affine=threshimg.affine)
                bin_img.to_filename(maskimg)
            else:
                # if it already exists, just use existing
                if not os.path.exists(maskimg):
                    bin_img = nibabel.load(maskimg)
                    if self.verbose:
                        print('copying existing binary mask for',
                              self.teamID)

    def get_resampled_images(self, imgtype,
                             overwrite=False, replace_na=False):
        """
        resample images into common space using nilearn
        """
        self.has_resampled = True
        # use linear interpolation for binarized maps, then threshold at 0.5
        # this avoids empty voxels that can occur with NN interpolation
        interp_type = {'thresh': 'linear', 'unthresh': 'continuous'}
        data_dirname = {'thresh': 'thresh_mask_orig',
                        'unthresh': 'orig'}

        resampled_dir = self.dirs.get_output_dir('resampled')

        for hyp in hypotheses:
            infile = os.path.join(
                self.dirs.dirs[data_dirname[imgtype]],
                self.datadir_label,
                'hypo%d_%s.nii.gz' % (hyp, imgtype))
            outfile = os.path.join(
                resampled_dir,
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
                 verbose=False, overwrite=False,
                 dataurl=None, testing=False):
        self.basedir = basedir
        self.dirs = NarpsDirs(basedir, dataurl=dataurl,
                              testing=testing)
        self.verbose = verbose
        self.teams = {}
        self.overwrite = overwrite
        self.started_at = datetime.datetime.now()
        self.testing = testing

        # create the full mask image if it doesn't already exist
        if not os.path.exists(self.dirs.full_mask_img):
            print('making full image mask')
            self.mk_full_mask_img(self.dirs)
        assert os.path.exists(self.dirs.full_mask_img)

        # get input dirs for orig data
        self.image_jsons = None
        self.input_dirs = self.get_input_dirs(self.dirs)

        # check images for each team
        self.complete_image_sets = {}
        self.get_orig_images(self.dirs)
        for imgtype in ['thresh', 'unthresh']:
            log_to_file(
                self.dirs.logfile,
                'found %d teams with complete original %s datasets' % (
                    len(self.complete_image_sets[imgtype]), imgtype))

        # set up metadata
        if metadata_file is None:
            self.metadata_file = os.path.join(
                self.dirs.dirs['orig'],
                'analysis_pipelines_for_analysis.xlsx')
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
        d = numpy.ones(mi.shape)
        full_mask = nibabel.Nifti1Image(d, affine=mi.affine)
        full_mask.to_filename(self.dirs.full_mask_img)

    def get_input_dirs(self, dirs, verbose=True, load_json=True):
        """
        get orig dirs
        - assumes that images.json is present for each valid dir
        """

        input_files = glob.glob(
            os.path.join(dirs.dirs['orig'], '*/hypo1_*thresh.nii.gz'))
        input_dirs = [os.path.dirname(i) for i in input_files]
        input_dirs = list(set(input_dirs))  # get unique dirs

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
        self.complete_image_sets = {
            'thresh': [],
            'unthresh': []}
        for teamID in self.teams:
            self.teams[teamID].get_orig_images()
            for imgtype in self.teams[teamID].images:
                if self.teams[teamID].has_all_images[imgtype]:
                    self.complete_image_sets[imgtype].append(teamID)

        # sort the teams - this is the order that will be used
        for imgtype in self.teams[teamID].images:
            self.complete_image_sets[imgtype].sort()

    def get_binarized_thresh_masks(self):
        """
        create binarized thresholded maps for each team
        """
        log_to_file(
            self.dirs.logfile,
            sys._getframe().f_code.co_name,
            headspace=2)
        for teamID in self.complete_image_sets['thresh']:
            self.teams[teamID].create_binarized_thresh_masks()

    def get_resampled_images(self, overwrite=None):
        """
        resample all images into FSL MNI space
        """
        log_to_file(
            self.dirs.logfile,
            sys._getframe().f_code.co_name,
            headspace=2)
        if overwrite is None:
            overwrite = self.overwrite
        for imgtype in ['thresh', 'unthresh']:
            for teamID in self.complete_image_sets[imgtype]:
                self.teams[teamID].get_resampled_images(imgtype=imgtype)

    def check_image_values(self, overwrite=None):
        """
        get # of nonzero and NA voxels for each image
        """
        log_to_file(
            self.dirs.logfile,
            sys._getframe().f_code.co_name,
            headspace=2)
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
        for teamID in self.complete_image_sets['thresh']:
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
                             create_voxel_map=False,
                             imgtypes=None,
                             overwrite=None):
        """
        create images concatenated across teams
        ordered by self.complete_image_sets
        create_voxel_map: will create a map showing
        proportion of nonzero teams at each voxel
        """
        log_to_file(
            self.dirs.logfile,
            sys._getframe().f_code.co_name,
            headspace=2)
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
            concat_dir = self.dirs.get_output_dir(
                '%s_concat_%s' % (imgtype, datatype))
            for hyp in range(1, 10):
                outfile = os.path.join(
                    concat_dir,
                    'hypo%d.nii.gz' % hyp)
                if self.verbose:
                    print(outfile)
                if not os.path.exists(outfile) or overwrite:
                    if self.verbose:
                        print('%s - hypo %d: creating concat file' % (
                            imgtype, hyp))
                    concat_teams = [
                        teamID for teamID in self.complete_image_sets[imgtype]
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
                    if create_voxel_map:
                        concat_data = nibabel.load(outfile).get_data()
                        voxel_map = numpy.mean(
                            numpy.abs(concat_data) > 1e-6, 3)
                        voxel_img = nibabel.Nifti1Image(
                            voxel_map, affine=concat_img.affine)
                        mapfile = outfile.replace(
                            '.nii.gz', '_voxelmap.nii.gz'
                        )
                        assert mapfile != outfile
                        voxel_img.to_filename(mapfile)

                    # save team ID and files to a label file for provenance
                    labelfile = outfile.replace('.nii.gz', '.labels')
                    with open(labelfile, 'w') as f:
                        for i, team in enumerate(concat_teams):
                            f.write('%s\t%s%s' % (
                                team,
                                self.all_maps[imgtype][datatype][i],
                                os.linesep))
                else:
                    if self.verbose:
                        print('%s - hypo %d: using existing file' % (
                            imgtype, hyp))
        return(self.all_maps)

    def create_mean_thresholded_images(self, datatype='resampled',
                                       overwrite=None, thresh=1e-5):
        """
        create overlap maps for thresholded images
        """
        log_to_file(
            self.dirs.logfile,
            sys._getframe().f_code.co_name,
            headspace=2)
        func_args = inspect.getargvalues(
            inspect.currentframe()).locals
        log_to_file(
            self.dirs.logfile,
            stringify_dict(func_args))

        imgtype = 'thresh'
        if overwrite is None:
            overwrite = self.overwrite
        output_dir = self.dirs.get_output_dir('overlap_binarized_thresh')
        concat_dir = self.dirs.get_output_dir(
            '%s_concat_%s' % (imgtype, datatype))

        for hyp in range(1, 10):
            outfile = os.path.join(
                output_dir,
                'hypo%d.nii.gz' % hyp)
            if not os.path.exists(outfile) or overwrite:
                if self.verbose:
                    print('%s - hypo %d: creating overlap file' % (
                        imgtype, hyp))
                concat_file = os.path.join(
                    concat_dir,
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
            self.dirs.logfile,
            sys._getframe().f_code.co_name,
            headspace=2)
        func_args = inspect.getargvalues(
            inspect.currentframe()).locals
        log_to_file(
            self.dirs.logfile,
            stringify_dict(func_args))

        if overwrite is None:
            overwrite = self.overwrite
        for teamID in self.complete_image_sets['unthresh']:
            if not hasattr(self.teams[teamID], 'rectify'):
                print('no rectification data for %s, skipping' % teamID)
                continue
            for hyp in range(1, 10):
                if hyp not in self.teams[teamID].rectify:
                    print('no rectification data for %s hyp%d, skipping' % (
                        teamID, hyp))
                    continue
                rectify = self.teams[teamID].rectify[hyp]
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
                    f.write('%s\t%s%s' % (l[0], l[1], os.linesep))

    def compute_image_stats(self, datatype='zstat', overwrite=None):
        """
        compute std and range on statistical images
        """
        log_to_file(
            self.dirs.logfile,
            sys._getframe().f_code.co_name,
            headspace=2)
        func_args = inspect.getargvalues(
            inspect.currentframe()).locals
        log_to_file(
            self.dirs.logfile,
            stringify_dict(func_args))

        if overwrite is None:
            overwrite = self.overwrite

        # set up directories
        unthresh_concat_dir = self.dirs.get_output_dir(
            'unthresh_concat_%s' % datatype)
        unthresh_range_dir = self.dirs.get_output_dir(
            'unthresh_range_%s' % datatype)
        unthresh_std_dir = self.dirs.get_output_dir(
            'unthresh_std_%s' % datatype)

        for hyp in range(1, 10):

            unthresh_file = os.path.join(
                unthresh_concat_dir,
                'hypo%d.nii.gz' % hyp)

            range_outfile = os.path.join(
                unthresh_range_dir,
                'hypo%d.nii.gz' % hyp)

            std_outfile = os.path.join(
                unthresh_std_dir,
                'hypo%d.nii.gz' % hyp)

            if not os.path.exists(range_outfile) \
                    or not os.path.exists(std_outfile) \
                    or overwrite:
                unthresh_img = nibabel.load(unthresh_file)
                unthresh_data = unthresh_img.get_data()
                concat_data = numpy.nan_to_num(unthresh_data)

                # compute range
                datarange = numpy.max(concat_data, axis=3) \
                    - numpy.min(concat_data, axis=3)
                range_img = nibabel.Nifti1Image(
                    datarange,
                    affine=unthresh_img.affine)
                range_img.to_filename(range_outfile)

                # compute standard deviation
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
            self.dirs.logfile,
            sys._getframe().f_code.co_name,
            headspace=2)
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
                'narps_neurovault_images_details_responses_corrected.csv')
        print('using map_metadata_file:', map_metadata_file)
        unthresh_stat_type = get_map_metadata(map_metadata_file)
        metadata = get_metadata(self.metadata_file)

        n_participants = metadata[['n_participants', 'NV_collection_string']]

        n_participants.index = metadata.teamID

        unthresh_stat_type = unthresh_stat_type.merge(
            n_participants, left_index=True, right_index=True)

        for teamID in self.complete_image_sets['unthresh']:
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
            self.dirs.logfile,
            sys._getframe().f_code.co_name,
            headspace=2)
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
        for teamID in self.complete_image_sets['unthresh']:
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

        for teamID in self.complete_image_sets['thresh']:
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
        for teamID in self.complete_image_sets['thresh']:
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
