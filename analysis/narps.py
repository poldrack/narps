# main class for narps analysis

import numpy,pandas
import nibabel
import json
import os,glob
import nilearn.image
import nilearn.input_data
import nilearn.plotting
from collections import OrderedDict
import shutil
import warnings
import sklearn
import matplotlib.pyplot as plt
import seaborn
import pickle
from utils import get_masked_data,get_metadata,get_teamID_to_collectionID_dict,TtoZ

hypotheses= {1:'+gain: equal indiff',
            2:'+gain: equal range',
            3:'+gain: equal indiff',
            4:'+gain: equal range',
            5:'-loss: equal indiff',
            6:'-loss: equal range',
            7:'+loss: equal indiff',
            8:'+loss: equal range',
            9:'+loss:ER>EI'}

hypnums = [1,2,5,6,7,8,9]

# separate class to store base directories, since we need them in multiple places
class NarpsDirs(object):
    def __init__(self,basedir):
        # set up directories and template files
        self.dirs = {}
        self.dirs['base'] = basedir
        assert os.path.exists(basedir)

        self.dirs['output'] = os.path.join(self.dirs['base'],'maps')
        assert os.path.exists(self.dirs['output'])
 
        output_dirs = ['orig','resampled','rectified','zstat','thresh_mask_orig','templates']
        required = ['orig'] # must exist in order to run
        for o in output_dirs:
            self.dirs[o] = os.path.join(self.dirs['output'],o)
            if o in required:
                assert os.path.exists(self.dirs[o])
            else:
                if not os.path.exists(self.dirs[o]):
                    os.mkdir(self.dirs[o])

       
        self.MNI_mask = os.path.join(self.dirs['templates'],'MNI152_T1_2mm_brain_mask.nii.gz')
        assert os.path.exists(self.MNI_mask)

        self.MNI_template = os.path.join(self.dirs['templates'],'MNI152_T1_2mm.nii.gz')
        assert os.path.exists(self.MNI_template)

        self.full_mask_img = os.path.join(self.dirs['templates'],'MNI152_all_voxels.nii.gz')


class NarpsTeam(object):
    def __init__(self,teamID,NV_collection_id,dirs,verbose=False):
        self.dirs = dirs
        self.teamID = teamID
        self.NV_collection_id = NV_collection_id
        self.datadir_label = '%s_%s'%(NV_collection_id,teamID)
        self.input_dir = os.path.join(self.dirs.dirs['orig'],'%s_%s'%(NV_collection_id,teamID))
        assert os.path.exists(self.input_dir)

        self.verbose = verbose
        self.image_json = None
        self.jsonfile = None
        self.NV_collection_id = None
        self.has_all_images = None
        # create image directory structure
        output_dirs = {'thresh':['orig','resampled','thresh_mask_orig'],
                        'unthresh':['orig','resampled','rectified','zstat']}
        self.images = {}
        for imgtype in ['thresh','unthresh']:
            self.images[imgtype]={}
            for o in output_dirs[imgtype]:
                self.images[imgtype][o]={}
        self.n_mask_vox = {}
        self.n_nan_inmask_values = {}
        self.n_zero_inmask_values = {}
        self.get_orig_images()
        self.has_resampled = None
        self.has_binarized_masks = None

    def get_orig_images(self):
        self.has_all_images = True
        for hyp in hypotheses:
            for imgtype in self.images:
                imgfile = os.path.join(self.input_dir,'hypo%d_%s.nii.gz'%(hyp,imgtype))
                if os.path.exists(imgfile):
                    self.images[imgtype]['orig'][hyp] = imgfile
                else:
                    self.images[imgtype]['orig'][hyp] = None
                    self.has_all_images = False
        
    def create_binarized_thresh_masks(self,thresh=1e-6,overwrite=False,replace_na=False):
         self.has_binarized_masks = True
         if self.verbose:
             print('creating binarized masks for',self.teamID)
         
         for hyp in self.images['thresh']['orig']:
            img = self.images['thresh']['orig'][hyp]
            self.images['thresh']['thresh_mask_orig'][hyp] = img.replace("orig",'thresh_mask_orig')
            if not os.path.exists(os.path.dirname(self.images['thresh']['thresh_mask_orig'][hyp])):
                os.mkdir(os.path.dirname(self.images['thresh']['thresh_mask_orig'][hyp]))
            if not os.path.exists(self.images['thresh']['thresh_mask_orig'][hyp]) or overwrite:
                threshimg = nibabel.load(img)
                threshdata = threshimg.get_data()
                if replace_na: # probably don't want to do this, false by default
                    threshdata = numpy.nan_to_num(threshdata)
                threshdata_bin = numpy.zeros(threshdata.shape)
                threshdata_bin[numpy.abs(threshdata)>thresh]=1
                bin_img = nibabel.Nifti1Image(threshdata_bin,affine=threshimg.affine)
                bin_img.to_filename(self.images['thresh']['thresh_mask_orig'][hyp])
            else:
                bin_img = nibabel.load(self.images['thresh']['thresh_mask_orig'][hyp])
                if self.verbose:
                    print('using existing binary mask for',self.teamID)
                threshdata_bin = bin_img.get_data()
            self.n_mask_vox[hyp]=numpy.sum(threshdata_bin)


       

    def get_resampled_images(self,overwrite=False,replace_na=False):
        self.has_resampled = True
        # use linear interpolation for binarized maps, then threshold at 0.5
        # this avoids empty voxels that can occur with NN interpolation
        interp_type={'thresh':'linear','unthresh':'continuous'}

        for hyp in hypotheses:
            for imgtype in self.images:
                infile = os.path.join(self.dirs.dirs['orig'],self.datadir_label,'hypo%d_%s.nii.gz'%(hyp,imgtype))
                outfile = os.path.join(self.dirs.dirs['resampled'],self.datadir_label,'hypo%d_%s.nii.gz'%(hyp,imgtype))
                self.images[imgtype]['resampled'][hyp] = outfile
                if not os.path.exists(os.path.dirname(outfile)):
                    os.mkdir(os.path.dirname(outfile))
                if not os.path.exists(outfile) or overwrite:
                    if self.verbose:
                        print("resampling",infile)
                    # create resampled file
                    with warnings.catch_warnings(): # ignore nilearn warnings
                        warnings.simplefilter("ignore")
                        resampled = nilearn.image.resample_to_img(infile, self.dirs.MNI_template,
                                                    interpolation=interp_type[imgtype])

                    if imgtype=='thresh':
                        resampled = nilearn.image.math_img('img>0.5', img=resampled)

                    resampled.to_filename(outfile)

                else:
                    if self.verbose:
                        print('using existing resampled image for',self.teamID)

            


class Narps(object):
    def __init__(self,basedir,metadata_file=None):
        self.basedir = basedir
        assert os.path.exists(self.basedir)
        self.dirs = NarpsDirs(basedir)
 
        self.teams = {}

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
        print('found %d teams with complete original datasets'%len(self.complete_image_sets))

        # set up metadata
        if metadata_file is None:
            self.metadata_file = os.path.join(self.dirs.dirs['base'],'analysis_pipelines_SW.xlsx')
        else:
            self.metadata_file = metadata_file

        self.metadata = get_metadata(self.metadata_file)

        self.hypothesis_metadata = pandas.DataFrame(columns=['teamID','hyp','n_na','n_zero'])

        self.all_maps = {'thresh':{'resampled':None},
                         'unthresh':{'resampled':None}}
        self.rectified_list = []

    def mk_full_mask_img(self,dirs):
        # make full image mask (all voxels)

        mi = nibabel.load(self.dirs.MNI_mask)
        d = mi.get_data()
        d = numpy.ones(d.shape)
        full_mask = nibabel.Nifti1Image(d,affine = mi.affine)
        full_mask.to_filename(self.dirs.full_mask_img)

    # get orig dirs - assumes that images.json is present for each valid dir
    def get_input_dirs(self,dirs,verbose=True,load_json=True):
        input_jsons = glob.glob(os.path.join(dirs.dirs['orig'],'*/images.json'))
        if verbose:
            print('found',len(input_jsons),'input directories')
        if load_json:
            for i in input_jsons:
                collection_id = os.path.basename(os.path.dirname(i))
                NV_collection_id,teamID = collection_id.split('_')
                if not teamID in self.teams:
                    self.teams[teamID]=NarpsTeam(teamID,NV_collection_id,dirs)
                    self.teams[teamID].jsonfile = i
                    self.teams[teamID].NV_collection_id = collection_id
                with open(i) as f:
                    self.teams[teamID].image_json = json.load(f)

    def get_orig_images(self,dirs):
        self.complete_image_sets = []
        for teamID in self.teams:
            self.teams[teamID].get_orig_images()
            if self.teams[teamID].has_all_images:
                self.complete_image_sets.append(teamID)
        # sort the teams - this is the order that will be used
        self.complete_image_sets.sort()

    def get_binarized_thresh_masks(self):
        print('getting binarized/thresholded orig maps')
        for teamID in self.complete_image_sets:
            self.teams[teamID].create_binarized_thresh_masks()
       

    # resample all images into FSL MNI space
    def get_resampled_images(self,overwrite=False):
        print("getting resampled images...")
        for teamID in self.complete_image_sets:
            self.teams[teamID].get_resampled_images()

    # get # of nonzero and NA voxels for each image
    def check_image_values(self,overwrite = False):
        image_metadata_file = os.path.join(self.dirs.dirs['output'],'image_metadata_df.csv')
        if os.path.exists(image_metadata_file) and not overwrite:
            print('using cached image metdata')
            image_metadata_df = pandas.read_csv(image_metadata_file)
            return(image_metadata_df)
        # otherwise load from scractch
        image_metadata = []
        print("checking image values...")
        masker=nilearn.input_data.NiftiMasker(mask_img=self.dirs.MNI_mask)
        for teamID in self.complete_image_sets:
            for hyp in self.teams[teamID].images['thresh']['resampled']:
                threshfile = self.teams[teamID].images['thresh']['resampled'][hyp] 
                threshdata = masker.fit_transform(threshfile)
                image_metadata.append([teamID,hyp,numpy.sum(numpy.isnan(threshdata)),
                                numpy.sum(threshdata==0.0)])
    
        image_metadata_df = pandas.DataFrame(image_metadata,columns=['teamID','hyp','n_na','n_nonzero'])
        image_metadata_df.to_csv(image_metadata_file)
        return(image_metadata_df)

    # create images concatenated across teams
    # ordered by self.complete_image_sets
    def create_concat_images(self,datatype='resampled',imgtypes = ['thresh','unthresh'],
                                overwrite=False):
        for imgtype in imgtypes:
            self.dirs.dirs['concat_%s'%imgtype]=os.path.join(self.dirs.dirs['output'],'%s_concat_%s'%(imgtype,datatype))
            for hyp in range(1,10):
                outfile = os.path.join(self.dirs.dirs['concat_%s'%imgtype],'hypo%d.nii.gz'%hyp)
                if not os.path.exists(os.path.dirname(outfile)):
                    os.mkdir(os.path.dirname(outfile))
                if not os.path.exists(outfile) or overwrite:
                    print('%s - hypo %d: creating concat file'%(imgtype,hyp))
                    self.all_maps[imgtype][datatype] = [self.teams[teamID].images[imgtype][datatype][hyp] for teamID in self.complete_image_sets]
                    masker = nilearn.input_data.NiftiMasker(mask_img=self.dirs.MNI_mask)
                    concat_data = masker.fit_transform(self.all_maps[imgtype][datatype])
                    concat_img = masker.inverse_transform(concat_data)
                    concat_img.to_filename(outfile)
                else:
                    print('%s - hypo %d: using existing file'%(imgtype,hyp))

    # create overlap maps for thresholded iamges
    def create_thresh_overlap_images(self,datatype='resampled',overwrite=True,thresh=10e-6):
        imgtype = 'thresh'
        self.dirs.dirs['overlap_binarized_thresh']=os.path.join(self.dirs.dirs['output'],'overlap_binarized_thresh')
        for hyp in range(1,10):
            outfile = os.path.join(self.dirs.dirs['overlap_binarized_thresh'],'hypo%d.nii.gz'%hyp)
            if not os.path.exists(os.path.dirname(outfile)):
                os.mkdir(os.path.dirname(outfile))
            if not os.path.exists(outfile) or overwrite:
                print('%s - hypo %d: creating overlap file'%(imgtype,hyp))
                concat_file = os.path.join(self.dirs.dirs['concat_thresh'],'hypo%d.nii.gz'%hyp)
                concat_img=nibabel.load(concat_file)
                concat_data = concat_img.get_data()
                concat_data = (concat_data>thresh).astype('float')
                concat_mean = numpy.mean(concat_data,3)
                concat_mean_img = nibabel.Nifti1Image(concat_mean,affine=concat_img.header.get_best_affine())
                concat_mean_img.to_filename(outfile)

            else:
                print('%s - hypo %d: using existing file'%(imgtype,hyp))

    # create rectified images 
    # - for any maps where the signal within the thresholded mask is completely negative, rectify the
    # unthresholded map (multiply by -1)
    def create_rectified_images(self,overwrite=True):
        for teamID in self.complete_image_sets:
            for hyp in range(1,10):
                rectify=False
                # load data from unthresh map within positive voxels of thresholded mask
                thresh_file = self.teams[teamID].images['thresh']['thresh_mask_orig'][hyp]
                masker=nilearn.input_data.NiftiMasker(mask_img=thresh_file)
                unthresh_file = self.teams[teamID].images['unthresh']['orig'][hyp]
                # need to catch ValueError that occurs is mask is completely empty -
                # in that case just copy over the data
                try:
                    masked_data = masker.fit_transform(unthresh_file)
                    if numpy.max(masked_data)<=0:
                        rectify=True
                except ValueError:
                    pass

                outfile = os.path.join(self.dirs.dirs['rectified'],
                        self.teams[teamID].NV_collection_id,'hypo%d_unthresh.nii.gz'%hyp)
                if not os.path.exists(os.path.dirname(outfile)):
                    os.mkdir(os.path.dirname(outfile))
                if not os.path.exists(outfile) or overwrite:
                    if rectify:  # all values are negative
                            print('rectifying',self.teams[teamID].NV_collection_id,hyp)
                            img = nibabel.load(unthresh_file)
                            img_rectified = nilearn.image.math_img('img*-1',img=img)
                            img_rectified.to_filename(outfile)
                            self.rectified_list.append((teamID,hyp))
                    else:
                        shutil.copy(unthresh_file,outfile)

    # compute stats on rectified images - std deviation and range
    def compute_image_stats(self,datatype='rectified'):
        return None

    


class TestNarps(object):
    def test_narps_dirs(self):
        narpsDirs = NarpsDirs("/Users/poldrack/data_unsynced/NARPS")
    def test_narps_team(self):
        narpsDirs = NarpsDirs("/Users/poldrack/data_unsynced/NARPS")
        narpsTeam = NarpsTeam('C88N','ADFZYYLQ',narpsDirs)
    def test_narps_main_class(self):
        narps = Narps("/Users/poldrack/data_unsynced/NARPS")

if __name__ == "__main__":
    narpsDirs = NarpsDirs("/Users/poldrack/data_unsynced/NARPS")
    narps = Narps('/Users/poldrack/data_unsynced/NARPS')
    narps.get_binarized_thresh_masks()
    narps.get_resampled_images()
    image_metadata_df = narps.check_image_values()
    narps.create_concat_images()
    narps.create_rectified_images()


