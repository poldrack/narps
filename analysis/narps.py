# main class for narps analysis

import numpy,pandas
import nibabel
import json
import os,glob,datetime
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
from nipype.interfaces.fsl.model import SmoothEstimate

from utils import get_masked_data,get_metadata,get_teamID_to_collectionID_dict,TtoZ,get_map_metadata


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
        self.dirs['metadata'] = os.path.join(self.dirs['base'],'metadata')
        self.dirs['cached'] = os.path.join(self.dirs['base'],'cached')
        self.dirs['orig'] = os.path.join(self.dirs['base'],'orig')
        self.dirs['templates'] = os.path.join(self.dirs['base'],'templates')

        assert os.path.exists(self.dirs['output'])
        assert os.path.exists(self.dirs['templates'])
        if not os.path.exists(self.dirs['cached']):
            os.mkdir(self.dirs['cached'])

 
        output_dirs = ['resampled','rectified','zstat','thresh_mask_orig','concat_thresh']
        for o in output_dirs:
            self.dirs[o] = os.path.join(self.dirs['output'],o)
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
        if not os.path.exists(self.input_dir):
            print("Warning: Input dir (%s) does not exist"%self.input_dir)

        self.verbose = verbose
        self.image_json = None
        self.jsonfile = None
        self.has_all_images = None
        self.logs = {}

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
            self.images['thresh']['thresh_mask_orig'][hyp] = os.path.join(self.dirs.dirs['thresh_mask_orig'],os.path.basename(img))
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
    def __init__(self,basedir,metadata_file=None,verbose=False,overwrite=False):
        self.basedir = basedir
        assert os.path.exists(self.basedir)
        self.dirs = NarpsDirs(basedir)
        self.verbose = verbose
        self.teams = {}
        self.overwrite=overwrite
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
        print('found %d teams with complete original datasets'%len(self.complete_image_sets))

        # set up metadata
        if metadata_file is None:
            self.metadata_file = os.path.join(self.dirs.dirs['orig'],'analysis_pipelines_SW.xlsx')
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
                    self.teams[teamID]=NarpsTeam(teamID,NV_collection_id,dirs,verbose = self.verbose)
                    self.teams[teamID].jsonfile = i
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
        for teamID in self.complete_image_sets:
            self.teams[teamID].create_binarized_thresh_masks()
       

    # resample all images into FSL MNI space
    def get_resampled_images(self,overwrite=None):
        if overwrite is None:
            overwrite = self.overwrite
        for teamID in self.complete_image_sets:
            self.teams[teamID].get_resampled_images()

    # get # of nonzero and NA voxels for each image
    def check_image_values(self,overwrite = None):
        if overwrite is None:
            overwrite = self.overwrite
        image_metadata_file = os.path.join(self.dirs.dirs['metadata'],'image_metadata_df.csv')
        if os.path.exists(image_metadata_file) and not overwrite:
            print('using cached image metdata')
            image_metadata_df = pandas.read_csv(image_metadata_file)
            return(image_metadata_df)
        # otherwise load from scractch
        image_metadata = []
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
                                overwrite=None):
        if overwrite is None:
            overwrite = self.overwrite
        for imgtype in imgtypes:
            self.dirs.dirs['concat_%s'%imgtype]=os.path.join(self.dirs.dirs['output'],'%s_concat_%s'%(imgtype,datatype))
            for hyp in range(1,10):
                outfile = os.path.join(self.dirs.dirs['concat_%s'%imgtype],'hypo%d.nii.gz'%hyp)
                if not os.path.exists(os.path.dirname(outfile)):
                    os.mkdir(os.path.dirname(outfile))
                if not os.path.exists(outfile) or overwrite:
                    if self.verbose:
                        print('%s - hypo %d: creating concat file'%(imgtype,hyp))
                    concat_teams = [teamID for teamID in self.complete_image_sets if os.path.exists(self.teams[teamID].images[imgtype][datatype][hyp])]
                    self.all_maps[imgtype][datatype] = [self.teams[teamID].images[imgtype][datatype][hyp] for teamID in concat_teams]
                    masker = nilearn.input_data.NiftiMasker(mask_img=self.dirs.MNI_mask)
                    concat_data = masker.fit_transform(self.all_maps[imgtype][datatype])
                    concat_img = masker.inverse_transform(concat_data)
                    concat_img.to_filename(outfile)
                else:
                    if self.verbose:
                        print('%s - hypo %d: using existing file'%(imgtype,hyp))
        return(self.all_maps)
    # create overlap maps for thresholded iamges
    def create_thresh_overlap_images(self,datatype='resampled',overwrite=None,thresh=10e-6):
        imgtype = 'thresh'
        if overwrite is None:
            overwrite = self.overwrite
        self.dirs.dirs['overlap_binarized_thresh']=os.path.join(self.dirs.dirs['output'],'overlap_binarized_thresh')
        for hyp in range(1,10):
            outfile = os.path.join(self.dirs.dirs['overlap_binarized_thresh'],'hypo%d.nii.gz'%hyp)
            if not os.path.exists(os.path.dirname(outfile)):
                os.mkdir(os.path.dirname(outfile))
            if not os.path.exists(outfile) or overwrite:
                if self.verbose:
                    print('%s - hypo %d: creating overlap file'%(imgtype,hyp))
                concat_file = os.path.join(self.dirs.dirs['concat_thresh'],'hypo%d.nii.gz'%hyp)
                concat_img=nibabel.load(concat_file)
                concat_data = concat_img.get_data()
                concat_data = (concat_data>thresh).astype('float')
                concat_mean = numpy.mean(concat_data,3)
                concat_mean_img = nibabel.Nifti1Image(concat_mean,affine=concat_img.affine)
                concat_mean_img.to_filename(outfile)

            else:
                if self.verbose:
                    print('%s - hypo %d: using existing file'%(imgtype,hyp))

    # create rectified images 
    # use metadata provided by teams
    # - originally used data-driven method: 
    # for any maps where the signal within the thresholded mask is completely negative, rectify the
    # unthresholded map (multiply by -1)
    def create_rectified_images(self,map_metadata_file = None,overwrite=None):
        if map_metadata_file is None:
            map_metadata_file = os.path.join(self.dirs.dirs['orig'], 'narps_neurovault_images_details.csv')
        map_metadata = get_map_metadata(map_metadata_file)
        if overwrite is None:
            overwrite = self.overwrite
        for teamID in self.complete_image_sets:
            for hyp in range(1,10): 
                if hyp in [5,6]:
                    mdstring = map_metadata.query('teamID == "%s"'%teamID)['hyp%d_direction'%hyp].iloc[0]
                    rectify = mdstring.split()[0] == 'Negative'
                elif hyp == 9:
                    # manual fix for one team with reversed maps
                    if teamID in ['R7D1']:
                        mdstring = map_metadata.query('teamID == "%s"'%teamID)['hyp%d_direction'%hyp].iloc[0]
                        rectify=True
                else:  # just copy the other hypotheses directly
                    rectify = False

                # load data from unthresh map within positive voxels of thresholded mask
                thresh_file = self.teams[teamID].images['thresh']['resampled'][hyp]
                masker=nilearn.input_data.NiftiMasker(mask_img=self.dirs.MNI_mask)
                unthresh_file = self.teams[teamID].images['unthresh']['resampled'][hyp]
                # need to catch ValueError that occurs is mask is completely empty -
                # in that case just copy over the data
                
                self.teams[teamID].images['unthresh']['rectified'][hyp] = os.path.join(self.dirs.dirs['rectified'],
                        self.teams[teamID].datadir_label,'hypo%d_unthresh.nii.gz'%hyp)
                
                if not os.path.exists(os.path.dirname(self.teams[teamID].images['unthresh']['rectified'][hyp])):
                    os.mkdir(os.path.dirname(self.teams[teamID].images['unthresh']['rectified'][hyp]))
                if not os.path.exists(self.teams[teamID].images['unthresh']['rectified'][hyp]) or overwrite:
                    if rectify:  # values were flipped for negative contrasts
                            print('rectifying hyp',hyp,'for',teamID)
                            print(mdstring)
                            print('')
                            img = nibabel.load(unthresh_file)
                            img_rectified = nilearn.image.math_img('img*-1',img=img)
                            img_rectified.to_filename(self.teams[teamID].images['unthresh']['rectified'][hyp])
                            self.rectified_list.append((teamID,hyp))
                    else:
                        shutil.copy(unthresh_file,self.teams[teamID].images['unthresh']['rectified'][hyp])
        # write list of rectified teams to disk
        if len(self.rectified_list)>0:
            with open(os.path.join(self.dirs.dirs['metadata'],'rectified_images_list.txt'),'w') as f:
                for l in self.rectified_list:
                    f.write('%s\t%s\n'%(l[0],l[1]))
        
    # compute std and range on statistical images
    def compute_image_stats(self,datatype='zstat',overwrite=None):
        if overwrite is None:
            overwrite = self.overwrite
        for teamID in self.complete_image_sets:
            for hyp in range(1,10):

                unthresh_file = os.path.join(self.dirs.dirs['output'],'unthresh_concat_%s/hypo%d.nii.gz'%(datatype,hyp))

                range_outfile=os.path.join(self.dirs.dirs['output'],'unthresh_range_%s/hypo%d.nii.gz'%(datatype,hyp))
                if not os.path.exists(os.path.join(self.dirs.dirs['output'],'unthresh_range_%s'%datatype)):
                    os.mkdir(os.path.join(self.dirs.dirs['output'],'unthresh_range_%s'%datatype))

                std_outfile=os.path.join(self.dirs.dirs['output'],'unthresh_std_%s/hypo%d.nii.gz'%(datatype,hyp))
                if not os.path.exists(os.path.join(self.dirs.dirs['output'],'unthresh_std_%s'%datatype)):
                    os.mkdir(os.path.join(self.dirs.dirs['output'],'unthresh_std_%s'%datatype))

                if not os.path.exists(range_outfile) or not os.path.exists(std_outfile) or overwrite:
                    unthresh_img = nibabel.load(unthresh_file)
                    unthresh_data = unthresh_img.get_data()
                    concat_data=numpy.nan_to_num(unthresh_data)
                    datarange = numpy.max(concat_data,axis=3) - numpy.min(concat_data,axis=3)
                    range_img = nibabel.Nifti1Image(datarange,affine=unthresh_img.affine)
                    range_img.to_filename(range_outfile)
                    datastd = numpy.std(concat_data,axis=3)
                    std_img = nibabel.Nifti1Image(datastd,affine=unthresh_img.affine)
                    std_img.to_filename(std_outfile)

    # convert rectified images to z scores - if they are already z then just copy
    # use metadata supplied by teams to determine image type
    def convert_to_zscores(self,map_metadata_file=None,overwrite=None):
        if overwrite is None:
            overwrite = self.overwrite
        if map_metadata_file is None:
            map_metadata_file = os.path.join(self.dirs.dirs['orig'],'narps_neurovault_images_details.csv')
        unthresh_stat_type = get_map_metadata(map_metadata_file)
        metadata = get_metadata(self.metadata_file)
        
        n_participants=metadata[['n_participants','NV_collection_string']]

        n_participants.index = metadata.teamID

        unthresh_stat_type = unthresh_stat_type.merge(n_participants,left_index=True,right_index=True)
            
        for teamID in self.complete_image_sets:
            if not teamID in unthresh_stat_type.index:
                print('no map metadata for',teamID)
                continue
            n=unthresh_stat_type.loc[teamID,'n_participants']
            collection = unthresh_stat_type.loc[teamID,'NV_collection_string']
            for hyp in range(1,10):
                infile = self.teams[teamID].images['unthresh']['rectified'][hyp]
                if not os.path.exists(infile):
                    print('skipping',infile)
                    continue
                self.teams[teamID].images['unthresh']['zstat'][hyp] = os.path.join(self.dirs.dirs['zstat'],
                                        self.teams[teamID].datadir_label, 'hypo%d_unthresh.nii.gz'%hyp)
                if os.path.exists(self.teams[teamID].images['unthresh']['zstat'][hyp]) and not overwrite:
                    continue
                
                if unthresh_stat_type.loc[teamID,'unthresh_type'].lower() == 't':
                    if not os.path.exists(os.path.dirname(self.teams[teamID].images['unthresh']['zstat'][hyp])):
                        os.mkdir(os.path.dirname(self.teams[teamID].images['unthresh']['zstat'][hyp]))
                    print("converting %s (hyp %d) to z - %d participants"%(teamID,hyp,n))
                    TtoZ(infile,self.teams[teamID].images['unthresh']['zstat'][hyp],n-1)
                elif unthresh_stat_type.loc[teamID,'unthresh_type'] == 'z':
                    if not os.path.exists(os.path.dirname(self.teams[teamID].images['unthresh']['zstat'][hyp])):
                        os.mkdir(os.path.dirname(self.teams[teamID].images['unthresh']['zstat'][hyp]))
                    if not os.path.exists(self.teams[teamID].images['unthresh']['zstat'][hyp]):
                        print('copying',teamID)
                        shutil.copy(infile,
                            os.path.dirname(self.teams[teamID].images['unthresh']['zstat'][hyp]))
                else:
                    print('skipping %s - other data type'%teamID)

    # estimate smoothness of Z maps using FSL's smoothness estimation
    def estimate_smoothness(self,overwrite=None,imgtype='zstat'):
        if overwrite is None:
            overwrite = self.overwrite
        output_file = os.path.join(self.dirs.dirs['metadata'],'smoothness_est.csv') 
        if os.path.exists(output_file) and not overwrite:
            if self.verbose:
                print('using existing smoothness file')
            smoothness_df = pandas.read_csv(output_file)
            return(smoothness_df)
        
        est = SmoothEstimate()
        smoothness = []
        for teamID in self.complete_image_sets:
            for hyp in range(1,10):
                if not hyp in self.teams[teamID].images['unthresh'][imgtype]:
                    # fill missing data with nan
                    print('no zstat present for',teamID,hyp)
                    smoothness.append([teamID,hyp,numpy.nan,
                                                        numpy.nan,
                                                        numpy.nan])
                    continue
                infile = self.teams[teamID].images['unthresh'][imgtype][hyp]
                if not os.path.exists(infile):
                    print('no image present:',infile)
                    continue
                else:
                    if self.verbose:
                        print('estimating smoothness for hyp',hyp)
                    est.inputs.zstat_file = infile
                    est.inputs.mask_file = self.dirs.MNI_mask
                    est.terminal_output = 'file_split'
                    smoothest_output = est.run()
                    smoothness.append([teamID,hyp,smoothest_output.outputs.dlh, 
                                                smoothest_output.outputs.volume,
                                                smoothest_output.outputs.resels])
                    self.teams[teamID].logs['smoothest']=(smoothest_output.runtime.stdout,
                                                            smoothest_output.runtime.stderr)

                        
        smoothness_df = pandas.DataFrame(smoothness,columns=['teamID','hyp','dhl','volume','resels'])
        smoothness_df.to_csv(output_file)
        return(smoothness_df)               

    # serialize important info and save to file                 
    def write_data(self,save_data = True,outfile=None):
        info={}
        info['started_at']=self.started_at
        info['save_time']=datetime.datetime.now()
        info['dirs'] = self.dirs
        info['teamlist']=self.complete_image_sets
        info['teams']={}

        for teamID in self.complete_image_sets:
            info['teams'][teamID]={'images':self.teams[teamID].images,
                                'image_json':self.teams[teamID].image_json,
                                'input_dir':self.teams[teamID].input_dir,
                                'NV_collection_id':self.teams[teamID].NV_collection_id,
                                'jsonfile':self.teams[teamID].jsonfile}
        if save_data:
            if not outfile:
                outfile = os.path.join(self.dirs.dirs['cached'],'narps_prepare_maps.pkl')
            with open(outfile,'wb') as f:
                pickle.dump(info,f)
        return(info)

    def load_data(self,infile=None):
        if not infile:
            infile = os.path.join(self.dirs.dirs['cached'],'narps_prepare_maps.pkl')
        assert os.path.exists(infile)

        with open(infile,'rb') as f:
            info = pickle.load(f)

        self.dirs = info['dirs'] 
        self.complete_image_sets = info['teamlist']
        for teamID in self.complete_image_sets:
            self.teams[teamID]=NarpsTeam(teamID,info['teams'][teamID]['NV_collection_id'],
                                        info['dirs'] ,verbose = self.verbose)
            self.teams[teamID].jsonfile = info['teams'][teamID]['jsonfile']

            self.teams[teamID].images = info['teams'][teamID]['images']
            self.teams[teamID].image_json = info['teams'][teamID]['image_json']
            self.teams[teamID].input_dir = info['teams'][teamID]['input_dir']


class TestNarps(object):
    def test_narps_dirs(self):
        narpsDirs = NarpsDirs("/Users/poldrack/data_unsynced/NARPS")
    def test_narps_team(self):
        narpsDirs = NarpsDirs("/Users/poldrack/data_unsynced/NARPS")
        narpsTeam = NarpsTeam('C88N','ADFZYYLQ',narpsDirs)
    def test_narps_main_class(self):
        narps = Narps("/Users/poldrack/data_unsynced/NARPS")

    
if __name__ == "__main__":
    # team data (from neurovault) should be in <basedir>/maps/orig
    # some data need to be renamed before using - see rename.sh in individual dirs

    # set an environment variable called NARPS_BASEDIR with location of base directory
    if 'NARPS_BASEDIR' in os.environ:
        basedir = os.environ['NARPS_BASEDIR']
    else:
        basedir = '/data'
    assert os.path.exists(basedir)

    run_all = True

    # setup main class
    narps = Narps(basedir,overwrite=False)

    if run_all:
        print('getting binarized/thresholded orig maps')
        narps.get_binarized_thresh_masks()

        print("getting resampled images...")
        narps.get_resampled_images()

        print("creating concatenated thresholded images...")
        narps.create_concat_images(datatype='resampled',imgtypes = ['thresh'])

        print("checking image values...")
        image_metadata_df = narps.check_image_values()

        print("creating rectified images...")
        narps.create_rectified_images()

        print('Creating overlap images for thresholded maps...')
        narps.create_thresh_overlap_images()

        #print("creating concatenated rectified images...")
        #narps.create_concat_images(datatype='rectified',imgtypes = ['unthresh'])

        print('converting to z-scores')
        narps.convert_to_zscores()

        print("creating concatenated zstat images...")
        narps.create_concat_images(datatype='zstat',imgtypes = ['unthresh'])

        print("computing image stats...")
        narps.compute_image_stats()

        print('estimating image smoothness')
        smoothness_df = narps.estimate_smoothness()

        # save directory structure
        narps.write_data()

    else:
        narps.load_data()

