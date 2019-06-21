import os,glob
import nilearn.input_data
import numpy,pandas
import nibabel 
from scipy.stats import norm, t
import scipy.stats


def get_masked_data(hyp,mask_img,output_dir,imgtype='unthresh',dataset='zstat'):
    if imgtype == 'unthresh':
        hmaps = glob.glob(os.path.join(output_dir,'%s/*/hypo%d_unthresh.nii.gz'%(dataset,hyp)))
    elif imgtype == 'thresh':
        hmaps = glob.glob(os.path.join(output_dir,'%s/*/hypo%d_thresh.nii.gz'%(dataset,hyp)))
    else:
        raise Exception('bad imgtype argument')
    hmaps.sort()
    #combined_data = nilearn.image.concat_imgs(hmaps)
    masker = nilearn.input_data.NiftiMasker(mask_img=mask_img)
    maskdata=masker.fit_transform(hmaps) #combined_data)   
    maskdata = numpy.nan_to_num(maskdata)
    if imgtype=='thresh':
        maskdata = (maskdata>1e-4).astype('float')
    labels = [os.path.basename(os.path.dirname(i)).split('_')[1] for i in hmaps]
    return(maskdata,labels)

def get_metadata(metadata_file = '/Users/poldrack/data_unsynced/NARPS/metadata/analysis_pipelines_SW.xlsx',
                        index_var = 'teamID'):
    metadata = pandas.read_excel(metadata_file,header=1)
    metadata.teamID = [i.strip() for i in metadata.teamID]
    metadata.shape
    metadata.index = metadata[index_var]
    # fix issues with metadata
    metadata['used_fmriprep_data'] = [i.strip().split(',')[0] for i in metadata['used_fmriprep_data']]
    metadata['used_fmriprep_data'] = metadata['used_fmriprep_data'].replace({'Yas':'Yes'})
    # manual fixes to textual responses
    metadata.loc['E6R3','n_participants']=108
    metadata.loc['J7F9','n_participants']=107
    metadata['n_participants']  = [int(i.split('\n')[0]) if isinstance(i, str) else i for i in metadata['n_participants']]
    return(metadata)

def get_tidy_metadata(metadata_file = '/Users/poldrack/data_unsynced/NARPS/metadata/decision_data.csv'):
    return(pandas.read_csv(metadata_file))

def get_map_metadata(map_metadata_file = '/Users/poldrack/data_unsynced/NARPS/metadata/narps_neurovault_images_details.csv'):
    """
    Timestamp	
    teamID (the four-characters string identifying your analysis team)	
    What software package did you use to compute the (unthresholded) statistical maps for your analysis ?	
    Our unthresholded images on neurovault represent (if other, please specify)	
    Our thresholded images on neurovault are with (if other, please specify)	
    What brain template you registered your images to? Please provide the exact file name for the image used as a template (e.g. in FSL, MNI152lin_T1_1mm.nii.gz)	
    For hypothesis 5 (negative parametric effect of loss in the vmPFC for the equal indifference group), does the direction of the values in your images match the direction of the contrast?	
    For hypothesis 6 (negative parametric effect of loss in the vmPFC for the equal range group), does the direction of the values in your images match the direction of the contrast?	
    For hypothesis 9 (greater positive response to losses in amygdala for equal range condition vs. equal indifference condition), does the direction of the values in your images match the direction of the contrast?	
    Any comments / further explanations?
    """
    map_info = pandas.read_csv(map_metadata_file,names=['timestamp','teamID','software','unthresh_type',
                                                        'thresh_type','MNItemplate','hyp5_direction',
                                                        'hyp6_direction','hyp9_direction','comments'],
                                                skiprows=1)
    # manual fixes
    map_info.teamID = [i.upper() for i in map_info.teamID]
    del map_info['timestamp']
    map_info.index = map_info.teamID
    map_info = map_info.drop_duplicates(subset='teamID',keep='last')
    map_info.loc[:,'unthresh_type']= [i.split('values')[0].strip() for i in map_info.unthresh_type]

    map_info.loc['E3B6','unthresh_type']='t'
    # for those that don't fit, set to NA
    map_info.loc[:,'unthresh_type']=[i if i in ['t','z'] else 'NA' for i in map_info.unthresh_type ]

    return(map_info)

def get_decisions(decisions_file = '/Users/poldrack/data_unsynced/NARPS/metadata/narps_results.xlsx',
                                tidy=False):
    colnames=[]
    for hyp in range(1,10):
        colnames += ['Decision%d'%hyp,'Confidence%d'%hyp,'Similar%d'%hyp]
    colnames += ['collection']
    decisions = pandas.read_excel(decisions_file,header=1)
    decisions.columns=colnames
    decisions['teamID']=decisions.index

    # make a tidy version
    if tidy:
        decisions_long = pandas.melt(decisions,id_vars=['teamID'],value_vars=decisions.columns.values[:27])
        decisions_long['vartype']= [i[:-1] for i in decisions_long['variable']]
        decisions_long['varnum']= [i[-1] for i in decisions_long['variable']]
        del decisions_long['variable']

        Decision_df = decisions_long.query('vartype =="Decision"')
        Similar_df = decisions_long.query('vartype =="Similar"')
        Confidence_df = decisions_long.query('vartype =="Confidence"')

        decision_df = Decision_df.merge(Similar_df,'left',on=['teamID','varnum'],suffixes=['_decision','_similar']).merge(Confidence_df,'left',on=['teamID','varnum'])
        del decision_df['vartype_decision']
        del decision_df['vartype_similar']
        del decision_df['vartype']
        decision_df.columns=['teamID', 'Decision', 'varnum', 'Similar', 'Confidence']
        decision_df['Decision']=(decision_df['Decision']=='Yes').astype('int')
        decision_df['Similar']=decision_df['Similar'].astype('int')
        decision_df['Confidence']=decision_df['Confidence'].astype('int')
        decision_df.head()
        return(decision_df)
    else:
        return(decisions)

def get_merged_metadata_decisions():
    metadata = get_metadata()
    decision_df = get_decisions(tidy=True)
    alldata_df = decision_df.merge(metadata,on='teamID',how='left')
    return(alldata_df)

# create dictionary mapping from teamID to collection ID
def get_teamID_to_collectionID_dict(metadata):
    teamid_to_collectionID_dict={}
    for i in metadata.index:
        idx = i.split()[0]
        teamid_to_collectionID_dict[idx]='_'.join([metadata.loc[i,'NV_collection_string'].strip(),idx])
    return(teamid_to_collectionID_dict)

def matrix_jaccard(mtx):
    jacmtx = numpy.zeros((mtx.shape[0],mtx.shape[0]))
    for i in range(mtx.shape[0]):
        for j in range(i+1,mtx.shape[0]):
            if i==j:
                 continue
            if numpy.sum(mtx[i,:])>0 and numpy.sum(mtx[j,:])>0:
                jacmtx[i,j]=sklearn.metrics.jaccard_score(mtx[i,:],mtx[j,:]) 
    
    jacmtx = numpy.nan_to_num(jacmtx)
    jacmtx = jacmtx + jacmtx.T
    jacmtx[numpy.diag_indices_from(jacmtx)]=1
    return(jacmtx)


# adapted from https://github.com/vsoch/TtoZ/blob/master/TtoZ/scripts.py
# takes a nibabel file object and converts from z to t using Hughett's transform

def TtoZ(tmapfile,outfile,df):
  
  mr = nibabel.load(tmapfile)
  data = mr.get_data()

  # Select just the nonzero voxels
  nonzero = data[data!=0]

  # We will store our results here
  Z = numpy.zeros(len(nonzero))

  # Select values less than or == 0, and greater than zero
  c  = numpy.zeros(len(nonzero))
  k1 = (nonzero <= c)
  k2 = (nonzero > c)

  # Subset the data into two sets
  t1 = nonzero[k1]
  t2 = nonzero[k2]

  # Calculate p values for <=0
  p_values_t1 = t.cdf(t1, df = df)
  z_values_t1 = norm.ppf(p_values_t1)

  # Calculate p values for > 0
  p_values_t2 = t.cdf(-t2, df = df)
  z_values_t2 = -norm.ppf(p_values_t2)
  Z[k1] = z_values_t1
  Z[k2] = z_values_t2

  # Write new image to file
  empty_nii = numpy.zeros(mr.shape)
  empty_nii[mr.get_data()!=0] = Z
  Z_nii_fixed = nibabel.nifti1.Nifti1Image(empty_nii,affine=mr.get_affine(),header=mr.get_header())
  nibabel.save(Z_nii_fixed,outfile)
  
# perform 1 sample t-teat for correlated observations, based on emails from JM and TN
def t_corr(y,Q=None):
    """
    perform a one-sample t-test on correlated data
    y = data (n observations X n vars)
    Q = "known" correlation across observations (use empirical correlation based on maps)
    """
    
    # equations in comments from Tom's email
    
    npts = y.shape[0]
    X = numpy.ones((npts,1))

    if len(y.shape)==1:
        y = y[:,numpy.newaxis]
    assert len(y.shape)<3

    if Q is None:
        #print('no Q specified, using identity (uncorrelated)')
        Q = numpy.eye(npts)

    y_hat = numpy.mean(y,axis=0) # bar{Y_i} = X’ Y_i/N
    
    R = numpy.eye(npts) - X.dot(numpy.linalg.inv(X.T.dot(X))).dot(X.T)

    # modified per Tom Nichols email
    s_hat_2 = numpy.sum(y**2,axis=0)/(numpy.trace(R.dot(Q))) 
    
    # modified per Tom Nichols email
    var_y_hat = X.T.dot(Q).dot(X)/(npts**2) #sigma^2_i   X’ Q X / N^2
    
    T = y_hat/numpy.sqrt(var_y_hat) # T_i = bar(Y_i) / sqrt(Var(bar{Y_i}))
    
    # R = I{n} - X(X'X)^{-1}X'

    # degrees of freedom = v = tr(RQ)^2/tr(RQRQ)
    df = (numpy.trace(R.dot(Q))**2)/numpy.trace(R.dot(Q).dot(R).dot(Q))
    p = 1 - scipy.stats.t.cdf(T,df=df)
    return(T,df,p)

