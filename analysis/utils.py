import os,glob
import nilearn.input_data
import numpy,pandas


def get_masked_data(hyp,mask_img,output_dir,imgtype='unthresh',dataset='rectified'):
    if imgtype == 'unthresh':
        hmaps = glob.glob(os.path.join(output_dir,'%s/*/hypo%d_unthresh.nii.gz'%(dataset,hyp)))
    elif imgtype == 'thresh':
        hmaps = glob.glob(os.path.join(output_dir,'%s/*/hypo%d_thresh.nii.gz'%(dataset,hyp)))
    else:
        raise Exception('bad imgtype argument')
       
    #combined_data = nilearn.image.concat_imgs(hmaps)
    masker = nilearn.input_data.NiftiMasker(mask_img=mask_img)
    maskdata=masker.fit_transform(hmaps) #combined_data)   
    maskdata = numpy.nan_to_num(maskdata)
    if imgtype=='thresh':
        maskdata = (maskdata>1e-4).astype('float')
    labels = [os.path.basename(os.path.dirname(i)).split('_')[1] for i in hmaps]
    return(maskdata,labels)

def get_metadata(metadata_file = '/Users/poldrack/data_unsynced/NARPS/analysis_pipelines_SW.xlsx',
                        index_var = 'teamID'):
    metadata = pandas.read_excel(metadata_file,header=1)
    metadata.shape
    metadata.index = metadata[index_var]
    # fix issues with metadata
    metadata['used_fmriprep_data'] = [i.strip().split(',')[0] for i in metadata['used_fmriprep_data']]
    metadata['used_fmriprep_data'] = metadata['used_fmriprep_data'].replace({'Yas':'Yes'})

    return(metadata)

def get_decisions(decisions_file = '/Users/poldrack/data_unsynced/NARPS/narps_results.xlsx',
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
