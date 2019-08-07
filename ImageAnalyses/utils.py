"""
utility functions for narps analysis
"""

import os
import glob
import nilearn.input_data
import numpy
import pandas
import nibabel
from scipy.stats import norm, t
import scipy.stats
from datetime import datetime


def stringify_dict(d):
    """create a pretty version of arguments for printing"""
    if 'self' in d:
        del d['self']
    s = 'Arguments:\n'
    for k in d:
        if not isinstance(d[k], str):
            d[k] = str(d[k])
        s = s + '%s: %s\n' % (k, d[k])
    return(s)


def log_to_file(fname, s, flush=False,
                add_timestamp=True,
                also_print=True,
                headspace=0):
    """ save string to log file"""
    if flush and os.path.exists(fname):
        os.remove(fname)
    if not isinstance(s, str):
        s = str(s)
    # add spacing before line
    if headspace > 0:
        s = os.linesep*headspace + s
    with open(fname, 'a+') as f:
        if also_print:
            print(s)
        f.write(s + os.linesep)
        if flush and add_timestamp:
            f.write(datetime.isoformat(
                datetime.now()) + 2 * os.linesep)


def get_masked_data(hyp, mask_img, output_dir,
                    imgtype='unthresh', dataset='zstat'):
    """
    load data from within mask
    """
    if imgtype == 'unthresh':
        hmaps = glob.glob(os.path.join(
            output_dir,
            '%s/*/hypo%d_unthresh.nii.gz' % (dataset, hyp)))
    elif imgtype == 'thresh':
        hmaps = glob.glob(os.path.join(
            output_dir,
            '%s/*/hypo%d_thresh.nii.gz' % (dataset, hyp)))
    else:
        raise Exception('bad imgtype argument')
    hmaps.sort()
    masker = nilearn.input_data.NiftiMasker(mask_img=mask_img)
    maskdata = masker.fit_transform(hmaps)  # combined_data
    maskdata = numpy.nan_to_num(maskdata)
    if imgtype == 'thresh':
        maskdata = (maskdata > 1e-4).astype('float')
    labels = [os.path.basename(os.path.dirname(i)).split('_')[1]
              for i in hmaps]
    return(maskdata, labels)


# load concatenated data - this is meant to replace
# get_masked_data()
def get_concat_data(hyp, mask_img, output_dir,
                    imgtype='unthresh', dataset='zstat',
                    vox_mask_thresh=None):
    """
    load data from within mask
    if vox_mask_thresh is specified, then the relevant
    file is loaded and only voxels with at least
    this proportion of teams present will be included
    """

    concat_file = os.path.join(
        output_dir,
        '%s_concat_%s' % (imgtype, dataset),
        'hypo%d.nii.gz' % hyp)
    assert os.path.exists(concat_file)

    labelfile = concat_file.replace('.nii.gz', '.labels')
    assert os.path.exists(labelfile)
    labels = []
    with open(labelfile, 'r') as f:
        for l in f.readlines():
            l_s = l.strip().split()
            labels.append(l_s[0])
    masker = nilearn.input_data.NiftiMasker(mask_img=mask_img)
    maskdata = masker.fit_transform(concat_file)  # combined_data
    maskdata = numpy.nan_to_num(maskdata)
    if imgtype == 'thresh':
        maskdata = (maskdata > 1e-4).astype('float')
    if vox_mask_thresh is not None:
        assert vox_mask_thresh >= 0 and vox_mask_thresh <= 1
        mask_file = concat_file.replace(
            '.nii.gz', '_voxelmap.nii.gz'
        )
        voxmaskdata = masker.fit_transform(mask_file)
        maskdata = maskdata[:, voxmaskdata[0, :] >= vox_mask_thresh]
    return(maskdata, labels)


def get_metadata(metadata_file,
                 index_var='teamID'):
    """ get team metadata"""
    metadata = pandas.read_excel(metadata_file, header=1)
    metadata.teamID = [i.strip() for i in metadata.teamID]
    metadata.index = metadata[index_var].values

    # fix issues with metadata
    metadata['used_fmriprep_data'] = [
        i.strip().split(',')[0]
        for i in metadata['used_fmriprep_data']]
    # manual fixes to textual responses
    metadata['used_fmriprep_data'] = metadata[
        'used_fmriprep_data'].replace({'Yas': 'Yes'})
    metadata.loc['E6R3', 'n_participants'] = 108
    metadata.loc['J7F9', 'n_participants'] = 107
    metadata['n_participants'] = [
        int(i.split('\n')[0]) if isinstance(i, str)
        else i for i in metadata['n_participants']]
    metadata['NV_collection_string'] = [
        os.path.basename(i.strip('/')) for i in
        metadata['NV_collection_link']]
    return(metadata)


def get_map_metadata(map_metadata_file):
    """
    get selected metadata from map metadata file
    """
    map_info = pandas.read_csv(
        map_metadata_file,
        names=['timestamp', 'teamID', 'software', 'unthresh_type',
               'thresh_type', 'MNItemplate', 'hyp5_direction',
               'hyp6_direction', 'hyp9_direction', 'comments'],
        skiprows=1)
    # manual fixes
    map_info.teamID = [i.upper() for i in map_info.teamID]
    del map_info['timestamp']
    map_info.index = map_info.teamID
    map_info = map_info.drop_duplicates(
        subset='teamID', keep='last')
    map_info.loc[:, 'unthresh_type'] = [
        i.split('values')[0].strip()
        for i in map_info.unthresh_type]
    # manual fixes
    map_info.loc['E3B6', 'unthresh_type'] = 't'
    map_info.loc['5G9K', 'unthresh_type'] = 't'
    map_info.loc['DC61', 'unthresh_type'] = 't'
    # for those that don't fit, set to NA
    map_info.loc[:, 'unthresh_type'] = [
        i if i in ['t', 'z'] else 'NA' for i in map_info.unthresh_type]

    return(map_info)


def get_decisions(decisions_file, tidy=False):
    colnames = ['teamID']
    for hyp in range(1, 10):
        colnames += ['Decision%d' % hyp,
                     'Confidence%d' % hyp,
                     'Similar%d' % hyp]
    colnames += ['collection']
    decisions = pandas.read_excel(decisions_file, skiprows=1,
                                  encoding='utf-8')
    decisions.columns = colnames

    # make a tidy version
    if tidy:
        decisions_long = pandas.melt(
            decisions,
            id_vars=['teamID'],
            value_vars=decisions.columns.values[1:28])
        decisions_long['vartype'] = [
            i[:-1] for i in decisions_long['variable']]
        decisions_long['varnum'] = [
            i[-1] for i in decisions_long['variable']]
        del decisions_long['variable']

        Decision_df = decisions_long.query('vartype =="Decision"')
        Similar_df = decisions_long.query('vartype =="Similar"')
        Confidence_df = decisions_long.query('vartype =="Confidence"')

        decision_df = Decision_df.merge(
            Similar_df,
            'left',
            on=['teamID', 'varnum'],
            suffixes=['_decision', '_similar']).merge(
                Confidence_df, 'left', on=['teamID', 'varnum'])

        del decision_df['vartype_decision']
        del decision_df['vartype_similar']
        del decision_df['vartype']

        decision_df.columns = ['teamID', 'Decision',
                               'varnum', 'Similar', 'Confidence']
        decision_df['Decision'] = (
            decision_df['Decision'] == 'Yes').astype('int')
        decision_df['Similar'] = decision_df['Similar'].astype('int')
        decision_df['Confidence'] = decision_df['Confidence'].astype('int')
        decision_df.head()
        return(decision_df)
    else:
        return(decisions)


def get_merged_metadata_decisions(metadata_file, decisions_file,):
    """ get all metadata in tidy format"""
    metadata = get_metadata(metadata_file)
    decision_df = get_decisions(decisions_file, tidy=True)
    alldata_df = decision_df.merge(metadata, on='teamID', how='left')
    return(alldata_df)


def get_teamID_to_collectionID_dict(metadata):
    """create dictionary mapping from teamID to collection ID"""
    teamid_to_collectionID_dict = {}
    for i in metadata.index:
        idx = i.split()[0]
        teamid_to_collectionID_dict[idx] = '_'.join(
            [metadata.loc[i, 'NV_collection_string'].strip(), idx])
    return(teamid_to_collectionID_dict)


def TtoZ(tmapfile, outfile, df):
    """
    takes a nibabel file object and converts from z to t
    using Hughett's transform
    adapted from:
    https://github.com/vsoch/TtoZ/blob/master/TtoZ/scripts.py
    """

    mr = nibabel.load(tmapfile)
    data = mr.get_data()

    # Select just the nonzero voxels
    nonzero = data[data != 0]

    # We will store our results here
    Z = numpy.zeros(len(nonzero))

    # Select values less than or == 0, and greater than zero
    c = numpy.zeros(len(nonzero))
    k1 = (nonzero <= c)
    k2 = (nonzero > c)

    # Subset the data into two sets
    t1 = nonzero[k1]
    t2 = nonzero[k2]

    # Calculate p values for <=0
    p_values_t1 = t.cdf(t1, df=df)
    z_values_t1 = norm.ppf(p_values_t1)

    # Calculate p values for > 0
    p_values_t2 = t.cdf(-t2, df=df)
    z_values_t2 = -norm.ppf(p_values_t2)
    Z[k1] = z_values_t1
    Z[k2] = z_values_t2

    # Write new image to file
    empty_nii = numpy.zeros(mr.shape)
    empty_nii[mr.get_data() != 0] = Z
    Z_nii_fixed = nibabel.nifti1.Nifti1Image(
        empty_nii,
        affine=mr.get_affine(),
        header=mr.get_header())
    nibabel.save(Z_nii_fixed, outfile)


def t_corr(y, res_mean=None, res_var=None, Q=None):
    """
    perform a one-sample t-test on correlated data
    y = data (n observations X n vars)
    res_mean = Common mean over voxels and results
    res_var  = Common variance over voxels and results
    Q = "known" correlation across observations
    - (use empirical correlation based on maps)
    """

    npts = y.shape[0]
    X = numpy.ones((npts, 1))

    if res_mean is None:
        res_mean = 0

    if res_var is None:
        res_var = 1

    if Q is None:
        Q = numpy.eye(npts)

    VarMean = res_var * X.T.dot(Q).dot(X) / npts**2

    # T  =  mean(y,0)/s-hat-2
    # use diag to get s_hat2 for each variable
    T = (numpy.mean(y, 0)-res_mean
         )/numpy.sqrt(VarMean)*numpy.sqrt(res_var) + res_mean

    # Assuming variance is estimated on whole image
    # and assuming infinite df
    p = 1 - scipy.stats.norm.cdf(T)

    return(T, p)


def randn_from_shape(shape):
    """
    take in a tuple defining a 4d matrix shape,
    and return a random matrix of that shape
    """
    assert len(shape) == 4
    return(
        numpy.random.randn(
            shape[0],
            shape[1],
            shape[2],
            shape[3]))
