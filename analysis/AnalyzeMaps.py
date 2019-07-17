#!/usr/bin/env python
# coding: utf-8
"""
Primary analysis of statistical maps
"""


import numpy
import pandas
import nibabel
import pickle
import os
import glob
import nilearn.image
import nilearn.input_data
import nilearn.plotting
import sklearn
import matplotlib.pyplot as plt
import seaborn
import scipy.cluster
import scipy.stats
from utils import get_masked_data
from narps import Narps, hypotheses, hypnums
from narps import NarpsDirs # noqa, flake8 issue

# create some variables used throughout

cut_coords = [-24, -10, 4, 18, 32, 52, 64]
cluster_colors = ['r', 'g', 'b', 'y', 'k']


def mk_overlap_maps(narps, verbose=True):
    """ create overlap maps for thresholded maps"""
    print('making overlap maps')
    masker = nilearn.input_data.NiftiMasker(
        mask_img=narps.dirs.MNI_mask)
    max_overlap = {}
    fig, ax = plt.subplots(7, 1, figsize=(12, 24))
    if verbose:
        print('Maximum voxel overlap:')
    for i, hyp in enumerate(hypnums):
        imgfile = os.path.join(
            narps.dirs.dirs['output'],
            'overlap_binarized_thresh/hypo%d.nii.gz' % hyp)
        nilearn.plotting.plot_stat_map(
            imgfile,
            threshold=0.1,
            display_mode="z",
            colorbar=True,
            title='hyp %d:' % hyp+hypotheses[hyp],
            vmax=1.,
            cmap='jet',
            cut_coords=cut_coords,
            axes=ax[i],
            figure=fig)

        # compute max and median overlap
        thresh_concat_file = os.path.join(
            narps.dirs.dirs['output'],
            'thresh_concat_resampled/hypo%d.nii.gz' % hyp)
        thresh_concat_data = masker.fit_transform(thresh_concat_file)
        overlap = numpy.mean(thresh_concat_data, 0)
        if verbose:
            print('hyp%d' % hyp, numpy.max(overlap))
        max_overlap[hyp] = overlap
    plt.savefig(os.path.join(narps.dirs.dirs['figures'], 'overlap_map.png'))
    return(max_overlap)


def mk_range_maps(narps):
    """ create maps of range of unthresholded values"""
    print('making range maps')
    fig, ax = plt.subplots(7, 1, figsize=(12, 24))
    for i, hyp in enumerate(hypnums):
        range_img = nibabel.load(
            os.path.join(
                narps.dirs.dirs['output'],
                'unthresh_range_%s/hypo%d.nii.gz' % (
                    unthresh_dataset_to_use, hyp)))
        nilearn.plotting.plot_stat_map(
            range_img,
            threshold=.1,
            display_mode="z",
            colorbar=True,
            title='Range: hyp %d:' % hyp+hypotheses[hyp],
            vmax=25,
            cut_coords=cut_coords,
            axes=ax[i])
    plt.savefig(os.path.join(
        narps.dirs.dirs['figures'], 'range_map.pdf'))


def mk_std_maps(narps):
    """ create maps of standard deviation of unthresholded values"""
    print('making standard deviation maps')
    # show std maps
    fig, ax = plt.subplots(7, 1, figsize=(12, 24))
    for i, hyp in enumerate(hypnums):
        std_img = nibabel.load(
            os.path.join(
                narps.dirs.dirs['output'],
                'unthresh_std_%s/hypo%d.nii.gz' % (
                    unthresh_dataset_to_use, hyp)))
        nilearn.plotting.plot_stat_map(
            std_img,
            threshold=.1,
            display_mode="z",
            colorbar=True,
            title='SD: hyp %d:' % hyp+hypotheses[hyp],
            vmax=4,
            cut_coords=cut_coords,
            axes=ax[i])
    plt.savefig(os.path.join(
        narps.dirs.dirs['figures'], 'std_map.pdf'))


def plot_individual_maps(
        narps,
        imgtype='unthresh',
        dataset='zstat'):
    """
    Display rectified unthresholded maps for each team
    save all hypotheses for each team to a separate file
    """
    if imgtype == 'unthresh':
        threshold = 2.
    else:
        threshold = 1e-5

    outdir = os.path.join(narps.dirs.dirs['figures'],
                          'team_maps_%s' % imgtype)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    nnz = []
    nonzero_volume = []
    dim_values = []
    missing_metadata = []

    # get all collection IDs
    collectionIDs = [
        os.path.basename(i) for i in glob.glob(
            os.path.join(narps.dirs.dirs['output'], '%s/*' % dataset))]

    # loop through each and create file
    for collection in collectionIDs:
        collection_string, teamID = collection.split('_')
        print('creating figure for team', teamID)
        hmaps = glob.glob(
            os.path.join(narps.dirs.dirs['output'],
                         '%s/%s/hypo*_unthresh.nii.gz' % (
                             dataset, collection)))
        hmaps.sort()

        fig, ax = plt.subplots(
            len(hypnums), 1, figsize=(12, len(hypnums)*2.5))
        print('making figure for team ', teamID)
        ctr = 0
        # load all maps and get dims
        for i, m in enumerate(hmaps):
            hyp = int(os.path.basename(
                m).split('_')[0].replace('hypo', ''))
            if hyp not in hypnums:
                print('skipping', hyp)
                continue
            img = nibabel.load(m)
            dims = img.header.get_data_shape()
            dim_values.append(dims)
            print(i, m)
            md = narps.metadata.query(
                'varnum==%d' % hyp).query(
                    'NV_collection_string == "%s"' %
                    collection_string).replace(numpy.nan, 'na')
            if md.shape[0] == 0:
                # try other identifier
                md = narps.metadata.query('varnum==%d' % hyp).query(
                    'teamID == "%s"' % teamID)
                if md.shape[0] == 0:
                    missing_metadata.append(collection)
                    continue

            # check for thresholding
            imgdata = img.get_data()
            nonzero_vox = numpy.nonzero(imgdata)
            n_nonzero_vox = len(nonzero_vox[0])
            nnz.append(n_nonzero_vox)
            vox_vol = numpy.prod(dims)
            nonzero_volume.append(n_nonzero_vox*vox_vol)

            if md['used_fmriprep_data'].values[0].find('Yes') > -1:
                prep_string = 'fmriprep'
            else:
                prep_string = 'other'
            nilearn.plotting.plot_stat_map(
                img,
                threshold=threshold,
                display_mode="z",
                colorbar=True,
                title='_'.join([
                    'hyp%d' % hyp, collection,
                    md['TSc_SW'].values[0],
                    prep_string]),
                cut_coords=cut_coords,
                axes=ax[ctr])
            ctr += 1
        plt.savefig(os.path.join(
            outdir, '%s.pdf' % teamID))


def mk_correlation_maps_unthresh(
        narps,
        corr_type='spearman',
        n_clusters={1: 4, 2: 3, 5: 4, 6: 3, 7: 4, 8: 4, 9: 3},
        dataset='zstat',
        distance_metric='euclidean'):
    """
    Create orrelation maps for unthresholded images
    These correlation matrices are clustered using Ward clustering,
    with the number of clusters for each hypotheses determined by
    visual examination.
    """

    dendrograms = {}
    membership = {}
    cc_unthresh = {}

    for i, hyp in enumerate(hypnums):
        print('creating correlation map for hypothesis', hyp)
        maskdata, labels = get_masked_data(
            hyp,
            narps.dirs.MNI_mask,
            narps.dirs.dirs['output'],
            dataset=dataset)

        # compute correlation of all datasets with mean
        if 'mean_corr' not in locals():
            mean_corr = pandas.DataFrame(
                numpy.zeros((len(labels), len(hypnums))),
                columns=['hyp%d' % i for i in hypnums],
                index=labels)
        meandata = numpy.mean(maskdata, 0)
        for t in range(maskdata.shape[0]):
            mean_corr.iloc[t, i] = scipy.stats.spearmanr(
                maskdata[t, :], meandata).correlation

        # cluster datasets
        if corr_type == 'spearman':
            cc = scipy.stats.spearmanr(maskdata.T).correlation
        else:  # use Pearson
            cc = numpy.corrcoef(maskdata)
        cc = numpy.nan_to_num(cc)
        df = pandas.DataFrame(cc, index=labels, columns=labels)

        ward_linkage = scipy.cluster.hierarchy.ward(cc)
        distances = scipy.spatial.distance.pdist(cc, distance_metric)

        clustlabels = [
            s[0] for s in
            scipy.cluster.hierarchy.cut_tree(
                ward_linkage,
                n_clusters=n_clusters[hyp])]

        # get decisions for column colors
        md = narps.metadata.query(
            'varnum==%d' % hyp).set_index('teamID')

        col_colors = [
            cluster_colors[md.loc[teamID, 'Decision']]
            for teamID in labels
            ]

        row_colors = [cluster_colors[s-1] for s in clustlabels]
        cm = seaborn.clustermap(
            df,
            cmap='vlag',
            figsize=(16, 16),
            method='ward',
            row_colors=row_colors,
            col_colors=col_colors,
            center=0,
            vmin=-1,
            vmax=1)
        plt.title('hyp %d:' % hyp+hypotheses[hyp])
        cc_unthresh[hyp] = (cc, labels)
        plt.savefig(os.path.join(
            narps.dirs.dirs['figures'],
            'hyp%d_%s_map_unthresh.pdf' % (hyp, corr_type)))
        dendrograms[hyp] = ward_linkage

        # get cluster membership
        membership[hyp] = {}
        for j in cm.dendrogram_row.reordered_ind:
            cl = clustlabels[j]
            if cl not in membership[hyp]:
                membership[hyp][cl] = []
            membership[hyp][cl].append(labels[j])

    # save cluster data to file so that we don't have to rerun everything
    with open(os.path.join(
            narps.dirs.dirs['output'],
            'unthresh_dendrograms_%s.pkl' % corr_type), 'wb') as f:
        pickle.dump((dendrograms, membership), f)

    # also save correlation info
    median_distance = mean_corr.median(1).sort_values()
    median_distance_df = pandas.DataFrame(
        median_distance,
        columns=['median_distance'])
    median_distance_df.to_csv(os.path.join(
        narps.dirs.dirs['metadata'],
        'median_pattern_distance.csv'))

    return((dendrograms, membership))


def analyze_clusters(
        narps,
        dendrograms=None,
        membership=None,
        dataset='zstat',
        corr_type='spearman',
        thresh=2.,
        rand_thresh=0.2):
    """
    Use dendrogram computed by seaborn clustermap to identify clusters,
    and then create separate mean statstical map for each cluster.
    """

    if dendrograms is None or membership is None:
        with open(os.path.join(
                narps.dirs.dirs['output'],
                'unthresh_dendrograms_%s.pkl' % corr_type), 'rb') as f:
            dendrograms, membership = pickle.load(f)

    mean_smoothing = {}
    mean_decision = {}

    masker = nilearn.input_data.NiftiMasker(
        mask_img=narps.dirs.MNI_mask)

    for i, hyp in enumerate(hypnums):
        print('hyp', hyp)
        clusters = list(membership[hyp].keys())
        clusters.sort()
        fig, ax = plt.subplots(len(clusters), 1, figsize=(12, 12))
        mean_smoothing[hyp] = {}
        mean_decision[hyp] = {}
        for i, cl in enumerate(clusters):
            # get all images for this cluster and average them
            member_maps = []
            member_smoothing = []
            member_decision = []
            for member in membership[hyp][cl]:
                cid = narps.teams[member].datadir_label
                infile = os.path.join(
                    narps.dirs.dirs['output'],
                    '%s/%s/hypo%d_unthresh.nii.gz' % (
                        dataset, cid, hyp))
                if os.path.exists(infile):
                    member_maps.append(infile)
                    member_smoothing.append(
                        narps.metadata.query(
                            'varnum==%d' % hyp).query(
                                'teamID=="%s"' % member)['fwhm'].iloc[0])
                    member_decision.append(
                        narps.metadata.query(
                            'varnum==%d' % hyp).query(
                                'teamID=="%s"' % member)['Decision'].iloc[0])

            print('cluster %d: found %d maps' % (cl, len(member_maps)))
            mean_smoothing[hyp][cl] = numpy.mean(numpy.array(member_smoothing))
            mean_decision[hyp][cl] = numpy.mean(numpy.array(member_decision))
            print('mean fwhm:', mean_smoothing[hyp][cl])
            print('pYes:', mean_decision[hyp][cl])
            maskdata = masker.fit_transform(member_maps)
            meandata = numpy.mean(maskdata, 0)
            mean_img = masker.inverse_transform(meandata)

            nilearn.plotting.plot_stat_map(
                mean_img,
                threshold=thresh,
                display_mode="z",
                colorbar=True,
                title='hyp%d - cluster%d (fwhm=%0.2f, pYes = %0.2f)' % (
                    hyp, cl, mean_smoothing[hyp][cl],
                    mean_decision[hyp][cl]
                ),
                cut_coords=cut_coords,
                axes=ax[i])

        plt.savefig(os.path.join(
            narps.dirs.dirs['figures'],
            'hyp%d_cluster_means.pdf' % hyp))

    # create a data frame containing cluster metadata
    print('creating cluster metadata df')
    cluster_metadata = {}
    cluster_metadata_df = pandas.DataFrame(
        columns=['hyp%d' % i for i in hypnums],
        index=narps.metadata.teamID)

    for i, hyp in enumerate(hypnums):
        cluster_metadata[hyp] = {}
        print('Hypothesis', hyp)
        clusters = list(membership[hyp].keys())
        clusters.sort()

        for i, cl in enumerate(clusters):
            print('cluster %d (%s)' % (cl, cluster_colors[i-1]))
            print(membership[hyp][cl])
            cluster_metadata[hyp][cl] = narps.metadata[
                narps.metadata.teamID.isin(membership[hyp][cl])]
            for m in membership[hyp][cl]:
                cluster_metadata_df.loc[m, 'hyp%d' % hyp] = cl
        print('')

    cluster_metadata_df = cluster_metadata_df.dropna()
    cluster_metadata_df.to_csv(os.path.join(
        narps.dirs.dirs['output'],
        'cluster_metadata_df.csv'))

    # compute clustering similarity across hypotheses

    randmtx = numpy.zeros((10, 10))
    for i, j in enumerate(hypnums):
        for k in hypnums[i:]:
            if j == k:
                continue
            randmtx[j, k] = sklearn.metrics.adjusted_rand_score(
                cluster_metadata_df['hyp%d' % j],
                cluster_metadata_df['hyp%d' % k])
            if randmtx[j, k] > rand_thresh:
                print(j, k, randmtx[j, k])

    numpy.savetxt(os.path.join(
        narps.dirs.dirs['output'],
        'cluster_membership_Rand_indices.csv'),
        randmtx)

    return(cluster_metadata_df)


def plot_distance_from_mean(narps):

    median_distance_df = pandas.read_csv(os.path.join(
        narps.dirs.dirs['metadata'],
        'median_pattern_distance.csv'))

    # Plot distance from mean across teams
    plt.bar(median_distance_df.index,
            median_distance_df.median_distance)
    plt.savefig(os.path.join(
        narps.dirs.dirs['figures'],
        'median_distance_sorted.png'))

    # This plot is limited to the teams with particularly
    # low median correlations (<.2)
    median_distance_low = median_distance_df.query(
        'median_distance < 0.2')
    print('found %d teams with r<0.2 with mean pattern' %
          median_distance_low.shape[0])
    print(median_distance_low.iloc[:, 0].values)

    median_distance_high = median_distance_df.query(
        'median_distance > 0.7')
    print('found %d teams with r>0.7 with mean pattern' %
          median_distance_high.shape[0])

# #### SKIP FOR NOW Similarity maps for thresholded images
# For each pair of thresholded images, compute the similarity
# of the thresholded/binarized maps using the Jaccard coefficient.
# def get_thresh_similarity(narps):
#     cc_thresh={}
#     get_jaccard = False
#     for hyp in [1,2,5,6,7,8,9]:
#         maskdata,labels = get_masked_data(hyp,
#           mask_img,output_dir,imgtype='thresh')
#         cc = matrix_jaccard(maskdata)
#         df = pandas.DataFrame(cc,index=labels,columns=labels)
#         cc_thresh[hyp]=df
#     for hyp in [1,2,5,6,7,8,9]:
#         df = cc_thresh[hyp]
#         seaborn.clustermap(df,cmap='jet',figsize=(16,16),method='ward')
#         plt.title(hypotheses[hyp])
#         plt.savefig(os.path.join(figure_dir,'hyp%d_jaccard_map_thresh.pdf'%hyp))


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

    # which dataset to use for analyses
    unthresh_dataset_to_use = 'zstat'

    # setup main class
    narps = Narps(basedir)
    narps.load_data()

    # Load full metadata and put into narps structure
    narps.metadata = pandas.read_csv(
        os.path.join(narps.dirs.dirs['metadata'], 'all_metadata.csv'))

    # create maps showing overlap of thresholded images
    mk_overlap_maps(narps)

    mk_range_maps(narps)

    mk_std_maps(narps)

    plot_individual_maps(narps, imgtype='unthresh', dataset='zstat')

    corr_type = 'spearman'
    dendrograms, membership = mk_correlation_maps_unthresh(
        narps, corr_type=corr_type)

    # if variables don't exist then load them
    cluster_metadata_df = analyze_clusters(narps, corr_type=corr_type)

    plot_distance_from_mean(narps)
