#!/usr/bin/env python
# coding: utf-8
"""
Primary analysis of statistical maps
"""


import numpy
import argparse
import pandas
import nibabel
import os
import json
import glob
import nilearn.image
import nilearn.input_data
import nilearn.plotting
import sklearn
import sys
import inspect
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import seaborn
import scipy.cluster
import scipy.stats
from scipy.spatial.distance import pdist, squareform
from utils import get_masked_data, log_to_file, stringify_dict
from narps import Narps, hypotheses, hypnums
from narps import NarpsDirs # noqa, flake8 issue

# create some variables used throughout

cut_coords = [-24, -10, 4, 18, 32, 52, 64]
cluster_colors = ['r', 'g', 'b', 'y', 'k']


def mk_overlap_maps(narps, verbose=True):
    """ create overlap maps for thresholded maps"""
    func_name = sys._getframe().f_code.co_name
    logfile = os.path.join(
        narps.dirs.dirs['logs'],
        'AnalyzeMaps-%s.txt' % func_name)
    log_to_file(
        logfile, '%s' %
        func_name,
        flush=True)
    log_to_file(logfile, 'Maximum voxel overlap:')

    masker = nilearn.input_data.NiftiMasker(
        mask_img=narps.dirs.MNI_mask)
    max_overlap = {}
    fig, ax = plt.subplots(7, 1, figsize=(12, 24))
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
        log_to_file(logfile, 'hyp%d: %f' % (hyp, numpy.max(overlap)))
        max_overlap[hyp] = overlap
    plt.savefig(os.path.join(narps.dirs.dirs['figures'], 'overlap_map.png'))
    plt.close()
    return(max_overlap)


def mk_range_maps(narps, dataset='zstat'):
    """ create maps of range of unthresholded values"""

    fig, ax = plt.subplots(7, 1, figsize=(12, 24))
    for i, hyp in enumerate(hypnums):
        range_img = nibabel.load(
            os.path.join(
                narps.dirs.dirs['output'],
                'unthresh_range_%s/hypo%d.nii.gz' % (
                    dataset, hyp)))
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
    plt.close(fig)


def mk_std_maps(narps, dataset='zstat'):
    """ create maps of standard deviation of unthresholded values"""
    print('making standard deviation maps')
    # show std maps
    fig, ax = plt.subplots(7, 1, figsize=(12, 24))
    for i, hyp in enumerate(hypnums):
        std_img = nibabel.load(
            os.path.join(
                narps.dirs.dirs['output'],
                'unthresh_std_%s/hypo%d.nii.gz' % (
                    dataset, hyp)))
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
    plt.close(fig)


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
        plt.close(fig)


def mk_correlation_maps_unthresh(
        narps,
        corr_type='spearman',
        n_clusters=None,
        dataset='zstat'):
    """
    Create correlation maps for unthresholded images
    These correlation matrices are clustered using Ward clustering,
    with the number of clusters for each hypotheses determined by
    visual examination.
    """
    func_args = inspect.getargvalues(
        inspect.currentframe()).locals
    func_name = sys._getframe().f_code.co_name
    logfile = os.path.join(
        narps.dirs.dirs['logs'],
        'AnalyzeMaps-%s.txt' % func_name)
    log_to_file(
        logfile, '%s' %
        func_name,
        flush=True)
    log_to_file(
        logfile,
        stringify_dict(func_args))

    if n_clusters is None:
        n_clusters = {1: 4, 2: 3, 5: 4, 6: 3, 7: 4, 8: 4, 9: 3}

    dendrograms = {}
    membership = {}
    cc_unthresh = {}
    output_dir = os.path.join(
        narps.dirs.dirs['output'],
        'correlation_unthresh')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i, hyp in enumerate(hypnums):
        print('creating correlation map for hypothesis', hyp)
        membership[str(hyp)] = {}
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
        df.to_csv(os.path.join(
            output_dir,
            '%s_unthresh_hyp%d.csv' % (corr_type, hyp)))

        ward_linkage = scipy.cluster.hierarchy.ward(cc)

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
        plt.close()
        dendrograms[hyp] = ward_linkage

        # get cluster membership
        for j in cm.dendrogram_row.reordered_ind:
            cl = clustlabels[j]
            if str(cl) not in membership[str(hyp)]:
                membership[str(hyp)][str(cl)] = []
            membership[str(hyp)][str(cl)].append(labels[j])

    # save cluster data to file so that we don't have to rerun everything
    with open(os.path.join(
              output_dir,
              'unthresh_cluster_membership_%s.json' % corr_type), 'w') as f:
        json.dump(membership, f)

    # also save correlation info
    median_distance = mean_corr.median(1).sort_values()
    median_distance_df = pandas.DataFrame(
        median_distance,
        columns=['median_distance'])
    median_distance_df.to_csv(os.path.join(
        narps.dirs.dirs['metadata'],
        'median_pattern_distance.csv'))

    log_to_file(logfile, 'median correlation between teams: %f' %
                numpy.median(cc[numpy.triu_indices_from(cc, 1)]))

    return((dendrograms, membership))


def analyze_clusters(
        narps,
        dendrograms,
        membership,
        dataset='zstat',
        corr_type='spearman',
        thresh=2.,
        rand_thresh=0.2):
    """
    Use dendrogram computed by seaborn clustermap to identify clusters,
    and then create separate mean statstical map for each cluster.
    """

    # if dendrograms is None or membership is None:
    #     with open(os.path.join(
    #             narps.dirs.dirs['output'],
    #             'unthresh_dendrograms_%s.pkl' % corr_type), 'rb') as f:
    #         dendrograms, membership = pickle.load(f)

    func_args = inspect.getargvalues(
        inspect.currentframe()).locals
    del func_args['membership']
    del func_args['dendrograms']
    func_name = sys._getframe().f_code.co_name
    logfile = os.path.join(
        narps.dirs.dirs['logs'],
        'AnalyzeMaps-%s.txt' % func_name)
    log_to_file(
        logfile, '%s' %
        func_name,
        flush=True)
    log_to_file(
        logfile,
        stringify_dict(func_args))

    mean_smoothing = {}
    mean_decision = {}
    cluster_metadata = {}
    cluster_metadata_df = pandas.DataFrame(
        columns=['hyp%d' % i for i in hypnums],
        index=narps.metadata.teamID)

    masker = nilearn.input_data.NiftiMasker(
        mask_img=narps.dirs.MNI_mask)

    for i, hyp in enumerate(hypnums):
        log_to_file(logfile, 'hyp %d' % hyp)
        # set cluster indices back to int, for consistency with above
        clusters = [int(x) for x in list(membership[str(hyp)].keys())]
        clusters.sort()

        fig, ax = plt.subplots(len(clusters), 1, figsize=(12, 12))
        cluster_metadata[hyp] = {}
        mean_smoothing[str(hyp)] = {}
        mean_decision[str(hyp)] = {}
        for j, cl in enumerate(clusters):
            # get all images for this cluster and average them
            member_maps = []
            member_smoothing = []
            member_decision = []
            for member in membership[str(hyp)][str(cl)]:
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
            log_to_file(logfile,
                        'hyp %d cluster %d (%s)' % (
                            hyp, cl, cluster_colors[j-1]))
            log_to_file(logfile, membership[str(hyp)][str(cl)])
            cluster_metadata[hyp][cl] = narps.metadata[
                narps.metadata.teamID.isin(membership[str(hyp)][str(cl)])]
            for m in membership[str(hyp)][str(cl)]:
                cluster_metadata_df.loc[m, 'hyp%d' % hyp] = cl

            log_to_file(logfile, 'found %d maps: %d' % (cl, len(member_maps)))
            mean_smoothing[str(hyp)][str(cl)] = numpy.mean(
                numpy.array(member_smoothing))
            mean_decision[str(hyp)][str(cl)] = numpy.mean(
                numpy.array(member_decision))
            log_to_file(logfile,
                        'mean fwhm: %f' % mean_smoothing[str(hyp)][str(cl)])
            log_to_file(logfile,
                        'pYes: %f' % mean_decision[str(hyp)][str(cl)])
            maskdata = masker.fit_transform(member_maps)
            meandata = numpy.mean(maskdata, 0)
            mean_img = masker.inverse_transform(meandata)

            nilearn.plotting.plot_stat_map(
                mean_img,
                threshold=thresh,
                display_mode="z",
                colorbar=True,
                title='hyp%d - cluster%d (fwhm=%0.2f, pYes = %0.2f)' % (
                    hyp, cl, mean_smoothing[str(hyp)][str(cl)],
                    mean_decision[str(hyp)][str(cl)]
                ),
                cut_coords=cut_coords,
                axes=ax[j])
            log_to_file(logfile, '')
        log_to_file(logfile, '')
        plt.savefig(os.path.join(
            narps.dirs.dirs['figures'],
            'hyp%d_cluster_means.pdf' % hyp))
        plt.close(fig)

    # save cluster metadata to data frame
    cluster_metadata_df = cluster_metadata_df.dropna()
    cluster_metadata_df.to_csv(os.path.join(
        narps.dirs.dirs['output'],
        'cluster_metadata_df.csv'))

    # compute clustering similarity across hypotheses
    log_to_file(logfile, 'Computing cluster similarity (Rand score)')
    log_to_file(logfile, 'pairs with adjusted Rand index > %f' % rand_thresh)

    randmtx = numpy.zeros((10, 10))
    for i, j in enumerate(hypnums):
        for k in hypnums[i:]:
            if j == k:
                continue
            randmtx[j, k] = sklearn.metrics.adjusted_rand_score(
                cluster_metadata_df['hyp%d' % j],
                cluster_metadata_df['hyp%d' % k])
            if randmtx[j, k] > rand_thresh:
                log_to_file(logfile, '%d, %d: %f' % (j, k, randmtx[j, k]))

    numpy.savetxt(os.path.join(
        narps.dirs.dirs['output'],
        'cluster_membership_Rand_indices.csv'),
        randmtx)

    return(cluster_metadata_df)


def plot_distance_from_mean(narps):

    func_name = sys._getframe().f_code.co_name
    logfile = os.path.join(
        narps.dirs.dirs['logs'],
        'AnalyzeMaps-%s.txt' % func_name)
    log_to_file(
        logfile, '%s' %
        func_name,
        flush=True)

    median_distance_df = pandas.read_csv(os.path.join(
        narps.dirs.dirs['metadata'],
        'median_pattern_distance.csv'))

    # Plot distance from mean across teams
    plt.bar(median_distance_df.index,
            median_distance_df.median_distance)
    plt.savefig(os.path.join(
        narps.dirs.dirs['figures'],
        'median_distance_sorted.png'))
    plt.close()

    # This plot is limited to the teams with particularly
    # low median correlations (<.2)
    median_distance_low = median_distance_df.query(
        'median_distance < 0.2')
    log_to_file(
        logfile,
        'found %d teams with r<0.2 with mean pattern' %
        median_distance_low.shape[0])
    log_to_file(logfile, median_distance_low.iloc[:, 0].values)

    median_distance_high = median_distance_df.query(
        'median_distance > 0.7')
    log_to_file(
        logfile,
        'found %d teams with r>0.7 with mean pattern' %
        median_distance_high.shape[0])


def get_thresh_similarity(narps, dataset='resampled'):
    """
    For each pair of thresholded images, compute the similarity
    of the thresholded/binarized maps using the Jaccard coefficient.
    Computation with zeros per https://stackoverflow.com/questions/37003272/how-to-compute-jaccard-similarity-from-a-pandas-dataframe # noqa
    also add computation of jaccard on only nonzero pairs
    (ala scipy)
    """

    func_args = inspect.getargvalues(
        inspect.currentframe()).locals
    func_name = sys._getframe().f_code.co_name
    logfile = os.path.join(
        narps.dirs.dirs['logs'],
        'AnalyzeMaps-%s.txt' % func_name)
    log_to_file(
        logfile, '%s' %
        func_name,
        flush=True)
    log_to_file(
        logfile,
        stringify_dict(func_args))

    output_dir = os.path.join(
        narps.dirs.dirs['output'],
        'jaccard_thresh')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for hyp in hypnums:
        print('creating Jaccard map for hypothesis', hyp)
        maskdata, labels = get_masked_data(
            hyp,
            narps.dirs.MNI_mask,
            narps.dirs.dirs['output'],
            imgtype='thresh',
            dataset=dataset)
        jacsim = 1 - pairwise_distances(maskdata,  metric="hamming")
        jacsim_nonzero = 1 - squareform(pdist(maskdata, 'jaccard'))
        df = pandas.DataFrame(jacsim, index=labels, columns=labels)
        df.to_csv(os.path.join(
            output_dir, 'jacsim_thresh_hyp%d.csv' % hyp))
        df_nonzero = pandas.DataFrame(
            jacsim_nonzero,
            index=labels,
            columns=labels)
        df_nonzero.to_csv(os.path.join(
            output_dir, 'jacsim_nonzero_thresh_hyp%d.csv' % hyp))
        seaborn.clustermap(
            df,
            cmap='jet',
            figsize=(16, 16),
            method='ward')
        plt.title(hypotheses[hyp])
        plt.savefig(os.path.join(
            narps.dirs.dirs['figures'],
            'hyp%d_jaccard_map_thresh.pdf' % hyp))
        plt.close()
        seaborn.clustermap(
            df_nonzero,
            cmap='jet',
            figsize=(16, 16),
            method='ward')
        plt.title(hypotheses[hyp])
        plt.savefig(os.path.join(
            narps.dirs.dirs['figures'],
            'hyp%d_jaccard_nonzero_map_thresh.pdf' % hyp))
        plt.close()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Analyze NARPS data')
    parser.add_argument('-b', '--basedir',
                        help='base directory')
    parser.add_argument('-d', '--detailed',
                        action='store_true',
                        help='generate detailed team-level figures')
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

    if args.detailed:
        plot_individual_maps(
            narps,
            imgtype='unthresh',
            dataset='zstat')

    corr_type = 'spearman'
    dendrograms, membership = mk_correlation_maps_unthresh(
        narps, corr_type=corr_type)

    # if variables don't exist then load them
    cluster_metadata_df = analyze_clusters(
        narps,
        dendrograms,
        membership,
        corr_type=corr_type)

    plot_distance_from_mean(narps)

    get_thresh_similarity(narps)
