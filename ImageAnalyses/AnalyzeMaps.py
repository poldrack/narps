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
import matplotlib.pyplot as plt
import seaborn
import scipy.cluster
import scipy.stats
from utils import get_concat_data, log_to_file, stringify_dict,\
    matrix_pct_agreement
from narps import Narps, hypnums
from narps import NarpsDirs # noqa, flake8 issue

# create some variables used throughout

cut_coords = [-24, -10, 4, 18, 32, 52, 64]
cluster_colors = ['c', 'm', 'y', 'k', 'b']
cluster_colornames = {
    'c': 'cyan',
    'm': 'magenta',
    'b': 'blue',
    'y': 'yellow',
    'k': 'black'}

# set up full names for figures
hypotheses_full = {
    1: '+gain: equal indifference',
    2: '+gain: equal range',
    3: '+gain: equal indifference',
    4: '+gain: equal range',
    5: '-loss: equal indifference',
    6: '-loss: equal range',
    7: '+loss: equal indifference',
    8: '+loss: equal range',
    9: '+loss: ER > EI'}


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
    fig, ax = plt.subplots(4, 2, figsize=(15, 10))
    axis_y = [0, 0, 0, 0, 1, 1, 1, 1]
    axis_x = [0, 1, 2, 3, 0, 1, 2, 3]
    for i, hyp in enumerate(hypnums):
        imgfile = os.path.join(
            narps.dirs.dirs['output'],
            'overlap_binarized_thresh/hypo%d.nii.gz' % hyp)
        nilearn.plotting.plot_stat_map(
            imgfile,
            threshold=0.1,
            display_mode="z",
            colorbar=True,
            title='H%d:' % hyp+hypotheses_full[hyp],
            vmax=1.,
            cmap='jet',
            cut_coords=cut_coords,
            axes=ax[axis_x[i], axis_y[i]],
            annotate=False,
            figure=fig)

        # compute max and median overlap
        thresh_concat_file = os.path.join(
            narps.dirs.dirs['output'],
            'thresh_concat_resampled/hypo%d.nii.gz' % hyp)
        thresh_concat_data = masker.fit_transform(thresh_concat_file)
        overlap = numpy.mean(thresh_concat_data, 0)
        log_to_file(logfile, 'hyp%d: %f' % (hyp, numpy.max(overlap)))
        max_overlap[hyp] = overlap
    # clear axis for last space
    ax[3, 1].set_axis_off()
    plt.savefig(
        os.path.join(narps.dirs.dirs['figures'], 'overlap_map.png'),
        bbox_inches='tight')
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
            title='Range: H%d:' % hyp+hypotheses_full[hyp],
            vmax=25,
            cut_coords=cut_coords,
            axes=ax[i])
    plt.savefig(os.path.join(
        narps.dirs.dirs['figures'], 'range_map.png'),
        bbox_inches='tight')
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
            title='SD: H%d:' % hyp+hypotheses_full[hyp],
            vmax=4,
            cut_coords=cut_coords,
            axes=ax[i])
    plt.savefig(os.path.join(
        narps.dirs.dirs['figures'], 'std_map.png'),
        bbox_inches='tight')
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

    outdir = narps.dirs.get_output_dir(
        'team_maps_%s' % imgtype,
        base='figures')

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
            outdir, '%s.png' % teamID),
            bbox_inches='tight')
        plt.close(fig)


def mk_correlation_maps_unthresh(
        narps,
        corr_type='spearman',
        n_clusters=None,
        dataset='zstat',
        vox_mask_thresh=1.0):
    """
    Create correlation maps for unthresholded images
    These correlation matrices are clustered using Ward clustering,
    with the number of clusters for each hypotheses determined by
    visual examination.
    vox_mask_thresh controls which voxels are analyzed in terms
    of proportion of teams with signal in voxel.  defaults to 100%
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
        n_clusters = {1: 3, 2: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3}

    dendrograms = {}
    membership = {}
    cc_unthresh = {}
    output_dir = narps.dirs.get_output_dir('correlation_unthresh')

    for i, hyp in enumerate(hypnums):
        print('creating correlation map for hypothesis', hyp)
        membership[str(hyp)] = {}
        maskdata, labels = get_concat_data(
            hyp,
            narps.dirs.MNI_mask,
            narps.dirs.dirs['output'],
            dataset=dataset,
            vox_mask_thresh=vox_mask_thresh,
            logfile=logfile)

        # compute correlation of all datasets with mean
        if 'mean_corr' not in locals():
            mean_corr = pandas.DataFrame(
                numpy.zeros((len(labels), len(hypnums))),
                columns=['H%d' % i for i in hypnums],
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

        # add 1 to cluster labels so they start at 1
        # rather than zero - for clarity in paper
        clustlabels = [
            s[0] + 1 for s in
            scipy.cluster.hierarchy.cut_tree(
                ward_linkage,
                n_clusters=n_clusters[hyp])]
        print('clustlabels:', clustlabels)
        # get decisions for column colors
        md = narps.metadata.query(
            'varnum==%d' % hyp).set_index('teamID')

        decision_colors = ['r', 'g']
        col_colors = [
            decision_colors[md.loc[teamID, 'Decision']]
            for teamID in labels
            ]

        row_colors = [cluster_colors[s] for s in clustlabels]
        print('row_colors:', row_colors)
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
        plt.title('H%d:' % hyp+hypotheses_full[hyp])
        cc_unthresh[hyp] = (cc, labels)
        plt.savefig(os.path.join(
            narps.dirs.dirs['figures'],
            'hyp%d_%s_map_unthresh.png' % (hyp, corr_type)),
            bbox_inches='tight')
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
    median_corr = mean_corr.median(1).sort_values()
    median_corr_df = pandas.DataFrame(
        median_corr,
        columns=['median_corr'])
    median_corr_df.to_csv(os.path.join(
        narps.dirs.dirs['metadata'],
        'median_pattern_corr.csv'))

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
        vmax=5.,
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
    # remove these to keep logs more tractable
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
            log_to_file(
                logfile,
                'hyp %d cluster %d (%s)' % (
                    hyp, cl, cluster_colors[j+1]))
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
            mean_filename = os.path.join(
                narps.dirs.dirs['output'],
                'cluster_maps/hyp%d_cluster%d_mean.nii.gz' % (hyp, cl)
            )
            if not os.path.exists(os.path.dirname(mean_filename)):
                os.mkdir(os.path.dirname(mean_filename))
            mean_img.to_filename(mean_filename)
            nilearn.plotting.plot_stat_map(
                mean_img,
                threshold=thresh,
                vmax=vmax,
                display_mode="z",
                colorbar=True,
                title='H%d - cluster %d [%s] (pYes = %0.2f)' % (
                    hyp, cl,
                    cluster_colornames[cluster_colors[j+1]],
                    mean_decision[str(hyp)][str(cl)]
                ),
                cut_coords=cut_coords,
                axes=ax[j])
            log_to_file(logfile, '')
        log_to_file(logfile, '')
        plt.savefig(os.path.join(
            narps.dirs.dirs['figures'],
            'hyp%d_cluster_means.png' % hyp),
            bbox_inches='tight')
        plt.close(fig)

    # save cluster metadata to data frame
    cluster_metadata_df = cluster_metadata_df.dropna()
    cluster_metadata_df.to_csv(os.path.join(
        narps.dirs.dirs['metadata'],
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
        'median_pattern_corr.csv'))

    # Plot distance from mean across teams
    plt.bar(median_distance_df.index,
            median_distance_df.median_distance)
    plt.savefig(os.path.join(
        narps.dirs.dirs['figures'],
        'median_distance_sorted.png'),
        bbox_inches='tight')
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

    for hyp in hypnums:
        print('analyzing thresh similarity for hypothesis', hyp)
        maskdata, labels = get_concat_data(
            hyp,
            narps.dirs.MNI_mask,
            narps.dirs.dirs['output'],
            imgtype='thresh',
            dataset=dataset)

        pctagree = matrix_pct_agreement(maskdata)
        median_pctagree = numpy.median(
            pctagree[numpy.triu_indices_from(pctagree, 1)])
        log_to_file(
            logfile,
            'hyp %d: mean pctagree similarity: %f' %
            (hyp, median_pctagree))

        df_pctagree = pandas.DataFrame(pctagree, index=labels, columns=labels)
        df_pctagree.to_csv(os.path.join(
            narps.dirs.dirs['metadata'],
            'pctagree_hyp%d.csv' % hyp))

        seaborn.clustermap(
            df_pctagree,
            cmap='jet',
            figsize=(16, 16),
            method='ward')
        plt.title(hypotheses_full[hyp])
        plt.savefig(os.path.join(
            narps.dirs.dirs['figures'],
            'hyp%d_pctagree_map_thresh.png' % hyp),
            bbox_inches='tight')
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
    parser.add_argument('-t', '--test',
                        action='store_true',
                        help='use testing mode (no processing)')
    parser.add_argument(
        '--skip_maps',
        action='store_true',
        help='skip creation of overlap/range/std maps')
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
    if not args.test:
        if not args.skip_maps:
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
