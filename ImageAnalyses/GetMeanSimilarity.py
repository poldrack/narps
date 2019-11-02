#!/usr/bin/env python
# coding: utf-8
"""
compute mean similarity overall and within clusters

"""

import os
import argparse
import numpy
import pandas
import json
import matplotlib.pyplot as plt
from narps import Narps, hypnums

hypnames = ['%d' % i for i in hypnums]
hypnames[:2] = ['1/3', '2/4']


def get_similarity_summary(narps, corrtype='spearman'):
    corr_summary = []
    for i, hyp in enumerate(hypnums):
        print('hyp', hyp)
        # load correlations and cluster info
        corrdata = pandas.read_csv(
            os.path.join(
                narps.dirs.dirs['output'],
                'correlation_unthresh',
                '%s_unthresh_hyp%d.csv' % (corrtype, hyp)),
            index_col=0
        )
        jsonfile = os.path.join(
            narps.dirs.dirs['output'],
            'correlation_unthresh',
            'unthresh_cluster_membership_spearman.json')
        with open(jsonfile) as f:
            clusterinfo = json.load(f)

        # overall correlation
        corrvals = corrdata.values
        corrvals_triu = corrvals[numpy.triu_indices_from(corrvals, 1)]
        corr_summary.append([hypnames[i],
                             'mean',
                             corrvals.shape[0],
                             numpy.mean(corrvals_triu)])
        # plot histogram without zeros
        plt.figure(figsize=(8, 8))
        plt.hist(corrvals_triu, 50, (-1, 1))
        histfile = os.path.join(
            narps.dirs.dirs['figures'],
            'correlation_unthresh',
            '%s_unthresh_hyp%d_mean.png' % (corrtype, hyp))
        if not os.path.exists(os.path.dirname(histfile)):
            os.mkdir(os.path.dirname(histfile))
        plt.savefig(histfile)
        plt.close()

        # per-cluster correlation
        ci = clusterinfo['%d' % hyp]
        for cluster in ci:
            clusterdata = corrdata.loc[ci[cluster]][ci[cluster]]
            assert (clusterdata.index == clusterdata.columns).all()
            cluster_corrvals = clusterdata.values
            cluster_corrvals_triu = cluster_corrvals[
                numpy.triu_indices_from(cluster_corrvals, 1)]
            corr_summary.append([hypnames[i],
                                 'cluster%s' % cluster,
                                 len(ci[cluster]),
                                 numpy.mean(cluster_corrvals_triu)])
            # plot histogram without zeros
            plt.figure(figsize=(8, 8))
            plt.hist(cluster_corrvals_triu, 50, (-1, 1))
            histfile = os.path.join(
                narps.dirs.dirs['figures'],
                'correlation_unthresh',
                '%s_unthresh_hyp%d_cluster%s.png' % (corrtype, hyp, cluster))
            if not os.path.exists(os.path.dirname(histfile)):
                os.mkdir(os.path.dirname(histfile))
            plt.savefig(histfile)
            plt.close()
    results_df = pandas.DataFrame(corr_summary)
    results_df.columns = ['hyp', 'group', 'Cluster size', 'Correlation']
    results_df_wide = results_df.pivot(
        index='hyp', columns='group',
        values=['Correlation', 'Cluster size'])
    results_df_wide.columns = [
        '%s (%s)' % (col[0], col[1]) for col in results_df_wide.columns.values]
    del results_df_wide['Cluster size (mean)']
    results_df_wide['Hyp'] = results_df_wide.index
    results_df_wide = results_df_wide[
        ['Hyp', 'Correlation (mean)',
         'Correlation (cluster1)', 'Cluster size (cluster1)',
         'Correlation (cluster2)', 'Cluster size (cluster2)',
         'Correlation (cluster3)', 'Cluster size (cluster3)']]

    results_df_wide.to_csv(os.path.join(
        narps.dirs.dirs['output'],
        'correlation_unthresh',
        'mean_unthresh_correlation_by_cluster.csv'),
        index=False)
    return(results_df_wide)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Get similarity summary')
    parser.add_argument('-b', '--basedir',
                        help='base directory')
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

    narps = Narps(basedir)

    if not args.test:
        corr_summary = get_similarity_summary(narps)
