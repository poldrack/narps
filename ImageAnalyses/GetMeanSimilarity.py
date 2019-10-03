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

from narps import Narps, hypnums

hypnames = ['%d' % i for i in hypnums]
hypnames[:2] = ['1/3', '2/4']


def get_similarity_summary(narps, corrtype='spearman'):
    corr_summary = []
    for i, hyp in enumerate(hypnums):
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
        corrvals_triu = numpy.triu(corrvals, 1)
        corr_summary.append([hypnames[i],
                             'mean',
                             numpy.mean(corrvals_triu)])

        # per-cluster correlation
        ci = clusterinfo['%d' % hyp]
        for cluster in ci:
            clusterdata = corrdata.loc[ci[cluster]][ci[cluster]]
            assert (clusterdata.index == clusterdata.columns).all()
            cluster_corrvals_triu = numpy.triu(clusterdata, 1)
            corr_summary.append([hypnames[i],
                                 'cluster%s' % cluster,
                                 numpy.mean(cluster_corrvals_triu)])
    results_df = pandas.DataFrame(corr_summary)
    results_df.columns = ['hyp', 'group', 'correlation']
    results_df_wide = results_df.pivot(
        index='hyp', columns='group', values='correlation')
    results_df_wide = results_df_wide[
        ['mean', 'cluster1', 'cluster2', 'cluster3']]
    results_df_wide.columns = [
        'All teams',
        'Cluster 1',
        'Cluster 2',
        'Cluster 3'
    ]
    results_df_wide.to_csv(os.path.join(
        narps.dirs.dirs['output'],
        'correlation_unthresh',
        'mean_unthresh_correlation_by_cluster.csv'))
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
