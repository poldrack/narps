#!/usr/bin/env python
# coding: utf-8
"""
create supplementary figure 1
showing decisions and confidence/similarity
- also add details from models, and sort by overall p(yes)
"""


import os
import pandas
import seaborn
import matplotlib.pyplot as plt
from matplotlib import colors

from narps import Narps
from narps import NarpsDirs # noqa, flake8 issue
from utils import log_to_file


def get_all_metadata(narps):
    metadata = pandas.read_csv(
        os.path.join(
            narps.dirs.dirs['metadata'],
            'all_metadata.csv'))
    return(metadata)


def mk_supp_figure1(narps, metadata):
    decision_wide = metadata.pivot(
        index='teamID',
        columns='varnum',
        values='Decision')
    confidence_wide = metadata.pivot(
        index='teamID',
        columns='varnum',
        values='Confidence')

    # sort by mean acceptance
    decision_wide['mean'] = decision_wide.mean(axis=1)
    decision_wide = decision_wide.sort_values(
        'mean', ascending=False)
    del decision_wide['mean']

    # merge with analysis metadata
    metadata_selected = metadata.query('varnum==1')[
        ['teamID', 'fwhm', 'package',
         'used_fmriprep_data',
         'testing', 'movement_modeling']
    ]
    metadata_selected.index = metadata_selected.teamID
    decision_wide_merged = decision_wide.join(
        metadata_selected
    )
    metadata_merged = decision_wide_merged.drop(
        columns=[i for i in range(1, 10)] + ['teamID'])
    # make everything into short strings
    metadata_merged['movement_modeling'] = [
        ['No', 'Yes'][i] for i in metadata_merged.movement_modeling.values]
    metadata_merged['fwhm'] = [
        '%0.2f' % i for i in metadata_merged.fwhm.values]
    metadata_merged.fwhm.replace(
        {'nan': ''}, inplace=True)
    testing_convert = {'parametric': 'P',
                       'permutations': 'NP',
                       'randomise': 'NP',
                       'ARI': 'Other',
                       'Other': 'Other'}
    metadata_merged['testing'] = [
        testing_convert[i] for i in metadata_merged.testing.values]
    metadata_merged.rename(columns={
        'used_fmriprep_data': 'fmriprep'},
        inplace=True)
    cmap = colors.ListedColormap(
        ['#CD5C5C', '#9dc183'])

    plt.figure(figsize=(12, 18))
    plt.subplot(1, 2, 1)
    h = seaborn.heatmap(
        decision_wide,
        cmap=cmap,
        annot=confidence_wide,
        fmt="d",
        annot_kws={'size': 12},
        cbar=False)
    h.axes.set_yticklabels(h.axes.get_ymajorticklabels(), fontsize=12)
    h.axes.set_xticklabels(h.axes.get_xmajorticklabels(), fontsize=14)
    plt.subplots_adjust(bottom=0.05, top=0.99)
    plt.tight_layout()
    plt.xlabel('Hypothesis number', fontsize=16)
    plt.ylabel('Team ID', fontsize=16)

    # now plot modeling info for teams
    plt.subplot(1, 2, 2)
    hmap_data = metadata_merged.copy()
    hmap_data.iloc[:, :] = 1
    seaborn.heatmap(
        hmap_data,
        annot=metadata_merged,
        fmt='',
        annot_kws={'size': 12},
        cbar=False,
        cmap=colors.ListedColormap(
            ['#FFFFFF']),
        yticklabels=False)
    ax = plt.gca()
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(
        narps.dirs.dirs['figures'],
        'SuppFigure1.png'))


if __name__ == "__main__":
    # set an environment variable called NARPS_BASEDIR
    # with location of base directory
    if 'NARPS_BASEDIR' in os.environ:
        basedir = os.environ['NARPS_BASEDIR']
    else:
        basedir = '/data'

    # setup main class
    narps = Narps(basedir)
    narps.load_data()

    logfile = os.path.join(
        narps.dirs.dirs['logs'],
        'MakeSupplementaryFigure1.txt')

    log_to_file(
        logfile,
        'running MakeSupplementaryFigure1.py',
        flush=True)

    metadata = get_all_metadata(narps)
    mk_supp_figure1(narps, metadata)
