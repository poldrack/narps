#!/usr/bin/env python
# coding: utf-8
"""
create supplementary figure 1
showing decisions and confidence/similarity
"""


import os
import pandas
import seaborn
import matplotlib.pyplot as plt
from matplotlib import colors

from narps import Narps
from narps import NarpsDirs # noqa, flake8 issue
from utils import log_to_file


def get_all_metadata():
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
    # similarity_wide = metadata.pivot(
    #     index='teamID',
    #     columns='varnum',
    #     values='Similar')

    # from Rotem:
    # I think maybe change the colors to more gentle shades
    # such that the numbers will be clearer? E.g.
    # 9dc183 for green and #CD5C5C fir red?

    cmap = colors.ListedColormap(
        ['#CD5C5C', '#9dc183'])

    plt.figure(figsize=(6, 18))
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

    metadata = get_all_metadata()
    mk_supp_figure1(narps, metadata)
