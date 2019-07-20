#!/usr/bin/env python
# coding: utf-8
"""
create supplementary figure 1
showing decisions and confidence/similarity
"""


import os
import sys
import pandas
import seaborn
import matplotlib.pyplot as plt
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

    plt.figure(figsize=(6, 20))
    h = seaborn.heatmap(
        decision_wide,
        cmap='prism',
        annot=confidence_wide,
        fmt="d",
        annot_kws={'size': 7},
        cbar=False)
    h.axes.set_yticklabels(h.axes.get_ymajorticklabels(), fontsize=7)
    plt.subplots_adjust(bottom=0.05, top=0.99)
    plt.tight_layout()
    plt.xlabel('Hypothesis number')
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
        '%s.txt' % sys.argv[0].split('.')[0])
    log_to_file(
        logfile, 'running %s' %
        sys.argv[0].split('.')[0],
        flush=True)

    metadata = get_all_metadata()
    mk_supp_figure1(narps, metadata)
