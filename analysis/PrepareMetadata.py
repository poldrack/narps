#!/usr/bin/env python
# coding: utf-8
"""
Consolidates the preparation of metadata for the analyses.
It requires that narps.py has already been run.
"""

import os
import pandas
from narps import Narps
from utils import get_merged_metadata_decisions


def package_recoder(p):
    others = ['nistats', 'PALM', 'randomise']
    if not isinstance(p, str):
        return('Other')
    if p.find('SPM') == 0:
        return('SPM')
    elif p in others:
        return('Other')
    else:
        return p


def prepare_metadata(narps):
    # get original image and decision metadata
    alldata_df = get_merged_metadata_decisions(
        narps.metadata_file,
        os.path.join(narps.dirs.dirs['orig'], 'narps_results.xlsx'))
    print('found merged metadata for %d teams' %
          alldata_df.teamID.unique().shape[0])

    # change type of varnum to int
    alldata_df['varnum'] = alldata_df['varnum'].astype('int')

    # recode variables to make analysis cleaner
    alldata_df['software'] = [
        package_recoder(x) for x in alldata_df['TSc_SW']]

    # load smoothness data
    smoothness_df = pandas.read_csv(
        os.path.join(
            narps.dirs.dirs['metadata'],
            'smoothness_est.csv'))
    print("found smoothness data for %d teams" %
          len(smoothness_df.teamID.unique()))
    print('missing smoothness data for:')
    print(set(narps.complete_image_sets).difference(
          set(smoothness_df.teamID.unique())))

    alldata_df = pandas.merge(
        alldata_df, smoothness_df,
        how='left',
        left_on=['teamID', 'varnum'],
        right_on=['teamID', 'hyp'])

    # average FWHM estimated as:
    # AvgFWHM = RESELS^(1/3)
    # (multplied by 2 since this value is in voxels
    # rather than mm) per:
    # https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=FSL;e792b5da.0803

    alldata_df['fwhm'] = [i**(1/3.)*2 for i in alldata_df.resels]

    # save data for loading into R
    alldata_df.to_csv(os.path.join(
        narps.dirs.dirs['metadata'], 'all_metadata.csv'))


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

    overwrite = False

    # setup main class
    narps = Narps(basedir, overwrite=overwrite)

    prepare_metadata(narps)
