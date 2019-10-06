"""
get stats for thresholded maps
"""

import os
import glob
import numpy
import pandas
import argparse
from narps import Narps
from ValueDiagnostics import compare_thresh_unthresh_values


def get_thresh_voxel_stats(basedir):
    # load cluster maps
    data_dir = os.path.join(
        basedir,
        'image_diagnostics_zstat')
    datafiles = glob.glob(os.path.join(
        data_dir, '*'
    ))
    datafiles.sort()
    all_data = None
    print('found %d data files' % len(datafiles))
    for d in datafiles:
        df = pandas.read_csv(d)
        if all_data is None:
            all_data = df
        else:
            all_data = pandas.concat([all_data, df])
    hypnums = all_data.hyp.unique()
    results_df = pandas.DataFrame({
        'Hyp #': hypnums,
        'Minimum sig voxels': None,
        'Maximum sig voxels': None,
        'Median sig voxels': None,
        'N empty images': None})

    for hyp in hypnums:
        hypdata = all_data.query('hyp == %d' % hyp)
        results_df.loc[
            results_df['Hyp #'] == hyp,
            'Minimum sig voxels'] = hypdata.n_thresh_vox.min()
        results_df.loc[
            results_df['Hyp #'] == hyp,
            'Maximum sig voxels'] = hypdata.n_thresh_vox.max()
        results_df.loc[
            results_df['Hyp #'] == hyp,
            'Median sig voxels'] = hypdata.n_thresh_vox.median()
        results_df.loc[
            results_df['Hyp #'] == hyp,
            'N empty images'] = numpy.sum(hypdata.n_thresh_vox == 0)

    results_df.to_csv(os.path.join(
        basedir, 'metadata/thresh_voxel_statistics.csv'),
        index=False)
    all_data.to_csv(os.path.join(
        basedir, 'metadata/thresh_voxel_data.csv'
    ))
    print(results_df)
    return(None)


# run diagnostics on zstat images
def get_zstat_diagnostics(narps,
                          verbose=True,
                          overwrite=False):
    for teamID in narps.teams:
        collectionID = '%s_%s' % (
            narps.teams[teamID].NV_collection_id,
            teamID)
        if verbose:
            print(collectionID)
        logfile = os.path.join(
            narps.dirs.dirs['logs'],
            'zstat_diagnostics.log')

        diagnostics_file = os.path.join(
            narps.dirs.dirs['image_diagnostics_zstat'],
            '%s.csv' % collectionID)
        if not os.path.exists(diagnostics_file)\
                or overwrite:
            image_diagnostics = compare_thresh_unthresh_values(
                narps.dirs, collectionID, logfile,
                unthresh_dataset='zstat',
                thresh_dataset='zstat')
            if image_diagnostics is not None:
                image_diagnostics.to_csv(diagnostics_file)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Get stats for thresholded maps')
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

    # setup main class
    narps = Narps(basedir)
    narps.load_data()

    # Load full metadata and put into narps structure
    narps.metadata = pandas.read_csv(
        os.path.join(narps.dirs.dirs['metadata'], 'all_metadata.csv'))
    if not args.test:

        get_zstat_diagnostics(narps)

        get_thresh_voxel_stats(narps.basedir)
