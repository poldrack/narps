"""
get stats for thresholded maps
"""

import os
import glob
import numpy
import pandas


def get_thresh_voxel_stats(basedir):
    # load cluster maps
    data_dir = os.path.join(
        basedir,
        'image_diagnostics')
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
        'hyp': hypnums,
        'min_nvox': None,
        'max_nvox': None,
        'median_nvox': None,
        'n_empty': None})

    for hyp in hypnums:
        hypdata = all_data.query('hyp == %d' % hyp)
        results_df.loc[
            results_df.hyp == hyp,
            'min_nvox'] = hypdata.n_thresh_vox.min()
        results_df.loc[
            results_df.hyp == hyp,
            'max_nvox'] = hypdata.n_thresh_vox.max()
        results_df.loc[
            results_df.hyp == hyp,
            'median_nvox'] = hypdata.n_thresh_vox.median()
        results_df.loc[
            results_df.hyp == hyp,
            'n_empty'] = numpy.sum(hypdata.n_thresh_vox == 0)

    print(results_df)
    return(results_df)


if __name__ == "__main__":

    basedir = os.environ['NARPS_BASEDIR']
    results_df = get_thresh_voxel_stats(basedir)
    results_df.to_csv(os.path.join(
        basedir, 'metadata/thresh_voxel_statistics.csv'
    ))
