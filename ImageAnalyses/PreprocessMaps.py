"""
Run the main preprocessing using the
functions defined in narps.py
"""

import argparse
import os
from narps import Narps
from SimulateData import setup_simulated_data,\
    make_orig_image_sets
import pandas


if __name__ == "__main__":
    # team data (from neurovault) should be in
    # # <basedir>/orig
    # some data need to be renamed before using -
    # see rename.sh in individual dirs

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Process NARPS data')
    parser.add_argument('-u', '--dataurl',
                        help='URL to download data')
    parser.add_argument('-b', '--basedir',
                        help=('base directory. '
                              'If not set, defaults to '
                              'env variable NARPS_BASEDIR - '
                              'that that env var is unset then'
                              'defaults to /data'))
    parser.add_argument('-s', '--simulate',
                        action='store_true',
                        help=('use simulated data - '
                              'requires that full analysis '
                              'has already been run'))
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

    if args.dataurl is not None:
        dataurl = args.dataurl
    elif 'DATA_URL' in os.environ:
        dataurl = os.environ['DATA_URL']
        print('reading data URL from environment')
    else:
        dataurl = None
        print('info.json no present - data downloading will be disabled')

    # set up simulation if specified
    if args.simulate:
        print('using simulated data')

        # load main class from real analysis
        narps_orig = Narps(basedir, 
                           overwrite=False)

        # create simulated data
        # setup main class from original data
        narps = Narps(basedir)
        narps.load_data()

        # Load full metadata and put into narps structure
        narps.load_metadata()
        narps.metadata = pandas.read_csv(
            os.path.join(narps.dirs.dirs['metadata'], 'all_metadata.csv'))

        basedir = setup_simulated_data(narps, verbose=False)

        make_orig_image_sets(narps, basedir)

        # doublecheck basedir name
        assert basedir.find('_simulated') > -1

    # setup main class

    narps = Narps(basedir, dataurl=dataurl)

    assert len(narps.complete_image_sets['thresh']) > 0

    if args.test:
        print('testing mode, exiting after setup')

    else:  # run all modules

        print('getting binarized/thresholded orig maps')
        narps.get_binarized_thresh_masks()

        print("getting resampled images...")
        narps.get_resampled_images()

        print("creating concatenated thresholded images...")
        narps.create_concat_images(datatype='resampled',
                                   imgtypes=['thresh'])

        print("checking image values...")
        image_metadata_df = narps.check_image_values()

        print("creating rectified images...")
        narps.create_rectified_images()

        print('Creating overlap(mean) images for thresholded maps...')
        narps.create_mean_thresholded_images()

        print('converting to z-scores')
        narps.convert_to_zscores()

        print("creating concatenated zstat images...")
        narps.create_concat_images(datatype='zstat',
                                   imgtypes=['unthresh'],
                                   create_voxel_map=True)

        print("computing image stats...")
        narps.compute_image_stats()

        print('estimating image smoothness')
        smoothness_df = narps.estimate_smoothness()

        # save directory structure
        narps.write_data()
