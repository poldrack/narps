# tests for narps code
# - currently these are all just smoke tests

import pytest
import pandas
import os
from narps import Narps
from PrepareMetadata import prepare_metadata
# Use a fixed base dir so that we can
# access the results as a circleci artifact


@pytest.fixture(scope="session")
def narps():
    dataurl = os.environ['DATA_URL']
    basedir = '/tmp/data'
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    narps = Narps(basedir, dataurl=dataurl)
    narps.write_data()
    return(narps)

# tests


def test_narps_get_binarized_thresh_masks(narps):
    narps.get_binarized_thresh_masks()


def test_narps_get_resampled_images(narps):
    narps.get_resampled_images()


def test_narps_create_concat_images_thresh(narps):
    narps.create_concat_images(datatype='resampled',
                               imgtypes=['thresh'])


def test_narps_check_image_values(narps):
    _ = narps.check_image_values()


def test_narps_create_rectified_images(narps):
    narps.create_rectified_images()


def test_narps_create_thresh_overlap_images(narps):
    narps.create_mean_thresholded_images()


def test_narps_convert_to_zscores(narps):
    narps.convert_to_zscores()


def test_narps_create_concat_images_unthresh(narps):
    narps.create_concat_images(datatype='zstat',
                               imgtypes=['unthresh'],
                               create_voxel_map=True)


def test_narps_compute_image_stats(narps):
    narps.compute_image_stats()


def test_narps_estimate_smoothness(narps):
    _ = narps.estimate_smoothness()


# PrepareMetadata
def test_prepare_metadata(narps):
    prepare_metadata(narps)
    # Load full metadata and put into narps structure
    narps.metadata = pandas.read_csv(
        os.path.join(narps.dirs.dirs['metadata'], 'all_metadata.csv'))
