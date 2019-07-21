"""
tests for narps code
- currently these are all just smoke tests
"""

# contents of conftest.py
import pytest
import pandas
import os
from narps import Narps
from AnalyzeMaps import mk_overlap_maps
from PrepareMetadata import prepare_metadata

# use a session-scoped fixture to 
# save data for entire session

@pytest.fixture(scope="session")
def basedir(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data")
    return fn

# tests
def test_narps_main_class(basedir):
    narps = Narps(basedir)
    narps.write_data()

def test_narps_get_binarized_thresh_masks(basedir):
    narps = Narps(basedir)
    narps.load_data()
    narps.get_binarized_thresh_masks()
    narps.write_data()

def test_narps_get_resampled_images(basedir):
    narps = Narps(basedir)
    narps.load_data()
    narps.get_resampled_images()
    narps.write_data()

def test_narps_create_concat_images_thresh(basedir):
    narps = Narps(basedir)
    narps.load_data()
    narps.create_concat_images(datatype='resampled',
                               imgtypes=['thresh'])

def test_narps_check_image_values(basedir):
    narps = Narps(basedir)
    narps.load_data()
    image_metadata_df = narps.check_image_values()

def test_narps_create_rectified_images(basedir):
    narps = Narps(basedir)
    narps.load_data()
    narps.create_rectified_images()
    narps.write_data()

def test_narps_create_thresh_overlap_images(basedir):
    narps = Narps(basedir)
    narps.load_data()
    narps.create_thresh_overlap_images()

def test_narps_convert_to_zscores(basedir):
    narps = Narps(basedir)
    narps.load_data()
    narps.convert_to_zscores()
    narps.write_data()

def test_narps_create_concat_images_unthresh(basedir):
    narps = Narps(basedir)
    narps.load_data()
    narps.create_concat_images(datatype='zstat',
                               imgtypes=['unthresh'])

def test_narps_compute_image_stats(basedir):
    narps = Narps(basedir)
    narps.load_data()
    narps.compute_image_stats()

def test_narps_estimate_smoothness(basedir):
    narps = Narps(basedir)
    narps.load_data()
    _ = narps.estimate_smoothness()
    narps.write_data()

def test_prepare_metadata(basedir):
    narps = Narps(basedir)
    narps.load_data()
    prepare_metadata(narps)
    
def test_analyze_maps(basedir):
    # setup main class
    narps = Narps(basedir)
    narps.load_data()

    # Load full metadata and put into narps structure
    narps.metadata = pandas.read_csv(
        os.path.join(narps.dirs.dirs['metadata'], 'all_metadata.csv'))

    # create maps showing overlap of thresholded images
    mk_overlap_maps(narps)
