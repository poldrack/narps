# tests for narps code
# - currently these are all just smoke tests

import pytest
import pandas
import os
from narps import Narps, setup_simulated_data,\
    make_orig_image_sets
from AnalyzeMaps import mk_overlap_maps,\
    mk_range_maps, mk_std_maps,\
    mk_correlation_maps_unthresh, analyze_clusters,\
    plot_distance_from_mean, get_thresh_similarity
from PrepareMetadata import prepare_metadata
from ConsensusAnalysis import run_ttests, mk_figures
from MakeSupplementaryFigure1 import mk_supp_figure1,\
    get_all_metadata

# use a session-scoped fixture to
# save data for entire session
@pytest.fixture(scope="session")
def basedir(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data")
    return fn


@pytest.fixture(scope="session")
def narps(basedir):
    narps = Narps(basedir)
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
    narps.create_thresh_overlap_images()


def test_narps_convert_to_zscores(narps):
    narps.convert_to_zscores()


def test_narps_create_concat_images_unthresh(narps):
    narps.create_concat_images(datatype='zstat',
                               imgtypes=['unthresh'])


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


# AnalyzeMaps
def test_mk_overlap_maps(narps):
    # create maps showing overlap of thresholded images
    mk_overlap_maps(narps)


def test_mk_range_maps(narps):
    mk_range_maps(narps)


def test_mk_std_maps(narps):
    mk_std_maps(narps)


def test_unthresh_correlation_analysis(narps):
    # conbine these into a single test
    # since they share data
    corr_type = 'spearman'
    dendrograms, membership = mk_correlation_maps_unthresh(
        narps, corr_type=corr_type)

    _ = analyze_clusters(
        narps,
        dendrograms,
        membership,
        corr_type=corr_type)


def test_plot_distance_from_mean(narps):
    plot_distance_from_mean(narps)


def test_get_thresh_similarity(narps):
    get_thresh_similarity(narps)


# ConsensusAnalysis
def test_consensus_analysis(narps):
    logfile = os.path.join(
        narps.dirs.dirs['logs'],
        'ConsensusAnalysis.txt')

    narps.dirs.dirs['consensus'] = os.path.join(
        narps.dirs.dirs['output'],
        'consensus_analysis')

    if not os.path.exists(narps.dirs.dirs['consensus']):
        os.mkdir(narps.dirs.dirs['consensus'])

    run_ttests(narps, logfile)
    mk_figures(narps, logfile)


# MakeSupplementaryFigure1
def test_mk_suppfigure1(narps):
    metadata = get_all_metadata(narps)
    mk_supp_figure1(narps, metadata)


# simulated data analysis
def test_simulated_data(narps):
    narps.metadata = pandas.read_csv(
        os.path.join(narps.dirs.dirs['metadata'],
                     'all_metadata.csv'))

    basedir = setup_simulated_data(narps, verbose=False)

    make_orig_image_sets(narps, basedir)
