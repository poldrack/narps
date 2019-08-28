# tests for narps code
# - currently these are all just smoke tests

import pytest
import os
import pandas
from narps import Narps
from AnalyzeMaps import mk_overlap_maps,\
    mk_range_maps, mk_std_maps,\
    mk_correlation_maps_unthresh, analyze_clusters,\
    plot_distance_from_mean, get_thresh_similarity
from MetaAnalysis import get_thresholded_Z_maps
from ThreshVoxelStatistics import get_thresh_voxel_stats
# Use a fixed base dir so that we can
# access the results as a circleci artifact


@pytest.fixture(scope="session")
def narps():
    basedir = '/tmp/data'
    assert os.path.exists(basedir)
    narps = Narps(basedir)
    narps.load_data()
    narps.metadata = pandas.read_csv(
        os.path.join(narps.dirs.dirs['metadata'], 'all_metadata.csv'))
    return(narps)


# tests
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


# this was created for ALE but we do it earlier here
def test_thresh_zmap(narps):
    # create thresholded versions of Z maps
    narps = get_thresholded_Z_maps(
        narps)


def test_thresh_voxel_stats(narps):
    get_thresh_voxel_stats(narps.basedir)
