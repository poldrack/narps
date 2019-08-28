# tests for narps code
# - currently these are all just smoke tests

import pytest
import os
import pandas
from narps import Narps

from ConsensusAnalysis import run_ttests, mk_figures
from MakeSupplementaryFigure1 import mk_supp_figure1,\
    get_all_metadata
from ClusterImageCorrelation import\
    cluster_image_correlation
from ThreshVoxelStatistics import get_thresh_voxel_stats
from MakeCombinedClusterFigures import\
    make_combined_cluster_figures

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


# compute cluster similarity
def test_cluster_image_correlations(narps):
    cluster_image_correlation(
        narps.basedir,
        'TomEtAl',
        ['Task', 'Gain', 'Loss'])
    cluster_image_correlation(
        narps.basedir,
        'NARPS_mean',
        ['Task'])


# compute thresh statisics
def test_get_thresh_voxel_stats(narps):
    get_thresh_voxel_stats(narps.basedir)


# make combined figures
def test_make_combined_cluster_figures(narps):
    make_combined_cluster_figures(narps.basedir)
