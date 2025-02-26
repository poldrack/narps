# tests for narps code
# - currently these are all just smoke tests

import pytest
import os
import pandas

from narps.ConsensusAnalysis import run_ttests, mk_figures
from narps.MakeSupplementaryFigure1 import mk_supp_figure1,\
    get_all_metadata
from narps.ClusterImageCorrelation import\
    cluster_image_correlation
from narps.ThreshVoxelStatistics import get_thresh_voxel_stats
from narps.MakeCombinedClusterFigures import\
    make_combined_cluster_figures
from tests import narps
# Use a fixed base dir so that we can
# access the results as a circleci artifact

# tests
# ConsensusAnalysis
@pytest.mark.order(1)
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




# compute cluster similarity
@pytest.mark.order(2)
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
@pytest.mark.order(3)
def test_get_thresh_voxel_stats(narps):
    get_thresh_voxel_stats(narps.basedir)


# make combined figures
@pytest.mark.order(4)
def test_make_combined_cluster_figures(narps):
    make_combined_cluster_figures(narps.basedir)


# MakeSupplementaryFigure1
@pytest.mark.order(5)
def test_mk_suppfigure1(narps):
    metadata = get_all_metadata(narps)
    mk_supp_figure1(narps, metadata)
