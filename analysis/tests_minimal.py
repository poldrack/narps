# tests for narps code
# - minimal set for configuration, not for production use

import pytest
import pandas
import os
from narps import Narps
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


def test_narps(narps):
    pass
