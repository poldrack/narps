# tests for narps code
# - currently these are all just smoke tests

import pytest
import os
import pandas
from narps import Narps

from ThresholdingSim import run_all_analyses, make_plot

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
# ThresholdSim
def test_thresholding_sim(narps):
    narps.dirs.get_output_dir('ThresholdSimulation',
                              base='figures')

    all_results = run_all_analyses(narps)
    make_plot(narps, all_results)
