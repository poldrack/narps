# tests for narps code
# - currently these are all just smoke tests

import pytest
import os
import pandas
from narps import Narps
from SimulateData import setup_simulated_data,\
    make_orig_image_sets
# Use a fixed base dir so that we can
# access the results as a circleci artifact


@pytest.fixture(scope="session")
def narps():
    basedir = '/tmp/data'
    assert os.path.exists(basedir)
    narps = Narps(basedir)
    narps.load_data()
    return(narps)


# tests
# simulated data analysis
def test_simulated_data(narps):
    narps.metadata = pandas.read_csv(
        os.path.join(narps.dirs.dirs['metadata'],
                     'all_metadata.csv'))

    basedir = setup_simulated_data(narps, verbose=False)

    make_orig_image_sets(narps, basedir)
