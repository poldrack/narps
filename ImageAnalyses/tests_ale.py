# tests for narps code
# - currently these are all just smoke tests

import pytest
import os
import pandas
from narps import Narps
from MetaAnalysis import extract_peak_coordinates,\
    run_ALE, save_results,\
    make_figures, make_combined_figure
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
    _ = narps.dirs.get_output_dir('ALE')
    return(narps)


# tests
# run ALE meta-analysis

def test_run_ALE(narps):
    _ = narps.dirs.get_output_dir('ALE')

    # extract peak coordinates
    for hyp in range(1, 10):
        ds_dict = extract_peak_coordinates(
            narps,
            hyp)

        # Performing ALE
        res = run_ALE(ds_dict, hyp, narps)
        images = save_results(hyp, res, narps)
        make_figures(narps, hyp, images)


def test_combined_figure(narps):
    # make a figure with all hypotheses
    make_combined_figure(narps)
