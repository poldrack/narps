# tests for narps code
# - minimal set for configuration, not for production use

import pytest
from narps import Narps

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
