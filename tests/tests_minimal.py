# tests for narps code
# - minimal set for configuration, not for production use

import pytest
from narps.narps import Narps
from conftest import basedir


@pytest.fixture(scope="session")
def narps(basedir):
    narps = Narps(basedir)
    narps.write_data()
    return(narps)

# tests


def test_narps(narps):
    assert narps is not None
