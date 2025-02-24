import pytest


# use a session-scoped fixture to
# save data for entire session
@pytest.fixture(scope="session")
def basedir(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data")
    yield fn
