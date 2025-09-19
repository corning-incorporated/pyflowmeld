"""Root configuration for all tests."""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil
import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, '..', 'pyflowmeld_src')
src_path = os.path.abspath(src_path)

if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.fixture(scope="session")
def package_root():
    """Return the root directory of the package."""
    return Path(__file__).parent.parent / "pyflowmeld"


@pytest.fixture
def temp_dir():
    """Create a temporary directory that's automatically cleaned up."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)