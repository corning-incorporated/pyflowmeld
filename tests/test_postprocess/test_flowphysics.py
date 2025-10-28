# ################################################################################## #
#  Copyright 2025. Corning Incorporated. All rights reserved.                        #
#                                                                                    #
#  This software may only be used in accordance with the identified license(s).      #
#                                                                                    #   
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        #
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          #
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL           #
#  CORNING BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN        #
#  ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN                 #
#  CONNECTION WITH THE SOFTWARE OR THE USE OF THE SOFTWARE.                          #
#  ################################################################################# #
# Authors:                                                                           #
# Hamed Haddadi Staff Scientist                                                      #
#               haddadigh@corning.com                                                #
# David Heine   Principal Scientist and Manager                                      #
#               heinedr@corning.com                                                  #
# ################################################################################## #

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import os

from pyflowmeld.postprocess.flowphysics import PhaseFlow


# Fixtures
@pytest.fixture
def mock_domain_size():
    """Standard domain size for testing."""
    return (40, 40, 40)


@pytest.fixture
def mock_padding():
    """Standard padding for testing."""
    return [2, 2, 2, 2, 2, 2]

@pytest.fixture
def mock_delta_p():
    """ expected delta p"""
    return np.linspace(0.1, 0.8, 20)


@pytest.fixture
def mock_density_files(temp_dir, mock_domain_size, mock_delta_p):
    """Create mock density files in the temporary directory."""
    # Create mock f1 and f2 density files for 5 time steps
    for i in range(20):
        f1_file = os.path.join(temp_dir, f'f1_rho_{i:04d}.dat')
        f2_file = os.path.join(temp_dir, f'f2_rho_{i:04d}.dat')
        
        data = np.random.rand(np.prod(mock_domain_size)) * 1.0 + 0.25  
        np.savetxt(f1_file, data)
        np.savetxt(f2_file, data)
    
    # Create simulation.dat file
    sim_file = os.path.join(temp_dir, 'simulation.dat')
    with open(sim_file, 'w') as f:
        f.write("Line 1\n")
        f.write("Line 2\n")
        f.write(f"Delta P: {' '.join([str(elem) for elem in mock_delta_p])}\n")
    
    return temp_dir


@pytest.fixture
def phase_flow_instance(mock_density_files, mock_domain_size, mock_padding):
    """Create a PhaseFlow instance with mocked data."""
    return PhaseFlow(
        data_path=mock_density_files,
        domain_size=mock_domain_size,
        padding=mock_padding,
        rho_f1_stem='f1_rho_',
        rho_f2_stem='f2_rho_',
        isolated_volume=0
    )


# Test initialization
def test_phaseflow_init_valid_inputs(mock_density_files, 
                                     mock_domain_size,
                                       mock_padding, 
                                        mock_delta_p):
    """Test PhaseFlow initialization with valid inputs."""
    pf = PhaseFlow(
        data_path=mock_density_files,
        domain_size=mock_domain_size,
        padding=mock_padding
    )
    
    assert pf.domain_size == mock_domain_size
    assert pf._padding == mock_padding
    assert pf.isolated_volume == 0
    assert len(pf.rho1_files) == 20
    assert len(pf.rho2_files) == 20
    assert pf.f1_saturation == []
    assert pf.f2_saturation == []
    assert pf.medium is None
    assert np.allclose(pf.delta_p, mock_delta_p, rtol = 1e-5, atol = 1e-6) 

def test_phaseflow_init_invalid_data_path():
    """Test PhaseFlow initialization with invalid data path."""
    with pytest.raises(ValueError, match="Invalid data path"):
        PhaseFlow(
            data_path="/nonexistent/path",
            domain_size=(10, 10, 10),
            padding=[0]*6
        )

def test_phaseflow_init_invalid_padding(temp_dir):
    """Test PhaseFlow initialization with invalid padding."""
    with pytest.raises(ValueError, match="padding must be a sequence of 6 integers"):
        PhaseFlow(
            data_path=temp_dir,
            domain_size=(10, 10, 10),
            padding=[0, 0, 0]  # Only 3 values
        )

def test_phaseflow_init_no_density_files(temp_dir):
    """Test PhaseFlow initialization when no density files are found."""
    with pytest.raises(ValueError, match="No matching density files found"):
        PhaseFlow(
            data_path=temp_dir,
            domain_size=(10, 10, 10),
            padding=[0]*6
        )

def test_padding_getter(phase_flow_instance):
    """Test getting padding property."""
    assert phase_flow_instance.padding == [2, 2, 2, 2, 2, 2]


def test_padding_setter_with_int(phase_flow_instance):
    """Test setting padding with single integer."""
    phase_flow_instance.padding = 4
    assert phase_flow_instance.padding == [4, 4, 4, 4, 4, 4]
    assert phase_flow_instance.remove_padding is True


def test_padding_setter_with_list(phase_flow_instance):
    """Test setting padding with list."""
    new_padding = [1, 2, 3, 4, 5, 6]
    phase_flow_instance.padding = new_padding
    assert phase_flow_instance.padding == new_padding


def test_padding_setter_with_none(phase_flow_instance):
    """Test setting padding to None."""
    phase_flow_instance.padding = None
    assert phase_flow_instance.padding is None
    assert phase_flow_instance.remove_padding is False


def test_padding_setter_invalid_value(phase_flow_instance):
    """Test setting padding with invalid value."""
    with pytest.raises(ValueError, match="Invalid padding value"):
        phase_flow_instance.padding = [1, 2, 3]  # Only 3 values


# Test static methods
def test_median_method():
    """Test the median static method."""
    data = np.array([1, 2, 3, 4, 5])
    result = PhaseFlow.median(data)
    expected = 0.5 * (1 + 5)  # (min + max) / 2
    assert result == expected


def test_nonzero_min_method():
    """Test the nonzero_min static method."""
    data = np.array([0, 0, 2, 3, 4])
    result = PhaseFlow.nonzero_min(data)
    assert result == 2


def test_nonzero_min_all_zeros():
    """Test nonzero_min with all zeros."""
    data = np.array([0, 0, 0])
    with pytest.raises(ValueError, match="Array has no non-zero values"):
        PhaseFlow.nonzero_min(data)

@patch('pyflowmeld.postprocess.flowphysics.PhaseFlow._get_slice')
def test_generate_slice_csv(mock_get_slice, phase_flow_instance, temp_dir):
    """Test _generate_slice_csv method."""
    # Setup mock data
    phase_flow_instance.res_x = 8  # After padding
    phase_flow_instance.res_y = 8
    phase_flow_instance.res_z = 8
    
    # Create coordinate arrays
    phase_flow_instance.xx, phase_flow_instance.yy, phase_flow_instance.zz = np.meshgrid(
        np.arange(8), np.arange(8), np.arange(8), indexing='ij'
    )
    
    # Create mock medium
    phase_flow_instance.medium = np.ones((8, 8, 8))
    
    # Mock slice return
    mock_get_slice.return_value = (slice(4, 5), slice(None), slice(None))
    
    # Create test distribution
    f_dist = np.random.rand(8, 8, 8)
    
    # Call method
    phase_flow_instance._generate_slice_csv(
        slices={'x': 4},
        f_dist=f_dist,
        cycle=0,
        phase_id='wetting'
    )
    
    # Check if CSV file was created
    expected_file = os.path.join(
        phase_flow_instance.save_path,
        'wetting_density_on_slice_x_4_0.csv'
    )
    assert os.path.exists(expected_file)
    
    # Check medium file for cycle 0
    medium_file = os.path.join(
        phase_flow_instance.save_path,
        'medium_on_slice_x_4.csv'
    )
    assert os.path.exists(medium_file)