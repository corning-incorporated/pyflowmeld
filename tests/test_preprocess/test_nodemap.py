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

""" 
test suite to test classes in the nodemap preprocess module
"""
import pytest 
import numpy as np
from pathlib import Path 

from pyflowmeld.preprocess._base import NodeMap
from pyflowmeld.preprocess.nodemap import DryingNodeMap 


class ConcreteNodeMap(NodeMap):
    def add_phases(self):
        pass 


#--- Test class for NodeMap Base class ---#
class TestNodeMap:
    """
    Test class for NodeMap base class
    """
    #-- add test methods here --#
    def test_node_map_initialization(self, temp_dir):
        shape = (20,20,20)
        domain = np.ones(shape, dtype = int)
        nm = ConcreteNodeMap(domain = domain, file_stem = "testcase", save_path = temp_dir)
        assert np.array_equal(nm.domain, domain)
        assert nm.file_stem == "testcase"
        assert nm.save_path == Path(temp_dir)
        assert (nm.padding == np.zeros(6)).all(), "padding is not all zero"
        assert (nm.side_walls == np.zeros(6)).all(), "side walls are not all zero"
        assert (nm.geometry_side_walls == np.zeros(6)).all(), "geometry side walls are not all zero"
        assert nm.domain_shape ==  shape
        assert np.isclose(nm.void_fraction, 1 - np.sum(domain)/np.prod(shape))

    
    #-- testing add padding --#
    @pytest.mark.parametrize(
    "padding, expected_shape",
    [
        (None, (5, 5, 5)),             
        (0, (5, 5, 5)),                
        ([0, 0, 0, 0, 0, 0], (5, 5, 5)),
        (1, (7, 7, 7)),                
        ([1, 2, 0, 0, 0, 0], (8, 5, 5)),  
        ([1, 2, 3, 4, 0, 0], (8, 12, 5)), 
        ([0, 0, 0, 0, 5, 6], (5, 5, 16)), 
    ],           
    )
    def test_add_padding_works_for_all_valid_paddings(self, 
        temp_dir,
        padding, 
        expected_shape):
        """ 
        test add_padding works for all valid paddings
        """
        domain = np.ones((5,5,5), dtype = int)
        nm = ConcreteNodeMap(domain = domain, file_stem = "pad", 
            save_path = temp_dir, padding = padding)
        nm.add_padding()
        assert nm.domain.shape == expected_shape, "paddings were not applied correctly" 


    @pytest.mark.parametrize(
        "bad_padding",
        [
            [1, 2, 3],            
            [1, 2, 3, 4, 5, 6, 7],  
            "abc",                
            -1,                   
            [0, 1, -2, 0, 0, 0],  
        ],
    )
    def test_add_padding_raises_value_error_for_invalid_paddings(self, temp_dir, bad_padding):
        """
        Test ValueError or failure for invalid paddings. Guards already added in add_padding
        """
        domain = np.ones((4, 4, 4), dtype=int)
        if isinstance(bad_padding, (int, list)):
            if isinstance(bad_padding, int) and bad_padding < 0:
                # Negative integer should not be accepted for padding
                with pytest.raises(ValueError):
                    _ = ConcreteNodeMap(domain=domain, file_stem="bad", save_path=temp_dir, padding=bad_padding)
            elif isinstance(bad_padding, list) and any(n < 0 for n in bad_padding):
                with pytest.raises(ValueError):
                    _ = ConcreteNodeMap(domain=domain, file_stem="bad", save_path=temp_dir, padding=bad_padding)
            else:
                with pytest.raises(ValueError):
                    _ = ConcreteNodeMap(domain=domain, file_stem="bad", save_path=temp_dir, padding=bad_padding)
        else:
            with pytest.raises(ValueError):
                _ = ConcreteNodeMap(domain=domain, file_stem="bad", save_path=temp_dir, padding=bad_padding)


    def test_call_basic_function(self, temp_dir):
        """
        Tests the basic functionality of the __call__ and sequence of actions
        """
        shape = (50,50,50)
        domain = np.zeros(shape, dtype = int)
        cube_start, cube_end = 20,30
        domain[cube_start:cube_end, cube_start:cube_end, cube_start:cube_end] = 1
        nm = ConcreteNodeMap(domain=domain, file_stem="calltest", save_path=temp_dir)
        nm(slice_direction="z", bounce_method="circ", separate=True, vtk=False, multiphase=False)
        exported_files = [
        "node_map_calltest.dat",
        "node_map_calltest_info.txt",
        "orig_domain_no_pad_calltest.dat",
        "node_map_calltest_inert_fluid.csv",
        "node_map_calltest_solids.csv",
        "node_map_calltest_boundary.csv",]
        for fname in exported_files:
            assert (Path(temp_dir) / fname).exists(), f"Missing output file: {fname}"

        # Domain should contain both fluid and solid (not all zeros or all ones)
        vals, counts = np.unique(nm.domain, return_counts=True)
        assert 0 in vals and 1 in vals, "Domain should contain both fluid and solid nodes after processing"

    def test_add_sidewalls_sets_sidewall_faces(self, temp_dir):
        """
        Tests that add_sidewalls correctly sets the six faces of the domain
        according to side wall thickness configuration.
        """
        shape = (10, 10, 10)
        domain = np.zeros(shape, dtype=int)
        side_walls = [2, 1, 1, 2, 2, 1]  # [x_min, x_max, y_min, y_max, z_min, z_max]
        nm = ConcreteNodeMap(domain=domain, file_stem="swall", save_path=temp_dir, side_walls=side_walls)
        nm._add_sidewalls("domain")
        assert (nm.domain[0:2,:,:] == 1).all(), "x_min sidewall not set correctly"
        assert (nm.domain[-1,:,:] == 1).all(), "x_max sidewall not set correctly"
        assert (nm.domain[:,0,:] == 1).all(), "y_min sidewall not set correctly"
        assert (nm.domain[:,-2:,:] == 1).all(), "y_max sidewall not set correctly"
        assert (nm.domain[:,:,0:2] == 1).all(), "z_min sidewall not set correctly"
        assert (nm.domain[:,:,-1] == 1).all(), "z_max sidewall not set correctly"
        inner = nm.domain[2:-1, 1:-2, 2:-1]
        if inner.size > 0:
            assert (inner == 0).all(), "Inner non-sidewall region was overwritten"

    def test_from_file_shape_inference_and_loading(self, temp_dir):
        """
        Tests NodeMap.from_file: infers shape from header and loads correct domain.
        The test tests three conditions:
        1) header is passed by user 
        2) header is inferred from file header
        3) header is not passed at all, in which case an exception must be raised 
        """
        shape = (4, 3, 2)
        domain = np.arange(np.prod(shape)).reshape(shape)
        flat_str = "\n".join(str(x) for x in domain.flatten())
        file_path = Path(temp_dir) / "test_domain_file.dat"

        with open(file_path, "w") as f:
            f.write("# {} {} {}\n".format(*shape))
            f.write(flat_str)
            f.write("\n")

        nm = ConcreteNodeMap.from_file(file_path)
        assert nm.domain_shape == shape
        assert np.array_equal(nm.domain, domain), "Domain loaded from file is incorrect"

        nm2 = ConcreteNodeMap.from_file(file_path, domain_size=shape)
        assert nm2.domain_shape == shape
        assert np.array_equal(nm2.domain, domain), "Domain loaded with explicit shape is incorrect"

        file_path2 = Path(temp_dir) / "test_domain_no_header.dat"
        with open(file_path2, "w") as f:
            f.write(flat_str)
            f.write("\n")

        with pytest.raises(ValueError):
            _ = ConcreteNodeMap.from_file(file_path2)  
        nm3 = ConcreteNodeMap.from_file(file_path2, domain_size=shape)
        assert nm3.domain_shape == shape
        assert np.array_equal(nm3.domain, domain)

#--------------------------------------#
#    tests for DryingNodeMap class     #
# -------------------------------------# 
class TestDryingNodeMap:
    """
    Tests for DryingNodeMap class
    """
    def test_initialization_defaults_and_attrs(self, temp_dir):
        shape = (50, 50, 50)
        domain = np.random.randint(0,2, shape)
        nm = DryingNodeMap(domain=domain, file_stem="dry_default", save_path=temp_dir)
        zone = nm.gap_from_edge
        assert (zone.x_min, zone.y_min, zone.z_min, zone.x_max, zone.y_max, zone.z_max) == (2, 2, 2, 2, 2, 2)
        assert np.array_equal(nm.domain, domain)
        assert nm.file_stem == "dry_default"
        assert nm.save_path == Path(temp_dir)
        assert nm.domain_shape == shape

