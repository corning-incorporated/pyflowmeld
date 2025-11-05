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
from datetime import datetime 

from pyflowmeld.preprocess._base import NodeMap, ZoneConfig
from pyflowmeld.preprocess.nodemap import DryingNodeMap, drying_nodemap_from_benchmark
 

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
    
    #-- testing class instatiations --#
    def test_from_domain_array_basic_function(self, temp_dir):
        domain = np.random.randint(0,2,(20,30,20))
        nmap = ConcreteNodeMap.from_domain_array(domain, file_stem = "from_array", 
            save_path =temp_dir, padding = [4,4,0,0,0,0], 
                side_walls = [0,0,2,2,2,2],)
        assert isinstance(nmap, ConcreteNodeMap), "from_domain_array does not instantiate teh class"
        assert nmap.save_path == Path(temp_dir)
        assert np.array_equal(nmap.padding, np.array([4,4,0,0,0,0])), "incorrect padding"
        assert np.array_equal(nmap.side_walls, np.array([0,0,2,2,2,2])), "incorrect side walls"
        assert nmap.domain_shape == (20,30,20), "shape not correct"

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
    
    def test_add_phases_gap_exceeds_domain(self, temp_dir):
        """Test that add_phases with excessive gap_from_edge does not write outside domain."""
        shape = (20, 20, 20)
        domain = np.zeros(shape, dtype = int)
        large_gap = ZoneConfig(x_min=3, x_max=4, y_min=3, y_max=4, z_min=3, z_max=4)
        nm = DryingNodeMap(domain=domain, file_stem="dry_gap", save_path=temp_dir, gap_from_edge=large_gap)
        nm.add_phases()
    
        x_start, x_end = 3, 16
        y_start, y_end = 3, 16
        z_start, z_end = 3, 16

        inner = nm.domain[x_start:x_end, y_start:y_end, z_start:z_end]
        assert np.all(inner == 3), "Inner fluid region should be filled with 3s"

        outside = nm.domain.copy()
        outside[x_start:x_end, y_start:y_end, z_start:z_end] = 0
        assert not (outside == 3).any(), "No fluid should be added outside the specified region"
    
#------------------------------------------------------- #
# test the helper function drying_nodemap_from_benchmark #
#------------------------------------------------------- #
class TestDryingNodemapFromBenchmark:
    """
    Tests for drying_nodemap_from_benchmark function in nodemap.py
    """

    def test_overlapping_spheres_basic_function(self, temp_dir):
        # Call with valid arguments
        shape = (12, 8, 6)
        out_dir = temp_dir / "drying_bench"
        drying_nodemap_from_benchmark(
            "overlapping-spheres",
            save_path=str(out_dir),
            padding=[1, 1, 1, 1, 1, 1],
            side_walls=2,
            num_spheres=5,
            domain_size=shape,
            porosity=0.35,
            delta=1.0,
            separate=True,
            vtk=False,
            overwrite=False,
        )
        files = list(out_dir.glob('**/*'))
        file_exists = any(f.suffix in ['.dat', '.vtk'] for f in files)
        assert file_exists, f"No output .dat or .vtk files created in {out_dir} or its subfolders"

    def test_output_directory_conflict(self, temp_dir):
        shape = (6, 6, 6)
        out_dir = temp_dir / "existing_bench"

        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
        expected_subdir = out_dir / f"overlapping-spheres_{timestamp}"
        expected_subdir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileExistsError):
            drying_nodemap_from_benchmark(
                "overlapping-spheres",
                save_path=str(out_dir),
                num_spheres=2,
                domain_size=shape,
                porosity=0.1,
                overwrite=False,
            )

    def test_benchmark_with_drainage_domain(self, temp_dir):
        shape = (10, 8, 7)
        out_dir = temp_dir / "drainage"
        drying_nodemap_from_benchmark(
            "overlapping-spheres",
            save_path=str(out_dir),
            num_spheres=3,
            domain_size=shape,
            porosity=0.25,
            include_drainage_domain=True,
            drainage_swap_axes=(0, 1),  
        )
        drainage_subdirs = [d for d in out_dir.parent.iterdir() if d.is_dir() and d.name.startswith("drainage_")]
        assert drainage_subdirs, f"No drainage output subdirectory found in {out_dir.parent}"

        target_dir = max(drainage_subdirs, key=lambda d: d.stat().st_mtime)
        files = list(target_dir.glob('sphere_pack_drainage.*'))
        print(f"Drainage files found in {target_dir}:", [str(f) for f in files])
        assert files, f"Drainage domain files not found in {target_dir}"

