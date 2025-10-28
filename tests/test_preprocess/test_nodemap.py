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

from pyflowmeld.preprocess.nodemap import NodeMap


class ConcreteNodeMap(NodeMap):
    def add_phases(self):
        pass 

    @classmethod 
    def from_file(cls, *args, **kwargs):
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

        expected_file = Path(temp_dir) / "orig_domain_no_pad_testcase.dat"
        assert expected_file.exists()
        assert nm.which_sidewall == []
    
    #-- testing add padding --#
    @pytest.mark.parametrize(
    "padding, expected_shape",
    [
        (None, (5, 5, 5)),             # No padding
        (0, (5, 5, 5)),                # 0-padding everywhere
        ([0, 0, 0, 0, 0, 0], (5, 5, 5)),
        (1, (7, 7, 7)),                # pad 1 on each side, all axes: original + 2*1 each
        ([1, 2, 0, 0, 0, 0], (8, 5, 5)),  # different min/max on x
        ([1, 2, 3, 4, 0, 0], (8, 12, 5)), # nonzero on y
        ([0, 0, 0, 0, 5, 6], (5, 5, 16)), # only z
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

    



