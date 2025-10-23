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
# ##################################################### #
# Authors:                                              #
# Hamed Haddadi Staff Scientist                         #
#               haddadigh@corning.com                   #
# David Heine   Principal Scientist and Manager         #
#               heinedr@corning.com                     #
# ##################################################### #
import pytest 
import numpy as np 
from os import path 
from pyflowmeld.utils.tools import load_reshape_trim, refine_array

#-- test basic functions of load_reshape_trim --#
def test_load_reshape_trim_works_with_no_trim(temp_dir):
    """
    Tests if the main file has no padding
    The function should preserve the original shape
    Test:
        1) first creates a flat array
        2) function reads tha array file and reshapes and trims it
    """
    shape = (8,12,32)
    arr = np.arange(np.prod(shape)).reshape(shape)
    file_path = path.join(temp_dir, "array.txt")
    np.savetxt(file_path, arr.flatten())

    result = load_reshape_trim(file_path, shape, trim = [0]*6)
    assert np.array_equal(result, arr), "recovered array is not equal to the original shape"
    assert result.shape == shape, "recovered shape is not equal to the shape of original array"


#--- Test class for array refinement tests ---#
class TestRefineArray:
    """
    All tests for refine_array function 
    """
    def test_refine_array_preserves_density(self):
        """
        Tests if the refined array has a same density as the original array
        That means the ratio of solid to the total domain volume must always be preserved 
        """
        np.random.seed(42)
        original = (np.random.rand(5, 5, 5) > 0.7).astype(int)

        factor = 4
        refined = refine_array(original, factor=factor)

        orig_ones = np.count_nonzero(original)
        orig_total = original.size
        refined_ones = np.count_nonzero(refined)
        refined_total = refined.size

        # For each 1 in original, refine_array turns it into factor^3 ones
        assert refined_ones == orig_ones * factor ** 3

        # The ratio of 1s to total must be the same
        orig_ratio = orig_ones / orig_total
        refined_ratio = refined_ones / refined_total
        assert np.isclose(orig_ratio, refined_ratio), \
            f"Original ratio {orig_ratio}, refined ratio {refined_ratio}"   

    @pytest.mark.parametrize("factor", [1, 2, 3, 4, 5])
    def test_refined_array_density_and_shape(self, factor):
        np.random.seed(0)
        original = (np.random.rand(3, 3, 3) > 0.5).astype(int)
        refined = refine_array(original, factor=factor)

        # Shape should match expectation
        assert refined.shape == tuple(np.array(original.shape) * factor)

        # Count and ratio checks
        orig_ones = np.count_nonzero(original)
        orig_total = original.size
        refined_ones = np.count_nonzero(refined)
        refined_total = refined.size

        assert refined_ones == orig_ones * factor ** 3
        orig_ratio = orig_ones / orig_total
        refined_ratio = refined_ones / refined_total
        assert np.isclose(orig_ratio, refined_ratio)