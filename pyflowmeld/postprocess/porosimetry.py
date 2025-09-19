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

import numpy as np
from os import PathLike 
from typing import Sequence, Literal, Optional  
from .. utils import tools

def porosity_is(file_info: PathLike,
    domain_size: Sequence,
    padding: Optional[Sequence] = None) -> float:
    """
    Computes porosity (void fraction) from LBM input data files.

    Porosity is defined as the fraction of the domain's volume that is void (empty space).
    Solid nodes are marked with values 1 and 2.

    Args:
        file_info (PathLike): Path to the file containing the domain data.
        domain_size (Sequence[int]): Shape of the domain as a sequence (e.g., [NX, NY, NZ]).
        padding (Optional[Sequence[int]]): Padding sequence to be removed 
            ([x_min, x_max, y_min, y_max, z_min, z_max]). Default is None.

    Returns:
        float: The void fraction (porosity) of the domain.

    Examples:
        >>> # Assuming domain is 100x100x100 and solid nodes are marked with 1 or 2
        >>> from postprocess.porosimetry import porosity_is
        >>> porosity = porosity_is("domain_data.txt", [100, 100, 100], [2, 2, 2, 2, 2, 2])
        >>> print(porosity)
        0.45
    """
    domain = np.loadtxt(file_info).reshape(domain_size)
    if padding is not None:
        domain = tools.remove_padding(domain = domain, padding = padding)
    solid_nodes = (domain == 1) | (domain == 2)
    void_fraction = 1 - np.sum(solid_nodes)
    return void_fraction 

def surface_area_is(file_info: PathLike,
                     domain_size: Sequence,
                         padding: Optional[Sequence] = None,
                             sum_axis: Literal['x', 'y', 'z'] = 'y') -> float:
    """
    Computes the surface area fraction of a slice along a specific axis 
    from LBM input data files.

    Surface area fraction is defined as the proportion of void space 
    for all slices along the specified axis.

    Args:
        file_info (PathLike): Path to the file containing the domain data.
        domain_size (Sequence[int]): Shape of the domain as a sequence (e.g., [NX, NY, NZ]).
        padding (Optional[Sequence[int]]): Padding sequence to be removed 
            ([x_min, x_max, y_min, y_max, z_min, z_max]). Default is None.
        sum_axis (Literal['x', 'y', 'z']): Axis along which slices are summed. 
            Default is 'y'.

    Returns:
        float: The mean surface area fraction across all slices on the specified axis.

    Examples:
        >>> # Compute surface area fraction along the 'y' axis for a 100x100x100 domain
        >>> from postprocess.porosimetry import surface_area_is
        >>> surface_area = surface_area_is("domain_data.txt", [100, 100, 100], [2, 2, 2, 2, 2, 2], 'y')
        >>> print(surface_area)
        0.35
    """
    axis = {'y': (0,2), 'x':(1,2), 'z':(0,1)}[sum_axis]
    domain = np.loadtxt(file_info).reshape(domain_size)
    if padding is not None:
        domain = tools.remove_padding(domain = domain, padding = padding)
    solid_nodes = (domain == 1) | (domain == 2)
    total_area = np.prod([domain.shape[i] for i in axis])
    surface_area_fraction = 1 - np.mean(solid_nodes, axis=axis) / total_area
    return surface_area_fraction

