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

import numpy as np
import pandas as pd
from os import listdir, path, makedirs, PathLike 
from natsort import natsorted
from typing import List, Tuple, Sequence, Optional, Literal    
import re

def get_slice(
    domain_size: Sequence[int],
    direction: Literal['x', 'y', 'z'],
    coordinate: int
) -> Tuple[slice, slice, slice]:
    """
    Compute slice indices for a given direction and coordinate in the simulation domain.

    Parameters
    ----------
    domain_size : Sequence[int]
        Dimensions of the simulation domain as (nx, ny, nz).
    direction : {'x', 'y', 'z'}
        Direction of the slice ('x', 'y', or 'z').
    coordinate : int
        Coordinate value along the given direction.

    Returns
    -------
    Tuple[slice, slice, slice]
        A slicing tuple that can be applied to extract data in the specified direction.

    Raises
    ------
    ValueError
        If `direction` is invalid or `coordinate` is out of bounds.

    Example
    -------
    ```python
    domain_size = (100, 100, 100)
    direction = 'x'
    coordinate = 10

    result = get_slice(domain_size, direction, coordinate)
    print(result)  # Output: (slice(10, 11), slice(0, 100), slice(0, 100))
    ```
    """
    if direction not in {'x', 'y', 'z'}:
        raise ValueError(f"Invalid direction '{direction}'. Must be one of 'x', 'y', 'z'.")
    if coordinate < 0 or coordinate >= domain_size['xyz'.index(direction)]:
        raise ValueError(
            f"Coordinate '{coordinate}' is out of bounds for direction '{direction}'. "
            f"Valid range: 0 <= coordinate < {domain_size['xyz'.index(direction)]}"
        )
    
    # Slice mapping based on the direction
    slice_map = {
        'x': (slice(coordinate, coordinate + 1), slice(0, domain_size[1]), slice(0, domain_size[2])),
        'y': (slice(0, domain_size[0]), slice(coordinate, coordinate + 1), slice(0, domain_size[2])),
        'z': (slice(0, domain_size[0]), slice(0, domain_size[1]), slice(coordinate, coordinate + 1)),
    }
    return slice_map[direction]


# ##### writes a dataframe to vtk file ##### #
def to_vtk(file_name: PathLike, data_frame: pd.DataFrame) -> None:
    scalars = [col for col in data_frame.columns if col not in ['x', 'y', 'z']]
    with open(file_name, 'w+') as f:                        
        f.write('# vtk DataFile Version 4.2 \n')
        f.write('paraview vtk output\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        f.write('POINTS '+str(len(data_frame.index))+' float'+'\n')
            
        reduced_frame = data_frame.loc[:, ['x','y','z']]
        reduced_frame.to_csv(f, sep=' ', index = None, header=None, float_format='%.8f')  
        f.write('\n'*3)        
        
        if len(scalars) > 0:
            f.write('POINT_DATA '+str(len(data_frame.index))+'\n')   
            for col in scalars:
                f.write('SCALARS '+col+' float\n')
                f.write('LOOKUP_TABLE default\n')
                reduced_frame = data_frame[col]
                reduced_frame.to_csv(f, sep=' ', index= None, header=None, float_format='%.5f')     

# #### returns a clean tuple from an input list #### 
cleantuple = lambda args: tuple([arg for arg in args if arg != ''])

# #### generates RGB #### #
def generate_RGB(number: float) -> List[Tuple]:
    _RGB = [((1/255)*np.random.randint(0, 255), (1/255)*np.random.randint(0, 255),\
         (1/255)*np.random.randint(0, 255)) for\
         _ in range(number)]
    return _RGB


# ###### list files in a directory ###### #
list_files = lambda file_path, file_name: natsorted([path.join(file_path, _file) for _file\
    in listdir(file_path) if file_name in _file], key=lambda y: y.lower()) 

find_file = lambda file_path, file_stem: [_file for _file in listdir(file_path) if file_stem in _file][0]


# ##### make a directory ###### #
def make_dir(dir_name: PathLike) -> PathLike:
    if not path.exists(dir_name):
        makedirs(dir_name)
    return dir_name 

exists = lambda file_info: file_info if path.exists(file_info) else None


# #### if a string is float converticle #### #
def can_be_float(string: str) -> bool:
    if re.match(r'^-?\d+(?:\.\d+)$', string) is not None:
        return True 
    else:
        return False 

# takes a sequence (x,y,z,val) of arrays and outputs them to vtk
def arrays_to_vtk(arrays: Tuple[np.ndarray],
                     columns: List[str], file_name: PathLike, 
                       screen_by_values: Optional[Sequence] = None, 
            trim: Optional[Sequence] = None) -> None:
    out_arrays = np.hstack([arr.flatten()[:,np.newaxis] for arr in arrays])
    out_df = pd.DataFrame(out_arrays, columns = columns)
    
    if screen_by_values is not None:
        out_df = out_df[out_df[columns[-1]].isin(screen_by_values)]
    
    if trim is not None:
        u_min, u_max, v_min, v_max, w_min, w_max = trim
        u,v,w = tuple(columns[:3])
        out_df = out_df[(out_df[u] >= u_min) & (out_df[u] <= u_max) &
                        (out_df[v] >= v_min) & (out_df[v] <= v_max) &
                        (out_df[w] >= w_min) & (out_df[w] <= w_max)]
    
    with open(file_name, 'w+') as f:
        f.write('# vtk DataFile Version 4.2 \n')
        f.write('paraview vtk output\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        f.write('POINTS '+str(len(out_df.index))+' float'+'\n')
        out_df.iloc[:,0:3].to_csv(f, sep=' ', index = None, header=None, float_format='%.8f')  
        f.write('\n'*3)

        f.write('POINT_DATA ' + str(len(out_df.index)) + '\n')
        f.write('SCALARS ' + columns[-1] +' float\n')
        f.write('LOOKUP_TABLE default\n')
        out_df.iloc[:,3].to_csv(f, sep=' ', index = None, header = None, float_format='%.5f')

def df_to_vtk(frame: pd.DataFrame, file_name: PathLike):
    
    with open(file_name, 'w+') as f:
        f.write('# vtk DataFile Version 4.2 \n')
        f.write('paraview vtk output\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        f.write('POINTS '+str(len(frame.index))+' float'+'\n')
        frame.iloc[:,0:3].to_csv(f, sep=' ', index = None, header=None, float_format='%.8f')  
        f.write('\n'*3)

        f.write('POINT_DATA ' + str(len(frame.index)) + '\n')
        f.write('SCALARS ' + list(frame.columns)[-1] +' float\n')
        f.write('LOOKUP_TABLE default\n')
        frame.iloc[:,3].to_csv(f, sep=' ', index = None, header = None, float_format='%.5f')

    
def true_division(_num: np.ndarray, _denum: np.ndarray) -> np.ndarray:
    with np.errstate(divide = 'ignore', invalid='ignore'):
        _num = np.true_divide(_num,  _denum)
        _num[~np.isfinite(_num)] = 0
    return _num

def is_float(element: str) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False 

# ==== geometry tools ====#
# -- array refinement to increase resolution --#
def refine_array(domain: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    It upscales a mask (zero/nonzero) array by a given factor, 
    where each "on" voxel in the coarse domain becomes a solid block of 1s in the refined, larger domain.
    It's commonly used to refine or increase the resolution of binary masks.

    Parameters
    ----------
    domain : np.ndarray
        A 3D numpy array (usually binary/mask).
    factor : int
        Refinement factor for each spatial dimension.
        
    Returns
    -------
    np.ndarray
        Refined 3D array, shape is (l*factor, w*factor, h*factor).
    """
    if not (isinstance(domain, np.ndarray) and domain.ndim == 3):
        raise ValueError("domain must be a 3D numpy array to refine")
    if factor < 1:
        raise ValueError("domain compression not possible")
    if factor == 1:
        return domain.copy()

    return np.kron(domain, np.ones((factor, factor, factor), dtype = domain.dtype))

# remove paddings
def remove_padding(domain: np.ndarray, padding: Sequence) -> np.ndarray:
    if padding[0] != 0:
        domain = domain[padding[0]:,:,:]
    if padding[1] != 0:
        domain = domain[:-padding[1],:,:]
    if padding[2] != 0:
        domain = domain[:,padding[2]:,:]
    if padding[3] != 0:
        domain = domain[:,:-padding[3],]
    if padding[4] != 0:
        domain = domain[:,:,padding[4]:]
    if padding[5] != 0:
        domain = domain[:,:,:-padding[5]]
    return domain 

#-- loads the array from a file and reshapes and trims it --#
def load_reshape_trim(file_info: PathLike, domain_size: np.ndarray,
                         trim: Sequence = [0]*6) -> np.ndarray:
    """
    Loads an array from file, reshapes to domain_size, 
    and trims (removes) specified amount from each edge/padding.

    Parameters
    ----------
    file_info : PathLike
        Path to file to load (assumed numeric text, np.loadtxt compatible)
    domain_size : Sequence[int]
        Shape to reshape to (e.g. (nx, ny, nz))
    trim : Sequence[int]
        Padding to remove (left, right, front, back, bottom, top)

    Returns
    -------
    np.ndarray
        The trimmed array of shape:
            (nx - (left + right), ny - (front + back), nz - (bottom + top))
    """
    arr = np.loadtxt(file_info).reshape(domain_size)
    if len(trim) != 6:
        raise ValueError("Trim argument must be a sequence of len 6: (x0,x1,y0,y1,z0,z1)")
    x0,x1,y0,y1,z0,z1 = trim 
    slices = (
        slice(x0, None if x1 == 0 else -x1), 
        slice(y0, None if y1 == 0 else -y1), 
        slice(z1, None if z1 == 0 else -z1)
    )
    return arr[slices]


