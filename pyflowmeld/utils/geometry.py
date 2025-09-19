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
from pathlib import Path 
from os import path, PathLike 
from typing import Optional, Iterable   
import timeit 
from typing import Sequence, Optional    


def flip_index(domain: np.ndarray, reverse: bool = True):
    """ flips domain index: 0 -> 1 or 1 -> 0 """
    index = {True:(0,1) , False: (1,0)}[reverse]
    domain = np.where(np.logical_or(domain == 1, domain == 2), index[0], index[1])
    return domain 

def slice_domain(domain: np.ndarray, slice_init_points: Sequence, slice_size: Sequence) -> np.ndarray:
    """ slice a domain using initial points and the slice length"""
    if len(slice_init_points) == 3:
        init_x, init_y, init_z = slice_init_points
        res_x, res_y, res_z = slice_size 
        domain = domain[init_x:init_x + res_x, init_y:init_y + res_y, init_z:init_z + res_z]
    elif len(slice_init_points) == 2:
        init_x, init_y = slice_init_points
        res_x, res_y = slice_size 
        domain = domain[init_x:init_x + res_x, init_y:init_y + res_y]
    else:
        raise ValueError(f"incorrect values passed for slice init points {slice_init_points}")
    return domain 

def read_dat_file(file_info: PathLike) -> np.ndarray:
    """
    reads a dat file formatted according to 
    x_resolution y_resolution z_resolution
    0
    1
    0
    ...
    Note that the number of rows after the header must be equal to x_resolution*y_resolution*z_resolution  
    """
    lines = open(file_info).read().splitlines()
    header = lines[0]
    if '#' in header:
        res_x, res_y, res_z = tuple([int(elem) for elem in header.split(' ')[1:]])
    else:
        res_x, res_y, res_z = tuple([int(elem) for elem in header.split(' ') if elem.isdigit()]) 
    
    start = timeit.default_timer()
    domain = np.loadtxt(file_info, dtype = int, skiprows = 2).reshape(res_x, res_y, res_z).astype(int)
    end = timeit.default_timer()
    print(f'finished reading the standard file in {end - start} ...')
    return domain 


def domain_to_df(domain: np.ndarray) -> pd.DataFrame:
    """ converts domain array to dataframe """
    res_x, res_y, res_z = domain.shape 
    xx, yy, zz = np.meshgrid(np.arange(0, res_x),
            np.arange(0, res_y),
        np.arange(0, res_z), indexing = 'ij')
    domain_df = pd.DataFrame(np.c_[xx.reshape((-1,1)),
                        yy.reshape((-1,1)),
                    zz.reshape((-1,1)),
                domain.reshape((-1,1))], columns = ['x', 'y', 'z', 'domain'])
    return domain_df[domain_df['domain'] != 0]    

def to_vtk(domain_df: pd.DataFrame, file_name: PathLike) -> None:
    """ converts dataframe to vtk file """
    with open(file_name, 'w+') as f:
        f.write('# vtk DataFile Version 4.2 \n')
        f.write('paraview vtk output\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        f.write('POINTS '+str(len(domain_df.index))+' float'+'\n')
        domain_df[['x','y','z']].to_csv(f, sep=' ', index = None, header=None, float_format='%.8f')  
        f.write('\n'*3)

def add_side_walls(domain: np.ndarray, side_walls: Sequence) -> np.ndarray:
    """ adds side walls to a domain by setting domain indices to 1 """
    if side_walls[0] != 0:
        domain[:side_walls[0], :, :] = 1
    if side_walls[1] != 0:
        domain[-side_walls[1]:, :, :] = 1 
    if side_walls[2] != 0:
        domain[:, :side_walls[2], :] = 1 
    if side_walls[3] != 0:
        domain[:, -side_walls[3]:, :] = 1 
    if side_walls[4] != 0:
        domain[:, :, :side_walls[4]] = 1 
    if side_walls[5] != 0:
        domain[:,:,-side_walls[5]:] = 1 
    return domain 

def solid_content(domain: np.ndarray) -> float:
    """ computes solid content of a domain """
    res_x, res_y, res_z = domain.shape 
    return np.sum(domain)/(res_x*res_y*res_z)

# ## grid refinement ## #
def porosity(domain: np.ndarray) -> float:
    """ computes domain porosity """
    res_x, res_y, res_z = domain.shape 
    return 1 - (np.sum(domain))/(res_x*res_y*res_z)


def compress(domain: np.ndarray, factor: int = 4, save_path: Optional[PathLike] = None) -> np.ndarray:
    """
    compressed a domain to smaller size without loss of domain morphology
    For example: a 100x100x100 domain with a cylinderical solid obstacle of radius = 30 can be compressed to
        a 50x50x50 domain of radius = 15 if factor = 2 is chosen
    """
    if factor > 1:
        indices = np.where(domain != 0)
        domain_shape = domain.shape 
        if len(domain_shape) == 3:
            l, w, h = domain_shape 
            comp_domain = np.zeros((l//factor, w//factor, h//factor))
            for i,j,k in zip(*indices):
                comp_slice = (slice(i//factor, (i + 1)//factor), 
                                slice(j//factor, (j + 1)//factor), 
                                    slice(k//factor, (k + 1)//factor))
                comp_domain[comp_slice] = 1 
            if save_path is not None:
                np.savetxt(path.join(save_path, 'compressed_domain.dat'),
                             comp_domain.flatten(), fmt = '%d')
        else:
            raise NotImplementedError('2d domains not implemented')
        return comp_domain 
    else:
        return domain 

def refine(domain: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    inverse operation as compress 
    """
    if factor > 1:
        indices = np.where(domain != 0)
        domain_shape = domain.shape 
        if len(domain_shape) == 3:
            l, w, h = domain_shape  
            fine_domain = np.zeros((l*factor, w*factor, h*factor))
            for i,j,k in zip(*indices):
                fine_slice = (slice(i*factor, (i + 1)*factor), 
                                    slice(j*factor, (j + 1)*factor), 
                                        slice(k*factor, (k + 1)*factor))
                fine_domain[fine_slice] = 1
        elif len(domain_shape) == 2:
            l, w = domain_shape 
            fine_domain = np.zeros((l*factor, w*factor))
            for i,j in zip(*indices):
                fine_slice = (slice(i*factor, (i + 1)*factor), 
                                    slice(j*factor, (j + 1)*factor))
                fine_domain[fine_slice] = 1
        return fine_domain 
    elif factor == 1:
        return domain
    else:
        raise ValueError('refine factor should be 1 or larger')

def slice_refine_dat(domain_file: str, domain_size: Iterable, refine_factor: int, slice_size: Optional[int] = None,
                      min_slice: Optional[Iterable] = None):
    """
    loads a domain, takes a slice if slize_size is not None and refines the domain
    note that it converts all nodes types to 0,1
    """
    
    file_path = Path(domain_file)
    save_path = file_path.parent.absolute()
    file_stem = file_path.name.split('.')[0]

    domain = np.loadtxt(domain_file).reshape(domain_size)
    domain = np.where(np.logical_or(domain == 1, domain == 2), 1, 0)

    if slice_size is not None:
        min_x, min_y, min_z = {True: (0,0,0), False: min_slice}[min_slice is None]
        domain = domain[min_x: min_x + slice_size, min_y: min_y + slice_size, min_z: min_z + slice_size]
    domain = refine(domain, refine_factor)

    file_name = file_stem + '_size_' + str(slice_size*3) + '_refined_by_' + str(refine_factor) + '.dat' 
    np.savetxt(path.join(save_path, file_name), domain.flatten(), fmt = '%d')


