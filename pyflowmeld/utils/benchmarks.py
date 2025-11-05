
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

import sys 
from os import path, makedirs, PathLike
from pathlib import Path   
from typing import Optional, Sequence, Tuple, Union   
import numpy as np 
import pandas as pd

from pyflowmeld import find_package_root

package_root = find_package_root(Path(__file__))
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

try:
    from pyflowmeld.utils import tools
except ModuleNotFoundError:
    from . utils import tools 


# ####################################### #
# Spheres with controlled porosity        #
# ####################################### #
class OverlappingSpherePack:
    """
    generates overlapping spheres  with controlled porosity
    drainage_swap_axes: if not None generates a geometry for drainage; 
    walls for drainage are added by the nodemap generator 
    poly_max_factor: factor to multiply the radii for incorporating polydispersity: a number larger than 1
    In this method, seeds inflate until reaching a designated porosity
    """
    def __init__(
        self, 
        domain_size: Tuple,
        num_spheres: int,
        porosity: float = 0.5,
        delta: float = 1.0,
        save: bool = False,
        save_path: Optional[Path] = None, 
        poly_max_factor: Optional[float] = None,
        drainage_swap_axes: Optional[Tuple] = None,
        drainage_side_walls: Optional[Union[Sequence, int]] = None):

        if not isinstance(num_spheres, int) or num_spheres <= 0:
            raise ValueError("num_spheres must be a positive integer.")
        if poly_max_factor is not None and (not isinstance(poly_max_factor, (float, int)) or poly_max_factor <= 1):
            raise ValueError("poly_max_factor must be > 1.")
        if drainage_swap_axes is not None:
            if (not isinstance(drainage_swap_axes, (tuple, list)) or 
                len(drainage_swap_axes) != 2 or
                not all(ax in [0,1,2] for ax in drainage_swap_axes)):
                raise ValueError("drainage_swap_axes must be a tuple of two axis indices (0,1,2).")

        if drainage_side_walls is None:
            self.drainage_side_walls = [0]*6
        elif isinstance(drainage_side_walls, int):
            if drainage_side_walls < 0:
                raise ValueError("drainage_side_walls must be non-negative.")
            self.drainage_side_walls = [drainage_side_walls]*6
        elif (isinstance(drainage_side_walls, (list, tuple)) and len(drainage_side_walls) == 6 
            and all(isinstance(w, int) and w >= 0 for w in drainage_side_walls)):
            self.drainage_side_walls = list(drainage_side_walls)
        else:
            raise ValueError("drainage_side_walls must be None, int, or sequence of six non-negative ints.")
        
        self.domain_shape = domain_size 
        self.domain = np.zeros(self.domain_shape)
        self.res_x, self.res_y, self.res_z = domain_size 
        self.domain_volume = np.prod(self.domain_shape)

        x_seeds = np.random.uniform(0, self.res_x, num_spheres)
        y_seeds = np.random.uniform(0, self.res_y, num_spheres)
        z_seeds = np.random.uniform(0, self.res_z, num_spheres)
        self.seed_coordinates = np.c_[x_seeds, y_seeds, z_seeds]

        # generates a polydisperse pack
        self.poly_radii = np.random.uniform(1, poly_max_factor, num_spheres) if poly_max_factor is not None else None  
        
        self.porosity = porosity
        self.delta = delta
        self.xx, self.yy, self.zz = np.meshgrid(np.arange(0, self.res_x), np.arange(0, self.res_y), 
                                                        np.arange(0, self.res_z), indexing = 'ij') 
        self.save_path = None
        if save:
            if save_path is None:
                raise ValueError("save_path must be specified if save is True.")
            sp = Path(save_path)
            sp.mkdir(parents=True, exist_ok=True)
            self.save_path = sp

        # Drainage config
        self._drainage_swap_axes = drainage_swap_axes
        self.drainage_domain = None
            
 
    def _solid_fraction_index(self, domain: np.ndarray) -> Tuple[float, Tuple]:
        solid_fraction = np.sum(domain)/self.domain_volume 
        solid_index = np.where(domain == 1)
        return solid_fraction, solid_index 

    def solid_content_monodisperse_is(
        self, 
        domain: np.ndarray,
        radius: float) -> Tuple[float, Tuple]:
        for coord in self.seed_coordinates:
            index = np.where((self.xx - coord[0])**2 +
                            (self.yy - coord[1])**2 + 
                            (self.zz - coord[2])**2 <= radius**2)
            domain[index] = 1 
        
        return self._solid_fraction_index(domain)
    
    def solid_content_polydisperse_is(
        self, domain: np.ndarray,
        radii: np.ndarray) -> Tuple[float, Tuple]:
        for coord, radius in zip(self.seed_coordinates, radii):
            index = np.where((self.xx - coord[0])**2 + (self.yy - coord[1])**2 + 
                                    (self.zz - coord[2])**2 <= radius**2) 
            domain[index] = 1 
        
        return self._solid_fraction_index(domain)

    def save_drainage_domain(
        self, 
        save_path: Path) -> None:
        if self.drainage_domain is None:
            raise ValueError("Drainage domain has not been generated")
        print("DEBUG: saving drainage domain")

        res_x, res_y, res_z = self.drainage_domain.shape 
        xx, yy, zz = np.meshgrid(np.arange(0, res_x,), 
                                    np.arange(0, res_y), 
                                        np.arange(0, res_z), indexing = 'ij')
        domain_df = pd.DataFrame(np.c_[xx.flatten()[:,None], yy.flatten()[:, None], 
                                            zz.flatten()[:, None], self.drainage_domain.flatten()[:, None]], 
                                                columns = ['x', 'y', 'z', 'domain'])
        domain_df = domain_df[domain_df['domain'] >= 1]
        
        vtk_name = path.join(save_path, 'sphere_pack_drainage.vtk')
        dat_name = path.join(save_path, 'sphere_pack_drainage.dat')

        print("file names", vtk_name)
        print("dat name ", dat_name)
        try:
            tools.to_vtk(file_name = vtk_name, data_frame = domain_df)
            np.savetxt(dat_name, self.drainage_domain.flatten(), fmt = '%d', newline = '\n', 
                        header = f'{res_x} {res_y} {res_z}', comments = '#')
        except Exception as e:
            raise IOError(f"failed to save the drainage domain {e}")
    
    def _generate_drainage_domain(self) -> None:
        drainage_domain = np.copy(self.domain)
        shape = drainage_domain.shape 
        slices = []

        for i in range(3):
            min_sw = self.drainage_side_walls[2*i] 
            max_sw = self.drainage_side_walls[2*i + 1]
            start = min_sw 
            end = shape[i] - max_sw if max_sw != 0 else shape[i] 
            slices.append(slice(start, end))

        drainage_domain = drainage_domain[slices[0], slices[1], slices[2]]
        self.drainage_domain = np.swapaxes(drainage_domain, self._drainage_swap_axes[0], self._drainage_swap_axes[1])
                
    #--- call inflates sphers until reaching a designated fraction ---#
    def __call__(self):
        """ Note: coarse implementation of count results in overstepping porosity limits"""
        solid_fraction = 0 
        count = 1
        while solid_fraction < self.porosity:
            domain = np.zeros(self.domain_shape)
            radius = count*self.delta 
            if self.poly_radii is not None:
                print('generating polydisperse ...')
                solid_fraction, solid_index = self.solid_content_polydisperse_is(domain = domain,
                                                radii = radius*self.poly_radii)
            else:
                solid_fraction, solid_index = self.solid_content_monodisperse_is(domain = domain, radius = radius)
            print(f'the solid fraction at radius {radius} is {solid_fraction}')
            count += 1
        
        self.domain[solid_index] = 1 

        if self._drainage_swap_axes:
            self._generate_drainage_domain()

        xx = self.xx[solid_index]
        yy = self.yy[solid_index]
        zz = self.zz[solid_index]
        if self.save_path:
            out_df = pd.DataFrame(np.c_[xx[:,None], yy[:, None], zz[:, None]], columns = ['x', 'y', 'z'])
            tools.to_vtk(file_name = path.join(self.save_path, 'sphere_pack.vtk'),
                             data_frame = out_df)
