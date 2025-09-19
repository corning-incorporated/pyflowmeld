
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

from os import path, makedirs, PathLike  
from typing import Optional, Sequence, Tuple   
import numpy as np 
import pandas as pd
from . import tools  

# ####################################### #
# Spheres with controlled porosity        #
# ####################################### #
class OverlappingSpherePack:
    """
    generates overlapping spheres 
    with controlled porosity
    drainage_swap_axes: if not None generates a geometry for drainage; 
    walls for drainage are added by the nodemap generator 
    poly_max_factor: factor to multiply the radii for incorporating polydispersity: a number larger than 1
    In this method, seeds inflate until reaching a designated porosity
    """
    def __init__(self, domain_size: np.ndarray, num_spheres: int,
                     porosity: float = 0.5, delta: float = 1.0, save: bool = False,
                         save_path: Optional[PathLike] = None, 
                        poly_max_factor: Optional[float] = None, drainage_swap_axes: Optional[Tuple] = None,
                          drainage_side_walls: Sequence = [0]*6):
        
        self.domain = np.zeros(domain_size)
        self.res_x, self.res_y, self.res_z = domain_size 
        self.domain_size = domain_size 
        self.domain_volume = self.res_x*self.res_y*self.res_z 
        x_seeds = np.random.uniform(0, self.res_x, num_spheres)
        y_seeds = np.random.uniform(0, self.res_y, num_spheres)
        z_seeds = np.random.uniform(0, self.res_z, num_spheres)
        self.seed_coordinates = np.c_[x_seeds[:,None], y_seeds[:, None], z_seeds[:, None]]

        # generates a polydisperse pack
        self.poly_radii = None 
        if poly_max_factor is not None:
            self.poly_radii = np.random.uniform(1, poly_max_factor, num_spheres)

        self.porosity = porosity
        self.delta = delta
        self.xx, self.yy, self.zz = np.meshgrid(np.arange(0, self.res_x), np.arange(0, self.res_y), 
                                                        np.arange(0, self.res_z), indexing = 'ij') 
        self.save_path = None 
        if save:
            if not path.exists(save_path):
                makedirs(save_path)
            self.save_path = save_path

        self._drainage_swap_axes = None 
        if drainage_swap_axes is not None:
            self._drainage_swap_axes = drainage_swap_axes 
            self.drainage_domain = None 
            self.side_walls = drainage_side_walls 
 
    def _solid_fraction_index(self, domain: np.ndarray) -> Tuple[float, Tuple]:
        solid_fraction = np.sum(domain)/self.domain_volume 
        solid_index = np.where(domain == 1)
        return solid_fraction, solid_index 

    def solid_content_monodisperse_is(self, domain: np.ndarray,
                                       radius: float) -> Tuple[float, Tuple]:
        for coord in self.seed_coordinates:
            index = np.where((self.xx - coord[0])**2 + (self.yy - coord[1])**2 + 
                                    (self.zz - coord[2])**2 <= radius**2)
            domain[index] = 1 
        
        return self._solid_fraction_index(domain)
    
    def solid_content_polydisperse_is(self, domain: np.ndarray,
                                       radii: np.ndarray) -> Tuple[float, Tuple]:
        for coord, radius in zip(self.seed_coordinates, radii):
            index = np.where((self.xx - coord[0])**2 + (self.yy - coord[1])**2 + 
                                    (self.zz - coord[2])**2 <= radius**2) 
            domain[index] = 1 
        
        return self._solid_fraction_index(domain)

    def save_drainage_domain(self, save_path: PathLike) -> None:
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
        tools.to_vtk(file_name = vtk_name, data_frame = domain_df)
        np.savetxt(dat_name, self.drainage_domain.flatten(), fmt = '%d', newline = '\n', 
                        header = f'{res_x} {res_y} {res_z}', comments = '#')
    
    def _generate_drainage_domain(self) -> None:
        drainage_domain = np.copy(self.domain)
        if self.side_walls[0] != 0:
            drainage_domain = drainage_domain[self.side_walls[0]:,:,:] 
        if self.side_walls[1] != 0:
            drainage_domain = drainage_domain[:-self.side_walls[1],:,:] 

        if self.side_walls[2] != 0 :
            drainage_domain = drainage_domain[:,self.side_walls[2]:,:] 
        if self.side_walls[3] != 0:
            drainage_domain = drainage_domain[:,:-self.side_walls[3],:] 
        
        if self.side_walls[4] != 0:
            drainage_domain = drainage_domain[:,:,self.side_walls[4]:] 
        if self.side_walls[5] != 0:
            drainage_domain = drainage_domain[:,:,:-self.side_walls[5]]
        
        self.drainage_domain = np.swapaxes(drainage_domain, self._drainage_swap_axes[0], 
                                                    self._drainage_swap_axes[1])
                
    def __call__(self):
        """ Note: coarse implementation of count results in overstepping porosity limits"""
        solid_fraction = 0 
        count = 1
        while solid_fraction < self.porosity:
            domain = np.zeros(self.domain_size)
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
