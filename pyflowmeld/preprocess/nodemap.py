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
import pandas as pd
from typing import Optional, Literal, Union, List, Any, Tuple, Iterable    
from abc import ABCMeta, abstractmethod
from scipy.ndimage import distance_transform_edt as edt 
from pathlib import Path
from os import path, makedirs, PathLike 
from .. utils import geometry, benchmarks, tools   
from functools import partial   
from datetime import datetime 
import sys 
import timeit 
from functools import partial 

class BounceBackGen:
    """
    This class generates fluid and bounce-back nodes (0, 1, 2) for lattice-based simulations
    by applying circular shifts on the provided domain data. The generated nodes are used
    to define different states within a simulation grid:
    - 0: Fluid nodes
    - 1: Bounce-back boundary nodes
    - 2: Solid nodes

    Circular shifts are applied row-wise on the array to determine the neighbor states
    in different directions, allowing for precise categorization of nodes. It relies on
    defined shift patterns for i (current), i+1 (next), and i-1 (previous) directions.

    Attributes:
        domain (numpy.ndarray): The 3D array representing the simulation grid.
        res_x, res_y, res_z (int): Dimensions of the domain grid.
        _grand_domain (list): Stores the processed layers of the bounce-back domain.

    Methods:
        process_inlet(): Processes the inlet direction and generates bounce-back nodes.
        process_stack(): Processes the intermediate layers of the domain and generates nodes.
        process_outlet(): Processes the outlet direction and generates bounce-back nodes.
        __call__(): Integrates processing of inlet, intermediate stack, and outlet nodes.

    Examples:
        >>> domain = np.random.choice([0, 1], size=(50, 50, 50))
        >>> bounce_gen = BounceBackGen(domain=domain)
        >>> bounce_back_nodes = bounce_gen()
        domain is generated in 0.012 seconds ...

        Here, `bounce_back_nodes` will be a 3D array containing processed node states
        based on the input `domain` configuration.
    """
    # i
    i_shifts = [1, -1, 1, -1, (1,1), (-1,1), (1, -1), (-1, -1)]
    i_axes = [1, 1, 0, 0, (0, 1), (0,1), (0, 1), (0, 1)]

    # i + 1
    fi_shifts = [1, (1,1), (-1,1), (1,1), (-1,1), (1,1,1), (-1,1,1), (1,-1,1), (-1,-1,1)]
    fi_axes = [2, (1,2), (1,2), (0,2), (0,2), (0,1,2), (0,1,2), (0,1,2), (0,1,2)]
    
    # i - 1
    bi_shifts = [-1, (1,-1), (-1,-1), (1,-1), (-1,-1), (1,1,-1), (-1,1,-1), (1,-1,-1), (-1,-1,-1)]
    bi_axes = [2, (1,2), (1,2), (0,2), (0,2), (0,1,2), (0,1,2), (0,1,2), (0,1,2)] 

    def __init__(self, domain: Optional[np.ndarray] = None ):
        self.domain = domain 
        self.res_x, self.res_y, self.res_z = self.domain.shape
        self._grand_domain = []
    
    _get_slice = lambda self, coord: (slice(coord, coord + 1), slice(0, self.res_y), slice(0, self.res_z))
    

    def process_inlet(self):
        slice_index = self._get_slice(0)
        nx, ny = np.squeeze(self.domain[slice_index]).shape 
        B = np.zeros((nx, ny))
        W = np.zeros((nx, ny, 2))
        
        W[:,:,0] = np.squeeze(self.domain[self._get_slice(0)])
        W[:,:,1] = np.squeeze(self.domain[self._get_slice(1)])  

        # i:
        i_shifts = [np.roll(W, value, axis = ax)[:,:,0] for value, ax in zip(self.i_shifts, self.i_axes)]
        # i + 1:
        fi_shifts = [np.roll(W, value, axis = ax)[:,:,0] for value, ax in zip(self.fi_shifts, self.fi_axes)]
        shifts = i_shifts + fi_shifts       
        # compact loop
        for i in range(nx):
            for j in range(ny):
                if W[i,j,0] == 1 and shifts[0][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,0] == 1 and shifts[1][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,0] == 1 and shifts[2][i,j] == 0:
                    B[i,j] = 1                    
                elif W[i,j,0] == 1 and shifts[3][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,0] == 1 and shifts[4][i,j] == 0:
                    B[i,j] = 1                      
                elif W[i,j,0] == 1 and shifts[5][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,0] == 1 and shifts[6][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,0] == 1 and shifts[7][i,j] == 0:
                    B[i,j] = 1  
                elif W[i,j,0] == 1 and shifts[8][i,j] == 0:
                    B[i,j] = 1 
                elif W[i,j,0] == 1 and shifts[9][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,0] == 1 and shifts[10][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,0] == 1 and shifts[11][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,0] == 1 and shifts[12][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,0] == 1 and shifts[13][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,0] == 1 and shifts[14][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,0] == 1 and shifts[15][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,0] == 1 and shifts[16][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,0] == 0:
                    B[i,j] = 0
                else:
                    B[i,j] = 2
        
        self._grand_domain.append(B)
    
    def process_stack(self):

        for n_stack in range(1, self.res_x - 1):
            nx, ny = np.squeeze(self.domain[self._get_slice(n_stack)]).shape 
            B = np.zeros((nx, ny))
            W = np.zeros((nx, ny, 3))
            W[:,:,0] = np.squeeze(self.domain[self._get_slice(n_stack - 1)])
            W[:,:,1] = np.squeeze(self.domain[self._get_slice(n_stack)])
            W[:,:,2] = np.squeeze(self.domain[self._get_slice(n_stack + 1)])
            # i:
            i_shifts = [np.roll(W, value, axis = ax)[:,:,0] for value,ax in zip(self.i_shifts, self.i_axes)]
            # i - 1:
            bi_shifts = [np.roll(W, value, axis = ax)[:,:,0] for value,ax in zip(self.bi_shifts, self.bi_axes)]
            # i + 1:
            fi_shifts = [np.roll(W, value, axis = ax)[:,:,0] for value,ax in zip(self.fi_shifts, self.fi_axes)]
            shifts = i_shifts + bi_shifts + fi_shifts             

            for i in range(nx):
                for j in range(ny):
                    if W[i,j,1] == 1 and shifts[0][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[1][i,j] == 0:
                        B[i,j] = 1                
                    elif W[i,j,1] == 1 and shifts[2][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[4][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[5][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[6][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[7][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[8][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[9][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[10][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[11][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[12][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[13][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[14][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[15][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[16][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[17][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[18][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[19][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[20][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[21][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[22][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[23][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[24][i,j] == 0:
                        B[i,j] = 1
                    elif W[i,j,1] == 1 and shifts[25][i,j] == 0:
                        B[i,j] = 1    
                    elif W[i,j,1] == 0:
                        B[i,j] = 0
                    else:
                        B[i,j] = 2

            self._grand_domain.append(B) 
    
    def process_outlet(self):
        nx, ny = np.squeeze(self.domain[self._get_slice(self.res_x - 1)]).shape 
        B = np.zeros((nx, ny))
        W = np.zeros((nx, ny, 2))    
        
        W[:,:,0] = np.squeeze(self.domain[self._get_slice(self.res_x - 2)])
        W[:,:,1] = np.squeeze(self.domain[self._get_slice(self.res_x - 1)])
       # i:
        i_shifts = [np.roll(W, value, axis = ax)[:,:,0] for value, ax in zip(self.i_shifts, self.i_axes)]
        # i-1:
        bi_shifts = [np.roll(W, value, axis = ax)[:,:,0] for value, ax in zip(self.bi_shifts, self.bi_axes)]
        shifts = i_shifts + bi_shifts         

        for i in range(nx):
            for j in range(ny):
                if W[i,j,1] == 1 and shifts[0][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[1][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[2][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[3][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[4][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[5][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[6][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[7][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[8][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[9][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[10][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[11][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[12][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[13][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[14][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[15][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 1 and shifts[16][i,j] == 0:
                    B[i,j] = 1
                elif W[i,j,1] == 0:
                    B[i,j] = 0
                else:
                    B[i,j] = 2

        self._grand_domain.append(B)  

    def __call__(self):
        start = timeit.default_timer()  
        self.process_inlet()
        self.process_stack()
        self.process_outlet()
        end = timeit.default_timer()
        print(f'domain is generated in {end - start} seconds ...')
        return np.stack(self._grand_domain, axis = 0)


# ################################## #
# Base class for node map generation #
# ################################## #
class NodeMap(metaclass = ABCMeta):
    """
    Base class for generating nodemaps for simulation domains in fluid dynamics and multiphase simulations.
    Domain and Geometry are normally different but can also be the same. 
    Domain is a result of adding padding to the geometry.
    This class supports:
        - Adding padding (x, y, z) to the simulation domain.
        - Creating sidewalls around the domain.
        - Supporting bounce-back conditions using circular shifts.
        - release boundaries, and centralized objects (e.g., droplets, bridges).
        - Exporting simulation domains to `.dat`, `.csv`, and `.vtk` formats.

    Attributes:
        domain (numpy.ndarray): The input 3D array representing the simulation domain.
        padding (numpy.array): Amount of padding added to each side of the domain ([x_min, x_max, y_min, y_max, z_min, z_max]).
        side_walls (numpy.array): Size of sidewalls added to the domain (default is zeros for no walls).
        geometry_side_walls (numpy.array): Sidewalls applied selectively to geometries.
        save_path (str): Directory to save the exported files.
        file_name (str): Stem name used for output files.
        fluid_nodes, boundary_nodes, solid_nodes (tuple): Node indices for fluid, boundary, and solid nodes.
        void_fraction (float): Fractional void in the domain computed from solid nodes.

    Methods:
        add_padding(): Adds padding to the domain (in x, y, z directions).
        add_sidewalls(side_wall_key): Adds sidewalls to the domain based on `side_walls` or `geometry_side_walls`.
        add_bounceback(): Updates the domain with bounce-back boundary conditions using the predefined method.
        add_phases(): Abstract method for adding simulation phases, implemented in subclasses.
        to_dat(), to_csv(), to_vtk(): Methods for exporting the nodemap domain in various formats.
        from_file(): Abstract method for constructing a NodeMap from a file.
        from_lbm_dat(), from_array(): Class methods for constructing nodemap instances from different sources.

    Examples:
        >>> domain = np.zeros((10, 10, 10))
        >>> nodemap = NodeMap(domain=domain, padding=[1, 1, 1, 1, 1, 1], save_path="./")

        This creates a domain with padded boundaries. 
        The nodemap is then exported to a `.dat` file in the specified directory.

    Notes:
        - This is an abstract base class intended for subclassing. The `add_phases` method must be implemented
          in any subclass defining specific nodemap behavior for multiphase simulations, drying simulations, etc.
     for all nodemap generators
    """
    def __init__(
        self, 
        domain: np.ndarray, 
        file_stem: Optional[str] = None, 
        save_path: Optional[Union[str, Path]] = None, 
        padding: Optional[Union[int, List[int]]] = None, 
        side_walls: Optional[Union[int, List[int]]] = None, 
        geometry_side_walls: Optional[Union[int, List[int]]] = None, 
        **kw: Any
    ) -> None:
        """
        Initialize the NodeMap object.

        Args:
            domain (np.ndarray): The 3D array representing the domain to process.
            file_stem (Optional[str]): The stem for file names during export.
            save_path (Optional[Union[str, Path]]): Directory to save exported files.
            padding (Optional[Union[int, List[int]]]): Padding values for each dimension 
                [x_min, x_max, y_min, y_max, z_min, z_max]. Default is zeros (no padding).
            side_walls (Optional[Union[int, List[int]]]): Thickness of sidewalls for each dimension
                [x_min, x_max, y_min, y_max, z_min, z_max]. Default is zeros (no walls).
            geometry_side_walls (Optional[Union[int, List[int]]]): encloses the actual geometry with side walls. 
                good to apply bounceback boundary condition on the geometry. 
            **kw (Any): Additional arguments for subclass-specific configurations.

        Raises:
            ValueError: If `domain` is not a valid 3D numpy array.
        """
        
        # Validate domain input
        if domain is None or not isinstance(domain, np.ndarray):
            raise ValueError("A valid 3D numpy array for domain must be provided.")
        self.domain = domain

        # Initialize file info
        self.file_stem = file_stem
        self.save_path = Path(save_path) if save_path else Path(".")  # Defaults to the current directory if save_path is None
        self.file_name = f'node_map_{file_stem}' if file_stem else 'node_map'

        # Initialize padding and sidewall parameters
        self.padding = self._normalize_input(padding, default=0)
        self.side_walls = self._normalize_input(side_walls, default=0)
        self.geometry_side_walls = self._normalize_input(geometry_side_walls, default=0)
        
        # Initialize additional attributes
        self.fluid_nodes: Optional[Tuple[np.ndarray, ...]] = None
        self.boundary_nodes: Optional[Tuple[np.ndarray, ...]] = None
        self.solid_nodes: Optional[Tuple[np.ndarray, ...]] = None
        self.total_number_fluid: Optional[int] = None
        self.domain_volume = np.prod(self.domain_shape)
        self.void_fraction = 1 - np.sum(self.domain)/self.domain_volume

        self.which_sidewall: List[str] = []
        if not all(elem == 0 for elem in self.geometry_side_walls):
            self.which_sidewall.append("geometry")
        if not all(elem == 0 for elem in self.side_walls):
            self.which_sidewall.append("domain")

        # Save original domain (no padding)
        np.savetxt(self.save_path / f'orig_domain_no_pad_{file_stem}.dat', self.domain.flatten(), fmt='%d')
        

    @staticmethod
    def _normalize_input(value, default=0):
        """
        Normalizes padding and side_wall inputs into numpy arrays with six elements.
        
        Args:
            value (None, int, or list): Value to normalize.
            default (int): Default value for each dimension if input is None.

        Returns:
            numpy.ndarray: Normalized array of length 6.
        """
        if value is None:
            return np.array([default] * 6)
        elif isinstance(value, int) and value >= 0:
            return np.array([value] * 6)
        elif isinstance(value, (tuple, list)):
            if len(value) == 6:
                arr = np.array(value)
                if np.any(arr < 0):
                    raise ValueError("array values must be non-negative.")
                return arr
            else:
                raise ValueError("Input must have exactly 6 values for dimensions.")
        else:
            raise ValueError("Invalid input type. Must be None, int, or list/tuple.")
        
    @property 
    def domain_shape(self):
        return self.domain.shape 

    def _append_padding(self, axis: int, padding_indices: Tuple[int, int]) -> None:
        """
        Adds padding along the specified axis of the domain.

        Args:
            axis (int): Axis along which padding is added (0 for x, 1 for y, 2 for z).
            padding_indices (Tuple[int, int]): Tuple containing padding values 
                                           for the lower and upper ends of the axis.
        """
        dims = list(self.domain_shape)
        dims[axis] = padding_indices[0]  # Lower padding
        pad_one = np.zeros(tuple(dims))

        dims[axis] = padding_indices[1]  # Upper padding
        pad_two = np.zeros(tuple(dims))

        self.domain = np.concatenate((pad_one, self.domain, pad_two), axis=axis)


    def add_padding(self) -> None:
        """
        Adds padding to the domain along all axes (x, y, z) based on `self.padding`.
        """
        if self.padding[0] != 0 or self.padding[1] != 0:
            self._append_padding(axis=0, padding_indices=(self.padding[0], self.padding[1]))
        if self.padding[2] != 0 or self.padding[3] != 0:
            self._append_padding(axis=1, padding_indices=(self.padding[2], self.padding[3]))
        if self.padding[4] != 0 or self.padding[5] != 0:
            self._append_padding(axis=2, padding_indices=(self.padding[4], self.padding[5]))
    
    def set_bounceback(self, bounce_method: Literal["circ", "edt"] = "circ"):
        """
        Identifies bounce-back boundary nodes in the domain and solid nodes based on the specified method.

        Args:
            bounce_method (Literal[str]): Method used for determining bounce-back conditions.
                - 'circ': Uses circular shifts to detect boundary and solid nodes. 
                          This approach analyzes neighbor states on a row-wise basis.
                - 'edt': Uses Euclidean Distance Transform to classify nodes. 
                         Nodes within a distance range are classified as boundary or solid.

        Sets:
            - self.boundary_nodes (Tuple[np.ndarray, ...]): Indices of boundary nodes where 
                                                       bounce-back conditions are applied.
            - self.solid_nodes (Tuple[np.ndarray, ...]): Indices of solid nodes.

        Raises:
            ValueError: If an invalid `bounce_method` is specified.

        Examples:
            >>> domain = np.random.choice([0, 1], size=(10, 10, 10))
            >>> nodemap = NodeMap(domain=domain)
            >>> nodemap.set_bounceback(bounce_method='circ')
            >>> print(nodemap.boundary_nodes)  # Prints indices of the boundary nodes
            >>> print(nodemap.solid_nodes)    # Prints indices of the solid nodes
        """
        if bounce_method == 'circ':
            bounce_domain = BounceBackGen(domain=self.domain)()
            self.boundary_nodes = np.where(bounce_domain == 1)
            self.solid_nodes = np.where(bounce_domain == 2)
        elif bounce_method == 'edt':
            bounce_domain = edt(self.domain)
            self.boundary_nodes = np.where((bounce_domain > 0)*(bounce_domain < 2))
            self.solid_nodes = np.where(bounce_domain >= 2)

    @abstractmethod 
    def add_phases(self):
        ... 

    def add_bounceback(self):
        self.domain[self.boundary_nodes] = 1
        self.domain[self.solid_nodes] = 2
    
    def add_sidewalls(self, side_wall_key: Literal["geometry", "domain"] = "domain"):
        """
        Adds sidewalls to the domain based on the specified key.

        Args:
            side_wall_key (Literal["geometry", "domain"]): Determines which set of sidewall thickness values 
                to use for adding sidewalls:
                - "geometry": Uses `self.geometry_side_walls` for sidewall dimensions.
                - "domain": Uses `self.side_walls` for sidewall dimensions.
                Default is "domain".

        Modifies:
            - self.domain: Updates the domain array to include boundary nodes (value = 1) for sidewalls.

        Behavior:
            Sidewalls are added along all six axes (x_min, x_max, y_min, y_max, z_min, z_max) based on the 
            thickness values provided by the relevant side wall set.

        Examples:
            >>> domain = np.zeros((10, 10, 10))
            >>> side_walls = [1, 1, 2, 2, 3, 3]
            >>> nodemap = NodeMap(domain=domain, side_walls=side_walls)
            >>> nodemap.add_sidewalls(side_wall_key="domain")
            >>> print(nodemap.domain.shape)
            # Sidewalls with thickness [1, 1, 2, 2, 3, 3] added to the domain.
        """
        side_walls = {"geometry": self.geometry_side_walls,
            "domain": self.side_walls}[side_wall_key]

        if side_walls[0] != 0:
            self.domain[0:side_walls[0],:,:] = 1 
        if side_walls[1] != 0:
            self.domain[-side_walls[1]:,:,:] = 1 

        if side_walls[2] != 0 :
            self.domain[:,0:side_walls[2],:] = 1 
        if side_walls[3] != 0:
            self.domain[:,-side_walls[3]:,:] = 1 
        
        if side_walls[4] != 0:
            self.domain[:,:,0:side_walls[4]] = 1 
        if side_walls[5] != 0:
            self.domain[:,:,-side_walls[5]:] = 1


    def to_dat(self):
        file_name = path.join(self.save_path, self.file_name + '.dat')
        np.savetxt(file_name, self.domain.flatten(), fmt = '%d')
    
    def _generate_csv(self, domain: np.ndarray):
        domain_df = pd.DataFrame(np.c_[self.xx.flatten()[:, np.newaxis], 
            self.yy.flatten()[:, np.newaxis], self.zz.flatten()[:, np.newaxis], domain.flatten()[:, np.newaxis]], 
                columns = ['x', 'y', 'z', 'domain'])
        return domain_df[domain_df['domain'] != 0]
    
    def to_vtk(self):
        file_name = path.join(self.save_path, self.file_name + '_full_domain.vtk')
        out_frame = pd.DataFrame(np.c_[self.xx.flatten()[:, np.newaxis], 
                    self.yy.flatten()[:, np.newaxis], self.zz.flatten()[:, np.newaxis], 
                        self.domain.flatten()[:, np.newaxis]], 
                            columns = ['x', 'y', 'z', 'solids'])
        out_frame = out_frame[(out_frame['solids'] == 1) | (out_frame['solids'] == 2)]
        with open(file_name, 'w+') as f:
            f.write('# vtk DataFile Version 4.2 \n')
            f.write('paraview vtk output\n')
            f.write('ASCII\n')
            f.write('DATASET POLYDATA\n')
            f.write('POINTS '+str(len(out_frame.index))+' float'+'\n')
            out_frame[['x','y','z']].to_csv(f, sep=' ', index = None, header=None, float_format='%.8f')  
            f.write('\n'*3)         
    
    def to_csv(self, separate: bool = False, multiphase: bool = True):
        """
        cvs file is used for visualization
        """
        if separate:
            solid_domain = np.where(self.domain == 2, 2, 0)
            boundary_domain = np.where(self.domain == 1, 1, 0)
            inert_domain = np.where(self.domain == 0, 4, 0)

            solid_df = self._generate_csv(domain = solid_domain)
            boundary_df = self._generate_csv(domain = boundary_domain)
            inert_df = self._generate_csv(domain = inert_domain)

            if not solid_df.empty:
                solid_df.to_csv(path.join(self.save_path, self.file_name + '_solids.csv'), sep = ',', 
                        header = True, index = False, float_format = '%.5f')
            if not boundary_df.empty:
                boundary_df.to_csv(path.join(self.save_path, self.file_name + '_boundary.csv'), sep = ',', 
                        header = True, index = False, float_format= '%.5f')
            
            inert_df.to_csv(path.join(self.save_path, self.file_name + '_inert_fluid.csv'), sep = ',', 
                    header = True, index = False, float_format= '%.5f')

            if multiphase:
                wetting_domain = np.where(self.domain == 3, 3, 0)
                wetting_df = self._generate_csv(domain = wetting_domain)
                wetting_df.to_csv(path.join(self.save_path, self.file_name + '_wetting_fluid.csv'), sep = ',', 
                    header = True, index = False, float_format= '%.5f')
        else:
            domain_df = pd.DataFrame(np.c_[self.xx.flatten()[:, np.newaxis], 
                    self.yy.flatten()[:, np.newaxis], self.zz.flatten()[:, np.newaxis], 
                        self.domain.flatten()[:, np.newaxis]], 
                            columns = ['x', 'y', 'z', 'domain'])
            domain_df.to_csv(path.join(self.save_path, self.file_name + '.csv'), 
                    sep = ',', header = True, index = False, 
                        float_format = '%.5f')
    
    # slice output of data
    def get_slice(self, direction: Literal['x', 'y', 'z'],
            coord: int, thickness: int):
        """
        Returns a tuple of slices to extract a region from the domain based on a given direction and coordinate.
        Utilizes the `get_slice` function from `tools.py`.

        Args:
            direction (Literal['x', 'y', 'z']): Direction of the slice ('x', 'y', or 'z').
            coordinate (int): Starting coordinate along the specified direction.
            thickness (int): Thickness of the slice along the specified direction (default is 1).

        Returns:
            Tuple[slice, slice, slice]: A tuple representing slices that can be used to index the domain.

        Raises:
            ValueError: If the direction is invalid or the coordinate is out of bounds.

        Example:
            >>> nodemap = NodeMap(domain=np.zeros((10, 10, 10)))
            >>> result = nodemap.get_slice('x', 5, thickness=2)
            >>> print(result)  # Output: (slice(5, 7), slice(0, 10), slice(0, 10))
        """
        resolution = (self.size_x, self.size_y, self.size_z)  # Extract domain resolution
        return tools.get_slice(direction, coord, resolution)

    def to_slice_csv(self, slice_direction: Literal['x', 'y', 'z']):
        res = {'x': self.size_x, 'y': self.size_y, 'z': self.size_z}[slice]
        slicer = partial(self._get_slice, direction = slice_direction)
        slice_dir = path.join(self.save_path, slice + '_slices')
        if not path.exists(slice_dir):
            makedirs(slice_dir)

        for s in range(res):
            s_slice = slicer(s)
            slice_df = pd.DataFrame(np.c_[self.xx[s_slice].flatten()[:, np.newaxis], 
                self.xx[s_slice].flatten()[:, np.newaxis],
                    self.zz[s_slice].flatten()[:, np.newaxis], 
                        self.domain.flatten()[:, np.newaxis]], columns = ['x', 'y', 'z', 'domain'])
            slice_df.to_csv(path.join(slice_dir, 'slice_' + str(s) + '.csv'), sep = ',', 
                            header = True, index = False, float_format = '%.5f')
    
    def _generate_coordinate_arrays(self):
        self.size_x, self.size_y, self.size_z = self.domain_shape 
        self.xx, self.yy, self.zz = np.meshgrid(np.arange(0, self.size_x), np.arange(0, self.size_y), 
                            np.arange(0, self.size_z), indexing = 'ij') 

    def _generate_saturation_info(self):
        self.total_number_fluid = np.sum(np.where(self.domain == 3, 1, 0)) 

    def add_file_info(self):
        file_info = path.join(self.save_path, self.file_name + '_info.txt')
        with open(file_info, 'w') as f:
            f.write(f'domain size: {self.domain.shape} \n')
            f.write(f"padding: {','.join([str(pad) for pad in self.padding])}\n")
            f.write(f'domain volume without padding is {self.domain_volume} & void fraction is {self.void_fraction} \n')

    def __call__(self, slice_direction: Literal['x', 'y', 'z'],
                 bounce_method: Literal['circ', 'edt'], 
                  separate: bool = True,
                      vtk: bool = True, multiphase: bool = True):
        """
        Generates a processed nodemap domain and exports it in multiple formats, including `.dat`, `.csv`, `.vtk`.

        Args:
            slice_direction (Optional[Literal["x", "y", "z"]]): Direction ('x', 'y', 'z') to export slices of the domain.
                                                             If None, slicing is skipped. Default is None.
            separate (bool): Whether to export separate `.csv` files for different node types (fluid, boundary, etc.).
                             Default is True.
            bounce_method (str): Method for setting bounce-back boundary conditions:
                                 - 'circ': Circular shifting approach for bounce-back detection.
                                 - 'edt': Euclidean distance transform-based detection.
                                 Default is 'circ'.
            vtk (bool): Whether to export the domain as a `.vtk` file for visualization in software like ParaView.
                        Default is True.
            multiphase (bool): If True, includes multiphase information (e.g., wetting fluid nodes) in exported files.
                               Default is True.

        Process Flow:
            1. Adds geometry-specific and domain-wide sidewalls to the domain based on configuration.
            2. Adds padding and bounce-back conditions based on input settings.
            3. Adds simulation phases, such as fluid and solid assignment, by interacting with subclasses.
            4. Exports the processed domain in `.dat`, `.csv`, and `.vtk` formats.
            5. If `slice_direction` is specified, exports slices of the domain to separate `.csv` files.

        Raises:
            ValueError: If `bounce_method` is invalid.

        Example Usage:
            --------------------------------------------
            Basic Usage:
            --------------------------------------------
            >>> domain = np.zeros((10, 10, 10))
            >>> nodemap = NodeMap(domain=domain, padding=[1, 1, 1, 1, 1, 1], save_path="./")
            >>> nodemap(slice_direction="z", vtk=True)  # Generates processed nodemap with slices along z-axis.

            --------------------------------------------
            Exporting Boundary Nodes with Bounce-Back:
            --------------------------------------------
            >>> domain = np.random.choice([0, 1, 2], size=(15, 15, 15))
            >>> nodemap = MultiPhaseNodeMap(domain=domain, set_phases={'method': 'drainage'})
            >>> nodemap(bounce_method="edt", separate=True)  # Exports `.dat`, `.csv` files with detailed boundary nodes.

            --------------------------------------------
            Skipping Slice and VTK Export:
            --------------------------------------------
            >>> domain = np.ones((20, 20, 20))
            >>> nodemap = DryingNodeMap(domain=domain, padding=[0, 0, 0, 0, 0, 0], save_path="./outputs")
            >>> nodemap(slice_direction=None, vtk=False)  # Creates nodemap but skips slicing and `.vtk` export.
        """
        # note: add_sidewalls and add_edges should be added before set_bounceback 
        if "geometry" in self.which_sidewall:
            self.add_sidewalls("geometry")
        self.add_padding()
        if "domain" in self.which_sidewall:
            self.add_sidewalls("domain")

        self._generate_coordinate_arrays()
        self.set_bounceback(bounce_method = bounce_method)
        self.add_phases()
        self.add_bounceback()
        self._generate_saturation_info()
        self.to_dat()
        self.to_csv(separate = separate, multiphase = multiphase)
        if vtk:
            self.to_vtk()
        self.add_file_info()
        if slice_direction is not None:
            self.to_slice_csv(slice_direction)

    @staticmethod 
    def _read_line(line):
        line = line.split(' ')
        if '#' in line:
            index = 1
        else:
            index = 0
        if len(line) > 2:
            return [int(elem) for elem in line[index:] if elem.isdigit()]
        else:
            return None 
        
    @classmethod
    @abstractmethod
    def from_file(
        cls, file_name: Union[str, PathLike], padding: Optional[Union[int, List[int]]] = None, save_path: Optional[Union[str, PathLike]] = None, **kw: Any
    ) -> "NodeMap":
        """
        Constructs a NodeMap instance from a domain file. Subclasses must implement this method.

        Args:
            file_name (Union[str, PathLike]): Path to the file containing the domain grid data.
            padding (Optional[Union[int, List[int]]]): Padding array specifying padding values. Default is None.
            save_path (Optional[Union[str, PathLike]]): Path to the directory for saving output files. Default is None.
            **kw (Any): Additional arguments for subclass-specific behavior.

        Returns:
            NodeMap: A NodeMap instance initialized based on the file.

        Notes:
            - This method is abstract and must be implemented by subclasses.
            - The implementation should be tailored to specific file formats.

        Example:
            >>> class CustomNodeMap(NodeMap):
            ...     @classmethod
            ...     def from_file(cls, file_name, padding=None, save_path=None, **kw):
            ...         # Custom implementation based on file format
            ...         pass
        """
        pass

    @classmethod
    def from_lbm_dat(
        cls,
        file_name: Union[str, PathLike],
        domain_size: Iterable[int],
        file_stem: Optional[str] = None,
        save_path: Optional[Union[str, PathLike]] = None,
        padding: Optional[Union[int, List[int]]] = None,
        side_walls: Optional[Union[int, List[int]]] = None,
        **kw: Any,
        ) -> "NodeMap":
        """
        Instantiates a NodeMap from an LBM-compatible `.dat` file containing 0s and 1s.

        Args:
            file_name (Union[str, PathLike]): Path to the `.dat` file containing grid data.
            domain_size (Iterable[int]): The size of the domain as (nx, ny, nz).
            file_stem (Optional[str]): File stem to name the outputs. Defaults to the stem of `file_name`.
            save_path (Optional[Union[str, PathLike]]): Directory where outputs and processed files will be saved.
                                                    Defaults to the parent directory of `file_name`.
            padding (Optional[Union[int, List[int]]]): Padding array specifying padding values for all six dimensions.
                                                   Defaults to None (no padding).
            side_walls (Optional[Union[int, List[int]]]): Thickness of the sidewalls for all six dimensions.
                                                     Defaults to None (no sidewalls).
            **kw (Any): Additional arguments for subclass-specific configurations.

        Returns:
            NodeMap: A NodeMap instance initialized with the domain grid read from the `.dat` file.

        Raises:
            FileNotFoundError: If `file_name` does not exist.
            ValueError: If the domain shape does not match `domain_size` or the `.dat` file format is invalid.

        Details:
            - The `.dat` file is assumed to contain flattened 3D grid data with 0s and 1s:
                - 0: Void (fluid)
                - 1: Solid
    
            - Domain resolution (`domain_size`) must match the actual dimensions of the file.

        Example:
            >>> file_path = "./data/lbm_domain.dat"
            >>> domain_size = (100, 100, 100)
            >>> node_map = NodeMap.from_lbm_dat(file_path, domain_size, padding=[1, 1, 1, 1, 0, 0])
            >>> print(node_map.domain.shape)  # Output: (102, 102, 100)
        """
        if not path.exists(file_name):
            raise FileNotFoundError(f"The file '{file_name}' does not exist.")
        try:
            domain = np.loadtxt(file_name, dtype=int).reshape(domain_size)
        except ValueError:
            raise ValueError(
                f"Could not reshape file '{file_name}' into the domain size {domain_size}. "
                f"Ensure the file format and domain resolution are correct."
            )
        domain = np.where(domain == 1, 1, 0)
        if save_path is None:
            save_path = Path(file_name).parent 
        padding_normalized = cls._normalize_input(padding, default=0)
        side_walls_normalized = cls._normalize_input(side_walls, default=0)

        if file_stem is None:
            file_stem = Path(file_name).stem

        init_kw = {
            "domain": domain,
            "file_stem": file_stem,
            "save_path": save_path,
            "padding": padding_normalized,
            "side_walls": side_walls_normalized,
            **kw,
            }
        return cls(**init_kw)

    @classmethod
    def from_array(
        cls,
        domain: np.ndarray,
        padding: Optional[Union[int, List[int]]] = None,
        save_path: Optional[Union[str, PathLike]] = None,
        file_stem: Optional[str] = None,
        **kw: Any
    ) -> "NodeMap":
        """
        Instantiates a NodeMap directly from a domain array.

        Args:
            domain (np.ndarray): A 3D array representing the simulation domain.
                    Each element indicates node type:
                    - 0 for void (fluid)
                    - 1 for solid
            padding (Optional[Union[int, List[int]]]): Padding array specifying padding values for all six dimensions.
                                                   Defaults to None (no padding).
            save_path (Optional[Union[str, PathLike]]): Directory where outputs and processed files will be saved.
                                                    Defaults to the current directory.
            file_stem (Optional[str]): File stem to name the outputs. Defaults to "node_map".
            **kw (Any): Additional keyword arguments for subclass-specific configurations.

        Returns:
            NodeMap: A NodeMap instance initialized with the provided domain array.

        Raises:
            ValueError: If the domain is not a valid 3D numpy array.

        Example:
            >>> domain = np.zeros((10, 10, 10), dtype=int)
            >>> node_map = NodeMap.from_array(
            ...     domain=domain,
            ...     padding=[1, 1, 1, 1, 0, 0],
            ...     save_path="./outputs",
            ...     file_stem="custom_map"
            ... )
            >>> print(node_map.domain.shape)  # Output: (12, 12, 10) after padding
        """
        if not isinstance(domain, np.ndarray) or domain.ndim != 3:
            raise ValueError("The domain must be a 3D numpy array.")

        save_path_normalized = Path(save_path) if save_path is not None else Path.cwd()

        padding_normalized = cls._normalize_input(padding, default=0)

        file_stem_normalized = file_stem if file_stem is not None else "node_map"

        init_kw = {
            "domain": domain,
            "padding": padding_normalized,
            "save_path": save_path_normalized,
            "file_stem": file_stem_normalized,
            **kw,
        }
        return cls(**init_kw) 


# ################################################## #
# generates nodemaps for multiphase flow simulations #
# ################################################## #
class MultiPhaseNodeMap(NodeMap):
    """
    A class for generating nodemaps for multiphase flow simulations such as drainage, 
    imbibition, invasion, and related scenarios.

    The class supports additional fluid phases within the simulation domain, including methods 
    to add droplets, bridges, or initiate fluid invasion from boundaries. The generated nodemap 
    also accommodates certain geometrical constraints like sidewalls.

    Attributes:
        set_phases (dict): Dictionary defining the method used for phase addition:
            - "drainage": Initializes fluid phase from the inlet boundary.
            - "imbibition": Fluid phase expands from the outlet boundary.
            - "invasion": Fluid phase initiates simultaneously from multiple boundaries.
            - "droplet": Initializes spherical fluid droplets in the domain.
            - "bridge": Adds bridge-shaped structures within the domain.
        domain (np.ndarray): Simulation domain, a 3D array.
        fluid_nodes (Optional[Tuple[np.ndarray, ...]]): Indices of fluid (empty) nodes.

    Methods:
        add_phases(): Adds fluid phases based on the set scenario parameters.

    Examples:
        --------------------------------------------
        Basic Drainage Simulation:
        --------------------------------------------
        >>> domain = np.zeros((100, 100, 100))
        >>> mp_nodemap = MultiPhaseNodeMap(
        ...     domain=domain, 
        ...     set_phases={"method": "drainage", "limit": 10}
        ... )
        >>> mp_nodemap.add_phases()

        --------------------------------------------
        Adding Fluid Droplets and Free Surface:
        --------------------------------------------
        >>> domain = np.zeros((80, 80, 80))
        >>> mp_nodemap = MultiPhaseNodeMap(
        ...     domain=domain,
        ...     set_phases={
        ...         "method": "droplet",
        ...         "drop_center": [(40, 40, 40)],
        ...         "drop_radius": [20],
        ...         "free_surface": {"direction": "z", "size": 5},
        ...     }
        ... )
        >>> mp_nodemap.add_phases()

        --------------------------------------------
        Invasion-Free Surface Case:
        --------------------------------------------
        >>> domain = np.zeros((60, 60, 60))
        >>> mp_nodemap = MultiPhaseNodeMap(
        ...     domain=domain,
        ...     set_phases={
        ...         "method": "invasion-free-surface",
        ...         "air_pocket": 5,
        ...     }
        ... )
        >>> mp_nodemap.add_phases()
    """
    def __init__(
        self, 
        domain: Optional[np.ndarray] = None, 
        file_stem: Optional[str] = None, 
        save_path: Optional[Union[str, Path]] = None, 
        side_walls: Optional[Union[int, List[int]]] = None,  
        geometry_side_walls: Optional[Union[int, List[int]]] = None, 
        padding: Optional[Union[int, List[int]]] = None, 
        set_phases: Optional[dict] = {'method': 'drainage'}
    ) -> None:
        """
        Initializes a MultiPhaseNodeMap instance.

        Args:
            domain (Optional[np.ndarray]): The 3D array representing the simulation domain.
            file_stem (Optional[str]): The stem name for stored files.
            save_path (Optional[Union[str, Path]]): Directory path for saving exported files.
            side_walls (Optional[Union[int, List[int]]]): Thickness of side walls around the domain.
            geometry_side_walls (Optional[Union[int, List[int]]]): Geometry-specific side walls.
            padding (Optional[Union[int, List[int]]]): Padding values for all six axes.
            set_phases (Optional[dict]): Parameters for fluid phase addition within the domain:
                - "method" (str): Specifies the method of phase addition.
                - Other optional keys for specific methods (e.g., limit, drop_center, air_pocket).

        Raises:
            ValueError: If `set_phases["method"]` is unsupported.
            Exception: If domain input is invalid (not a 3D numpy array).
        """
        super().__init__(
            domain=domain, 
            file_stem=file_stem, 
            save_path=save_path, 
            padding=padding, 
            side_walls=side_walls, 
            geometry_side_walls=geometry_side_walls, 
        )
        
        self.set_phases = set_phases
        valid_methods = ['drainage', 'imbibition', 'invasion', 'invasion-free-surface', 
                         'droplet', 'bridge', 'droplet_free_surface']
        method = self.set_phases.get("method")
        if method not in valid_methods:
            raise ValueError(
                f"Unsupported method '{method}' for phase addition. "
                f"Choose from {valid_methods}."
            )
        print(f"MultiPhaseNodeMap initialized with method: {method}")

    def add_phases(self) -> None:
        """
        Adds fluid phases to the simulation domain based on `set_phases` configuration.

        Supported Methods:
            - "drainage": Adds fluid phase from the inlet boundary.
            - "imbibition": Adds fluid phase from the outlet boundary.
            - "invasion": Adds fluid phase from all six boundaries.
            - "invasion-free-surface": Initializes invasion from boundaries with a free surface.
            - "droplet": Adds spherical fluid droplets within the domain.
            - "bridge": Adds fluid bridges spanning specific regions.

        Raises:
            ValueError: For unsupported method types or missing required parameters.
        """
        method = self.set_phases['method']
        def _apply_drainage(limit: Optional[int]) -> None:
            if limit:
                self.domain[:limit, :, :] = 3
            else:
                self.domain[:self.padding[0], :, :] = 3
        
        def _apply_imbibition(limit: Optional[int]) -> None:
            if limit:
                self.domain[:limit, :,:] = 3
            else:
                self.domain[:self.padding[1] + 3, :, :] = 3

        def _apply_invasion(limit: Optional[int]) -> None:
            if not limit:
                limit = self.padding[1] + 3
            self.domain[:limit, :, :] = 3
            self.domain[-limit:, :, :] = 3
            self.domain[:, :limit, :] = 3
            self.domain[:, -limit:, :] = 3
            self.domain[:, :, :limit] = 3
            self.domain[:, :, -limit:] = 3
        
        def _apply_invasion_free_surface(air_pocket: Optional[int]) -> None:
            self.domain[:self.padding[0], :, :] = 3
            self.domain[-self.padding[1]:, :, :] = 3
            self.domain[:, :self.padding[2], :] = 3
            self.domain[:, -self.padding[3]:, :] = 3
            self.domain[:, :, :self.padding[4]] = 3
            self.domain[:, :, -self.padding[5]:] = 3
            if air_pocket:
                self.domain[:, -air_pocket:, :] = 0

        def _apply_droplets(droplets: List[dict]) -> None:
            for drop in droplets:
                center = drop["drop_center"]
                radius = drop["drop_radius"]
                coords_x, coords_y, coords_z = center
                droplet_indices = np.where(
                    (self.xx - coords_x) ** 2 + (self.yy - coords_y) ** 2 + (self.zz - coords_z) ** 2 <= radius ** 2
                )
                self.domain[droplet_indices] = 3
                if drop.get("free_surface"):
                    free_surface_dir = drop["free_surface"]["direction"]
                    free_surface_size = drop["free_surface"]["size"]
                    coord = {"x": self.size_x - free_surface_size, 
                             "y": self.size_y - free_surface_size, 
                             "z": self.size_z - free_surface_size}[free_surface_dir]
                    free_slice = self.get_slice(direction=free_surface_dir, coord=coord, thickness=free_surface_size)
                    self.domain[free_slice] = 3
        
        def _apply_bridge(props: dict) -> None:
            bridge_cx, bridge_cz = props["bridge_center"]
            bridge_radius = props["bridge_radius"]
            bridge_bounds = props["bridge_bounds"]
            bridge_indices = np.where(
                ((self.xx - bridge_cx) ** 2 + (self.zz - bridge_cz) ** 2 <= bridge_radius ** 2) &
                (self.yy >= bridge_bounds[0]) & (self.yy <= bridge_bounds[1])
            )
            self.domain[bridge_indices] = 3

        # Execute based on method
        if method == "drainage":
            _apply_drainage(limit=self.set_phases.get("limit"))
        elif method == "imbibition":
            _apply_imbibition(limit=self.set_phases.get("limit"))
        elif method == "invasion":
            _apply_invasion(limit=self.set_phases.get("limit"))
        elif method == "invasion-free-surface":
            _apply_invasion_free_surface(air_pocket=self.set_phases.get("air_pocket"))
        elif method == "droplet":
            droplets = self.set_phases.get("droplets", [{"drop_center": self.set_phases["drop_center"], "drop_radius": self.set_phases["drop_radius"], "free_surface": self.set_phases.get("free_surface")}])
            _apply_droplets(droplets)
        elif method == "bridge":
            _apply_bridge(props=self.set_phases)
        else:
            raise ValueError(f"Unsupported method '{method}' provided in `set_phases`.")         

        self.fluid_nodes = np.where(self.domain == 0)


class DryingNodeMap(NodeMap):
    """
    A specialized class for generating LBM-compatible node maps for drying simulations. 

    The drying process involves releasing liquid into the simulation domain and allowing 
    equilibrium to occur between phases, such as water and air. This class supports the addition 
    of drying phases with optionally padded boundaries, geometric constraints, and release boundaries.

    It assists in configurations with predefined boundary releases, padding, and various geometries, 
    and supports export in `.dat`, `.csv`, and `.vtk` file formats for visualization.

    Attributes:
        domain (np.ndarray): The simulation domain, a 3D array.
        distance_from_edge (Optional[Union[int, List[int]]]): Distance from edges to exclude fluid phases.
        phase_release_boundary (Optional[List[int]]): Defines bounds for releasing fluid phase near specific boundaries.
        side_walls (Optional[Union[int, List[int]]]): Thickness of walls added around the domain.
        geometry_side_walls (Optional[Union[int, List[int]]]): Selective side walls for specific geometries.

    Methods:
        add_phases(): Adds water phase into the domain based on the specified configuration.
        add_file_info(): Writes domain and configuration information to a text file.

    Examples:
        --------------------------------------------
        Basic Drying Simulation with Padding:
        --------------------------------------------
        >>> domain = np.zeros((100, 100, 100))
        >>> dn_map = DryingNodeMap(
        ...     domain=domain, 
        ...     padding=[2, 2, 2, 2, 0, 0], 
        ...     distance_from_edge=[3, 3, 3, 3, 0, 0]
        ... )
        >>> dn_map.add_phases()

        --------------------------------------------
        Adding Release Boundary:
        --------------------------------------------
        >>> domain = np.zeros((60, 60, 60))
        >>> dn_map = DryingNodeMap(
        ...     domain=domain,
        ...     phase_release_boundary=[45, 0, 0, 0, 0, 0]
        ... )
        >>> dn_map.add_phases()

        --------------------------------------------
        Geometrical Sidewalls and Full Padding:
        --------------------------------------------
        >>> domain = np.ones((100, 100, 100))
        >>> dn_map = DryingNodeMap(
        ...     domain=domain,
        ...     padding=[4, 4, 4, 4, 4, 4],
        ...     side_walls=[2, 0, 2, 0, 2, 0]
        ... )
        >>> dn_map.add_phases()
    """
    def __init__(
        self, 
        domain: Optional[np.ndarray] = None, 
        file_stem: Optional[str] = None, 
        save_path: Optional[Union[str, Path]] = None,
        padding: Optional[Union[int, List[int]]] = None, 
        distance_from_edge: Optional[Union[int, List[int]]] = None, 
        side_walls: Optional[Union[int, List[int]]] = None, 
        geometry_side_walls: Optional[Union[int, List[int]]] = None, 
        phase_release_boundary: Optional[List[int]] = None
    ) -> None:
        """
        Initializes the DryingNodeMap class.

        Args:
            domain (Optional[np.ndarray]): A 3D numpy array representing the simulation domain grid.
            file_stem (Optional[str]): Name prefix for saved output files.
            save_path (Optional[Union[str, Path]]): Directory for saving the output files. Defaults to the current directory.
            padding (Optional[Union[int, List[int]]]): Padding values for all six dimensions [x_min, x_max, y_min, y_max, z_min, z_max].
                                                       Defaults to None (no padding).
            distance_from_edge (Optional[Union[int, List[int]]]): Distance from domain edges to restrict fluid phase addition.
            side_walls (Optional[Union[int, List[int]]]): Thickness of side walls around six dimensions.
            geometry_side_walls (Optional[Union[int, List[int]]]): Selective sidewalls for specific regions/geometries.
            phase_release_boundary (Optional[List[int]]): Defines boundaries for releasing fluid phases 
                                                          (expected list format [x_min, x_max, y_min, y_max, z_min, z_max]).

        Raises:
            ValueError: If `domain` is invalid (not a 3D numpy array).
            ValueError: If phase_release_boundary is not a list of six elements.
        """
        super().__init__(
            domain=domain, 
            file_stem=file_stem, 
            save_path=save_path,
            padding=padding,
            side_walls=side_walls, 
            geometry_side_walls=geometry_side_walls
        )

        self.distance_from_edge = self._normalize_input(distance_from_edge, default=0)

        if phase_release_boundary:
            if isinstance(phase_release_boundary, list) and len(phase_release_boundary) == 6:
                self.phase_release_boundary = phase_release_boundary
            else:
                raise ValueError(
                    "`phase_release_boundary` must be a list of six integers "
                    "specifying bounds for fluid phase release."
                )
        else:
            self.phase_release_boundary = [0] * 6

        self._set_boundaries()

    def _set_boundaries(self) -> None:
        """
        Sets domain boundaries dynamically based on padding and side wall parameters.
        """
        self.min_x = self.padding[0] + self.side_walls[0]
        self.min_y = self.padding[2] + self.side_walls[2]
        self.min_z = self.padding[4] + self.side_walls[4]

        self.max_x = self.domain_shape[0] - (self.padding[1] + self.side_walls[1])
        self.max_y = self.domain_shape[1] - (self.padding[3] + self.side_walls[3])
        self.max_z = self.domain_shape[2] - (self.padding[5] + self.side_walls[5])

    def _set_phase_release_standard(self):
        """
        Adds fluid phase to the domain, excluding areas defined by `distance_from_edge`.
        Modifies the domain by marking nodes within the restricted boundaries as fluid (value = 3).
        """
        min_x = self.min_x + self.distance_from_edge[0]
        max_x = self.max_x - self.distance_from_edge[1]

        min_y = self.min_y + self.distance_from_edge[2]
        max_y = self.max_y - self.distance_from_edge[3]

        min_z = self.min_z + self.distance_from_edge[4]
        max_z = self.max_z - self.distance_from_edge[5]

        self.domain[min_x:max_x, min_y:max_y, min_z:max_z] = 3    

    def _set_phase_release_boundary(self):
        """
        Adds fluid phase to the domain based on explicit boundaries defined in `phase_release_boundary`.
        Updates the domain by marking fluid nodes within these bounds as fluid (value = 3).
        """
        max_x = self.phase_release_boundary[0] or self.max_x
        max_y = self.phase_release_boundary[1] or self.max_y 
        max_z = self.phase_release_boundary[2] or self.max_z         
        self.domain[self.min_x:max_x, self.min_y:max_y, self.min_z:max_z] = 3 
  
    def add_phases(self):
        """
        Adds phases to the simulation domain for drying problems.

        This method dynamically sets the liquid phase (value = 3) in appropriate parts of the domain 
        based on user-defined configurations. It supports two main strategies:
            1. `_set_phase_release_standard`: Uses `distance_from_edge` to exclude fluid from the edges.
            2. `_set_phase_release_boundary`: Explicitly adds fluid phase based on predefined boundary limits.

        Behavior:
            - When `phase_release_boundary` is None, the method defaults to `_set_phase_release_standard`.
            - If `phase_release_boundary` is defined, `_set_phase_release_boundary` is used.

        Attributes Modified:
            - `domain` (numpy.ndarray): Updates the domain to assign fluid nodes as value = 3.
            - `fluid_nodes` (tuple): Sets indices of void (air) nodes where the domain value is 0.

        Notes:
            - `add_phases` is specialized for DryingNodeMap but overrides the abstract method 
            from the base `NodeMap` class in the hierarchy.

        Example Usage:
            --------------------------------------------
            Standard Phase Release:
            --------------------------------------------
            >>> domain = np.zeros((50, 50, 50))
            >>> dn_map = DryingNodeMap(
            ...     domain=domain, 
            ...     padding=[2, 2, 0, 0, 0, 0], 
            ...     distance_from_edge=[5, 5, 5, 5, 0, 0]
            ... )
            >>> dn_map.add_phases()

            --------------------------------------------
            Phase Release with Boundary:
            --------------------------------------------
            >>> domain = np.zeros((60, 60, 60))
            >>> dn_map = DryingNodeMap(
            ...     domain=domain, 
            ...     phase_release_boundary=[50, 0, 0, 0, 0, 0]
            ... )
            >>> dn_map.add_phases()
            >>> print(dn_map.domain)

            --------------------------------------------
            Full Padding with Custom Geometry:
            --------------------------------------------
            >>> domain = np.ones((40, 40, 40))
            >>> dn_map = DryingNodeMap(
            ...     domain=domain, 
            ...     padding=[4, 4, 4, 4, 4, 4], 
            ...     side_walls=[1, 1, 1, 1, 1, 1]
            ... )
            >>> dn_map.add_phases()
        """
        self._set_boundaries()
        if self.phase_release_boundary is None:
            self._set_phase_release_standard()
        else:
            self._set_phase_release_boundary()
        self.fluid_nodes = np.where(self.domain == 0)
    
    def add_file_info(self):
        """
        Writes detailed information about the domain and its configuration to a text file.
        This includes:
            - Domain dimensions and resolution.
            - Padding values added to the domain.
            - Side wall and geometry side wall thickness values.
            - Void fraction and domain volume.
            - Total count of fluid nodes (air nodes).
            - Simulation details specific to drying phase release.
        Notes:
            - The output file is saved to the specified `save_path` directory using the stem `file_name`.
        """
        file_info = path.join(self.save_path, self.file_name + '_info.txt')
        with open(file_info, 'w') as f:
            f.write(f'domain size: {self.domain_shape} \n')
            f.write(f'padding values were {self.padding} \n')
            f.write(f"Padding Values (x_min, x_max, y_min, y_max, z_min, z_max): {', '.join(map(str, self.padding))}\n")
            f.write(f"Side Wall Thickness (x_min, x_max, y_min, y_max, z_min, z_max): {', '.join(map(str, self.side_walls))}\n")
            f.write(f'void fraction is {self.void_fraction}')
            f.write(f'total number of fluid is {self.total_number_fluid}')
    
    def __call__(
        self, 
        separate: bool = False, 
        vtk: bool = False, 
        bounce_method: Literal["circ", "edt"] = "circ"
    ) -> None:
        """
        Generates and exports the processed drying nodemap.

        This method handles:
            - Full domain processing using sidewalls, padding, and drying phases.
            - Setting bounce-back boundary nodes (`circ` or `edt` method).
            - Exporting the processed domain in `.dat` format, `.csv` format, and `.vtk` format (if enabled).
            - Writing metadata to an info text file.

        Args:
            separate (bool, optional): 
                If True, creates separate `.csv` files for fluid (air), boundary, and solid nodes. Defaults to False.
            vtk (bool, optional): 
                If True, includes a `.vtk` export for visualization purposes (e.g., ParaView). Defaults to False.
            bounce_method (Literal["circ", "edt"], optional): 
                The method used to classify bounce-back regions:
                - "circ": Uses circular shifts to detect boundary and solid nodes.
                - "edt": Uses Euclidean Distance Transform to classify nodes.
                Defaults to "circ".

        Behavior:
            - Applies padding, sidewalls, release boundaries, and bounce-back conditions.
            - Generates fluid, boundary, and solid phases in the domain.
            - Exports the processed domain for documentation and visualization.

        Raises:
            ValueError: If an invalid `bounce_method` is provided.

        Example Usage:
            --------------------------------------------
            Basic Drying Simulation:
            --------------------------------------------
            >>> domain = np.ones((40, 40, 40))  # Initial empty domain
            >>> drying_map = DryingNodeMap(
            ...     domain=domain,
            ...     padding=[2, 2, 0, 0, 0, 0],
            ...     distance_from_edge=[3, 3, 3, 3, 0, 0]
            ... )
            >>> drying_map(separate=True, vtk=True, bounce_method="circ")

            --------------------------------------------
            Exporting with Sidewalls:
            --------------------------------------------
            >>> domain = np.random.choice([0, 1], size=(20, 20, 20))
            >>> drying_map = DryingNodeMap(
            ...     domain=domain, 
            ...     side_walls=[1, 1, 1, 1, 0, 0]
            ... )
            >>> drying_map(separate=False, vtk=False, bounce_method="edt")

        """
        super().__call__(separate=separate, vtk=vtk, bounce_method=bounce_method)
        self.add_file_info()


# ############################## #
#      helper functions          #
# ############################## #  
# #### drying nodemap helper functions #### #
def drying_nodemap_from_benchmark(
    benchmark: Optional[str] = None, 
    save_path: Optional[Union[str, Path]] = None, 
    distance_from_edge: Optional[Union[int, List[int]]] = None,
    padding: Optional[Union[int, List[int]]] = None, 
    side_walls: Optional[Union[int, List[int]]] = None, 
    geometry_side_walls: Optional[Union[int, List[int]]] = [0] * 6, 
    separate: bool = True, 
    bounce_method: Literal["circ", "edt"] = "circ", 
    vtk: bool = False, 
    **benchmark_kw: Any
) -> None:
    """
    Generates and exports drying LBM node maps based on a benchmark geometry.

    This helper function creates a simulation domain using a predefined benchmark 
    geometry (e.g., overlapping spheres) and processes it with drying nodemap configurations. 
    The result is exported in various formats (`.dat`, `.csv`, `.vtk`).

    Args:
        benchmark (Optional[str]): The name of the benchmark geometry to use.
            - `"overlapping-spheres"` is currently supported.
        save_path (Optional[Union[str, Path]]): Directory to save the output files. 
            If None, creates a directory based on the current date and time.
        distance_from_edge (Optional[Union[int, List[int]]]): Distance from edges 
            to restrict fluid phase addition.
        padding (Optional[Union[int, List[int]]]): Padding values for all six dimensions 
            [x_min, x_max, y_min, y_max, z_min, z_max].
        side_walls (Optional[Union[int, List[int]]]): Thickness of side walls around six dimensions.
        geometry_side_walls (Optional[Union[int, List[int]]]): Geometry-specific side walls 
            (default is zeros for no walls).
        separate (bool, optional): 
            If True, creates separate `.csv` files for fluid (air), boundary, and solid nodes. Defaults to True.
        bounce_method (Literal["circ", "edt"], optional): 
            The method used to classify bounce-back regions. Defaults to "circ".
        vtk (bool, optional): 
            If True, includes a `.vtk` export for visualization purposes. Defaults to False.
        **benchmark_kw (Any): Additional keyword arguments passed to the benchmark geometry generator.

    Raises:
        KeyError: If an invalid benchmark name is provided.
        FileNotFoundError: If the benchmark geometry fails to generate.
        ValueError: If critical parameters (e.g., padding) are invalid.

    Notes:
        - The generated benchmarks are highly customizable through `benchmark_kw`.
        - Supports multiple formats for exporting the processed domain, such as `.dat`, `.csv`, and `.vtk`.

    Example Usage:
        --------------------------------------------
        Basic Drying Simulation with Spheres:
        --------------------------------------------
        >>> drying_nodemap_from_benchmark(
        ...     benchmark="overlapping-spheres",
        ...     save_path="./sim_data",
        ...     padding=[1, 1, 1, 1, 0, 0],
        ...     distance_from_edge=[2, 2, 0, 0, 0, 0],
        ...     separate=True,
        ...     vtk=True
        ... )

        --------------------------------------------
        Drying Benchmark with Custom Sidewalls:
        --------------------------------------------
        >>> drying_nodemap_from_benchmark(
        ...     benchmark="overlapping-spheres",
        ...     side_walls=[1, 2, 3, 4, 0, 0],
        ...     separate=False
        ... )
    """
    
    _SUPPORTED_BENCHMARKS = ["overlapping-spheres"]
    if benchmark not in _SUPPORTED_BENCHMARKS:
        raise KeyError(
            f"The benchmark '{benchmark}' is not supported. Choose from {_SUPPORTED_BENCHMARKS}."
        )
    
    try:
        benchmark_obj = {'overlapping-spheres':benchmarks.OverlappingSpherePack}[benchmark](**benchmark_kw)
    except Exception as e:
        raise FileNotFoundError(f"Failed to initialize benchmark '{benchmark}': {e}")
    
    benchmark_obj()

    save_path = save_path + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M')
    if save_path is not None and not path.exists(save_path):
        makedirs(save_path)
    
    drying_map = DryingNodeMap.from_array(domain = benchmark_obj.domain, save_path = save_path, distance_from_edge = distance_from_edge,
                              side_walls = side_walls, geometry_side_walls = geometry_side_walls,
                                 file_stem = benchmark, padding = padding)
    drying_map(separate = separate, bounce_method = bounce_method, vtk = vtk)
    
    if hasattr(benchmark_obj, 'drainage_domain'):
        benchmark_obj.save_drainage_domain(save_path = save_path)
    
    if hasattr(benchmark_obj, 'domain_attributes'):
        benchmark_obj.save_attributes(save_path = save_path)
    print(f'save path is {save_path}')

 


















    











        
        
