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
import numpy as np
from dataclasses import dataclass 
from typing import Optional, Literal, Union, List, Any, Tuple    
from pathlib import Path
from os import path, makedirs 
from datetime import datetime

# module imports 
from pyflowmeld import find_package_root

package_root = find_package_root(Path(__file__))
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

try:
    from pyflowmeld.utils import benchmarks   
    from pyflowmeld.preprocess._base import NodeMap, ZoneConfig
except ModuleNotFoundError:
    from .. utils import benchmarks
    from . _base import NodeMap, ZoneConfig 

# ################################################## #
# generates nodemaps for multiphase flow simulations #
# ################################################## #
#----------------------#
#  Imbibition nodemap  #
#----------------------#
class Imbibition(NodeMap):
    """
    Generates nodemap for imbibitioon problem by adding 
    wetting phases: 3
    non wetting phase: 0
    solids: 1
    boundary: 2
    """
    def add_phases(self, limit: Optional[int] = None) -> None:
        """
        fills the domain up to the limit with the wetting fluid
        limit: int
        if limit is not specified, uses padding[0] + 3
        """
        if limit:
            self.domain[:limit, :, :] = 3
        else:
            self.domain[:self.padding[0] + 3, :, :] = 3
        self.fluid_nodes = np.where(self.domain == 0)

#----------------------#
#  Drainage nodemap    #
#----------------------#
class Drainage(NodeMap):
    """
    Generates nodemap for drainage problem by adding 
    non wetting phases: 3
    non wetting phase: 0
    solids: 1
    boundary: 2
    The logic is very similar to Imbibition; except that nonwetting fluid
    does not overlap with the domain st the start
    """
    def add_phases(self, limit: Optional[int] = None) -> None:
        """
        fills the domain up to the limit with the wetting fluid
        limit: int
        if limit is not specified, uses padding[0] + 3
        """
        if limit:
            self.domain[:limit, :, :] = 3
        else:
            self.domain[:self.padding[0], :, :] = 3
        self.fluid_nodes = np.where(self.domain == 0)

#----------------------------#
#  Droplet spread nodemap    #
#----------------------------#
@dataclass
class Droplet:
    center: Tuple[int, int, int]
    radius: int 

class DropletSpread(NodeMap):
    """
    Generates nodemap for droplet spread on a surface
    It can simulate more than one droplet
    """
    def add_phases(self, droplets: List[Droplet]) -> None:
        """
        adds droplet to the domain. 
        Domain must contain solid regions. such as a rough surface 
        or a surface with patterns
        """
        for drop in droplets:
            coord_x, coord_y, coord_z = drop.center 
            radius = drop.radius 
            self.domain[np.where(
            (self.xx - coord_x) ** 2 + 
            (self.yy - coord_y) ** 2 + 
            (self.zz - coord_z) ** 2 <= radius ** 2
                )] = 3

# ######################################### #
#   Nodemap class for Drying problems       #
# ######################################### #
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
    """
    def __init__(
        self, 
        domain: Optional[np.ndarray] = None, 
        file_stem: Optional[str] = None, 
        save_path: Optional[Union[str, Path]] = None,
        padding: Optional[Union[int, List[int]]] = None, 
        side_walls: Optional[Union[int, List[int]]] = None, 
        geometry_side_walls: Optional[Union[int, List[int]]] = None, 
        gap_from_edge: Optional[ZoneConfig] = None
    ) -> None:
        """
        Initializes the DryingNodeMap class.

        Args:
            domain (Optional[np.ndarray]): A 3D numpy array representing the simulation domain grid.
            file_stem (Optional[str]): Name prefix for saved output files.
            save_path (Optional[Union[str, Path]]): Directory for saving the output files. Defaults to the current directory.
            padding (Optional[Union[int, List[int]]]): Padding values for all six dimensions [x_min, x_max, y_min, y_max, z_min, z_max].
                                                       Defaults to None (no padding).
            side_walls (Optional[Union[int, List[int]]]): Thickness of side walls around six dimensions.
            geometry_side_walls (Optional[Union[int, List[int]]]): Sidewalls that wrap the geomrty itself.
            gap_from_edge (Optional[LiquidReleaseZone]), gap between the liquid and edges of the domain 
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

        self.gap_from_edge = ZoneConfig(
            x_min = 2,
            y_min = 2,
            z_min = 2,
            x_max = 2, 
            y_max = 2,
            z_max = 2) if gap_from_edge is None else gap_from_edge  
    
    @property
    def domain_boundaries(self) -> ZoneConfig:
        boundaries = {}
        axes = ['x', 'y', 'z']
        for i, axis in enumerate(axes):
            boundaries[f"{axis}_min"] = self.padding[2*i] + self.side_walls[2*i]
            boundaries[f"{axis}_max"] = self.domain_shape[i] - (self.padding[2*i+1] + self.side_walls[2*i+1])
        return ZoneConfig(**boundaries)

    def _add_liquid(self):
        """
        Adds fluid phase to the domain, excluding areas defined by liquid_zone
        Modifies the domain by marking nodes within the restricted boundaries as fluid (value = 3).
        """
        boundaries = self.domain_boundaries
        gap = self.gap_from_edge
        axes = ['x', 'y', 'z']
        mins = []
        maxs = []
        for axis in axes:
            mins.append(getattr(boundaries, f"{axis}_min") + getattr(gap, f"{axis}_min"))
            maxs.append(getattr(boundaries, f"{axis}_max") - getattr(gap, f"{axis}_max"))
        self.domain[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]] = 3   
  
    def add_phases(self):
        """
        add the liquid phase for drying simulations
        """
        self._add_liquid()
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
            f.write(f'total number of fluid nodes is {self.total_number_fluid_nodes}')
    
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
        """
        super().__call__(separate=separate, vtk=vtk, bounce_method=bounce_method)
        self.add_file_info()


# ############################## #
#      helper functions          #
# ############################## # 

# ------------------------------- #
#  helpers to generate benchmarks #
# ------------------------------- # 
BENCHMARK_REGISTRY = {
    "overlapping-spheres": benchmarks.OverlappingSpherePack,
}
 
def drying_nodemap_from_benchmark(
    benchmark: str, 
    save_path: Optional[Union[str, Path]] = None, 
    gap_from_edge: Optional[Union[int, List[int], ZoneConfig]] = None,
    padding: Optional[Union[int, List[int]]] = None, 
    side_walls: Optional[Union[int, List[int]]] = None, 
    geometry_side_walls: Optional[Union[int, List[int]]] = [0] * 6, 
    separate: bool = True, 
    bounce_method: Literal["circ", "edt"] = "circ", 
    vtk: bool = False, 
    **benchmark_kw: Any
    ) -> None:
    """
    generates drying nodemap from registered benchmarks
    """
    if benchmark not in BENCHMARK_REGISTRY:
        raise KeyError(f"requested benchmark {benchmark} does not exist in the registry")
    
    try:
        benchmark_obj = BENCHMARK_REGISTRY[benchmark](**benchmark_kw)
    except Exception as e:
        print(f"failed to create the benchmark object {e}")
    
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

 


















    











        
