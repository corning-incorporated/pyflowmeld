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
from dataclasses import dataclass 
from typing import Optional, Literal, Union, List, Any, Tuple    
from pathlib import Path
from .. utils import benchmarks   
from datetime import datetime 
from _base import NodeMap 


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
            drop_index = np.where(
            (self.xx - coord_x) ** 2 + 
            (self.yy - coord_y) ** 2 + 
            (self.zz - coord_z) ** 2 <= radius ** 2
                )
            self.domain[drop_index] = 3

#-------------------------------#
# Nodemap Generation for drying #
#-------------------------------#
@dataclass 
class ZoneConfig:
    """ 
    configuration for liquid release zone
    """
    x_min: int = 0
    x_max: int = 0
    y_min: int = 0
    y_max: int = 0
    z_min: int = 0
    z_max: int = 0 

#--------------------------#
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
        x_min = self.padding[0] + self.side_walls[0]
        y_min = self.padding[2] + self.side_walls[2]
        z_min = self.padding[4] + self.side_walls[4]

        x_max = self.domain_shape[0] - (self.padding[1] + self.side_walls[1])
        y_max = self.domain_shape[1] - (self.padding[3] + self.side_walls[3])
        z_max = self.domain_shape[2] - (self.padding[5] + self.side_walls[5])

        return ZoneConfig(
        x_min = x_min,
        y_min = y_min,
        z_min = z_min,
        x_max = x_max, 
        y_max = y_max, 
        z_max = z_max 
        )

    def _add_liquid(self):
        """
        Adds fluid phase to the domain, excluding areas defined by liquid_zone
        Modifies the domain by marking nodes within the restricted boundaries as fluid (value = 3).
        """
        boundaries = self.domain_boundaries 
        min_x = boundaries.x_min + self.gap_from_edge.x_min 
        max_x = boundaries.x_max - self.gap_from_edge.x_max 

        min_y = boundaries.y_min + self.gap_from_edge.y_min 
        max_y = boundaries.y_max - self.gap_from_edge.y_max 

        min_z = boundaries.z_min + self.gap_from_edge.z_min 
        max_z = boundaries.z_max - self.gap_from_edge.z_max 

        self.domain[min_x:max_x, min_y:max_y, min_z:max_z] = 3    
  
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

 


















    











        
