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

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Literal, Set 
import numpy as np
import porespy as ps
import openpnm as op
from tqdm import tqdm
from scipy.stats import binned_statistic
import pandas as pd

@dataclass
class PoreNetworkData:
    """Immutable data structure for pore network properties"""
    pore_coords: np.ndarray
    pore_diameters: np.ndarray
    pore_volumes: np.ndarray
    pore_surface_areas: np.ndarray
    throat_conns: np.ndarray
    throat_diameters: np.ndarray
    regions: np.ndarray
    region_ids: np.ndarray
    domain_shape: Tuple[int, int, int]
    porosity: float
    
    @property
    def num_pores(self) -> int:
        return len(self.pore_coords)
    
    @property
    def num_throats(self) -> int:
        return len(self.throat_conns)

@dataclass
class PhaseDistributionData:
    """Immutable data structure for phase distribution at a time step"""
    time_step: int
    phase_coordinates: np.ndarray
    pore_fills: np.ndarray
    phase_indices: np.ndarray
    saturation: float
    wet_pore_ids: np.ndarray

@dataclass
class NetworkConfig:
    """Configuration for network extraction"""
    method: Literal['snow', 'snow2'] = 'snow2'
    boundary_width: int = 0

@dataclass
class PhaseConfig:
    """Configuration for phase analysis"""
    physics: Literal['multiphase', 'drying'] = 'multiphase'
    phase: Literal['invading', 'defending'] = 'invading'
    threshold_tol: float = 0.1


class PoreNetworkExtractor:
    """Handles network extraction from binary images"""
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        self.config = config or NetworkConfig()
    
    def extract(self, binary_image: np.ndarray) -> PoreNetworkData:
        """Extract pore network from binary image"""
        # Convert to PoreSpy format (True for pores, False for solid)
        pn_image = binary_image.astype(bool)
        
        # Extract network using configured method
        if self.config.method == 'snow2':
            snow_output = ps.networks.snow2(
                pn_image, 
                boundary_width=self.config.boundary_width
            )
            network = op.io.network_from_porespy(snow_output.network)
            regions = snow_output.regions
        else:
            snow_output = ps.filters.snow_partitioning(pn_image)
            net = ps.networks.regions_to_network(snow_output.regions)
            network = op.io.network_from_porespy(net)
            regions = snow_output.regions
        
        # Calculate porosity
        domain_shape = binary_image.shape
        porosity = np.sum(binary_image) / np.prod(domain_shape)
        
        # Extract region information
        region_ids = np.unique(regions)[1:]  # Exclude 0 (solid)
        
        return PoreNetworkData(
            pore_coords=network['pore']['coords'],
            pore_diameters=network['pore']['inscribed_diameter'],
            pore_volumes=self._calculate_volumes(network['pore']['inscribed_diameter']),
            pore_surface_areas=network['pore']['surface_area'],
            throat_conns=network['throat']['conns'],
            throat_diameters=network['throat']['inscribed_diameter'],
            regions=regions,
            region_ids=region_ids,
            domain_shape=domain_shape,
            porosity=porosity
        )
    
    @staticmethod
    def _calculate_volumes(diameters: np.ndarray) -> np.ndarray:
        """Calculate spherical volumes from diameters"""
        return (4/3) * np.pi * (diameters/2)**3

# ####################### #
# Network Analyzer        #
# ####################### #
class PoreNetworkAnalyzer:
    """Enhanced analyzer with all original properties"""
    
    def __init__(self, network_data: PoreNetworkData):
        self.data = network_data
        self._adjacency_matrix = None
        self._connectivity = None
        self._weighted_connectivity = None
        self._connected_pore_ids = None
        self._disconnected_pore_ids = None
    
    # Original connectivity properties
    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Lazy calculation of adjacency matrix"""
        if self._adjacency_matrix is None:
            self._adjacency_matrix = self._build_adjacency_matrix()
        return self._adjacency_matrix
    
    @property
    def connectivity(self) -> np.ndarray:
        """Number of connections per pore"""
        if self._connectivity is None:
            adj_matrix = self.adjacency_matrix
            self._connectivity = np.sum(adj_matrix, axis=0)
        return self._connectivity
    
    @property
    def weighted_connectivity(self) -> np.ndarray:
        """Throat-diameter-weighted connectivity"""
        if self._weighted_connectivity is None:
            self._weighted_connectivity = self._build_weighted_connectivity()
        return self._weighted_connectivity
    
    @property
    def connected_pore_ids(self) -> np.ndarray:
        """IDs of pores with at least one connection"""
        if self._connected_pore_ids is None:
            self._connected_pore_ids = np.where(self.connectivity > 0)[0]
        return self._connected_pore_ids
    
    @property
    def disconnected_pore_ids(self) -> np.ndarray:
        """IDs of isolated pores"""
        if self._disconnected_pore_ids is None:
            self._disconnected_pore_ids = np.where(self.connectivity == 0)[0]
        return self._disconnected_pore_ids
    
    @property
    def disconnected_index(self) -> np.ndarray:
        """Binary array: 1 for disconnected pores, 0 for connected"""
        return np.where(self.connectivity == 0, 1, 0)
    
    @property
    def connected_index(self) -> np.ndarray:
        """Binary array: 1 for connected pores, 0 for disconnected"""
        return np.where(self.connectivity == 0, 0, 1)
    
    # Pore size properties
    @property
    def connected_pore_diameters(self) -> np.ndarray:
        """Diameters of connected pores only"""
        return self.data.pore_diameters[self.connected_pore_ids]
    
    @property
    def disconnected_pore_diameters(self) -> np.ndarray:
        """Diameters of disconnected pores only"""
        return self.data.pore_diameters[self.disconnected_pore_ids]
    
    @property
    def connected_pore_coordinates(self) -> np.ndarray:
        """Coordinates of connected pores only"""
        return self.data.pore_coords[self.connected_pore_ids]
    
    @property
    def disconnected_pore_coordinates(self) -> np.ndarray:
        """Coordinates of disconnected pores only"""
        return self.data.pore_coords[self.disconnected_pore_ids]
    
    # Volume properties
    @property
    def total_pore_volume(self) -> float:
        """Total volume of all pores (assuming spherical)"""
        return np.sum((4/3) * np.pi * (self.data.pore_diameters/2)**3)
    
    @property
    def total_connected_pore_volume(self) -> float:
        """Total volume of connected pores only"""
        return np.sum((4/3) * np.pi * (self.connected_pore_diameters/2)**3)
    
    @property
    def total_disconnected_pore_volume(self) -> float:
        """Total volume of disconnected pores only"""
        return np.sum((4/3) * np.pi * (self.disconnected_pore_diameters/2)**3)
    
    @property
    def total_disconnected_voxel_volume(self) -> float:
        """Total voxel volume of disconnected pores (cube of diameter)"""
        return np.sum(self.disconnected_pore_diameters**3)
    
    @property
    def domain_volume(self) -> int:
        """Total domain volume in voxels"""
        return np.prod(self.data.domain_shape)
    
    # Statistical properties
    @property
    def mean_pore_diameter(self) -> float:
        """Mean diameter of all pores"""
        return np.mean(self.data.pore_diameters)
    
    @property
    def std_pore_diameter(self) -> float:
        """Standard deviation of pore diameters"""
        return np.std(self.data.pore_diameters)
    
    @property
    def mean_connected_diameter(self) -> float:
        """Mean diameter of connected pores"""
        if len(self.connected_pore_diameters) > 0:
            return np.mean(self.connected_pore_diameters)
        return 0.0
    
    @property
    def mean_disconnected_diameter(self) -> float:
        """Mean diameter of disconnected pores"""
        if len(self.disconnected_pore_diameters) > 0:
            return np.mean(self.disconnected_pore_diameters)
        return 0.0
    
    @property
    def connectivity_fraction(self) -> float:
        """Fraction of pores that are connected"""
        return len(self.connected_pore_ids) / self.data.num_pores
    
    @property
    def mean_coordination_number(self) -> float:
        """Average number of connections per pore"""
        return np.mean(self.connectivity)
    
    # Helper methods
    def _build_adjacency_matrix(self) -> np.ndarray:
        """Build adjacency matrix from throat connections"""
        n_pores = self.data.num_pores
        adj_matrix = np.zeros((n_pores, n_pores))
        
        for p1, p2 in self.data.throat_conns:
            adj_matrix[p1, p2] = 1
            adj_matrix[p2, p1] = 1
        
        return adj_matrix
    
    def _build_weighted_connectivity(self) -> np.ndarray:
        """Build throat-diameter-weighted connectivity"""
        n_pores = self.data.num_pores
        weighted_adj = np.zeros((n_pores, n_pores))
        
        for i, (p1, p2) in enumerate(self.data.throat_conns):
            weight = self.data.throat_diameters[i]
            weighted_adj[p1, p2] = weight
            weighted_adj[p2, p1] = weight
        
        return np.sum(weighted_adj, axis=0)
    
    def mean_std(self, attribute: str) -> Tuple[float, float]:
        """Compute mean and standard deviation for any attribute"""
        if attribute == 'pore_diameter':
            values = self.data.pore_diameters
        elif attribute == 'pore_volume':
            values = self.data.pore_volumes
        elif attribute == 'throat_diameter':
            values = self.data.throat_diameters
        elif attribute == 'connectivity':
            values = self.connectivity
        else:
            raise ValueError(f"Unknown attribute: {attribute}")
        
        return np.mean(values), np.std(values)
    
    def get_connected_network(self) -> PoreNetworkData:
        """Extract only connected pores as a new network"""
        connected_ids = self.connected_pore_ids
        
        # Filter pore properties
        connected_coords = self.data.pore_coords[connected_ids]
        connected_diameters = self.data.pore_diameters[connected_ids]
        connected_volumes = self.data.pore_volumes[connected_ids]
        connected_surface_areas = self.data.pore_surface_areas[connected_ids]
        
        # Remap throat connections
        old_to_new = {old: new for new, old in enumerate(connected_ids)}
        connected_throats = []
        connected_throat_diams = []
        
        for i, (p1, p2) in enumerate(self.data.throat_conns):
            if p1 in old_to_new and p2 in old_to_new:
                connected_throats.append([old_to_new[p1], old_to_new[p2]])
                connected_throat_diams.append(self.data.throat_diameters[i])
        
        connected_throats = np.array(connected_throats)
        connected_throat_diams = np.array(connected_throat_diams)
        
        return PoreNetworkData(
            pore_coords=connected_coords,
            pore_diameters=connected_diameters,
            pore_volumes=connected_volumes,
            pore_surface_areas=connected_surface_areas,
            throat_conns=connected_throats,
            throat_diameters=connected_throat_diams,
            regions=self.data.regions,  # Keep original regions
            region_ids=self.data.region_ids,
            domain_shape=self.data.domain_shape,
            porosity=self.data.porosity
        )
    
    def compute_size_distribution(self, attribute: str = 'diameter', 
                                bins: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """Compute histogram of pore/throat sizes"""
        if attribute == 'diameter':
            values = self.data.pore_diameters
        elif attribute == 'volume':
            values = self.data.pore_volumes
        elif attribute == 'throat_diameter':
            values = self.data.throat_diameters
        else:
            raise ValueError(f"Unknown attribute: {attribute}")
        
        hist, bin_edges = np.histogram(values, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return hist, bin_centers

# ################################### #
# Properties Aggregator Class         #
# ################################### #
@dataclass
class NetworkProperties:
    """Aggregated network properties for easy access"""
    # Basic properties
    num_pores: int
    num_throats: int
    num_connected_pores: int
    num_disconnected_pores: int
    
    # Volume properties
    domain_volume: int
    total_pore_volume: float
    total_connected_pore_volume: float
    total_disconnected_pore_volume: float
    total_disconnected_voxel_volume: float
    
    # Statistical properties
    porosity: float
    connectivity_fraction: float
    mean_coordination_number: float
    mean_pore_diameter: float
    std_pore_diameter: float
    mean_connected_diameter: float
    mean_disconnected_diameter: float
    
    @classmethod
    def from_analyzer(cls, analyzer: PoreNetworkAnalyzer) -> 'NetworkProperties':
        """Create properties summary from analyzer"""
        return cls(
            num_pores=analyzer.data.num_pores,
            num_throats=analyzer.data.num_throats,
            num_connected_pores=len(analyzer.connected_pore_ids),
            num_disconnected_pores=len(analyzer.disconnected_pore_ids),
            domain_volume=analyzer.domain_volume,
            total_pore_volume=analyzer.total_pore_volume,
            total_connected_pore_volume=analyzer.total_connected_pore_volume,
            total_disconnected_pore_volume=analyzer.total_disconnected_pore_volume,
            total_disconnected_voxel_volume=analyzer.total_disconnected_voxel_volume,
            porosity=analyzer.data.porosity,
            connectivity_fraction=analyzer.connectivity_fraction,
            mean_coordination_number=analyzer.mean_coordination_number,
            mean_pore_diameter=analyzer.mean_pore_diameter,
            std_pore_diameter=analyzer.std_pore_diameter,
            mean_connected_diameter=analyzer.mean_connected_diameter,
            mean_disconnected_diameter=analyzer.mean_disconnected_diameter
        )
    
    def print_summary(self):
        """Print a formatted summary of network properties"""
        print(f"\n{'='*50}")
        print(f"PORE NETWORK PROPERTIES")
        print(f"{'='*50}")
        print(f"\nCounts:")
        print(f"  Total pores:         {self.num_pores}")
        print(f"  Connected pores:     {self.num_connected_pores} ({self.connectivity_fraction:.1%})")
        print(f"  Disconnected pores:  {self.num_disconnected_pores}")
        print(f"  Throats:             {self.num_throats}")
        
        print(f"\nVolumes:")
        print(f"  Domain volume:       {self.domain_volume:,} voxels")
        print(f"  Porosity:            {self.porosity:.3%}")
        print(f"  Total pore volume:   {self.total_pore_volume:.3e}")
        print(f"  Connected volume:    {self.total_connected_pore_volume:.3e}")
        print(f"  Disconnected volume: {self.total_disconnected_pore_volume:.3e}")
        
        print(f"\nStatistics:")
        print(f"  Mean coordination:   {self.mean_coordination_number:.2f}")
        print(f"  Mean diameter:       {self.mean_pore_diameter:.3f}")
        print(f"  Std diameter:        {self.std_pore_diameter:.3f}")
        print(f"  Mean connected:      {self.mean_connected_diameter:.3f}")
        print(f"  Mean disconnected:   {self.mean_disconnected_diameter:.3f}")
        print(f"{'='*50}\n")

class PhaseDistributionTracker:
    """Tracks phase positions within pore network over time"""
    
    def __init__(self, network_data: PoreNetworkData, config: Optional[PhaseConfig] = None):
        self.network = network_data
        self.config = config or PhaseConfig()
        self.analyzer = PoreNetworkAnalyzer(network_data)
        
        # Cache for pore voxel coordinates
        self._pore_voxel_cache: Dict[int, np.ndarray] = {}
        self._pore_voxel_set_cache: Dict[int, Set[tuple]] = {}
        
        # Build coordinate grid
        self._build_coordinate_grid()
        
        # Store history of phase distributions
        self.phase_history: List[PhaseDistributionData] = []
    
    def track_phase(self, rho1: np.ndarray, rho2: Optional[np.ndarray] = None, 
                   time_step: int = 0, use_connected_only: bool = True) -> PhaseDistributionData:
        """
        Track phase distribution at a given time step
        
        Args:
            rho1: Primary phase density field
            rho2: Secondary phase density field (for drying/multiphase)
            time_step: Current time step
            use_connected_only: Whether to track only connected pores
            
        Returns:
            PhaseDistributionData object containing results
        """
        # Get phase coordinates
        phase_coords = self._get_phase_coordinates(rho1, rho2)
        
        # Determine which pores to analyze
        if use_connected_only:
            pore_ids = self.analyzer.connected_pore_ids
        else:
            pore_ids = np.arange(self.network.num_pores)
        
        # Calculate pore fills
        pore_fills, phase_indices = self._calculate_pore_fills(phase_coords, pore_ids)
        
        # Expand arrays to full size if using connected only
        if use_connected_only:
            full_pore_fills = np.zeros(self.network.num_pores)
            full_phase_indices = np.zeros(self.network.num_pores)
            full_pore_fills[pore_ids] = pore_fills
            full_phase_indices[pore_ids] = phase_indices
            pore_fills = full_pore_fills
            phase_indices = full_phase_indices
        
        # Calculate saturation
        saturation = np.mean(pore_fills[pore_ids])
        
        # Get wet pore IDs
        wet_pore_ids = np.where(phase_indices > 0)[0]
        
        result = PhaseDistributionData(
            time_step=time_step,
            phase_coordinates=phase_coords,
            pore_fills=pore_fills,
            phase_indices=phase_indices,
            saturation=saturation,
            wet_pore_ids=wet_pore_ids
        )
        
        # Store in history
        self.phase_history.append(result)
        
        return result
    
    def track_phase_for_size_range(self, rho1: np.ndarray, rho2: Optional[np.ndarray], 
                                  time_step: int, size_range: Tuple[float, float]) -> PhaseDistributionData:
        """Track phase distribution for pores within a specific size range"""
        size_mask = ((self.network.pore_diameters >= size_range[0]) & 
                     (self.network.pore_diameters <= size_range[1]))
        pore_ids = np.where(size_mask)[0]
        
        phase_coords = self._get_phase_coordinates(rho1, rho2)
        
        pore_fills, phase_indices = self._calculate_pore_fills(phase_coords, pore_ids)
        
        full_pore_fills = np.zeros(self.network.num_pores)
        full_phase_indices = np.zeros(self.network.num_pores)
        full_pore_fills[pore_ids] = pore_fills
        full_phase_indices[pore_ids] = phase_indices
        
        saturation = np.mean(pore_fills)
        
        wet_pore_ids = pore_ids[phase_indices > 0]
        
        return PhaseDistributionData(
            time_step=time_step,
            phase_coordinates=phase_coords,
            pore_fills=full_pore_fills,
            phase_indices=full_phase_indices,
            saturation=saturation,
            wet_pore_ids=wet_pore_ids
        )
    
    def _build_coordinate_grid(self):
        """Build coordinate grid for the domain"""
        nx, ny, nz = self.network.domain_shape
        self.xx, self.yy, self.zz = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), 
            indexing='ij'
        )
    
    def _get_phase_coordinates(self, rho1: np.ndarray, 
                              rho2: Optional[np.ndarray]) -> np.ndarray:
        """Extract coordinates occupied by phase of interest"""
        if self.config.physics == 'drying' and rho2 is not None:
            rho_fraction = rho1 / (rho1 + rho2 + 1e-10)
            threshold = rho_fraction.max() * (1 - self.config.threshold_tol)
            phase_mask = rho_fraction > threshold
        else:
            threshold = 0.5 * (rho1.min() + rho1.max())
            if self.config.phase == 'invading':
                phase_mask = rho1 > threshold
            else:  
                phase_mask = rho2 > threshold if rho2 is not None else rho1 < threshold
        
        indices = np.where(phase_mask)
        return np.column_stack([
            self.xx[indices],
            self.yy[indices],
            self.zz[indices]
        ])
    
    def _calculate_pore_fills(self, phase_coords: np.ndarray, 
                             pore_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate filling fraction for specified pores"""
        n_pores = len(pore_ids)
        pore_fills = np.zeros(n_pores)
        phase_indices = np.zeros(n_pores)
        
        phase_set = set(map(tuple, phase_coords))
        
        if hasattr(self.network, 'regions') and self.network.regions is not None:
            pore_fills, phase_indices = self._calculate_fills_using_regions(
                phase_set, pore_ids
            )
        else:
            # Fallback to spherical approximation
            for i, pore_id in enumerate(tqdm(pore_ids, desc="Calculating pore fills")):
                pore_set = self._get_pore_voxel_set(pore_id)
                
                # Calculate overlap
                overlap = len(pore_set.intersection(phase_set))
                total = len(pore_set)
                
                if total > 0:
                    pore_fills[i] = overlap / total
                    if overlap > 0:
                        phase_indices[i] = 1
        
        return pore_fills, phase_indices
    
    def _calculate_fills_using_regions(self, phase_set: Set[tuple], 
                                      pore_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate pore fills using region labeling from SNOW algorithm"""
        n_pores = len(pore_ids)
        pore_fills = np.zeros(n_pores)
        phase_indices = np.zeros(n_pores)
        
        # Create mapping from pore ID to region label
        # Assuming pore IDs correspond to region labels - 1
        for i, pore_id in enumerate(tqdm(pore_ids, desc="Calculating pore fills (regions)")):
            region_label = pore_id + 1  # Region labels start at 1
            
            # Get all voxels in this region
            region_coords = np.where(self.network.regions == region_label)
            region_voxels = set(zip(region_coords[0], region_coords[1], region_coords[2]))
            
            # Calculate overlap with phase
            overlap = len(region_voxels.intersection(phase_set))
            total = len(region_voxels)
            
            if total > 0:
                pore_fills[i] = overlap / total
                if overlap > 0:
                    phase_indices[i] = 1
        
        return pore_fills, phase_indices
    
    def _get_pore_voxel_set(self, pore_id: int) -> Set[tuple]:
        """Get set of voxel coordinates for a pore (cached)"""
        if pore_id not in self._pore_voxel_set_cache:
            voxels = self._get_pore_voxels(pore_id)
            self._pore_voxel_set_cache[pore_id] = set(map(tuple, voxels))
        return self._pore_voxel_set_cache[pore_id]
    
    def _get_pore_voxels(self, pore_id: int) -> np.ndarray:
        """Get all voxel coordinates belonging to a pore (cached)"""
        if pore_id not in self._pore_voxel_cache:
            center = self.network.pore_coords[pore_id]
            radius = self.network.pore_diameters[pore_id] / 2
            
            # Find voxels within sphere
            dist_sq = ((self.xx - center[0])**2 + 
                      (self.yy - center[1])**2 + 
                      (self.zz - center[2])**2)
            
            indices = np.where(dist_sq <= radius**2)
            voxels = np.column_stack([
                self.xx[indices],
                self.yy[indices],
                self.zz[indices]
            ])
            self._pore_voxel_cache[pore_id] = voxels
        
        return self._pore_voxel_cache[pore_id]
    
    def clear_cache(self):
        """Clear cached pore voxel data to free memory"""
        self._pore_voxel_cache.clear()
        self._pore_voxel_set_cache.clear()

# ############################# #
# Phase Statistics Calculator   #
# ############################# #


class PhaseStatisticsCalculator:
    """Computes comprehensive statistics and histograms for phase distributions"""
    
    def __init__(self, network_data: PoreNetworkData):
        self.network = network_data
        self.analyzer = PoreNetworkAnalyzer(network_data)
    
    def compute_saturations(self, phase_data: PhaseDistributionData) -> Dict[str, float]:
        """
        Compute various saturation metrics
        
        Returns:
            Dictionary containing different saturation calculations
        """
        # Basic saturation (mean of pore fills)
        basic_saturation = phase_data.saturation
        
        # Volume-weighted saturation
        pore_volumes = self.network.pore_volumes
        volume_saturation = np.sum(phase_data.pore_fills * pore_volumes) / np.sum(pore_volumes)
        
        # Connected pores only saturation
        connected_ids = self.analyzer.connected_pore_ids
        connected_fills = phase_data.pore_fills[connected_ids]
        connected_saturation = np.mean(connected_fills)
        
        # Volume-weighted connected saturation
        connected_volumes = pore_volumes[connected_ids]
        connected_volume_saturation = (
            np.sum(connected_fills * connected_volumes) / np.sum(connected_volumes)
        )
        
        # Saturation by pore size classes
        size_class_saturations = self._compute_size_class_saturations(phase_data)
        
        return {
            'basic_saturation': basic_saturation,
            'volume_weighted_saturation': volume_saturation,
            'connected_saturation': connected_saturation,
            'connected_volume_saturation': connected_volume_saturation,
            'num_wet_pores': len(phase_data.wet_pore_ids),
            'fraction_wet_pores': len(phase_data.wet_pore_ids) / self.network.num_pores,
            'size_class_saturations': size_class_saturations
        }
    
    def compute_filling_histogram(self, phase_data: PhaseDistributionData,
                                bins: int = 40, 
                                use_connected_only: bool = True) -> Dict[str, np.ndarray]:
        """
        Compute histogram of pore filling vs size
        
        Args:
            phase_data: Phase distribution data
            bins: Number of histogram bins
            use_connected_only: Whether to use only connected pores
            
        Returns:
            Dictionary with histogram data
        """
        # Select pores to analyze
        if use_connected_only:
            pore_ids = self.analyzer.connected_pore_ids
        else:
            pore_ids = np.arange(self.network.num_pores)
        
        pore_sizes = self.network.pore_diameters[pore_ids]
        pore_fills = phase_data.pore_fills[pore_ids]
        
        # Compute statistics
        size_min, size_max = pore_sizes.min(), pore_sizes.max()
        
        # Mean filling per size bin
        fill_means, bin_edges, _ = binned_statistic(
            pore_sizes, pore_fills, 
            statistic='mean', bins=bins,
            range=(size_min, size_max)
        )
        
        # Count of pores per size bin
        count_hist, _, _ = binned_statistic(
            pore_sizes, pore_fills,
            statistic='count', bins=bins,
            range=(size_min, size_max)
        )
        
        # Sum of fills per size bin (for volume calculations)
        fill_sums, _, _ = binned_statistic(
            pore_sizes, pore_fills,
            statistic='sum', bins=bins,
            range=(size_min, size_max)
        )
        
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Handle NaN values
        fill_means = np.nan_to_num(fill_means, nan=0.0)
        
        return {
            'bin_centers': bin_centers,
            'fill_means': fill_means,
            'fill_sums': fill_sums,
            'pore_counts': count_hist,
            'saturation': phase_data.saturation,
            'bin_edges': bin_edges
        }
    
    def compute_wet_pore_distribution(self, phase_data: PhaseDistributionData,
                                    bins: int = 40) -> Dict[str, np.ndarray]:
        """
        Compute size distribution of wet pores only
        
        Returns:
            Dictionary with wet pore size distribution
        """
        wet_ids = phase_data.wet_pore_ids
        
        if len(wet_ids) == 0:
            return {
                'bin_centers': np.array([]),
                'counts': np.array([]),
                'fraction_wet': 0.0,
                'wet_sizes': np.array([])
            }
        
        wet_sizes = self.network.pore_diameters[wet_ids]
        
        # Compute histogram
        counts, bin_edges = np.histogram(wet_sizes, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Normalized histogram
        density, _ = np.histogram(wet_sizes, bins=bins, density=True)
        
        return {
            'bin_centers': bin_centers,
            'counts': counts,
            'density': density,
            'fraction_wet': len(wet_ids) / self.network.num_pores,
            'wet_sizes': wet_sizes,
            'mean_wet_size': np.mean(wet_sizes),
            'std_wet_size': np.std(wet_sizes)
        }
    
    def compute_mip_curve(self, phase_data_list: List[PhaseDistributionData],
                         size_bins: Optional[np.ndarray] = None,
                         n_bins: int = 50) -> Dict[str, np.ndarray]:
        """
        Compute mercury intrusion porosimetry style curves
        
        Args:
            phase_data_list: List of phase data at different pressures/times
            size_bins: Custom size bins, or None to auto-generate
            n_bins: Number of bins if auto-generating
            
        Returns:
            Dictionary with MIP curve data
        """
        if size_bins is None:
            size_min = self.network.pore_diameters.min()
            size_max = self.network.pore_diameters.max()
            size_bins = np.logspace(np.log10(size_min), np.log10(size_max), n_bins)
        
        n_steps = len(phase_data_list)
        n_size_bins = len(size_bins) - 1
        
        # Initialize result arrays
        intrusion_curve = np.zeros((n_steps, n_size_bins))
        cumulative_intrusion = np.zeros((n_steps, n_size_bins))
        
        for i, phase_data in enumerate(phase_data_list):
            cumulative_volume = 0
            total_volume = 0
            
            for j, (size_min, size_max) in enumerate(zip(size_bins[:-1], size_bins[1:])):
                # Find pores in size range
                mask = ((self.network.pore_diameters >= size_min) & 
                       (self.network.pore_diameters <= size_max))
                
                if np.any(mask):
                    # Calculate fraction filled
                    fills_in_range = phase_data.pore_fills[mask]
                    volumes_in_range = self.network.pore_volumes[mask]
                    
                    # Incremental intrusion
                    filled_volume = np.sum(fills_in_range * volumes_in_range)
                    range_volume = np.sum(volumes_in_range)
                    intrusion_curve[i, j] = filled_volume / range_volume if range_volume > 0 else 0
                    
                    # Cumulative intrusion
                    cumulative_volume += filled_volume
                    total_volume += range_volume
                    cumulative_intrusion[i, j] = cumulative_volume / total_volume if total_volume > 0 else 0
        
        return {
            'size_bins': 0.5 * (size_bins[:-1] + size_bins[1:]),
            'size_bin_edges': size_bins,
            'intrusion_curves': intrusion_curve,
            'cumulative_intrusion': cumulative_intrusion,
            'time_steps': [pd.time_step for pd in phase_data_list]
        }
    
    def compute_time_series_statistics(self, phase_data_list: List[PhaseDistributionData]) -> pd.DataFrame:
        """
        Compute statistics over time series of phase distributions
        
        Returns:
            DataFrame with time-series statistics
        """
        stats_list = []
        
        for phase_data in phase_data_list:
            saturations = self.compute_saturations(phase_data)
            
            # Basic statistics
            stats = {
                'time_step': phase_data.time_step,
                'basic_saturation': saturations['basic_saturation'],
                'volume_saturation': saturations['volume_weighted_saturation'],
                'connected_saturation': saturations['connected_saturation'],
                'num_wet_pores': saturations['num_wet_pores'],
                'fraction_wet_pores': saturations['fraction_wet_pores'],
                'mean_fill': np.mean(phase_data.pore_fills),
                'std_fill': np.std(phase_data.pore_fills),
                'max_fill': np.max(phase_data.pore_fills),
                'min_nonzero_fill': np.min(phase_data.pore_fills[phase_data.pore_fills > 0]) 
                                   if np.any(phase_data.pore_fills > 0) else 0
            }
            
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list)
    
    def _compute_size_class_saturations(self, phase_data: PhaseDistributionData,
                                      size_classes: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute saturation for different pore size classes"""
        if size_classes is None:
            # Default size classes (in diameter units)
            size_classes = np.array([0, 10, 20, 50, 100, np.inf])
        
        saturations = {}
        
        for i in range(len(size_classes) - 1):
            size_min, size_max = size_classes[i], size_classes[i + 1]
            mask = ((self.network.pore_diameters >= size_min) & 
                   (self.network.pore_diameters < size_max))
            
            if np.any(mask):
                class_fills = phase_data.pore_fills[mask]
                class_volumes = self.network.pore_volumes[mask]
                
                # Volume-weighted saturation for this size class
                saturation = np.sum(class_fills * class_volumes) / np.sum(class_volumes)
                saturations[f'{size_min:.0f}-{size_max:.0f}'] = saturation
        
        return saturations
    
    def compute_connectivity_statistics(self, phase_data: PhaseDistributionData) -> Dict[str, Any]:
        """
        Compute statistics related to phase connectivity
        
        Returns:
            Dictionary with connectivity-related statistics
        """
        wet_pore_ids = phase_data.wet_pore_ids
        
        if len(wet_pore_ids) == 0:
            return {
                'num_wet_clusters': 0,
                'largest_cluster_size': 0,
                'percolating': False,
                'wet_connectivity': 0.0
            }
        
        # Build subnetwork of wet pores
        wet_mask = np.zeros(self.network.num_pores, dtype=bool)
        wet_mask[wet_pore_ids] = True
        
        # Find connected components among wet pores
        wet_throats = []
        for i, (p1, p2) in enumerate(self.network.throat_conns):
            if wet_mask[p1] and wet_mask[p2]:
                wet_throats.append([p1, p2])
        
        if len(wet_throats) == 0:
            return {
                'num_wet_clusters': len(wet_pore_ids),
                'largest_cluster_size': 1,
                'percolating': False,
                'wet_connectivity': 0.0
            }
        
        # Analyze connectivity of wet network
        # (This is a simplified version - full implementation would use graph algorithms)
        wet_connectivity = len(wet_throats) / len(self.network.throat_conns)
        
        return {
            'num_wet_clusters': None,  # Would require graph analysis
            'largest_cluster_size': None,  # Would require graph analysis
            'percolating': None,  # Would require percolation analysis
            'wet_connectivity': wet_connectivity,
            'num_wet_throats': len(wet_throats),
            'fraction_wet_throats': len(wet_throats) / self.network.num_throats
        }

