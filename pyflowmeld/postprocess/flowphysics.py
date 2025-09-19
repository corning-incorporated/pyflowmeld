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

import threading 
import numpy as np 
import pandas as pd 
from scipy import ndimage 
import matplotlib
import matplotlib.pyplot as plt 
from os import path, listdir, makedirs, PathLike   
from natsort import natsorted
from typing import Iterable, Sequence, Optional, Literal, Union, Tuple, Dict, List   
from pyflowmeld.utils import tools
from functools import partial, wraps  
from datetime import datetime 
from tqdm import tqdm 

matplotlib.use('AGG')

def get_files(data_path = None, stem = None):
    return natsorted([path.join(data_path, _file) for _file in listdir(data_path)
                                    if stem in _file and '.dat' in _file], key = lambda name: name.lower())

class PhaseFlow:
    """
    Analyzes phase distributions and saturation evolution in multiphase flow simulations.
    
    This class processes density distributions from two-phase flow simulations to:
    - Calculate phase saturations over time
    - Generate slice visualizations
    - Export 3D phase maps
    - Track saturation profiles along specified directions
    
    Parameters
    ----------
    data_path : PathLike
        Path to directory containing simulation data files
    domain_size : Sequence
        Domain dimensions as (nx, ny, nz)
    padding : Sequence, optional
        Padding values as [x_min, x_max, y_min, y_max, z_min, z_max], default [0]*6
    rho_f1_stem : str, optional
        File stem for phase 1 density files, default 'f1_rho_'
    rho_f2_stem : str, optional
        File stem for phase 2 density files, default 'f2_rho_'
    save_path : PathLike, optional
        Output directory path. If None, creates timestamped folder in data_path
    isolated_volume : float, optional
        Volume of isolated regions to exclude from saturation calculations, default 0
    
    Attributes
    ----------
    f1_saturation : list
        Phase 1 saturation values over time
    f2_saturation : list
        Phase 2 saturation values over time
    medium : ndarray
        Binary array indicating solid phase (1) vs fluid phase (0)

    Methods
    -------
    __call__(time_range, slices, phase, saturation_profiles, thresh)
        Execute phase analysis for the specified configurations, including slice extraction, 3D maps,
        and saturation rate computations.

    _generate_saturation_rate()
        Generates and saves saturation rates over time based on processed data.

    Example
    -------
    ```python
    phase_flow = PhaseFlow(
        data_path="/path/to/data",
        domain_size=(100, 100, 100),
        padding=[5, 5, 5, 5, 5, 5],
        rho_f1_stem="phase1_density_",
        rho_f2_stem="phase2_density_"
    )

    # Run the analysis with specific slices and directions
    phase_flow(
        time_range=(0, 50),
        slices={"x": 10, "z": 30},
        phase="invading",
        saturation_profiles=["x", "z"],
        thresh="median"
    )
    ```
    """
    def __init__(self, data_path: PathLike, domain_size: Sequence, padding: Optional[Sequence] = None, 
                    rho_f1_stem: str = 'f1_rho_', rho_f2_stem: str = 'f2_rho_', 
                        save_path: Optional[PathLike] = None, isolated_volume: float = 0):
        
        self._validate_inputs(data_path, domain_size, padding)

        self._initialize_density_files(data_path, rho_f1_stem, rho_f2_stem)

        self.domain_size = domain_size
        self.isolated_volume = isolated_volume
        self._padding = padding 
        self.max_length = len(self.rho1_files)

        self.xx, self.yy, self.zz = None, None, None
        self.res_x, self.res_y, self.res_z = None, None, None

        self.save_path = self._initialize_save_path(data_path, save_path)
        
        self.f1_saturation, self.f2_saturation = [], []
        self.medium = None

    def _validate_inputs(self, data_path: PathLike, domain_size: Sequence, padding: Sequence):
        """Validate inputs to ensure correctness."""
        if not path.exists(data_path):
            raise ValueError(f"Invalid data path '{data_path}' does not exist.")
        if len(domain_size) != 3:
            raise ValueError("domain_size must be a sequence of 3 integers (nx, ny, nz).")
        if len(padding) != 6:
            raise ValueError("padding must be a sequence of 6 integers (x_min, x_max, y_min, y_max, z_min, z_max).")

    def _initialize_density_files(self, data_path: PathLike, rho_f1_stem: str, rho_f2_stem: str):
        """Initialize file paths for phase density data."""
        self.rho1_files = get_files(data_path=data_path, stem=rho_f1_stem)
        self.rho2_files = get_files(data_path=data_path, stem=rho_f2_stem)

        if not self.rho1_files or not self.rho2_files:
            raise ValueError("No matching density files found in the data path.")

        simulation_path = path.join(data_path, 'simulation.dat')
        if path.exists(simulation_path):
            self.delta_p = [
                float(elem) for elem in open(simulation_path).read().splitlines()[2].split(':')[1].split(' ')
                if tools.is_float(elem)
            ]
        else:
            self.delta_p = None

    def _initialize_save_path(self, data_path: PathLike, save_path: Optional[PathLike]) -> str:
        """Create or initialize the save path."""
        if save_path:
            save_path = f"{save_path}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        else:
            save_path = path.join(data_path, f'postprocess_on_{datetime.now().strftime("%Y-%m-%d-%H-%M")}')

        if not path.exists(save_path):
            makedirs(save_path)
        return save_path

    def _has_padding(self) -> bool:
        """Check if padding is enabled."""
        return self._padding is not None and sum(self._padding) != 0
    
    @property 
    def remove_padding(self):
        """ check if padding removal must be done """
        return self._has_padding()

    @property
    def padding(self) -> Sequence:
        """
        Get the padding values.

        Returns
        -------
        Sequence
            List of six padding values `[x_min, x_max, y_min, y_max, z_min, z_max]`.
        """
        return self._padding
    
    @padding.setter
    def padding(self, new_padding: Union[int, Sequence, None]) -> None:
        """
        Set the padding values with validation.

        Parameters
        ----------
        new_padding : Union[int, Sequence, None]
            Padding can be:
            - An integer (applies the same padding value to all six sides)
            - A sequence of six integers for custom padding
            - None to disable padding

        Raises
        ------
        ValueError
            If `new_padding` is neither an integer nor a sequence of six integers.
        """
        if new_padding is None:
            self._padding = None
        elif isinstance(new_padding, int):
            self._padding = [new_padding] * 6
        elif isinstance(new_padding, Iterable) and len(new_padding) == 6 and all(isinstance(val, int) for val in new_padding):
            self._padding = list(new_padding)
        else:
            raise ValueError(
                "Invalid padding value. Must be either an integer, a sequence of 6 integers, or None."
            )
    
    # ######### Helper methods ########## # 
    @staticmethod
    def median(data_array: np.ndarray) -> float:
        """
        Compute the median of a numerical array.

        Parameters
        ----------
        data_array : np.ndarray
            Array of numerical values.
        
        Returns
        -------
        float
            Median value, calculated as the mean of the minimum and maximum values.
        """
        return 0.5 * (data_array.min() + data_array.max())

    @staticmethod
    def nonzero_min(data_array: np.ndarray) -> float:
        """
        Compute the minimum non-zero value in a numerical array.

        Parameters
        ----------
        data_array : np.ndarray
            Array of numerical values.
        
        Returns
        -------
        float
            Smallest non-zero value in the array.
        
        Raises
        ------
        ValueError
            If the array has no non-zero values.
        """
        non_zero_values = data_array[data_array > 0]
        if len(non_zero_values) == 0:
            raise ValueError("Array has no non-zero values.")
        return non_zero_values.min()
    
    def _get_slice(self, direction: Literal['x', 'y', 'z'], coordinate: int) -> Tuple[slice, slice, slice]:
        """
        Wrapper for the module-level `get_slice` function for compatibility with class attributes.
        """
        return tools.get_slice(self.domain_size, direction, coordinate)

    def _generate_slice_csv(
        self,
        slices: Dict[str, int],
        f_dist: np.ndarray,
        cycle: int,
        phase_id: Literal['wetting', 'defending'] = 'wetting'
    ) -> None:
        """
        Generate and save phase distribution data for specific slices.

        Parameters
        ----------
        slices : dict
            Dictionary specifying slice direction ('x', 'y', 'z') and coordinate, e.g., {'x': 10, 'y': 20}.
        f_dist : np.ndarray
            Phase distribution array in the simulation domain.
        cycle : int
            Simulation cycle (used for naming output files).
        phase_id : {'wetting', 'defending'}, optional
            Type of phase for output data, default 'wetting'.

        Returns
        -------
        None
            CSV files for the slices are saved in the `save_path`.

        Raises
        ------
        ValueError
            If slices contain invalid directions or coordinates.
        """
        def validate_slices(slices_dict: Dict[str, int]):
            for direction, coordinate in slices_dict.items():
                if direction not in {'x', 'y', 'z'}:
                    raise ValueError(
                        f"Invalid direction '{direction}' in slices. Must be one of 'x', 'y', 'z'."
                    )
                bounds = {'x': self.res_x, 'y': self.res_y, 'z': self.res_z}
                if coordinate < 0 or coordinate >= bounds[direction]:
                    raise ValueError(
                        f"Coordinate {coordinate} is out of bounds for direction '{direction}'. "
                        f"Valid range: 0 <= coordinate < {bounds[direction]}"
                    )
        
        validate_slices(slices)

        for direction, coordinate in slices.items():
            slice_index = self._get_slice(direction, coordinate)
            xx = self.xx[slice_index]
            yy = self.yy[slice_index]
            zz = self.zz[slice_index]
            rho = f_dist[slice_index]

            phase_key = f"{phase_id}_density"
            rho_df = pd.DataFrame(
                np.c_[
                    xx.flatten()[:, np.newaxis],
                    yy.flatten()[:, np.newaxis],
                    zz.flatten()[:, np.newaxis],
                    rho.flatten()[:, np.newaxis]
                ],
                columns=['x', 'y', 'z', phase_key]
            )
            rho_df = rho_df[rho_df[phase_key] != 0.0]  

            filename = path.join(
                self.save_path,
                f"{phase_key}_on_slice_{direction}_{coordinate}_{cycle}.csv"
            )
            rho_df.to_csv(filename, sep=',', header=True, index=False, float_format='%.5f')

            if cycle == 0:
                domain = self.medium[slice_index]
                domain_df = pd.DataFrame(
                    np.c_[
                        xx.flatten()[:, np.newaxis],
                        yy.flatten()[:, np.newaxis],
                        zz.flatten()[:, np.newaxis],
                        domain.flatten()[:, np.newaxis]
                    ],
                    columns=['x', 'y', 'z', 'domain']
                )
                domain_df = domain_df[domain_df['domain'] == 1]  
                medium_filename = path.join(
                    self.save_path,
                    f"medium_on_slice_{direction}_{coordinate}.csv"
                )
                domain_df.to_csv(medium_filename, sep=',', header=True, index=False, float_format='%.5f')

    def _generate_saturation_profile(self, phase: str = 'wetting', directions: Optional[list] = None, 
                                     f_dist: np.ndarray = None, cycle: int = None) -> None:
        """
        Generate saturation profiles for the specified fluid phase.

        Parameters
        ----------
        phase : str, optional
            Fluid phase ('wetting' or 'defending'), default is 'wetting'.
        directions : list of str, optional
            List of directions for saturation profiles ('x', 'y', 'z').
        f_dist : np.ndarray
            Density distribution array for the current fluid phase.
        cycle : int
            Current simulation cycle (used for naming output files).

        Returns
        -------
        None
            Saturation profiles are saved as .dat files in the `save_path`.

        Raises
        ------
        ValueError
            If `directions` contains invalid directions or required inputs are missing.
        ZeroDivisionError
            If computed denominator values are zero during saturation calculations.
        """
        if directions is None or not all(dir in {'x', 'y', 'z'} for dir in directions):
            raise ValueError("Invalid directions. Must be a list containing 'x', 'y', or 'z'.")
        if f_dist is None or self.medium is None:
            raise ValueError("Inputs 'f_dist' and 'self.medium' must not be None.")
        if cycle is None:
            raise ValueError("The 'cycle' must be a valid integer.")

        if not path.exists(self.save_path):
            makedirs(self.save_path)

        def compute_saturation(direction: str) -> np.ndarray:
            axis = {'x': (1, 2), 'y': (0, 2), 'z': (0, 1)}[direction]
            total_medium = np.sum(self.medium, axis=axis)
            if np.any(total_medium == 0):
                raise ZeroDivisionError(f"Division by zero encountered in direction '{direction}'.")
            return tools.true_division(np.sum(f_dist, axis=axis), total_medium)

        for direction in directions:
            profile = compute_saturation(direction)
            file_name = path.join(self.save_path, f'{phase}_saturation_profile_direction_{direction}.dat')
            with open(file_name, 'a') as f:
                f.write(','.join([f'{val:.5f}' for val in profile]))
                f.write('\n')
    
    def _generate_saturation_rate(self) -> None:
        """
        Generate and save a CSV file for saturation rates over time.

        This method calculates saturation rates for phase 1 (`f1_saturation`) and phase 2 (`f2_saturation`)
        for the simulation cycles completed so far. The results are saved as a CSV file
        in the specified save directory.

        Returns
        -------
        None
            Saturation rates are saved as a CSV file in the `save_path`.

        Raises
        ------
        ValueError
            If no saturation data is available for either phase.
        """
        # Validate saturation data
        if not self.f1_saturation or not self.f2_saturation:
            raise ValueError("Saturation data for one or both phases is missing.")

        # Convert saturation data to a DataFrame
        saturation_data = {
            'f1_saturation': self.f1_saturation,
            'f2_saturation': self.f2_saturation,
        }
        saturation_df = pd.DataFrame(saturation_data)

        # Save the DataFrame to a CSV file
        output_file = path.join(self.save_path, 'saturation_rates.csv')
        saturation_df.to_csv(
            output_file,
            index_label='time',
            header=True,
            index=True,
            float_format='%.5f',
        )
        print(f"Saturation rates saved to {output_file}.")
            

    def __call__(
        self,
        time_range: Optional[Tuple[int, int]] = None,
        slices: Dict[str, int] = {},
        phase: Literal['invading', 'defending'] = 'invading',
        saturation_profiles: Optional[List[str]] = None,
        thresh: Literal['median', 'mean'] = 'median'
    ) -> None:
        """
        Execute phase analysis for the specified configurations.

        This method processes phase distribution data between the given range of time steps, computing
        attributes such as slice densities, saturation profiles, 3D maps, and saturation rates.

        Parameters
        ----------
        time_range : Optional[Tuple[int, int]]
            Range of simulation time steps to process (start, end). If None, processes all steps.
        slices : Dict[str, int]
            Slices to analyze specified as a dictionary of direction -> coordinate.
            Example: {'x': 10, 'y': 20}.
        phase : {'invading', 'defending'}, optional
            Phase type to analyze ('invading' or 'defending'). Defaults to 'invading'.
        saturation_profiles : Optional[List[str]], optional
            Directions for saturation profile generation (e.g., ['x', 'z']).
            Defaults to None.
        thresh : {'median', 'mean'}, optional
            Thresholding method for phase distributions. Defaults to 'median'.

        Returns
        -------
        None

        Example
        -------
        ```python
        phase_flow(
            time_range=(0, 50),
            slices={"x": 10, "z": 20},
            phase="invading",
            saturation_profiles=["x", "z"],
            thresh="median"
        )
        ```
        """
        self._validate_call_inputs(time_range, slices, phase, saturation_profiles, thresh)
        time_min, time_max = time_range if time_range else (0, len(self.rho1_files))

        
        # => main loop for all calculations <= #
        for count, rho_files in enumerate(tqdm(zip(self.rho1_files[time_min:time_max], self.rho2_files[time_min:time_max]))):
            print(f'processing cycle {count} ...')
            rho1 = tools.load_reshape_trim(rho_files[0], self.domain_size, self.padding)
            rho2 = tools.load_reshape_trim(rho_files[1], self.domain_size, self.padding)
                        
            thresh_rho1 = self.median(rho1) if thresh == "median" else rho1[rho1 > 0].mean()
            thresh_rho2 = self.median(rho2) if thresh == "mean" else rho2[rho2 > 0].mean() 
            
            f1_dist = np.where(rho1 > thresh_rho1, 1, 0)
            f2_dist = np.where(rho2 > thresh_rho2, 1, 0)

            if count == 0:
                self._initialize_geometry(rho1, f1_dist, f2_dist)

            if slices:
                self._generate_slice_csv(slices, f1_dist if phase == "invading" else f2_dist,
                        cycle = count, phase_id = phase)


            
            if saturation_profiles:
                self._generate_saturation_profile(phase = phase,
                                                directions = saturation_profiles,
                                                  f_dist = {'invading': f1_dist, 
                                                                'defending': f2_dist}[phase], cycle = count)    
            self.f1_saturation.append((np.sum(f1_dist) - self.isolated_volume)/(self.num_voids - self.isolated_volume))
            self.f2_saturation.append((np.sum(f2_dist) - self.isolated_volume)/(self.num_voids - self.isolated_volume))

            # will generate a saturation rate 
            self._generate_saturation_rate()

    def _validate_call_inputs(
        self,
        time_range: Optional[Tuple[int, int]],
        slices: Dict[str, int],
        phase: Literal['invading', 'defending'],
        saturation_profiles: Optional[List[str]],
        thresh: Literal['median', 'mean']
    ) -> None:
        """
        Validates inputs to the __call__ method.

        Parameters
        ----------
        time_range : Optional[Tuple[int, int]]
            Range of simulation time steps to process.
        slices : Dict[str, int]
            Slices to process as direction -> coordinate.
        phase : {'invading', 'defending'}
            Phase type to analyze.
        saturation_profiles : Optional[List[str]]
            Directions for saturation profile generation.
        thresh : {'median', 'mean'}
            Thresholding method.

        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        if time_range and (len(time_range) != 2 or not all(isinstance(val, int) for val in time_range)):
            raise ValueError("'time_range' must be a tuple of two integers (start, end).")
        if slices and not all(direction in {'x', 'y', 'z'} for direction in slices.keys()):
            raise ValueError("Invalid directions in 'slices'. Must be 'x', 'y', or 'z'.")
        if phase not in {'invading', 'defending'}:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'invading' or 'defending'.")
        if saturation_profiles and not all(profile in {'x', 'y', 'z'} for profile in saturation_profiles):
            raise ValueError("Invalid directions in 'saturation_profiles'. Must be 'x', 'y', or 'z'.")
        if thresh not in {'median', 'mean'}:
            raise ValueError(f"Invalid threshold '{thresh}'. Must be 'median' or 'mean'.")

    def _initialize_geometry(self, rho1: np.ndarray, f1_dist: np.ndarray, f2_dist: np.ndarray) -> None:
        """
        Initialize geometry-related attributes during the first cycle.

        Parameters
        ----------
        rho1 : np.ndarray
            Binarized distribution for phase 1.
        f1_dist : np.ndarray
            Phase 1 binary distribution array.
        f2_dist : np.ndarray
            Phase 2 binary distribution array.

        Returns
        -------
        None
        """
        self.res_x, self.res_y, self.res_z = rho1.shape
        self.xx, self.yy, self.zz = np.meshgrid(
            np.arange(0, self.res_x),
            np.arange(0, self.res_y),
            np.arange(0, self.res_z),
            indexing='ij'
        )
        self.medium = np.zeros_like(f1_dist)
        self.medium[np.logical_and(f1_dist == 0, f2_dist == 0)] = 1
        self.num_solids = np.sum(self.medium)
        self.num_voids = (self.res_x * self.res_y * self.res_z) - self.num_solids


# ###################################### #
#   MultiphaseProcessor                  #
# ###################################### #
class MultiphaseProcessor:
    """
    General-purpose processor for analyzing multiphase simulation data.

    This class provides tools to load, trim, and process multiphase flow simulation files.
    It serves as a foundational utility for subclasses implementing specific analyses.

    Parameters
    ----------
    data_path : PathLike
        Path to the directory containing simulation data files.
    domain_size : Sequence[int]
        Domain dimensions as (nx, ny, nz).
    trim : Sequence[int], optional
        Padding to trim from each side of the simulation domain, specified as
        [x_min, x_max, y_min, y_max, z_min, z_max]. Default is [0, 0, 0, 0, 0, 0].
    fluid_id : {'f1', 'f2'}, optional
        Identifier for the fluid phase being analyzed. Default is 'f1' (phase 1).
    save_path : PathLike, optional
        Directory for saving processed outputs. If None, a subdirectory named `multi_phase_process` is created
        within `data_path`. Default is None.
    **kwargs
        Additional arguments for potential subclass customization.

    Attributes
    ----------
    rho_files : list
        List of file paths containing density data for the specified fluid phase.
    domain_size : Sequence[int]
        Domain dimensions as (nx, ny, nz).
    trim : Sequence[int]
        Padding values to trim from the simulation domain.
    xx, yy, zz : ndarray
        Meshgrid arrays for spatial coordinates in the domain.

    Methods
    -------
    _compute_coordinates()
        Computes the trimmed spatial meshgrid arrays.
    binarize(rho: np.ndarray) -> Tuple[np.ndarray]:
        Converts density data into a binary array based on median value.
    """
    def __init__(
        self, 
        data_path: PathLike, 
        domain_size: Sequence[int], 
        trim: Sequence[int] = [0, 0, 0, 0, 0, 0], 
        fluid_id: str = 'f1', 
        save_path: Optional[PathLike] = None, 
        **kwargs
    ):
        """
        Initialize the `MultiphaseProcessor` class with simulation parameters.

        Parameters
        ----------
        data_path : PathLike
            Path to the directory containing simulation data files.
        domain_size : Sequence[int]
            Domain dimensions specified as (nx, ny, nz).
        trim : Sequence[int], optional
            Padding to trim from the simulation domain as [x_min, x_max, y_min, y_max, z_min, z_max].
            Default is [0, 0, 0, 0, 0, 0].
        fluid_id : {'f1', 'f2'}, optional
            Identifier for the fluid phase being analyzed. Default is 'f1'.
        save_path : PathLike, optional
            Directory for saving processed outputs. If None, a subdirectory is created in `data_path`.
        **kwargs
            Additional arguments for subclass-specific customization.

        Raises
        ------
        ValueError
            If `domain_size` is invalid, `trim` is invalid, or `data_path` does not exist.
        """
        # Validate domain size
        if not (isinstance(domain_size, (list, tuple)) and len(domain_size) == 3 and all(isinstance(n, int) and n > 0 for n in domain_size)):
            raise ValueError("'domain_size' must be a sequence of three positive integers (nx, ny, nz).")

        # Validate trim
        if not (isinstance(trim, (list, tuple)) and len(trim) == 6 and all(isinstance(n, int) and n >= 0 for n in trim)):
            raise ValueError("'trim' must be a sequence of six non-negative integers [x_min, x_max, y_min, y_max, z_min, z_max].")

        # Validate fluid_id
        if fluid_id not in {'f1', 'f2'}:
            raise ValueError(f"'fluid_id' must be one of 'f1' or 'f2'. Received: {fluid_id}")

        # Validate data path
        if not path.exists(data_path) or not path.isdir(data_path):
            raise ValueError(f"'data_path' does not exist or is not a valid directory. Received: {data_path}")

        # Initialize attributes
        self.domain_size = domain_size
        self.trim = trim
        self.fluid_id = fluid_id
        self.rho_files = get_files(data_path=data_path, stem={'f1': 'f1_rho_', 'f2': 'f2_rho_'}[fluid_id])
        
        # Ensure density files exist
        if not self.rho_files:
            raise ValueError(f"No density files found in '{data_path}' matching fluid_id '{fluid_id}'.")

        # Create save directory if not provided
        if save_path is None:
            save_path = path.join(data_path, 'multi_phase_process')
        
        # Make directory or validate existing path
        try:
            self.save_path = tools.make_dir(save_path)
        except Exception as e:
            raise IOError(f"Failed to create or access save directory '{save_path}'. Error: {str(e)}")
        
        # Compute meshgrid for the trimmed domain
        self.xx, self.yy, self.zz = [None] * 3
        self._compute_coordinates()

    def _compute_coordinates(self) -> None:
        """
        Compute the trimmed spatial meshgrid arrays (xx, yy, zz).

        This method creates coordinate arrays (`xx`, `yy`, `zz`) for the simulation domain
        after applying trimming (`self.trim`) to the original domain size (`self.domain_size`).

        Raises
        ------
        ValueError
            If the computed dimensions after trimming are invalid (non-positive).
        """
        # Calculate trimmed domain dimensions
        self.res_x = max(0, self.domain_size[0] - (self.trim[0] + self.trim[1]))
        self.res_y = max(0, self.domain_size[1] - (self.trim[2] + self.trim[3]))
        self.res_z = max(0, self.domain_size[2] - (self.trim[4] + self.trim[5]))

        # Validate dimensions
        if self.res_x <= 0 or self.res_y <= 0 or self.res_z <= 0:
            raise ValueError(
                f"Invalid trimmed domain dimensions: ({self.res_x}, {self.res_y}, {self.res_z}). "
                "Ensure trim values do not exceed domain size."
            )

        # Create meshgrid
        try:
            self.xx, self.yy, self.zz = np.meshgrid(
                np.arange(self.res_x), np.arange(self.res_y), np.arange(self.res_z), indexing="ij"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create meshgrid for dimensions ({self.res_x}, {self.res_y}, {self.res_z}). Error: {str(e)}"
            )
    

    @staticmethod
    def binarize(
        rho: np.ndarray, 
        threshold_method: Literal["median", "mean", "custom"] = "median", 
        custom_threshold: Optional[float] = None, 
        high_value: int = 1, 
        low_value: int = 0,
        ) -> np.ndarray:
        """
        Binarize a numerical array based on a threshold.

        Parameters
        ----------
        rho : np.ndarray
            Numerical array representing density or similar data to be binarized.
        threshold_method : {'median', 'mean', 'custom'}, optional
            Method for determining the threshold:
            - 'median': Use median value as threshold (default).
            - 'mean': Use mean value as threshold.
            - 'custom': Use a user-defined value passed via `custom_threshold`.
        custom_threshold : Optional[float], optional
            Custom threshold value. Must be provided if `threshold_method='custom'`.
        high_value : int, optional
            Value assigned to elements greater than the threshold. Default is 1.
        low_value : int, optional
            Value assigned to elements less than or equal to the threshold. Default is 0.

        Returns
        -------
        np.ndarray
            Binarized array where values are mapped to `high_value` or `low_value`.

        Raises
        ------
        ValueError
            If `rho` is not a valid numerical array.
            If `threshold_method='custom'` and `custom_threshold` is not provided.
        """
        # Validate input array
        if not isinstance(rho, np.ndarray) or rho.size == 0:
            raise ValueError("Input 'rho' must be a non-empty NumPy array.")

        # Determine threshold method
        if threshold_method == "median":
            threshold = 0.5 * (rho.min() + rho.max())  # Median approximation
        elif threshold_method == "mean":
            threshold = rho.mean()
        elif threshold_method == "custom":
            if custom_threshold is None:
                raise ValueError("A custom threshold must be provided when `threshold_method='custom'`.")
            threshold = custom_threshold
        else:
            raise ValueError(f"Invalid `threshold_method`: {threshold_method}. Must be 'median', 'mean', or 'custom'.")

        # Binarize array
        return np.where(rho > threshold, high_value, low_value)


# ################################# #
# front tracking                    #
# ################################# #

class FrontTracking(MultiphaseProcessor):
    """
    Tracks the advancing front in multiphase flow simulations.

    This class is used to compute the location of the advancing front for a given phase 
    during a simulation, based on density distribution data. It can process datasets 
    across simulation time steps and save results such as front positions and binary representations.

    Parameters
    ----------
    data_path : PathLike
        Path to the directory containing simulation data files.
    domain_size : Sequence[int]
        Domain dimensions as (nx, ny, nz).
    trim : Sequence[int], optional
        Padding to trim from each side of the simulation domain, specified as
        [x_min, x_max, y_min, y_max, z_min, z_max]. Default is [0, 0, 0, 0, 0, 0].
    fluid_id : {'f1', 'f2'}, optional
        Identifier for the analyzed fluid phase. Default is 'f1' (phase 1).
    save_path : PathLike, optional
        Directory for saving processed outputs. If None, a subdirectory is created in `data_path`.
    flow_direction : {'x', 'y', 'z'}, optional
        Direction of fluid flow in the simulation domain. Default is 'x'.

    Attributes
    ----------
    flow_direction : str
        Direction of fluid flow ('x', 'y', or 'z').
    save_path : PathLike
        Directory where processed data is saved.
    domain_size : Sequence[int]
        Domain dimensions as (nx, ny, nz).
    trim : Sequence[int]
        Padding applied to the trimmed domain dimensions.
    rho_files : list
        List of file paths containing density data for the specified fluid phase.
    xx, yy, zz : ndarray
        Meshgrid arrays for spatial coordinates in the simulation domain.

    Methods
    -------
    find_front(flat_array: np.ndarray, num_elems: int = 3) -> float
        Estimates the location of the advancing front based on the highest density values.
    _output_front(front: float, step: int) -> None
        Saves the advancing front location to a CSV file.
    _output_binary(rho: np.ndarray, step: int) -> None
        Saves binary density data to a file for the current time step.
    __call__(time_steps: Optional[Union[int, Sequence]] = None, 
             outputs: Literal['front', 'binary'] = 'front', 
             num_elems: int = 1, stop: int = 2) -> str
        Executes the front-tracking analysis for specified simulation steps and outputs results.

    Raises
    ------
    ValueError
        If inputs such as `time_steps` or `flow_direction` are invalid.

    Example
    -------
    ```python
    ft = FrontTracking(
        data_path="/path/to/data", 
        domain_size=(100, 100, 100), 
        fluid_id="f1", 
        flow_direction="x"
    )
    ft(time_steps=[0, 10], outputs="front", num_elems=5, stop=5)
    ```
    """
    def __init__(
        self, 
        data_path: PathLike, 
        domain_size: Sequence[int], 
        trim: Sequence[int] = [0, 0, 0, 0, 0, 0], 
        fluid_id: str = 'f1', 
        save_path: Optional[PathLike] = None, 
        flow_direction: Literal['x', 'y', 'z'] = 'x'
    ):
        """
        Initialize the FrontTracking class with simulation parameters.

        Parameters
        ----------
        data_path : PathLike
            Path to the directory containing simulation data files.
        domain_size : Sequence[int]
            Domain dimensions specified as (nx, ny, nz).
        trim : Sequence[int], optional
            Padding to trim from the simulation domain as [x_min, x_max, y_min, y_max, z_min, z_max].
            Default is [0, 0, 0, 0, 0, 0].
        fluid_id : {'f1', 'f2'}, optional
            Identifier for the analyzed fluid phase. Default is 'f1'.
        save_path : PathLike, optional
            Directory for saving processed outputs. If None, a subdirectory is created in `data_path`.
        flow_direction : {'x', 'y', 'z'}, optional
            Direction of fluid flow in the simulation domain. Default is 'x'.

        Raises
        ------
        ValueError
            If `fluid_id` or `flow_direction` is invalid, or `data_path` does not exist.
        IOError
            If the save directory cannot be created or accessed.
        """
        # Validate flow direction
        if flow_direction not in {'x', 'y', 'z'}:
            raise ValueError(
                f"Invalid `flow_direction` '{flow_direction}'. Must be one of 'x', 'y', or 'z'."
            )
        
        # Call parent constructor for base attributes
        super().__init__(data_path, domain_size, trim, fluid_id, save_path)

        # Initialize flow direction
        self.flow_direction = flow_direction

        # Log initialization (optional)
        print(f"Initializing FrontTracking with flow direction: {self.flow_direction}")
        print(f"Data will be saved in: {self.save_path}")
        print(f"Domain size: {self.domain_size}, Trim values: {self.trim}")

    
    def find_front(
        self, 
        flat_array: np.ndarray, 
        num_elems: int = 3, 
        method: Literal['top', 'percentile'] = 'top', 
        percentile: Optional[float] = None, 
        ascending: bool = False
        ) -> float:
        """
        Estimate the advancing front's position in the specified direction.

        Parameters
        ----------
        flat_array : np.ndarray
            Flattened numerical array of positions (e.g., x-positions for the advancing front).
        num_elems : int, optional
            Number of elements to consider for estimating the front position (default: 3).
        method : {'top', 'percentile'}, optional
            Method to use for estimating the front:
            - 'top': Use the top `num_elems` values (default).
            - 'percentile': Use the specified percentile value.
        percentile : float, optional
            Percentile value to use if `method='percentile'`. Must be between 0 and 100.
        ascending : bool, optional
            Sort order for the values (default: False, descending order).

        Returns
        -------
        float
            Estimated front position.

        Raises
        ------
        ValueError
            If `flat_array` is invalid or empty, or if `percentile` is not between 0 and 100.
            If `num_elems` exceeds the size of `flat_array`.
        """
        if not isinstance(flat_array, np.ndarray) or flat_array.size == 0:
            raise ValueError("Invalid input: 'flat_array' must be a non-empty NumPy array.")

        if method == 'percentile':
            if percentile is None or not (0 <= percentile <= 100):
                raise ValueError("Percentile must be specified and between 0 and 100 when using method='percentile'.")

        if method == 'top':
            if num_elems > flat_array.size:
                raise ValueError(f"'num_elems' exceeds the size of 'flat_array' ({flat_array.size}).")
        
            partitioned = np.partition(flat_array, -num_elems if not ascending else num_elems - 1)
            top_values = partitioned[-num_elems:] if not ascending else partitioned[:num_elems]
            return np.mean(top_values)
        elif method == 'percentile':
            return np.percentile(flat_array, percentile)
        else:
            raise ValueError(f"Invalid method '{method}'. Must be 'top' or 'percentile'.")
    
    def _output_front(self, front: float, step: int) -> None:
        with open(path.join(self.save_path, 'front.dat'), 'a') as f:
            f.write(f"{step},{front}\n")
    
    def _output_binary(self, rho: np.ndarray, step: int) -> None:
        np.savetxt(path.join(self.save_path, f'rho_{step}.dat'), rho.flatten(),
                    fmt = '%d', delimiter = ' ',
                      header = f'{self.res_x} {self.res_y} {self.res_z}', 
                        comments = '#')
    
    def __call__(
        self,
        time_steps: Optional[Union[int, Sequence]] = None,
        outputs: Literal['front', 'binary', 'both'] = 'front',
        num_elems: int = 3,
        stop_criteria: Literal['distance', 'steps'] = 'steps',
        stop_value: Union[int, float] = 2,
    ) -> str:
        """
        Executes the front-tracking analysis across specified simulation steps.

        This method processes density files for the given time range, computes the advancing front position, 
        and optionally saves binary density data or front positions.

        Parameters
        ----------
        time_steps : Optional[Union[int, Sequence]]
            Range of simulation time steps to process. Can be:
            - None: Process all available steps.
            - int: Process a specific time step.
            - Sequence[int]: Process multiple steps (e.g., [start, end]).
        outputs : {'front', 'binary', 'both'}, optional
            Specify the output to generate:
            - 'front': Save front positions only.
            - 'binary': Save binary density data only.
            - 'both': Save both front positions and binary density data. Default is 'front'.
        num_elems : int, optional
            Number of elements to consider for approximating the advancing front. Default is 3.
        stop_criteria : {'distance', 'steps'}, optional
            Criteria for stopping the simulation:
            - 'distance': Stop when the front reaches a specific maximum distance in the flow direction.
            - 'steps': Stop after processing a fixed number of steps. Default is 'steps'.
        stop_value : Union[int, float], optional
            Value associated with the stop criteria:
            - If `stop_criteria='distance'`, this is the maximum allowed distance.
            - If `stop_criteria='steps'`, this is the maximum number of steps to process. Default is 2.

        Returns
        -------
        str
            Final location of the front and the time step at which it occurred.

        Raises
        ------
        ValueError
            If `time_steps` references invalid indices.
            If other parameters like `stop_criteria` or `outputs` are invalid.
        IOError
            If any density file is inaccessible.

        Example
        -------
            ```python
            ft = FrontTracking(
            data_path="/path/to/data", 
            domain_size=(100, 100, 100), 
            fluid_id="f1", 
            flow_direction="x"
        )
        result = ft(time_steps=[0, 100], outputs="front", num_elems=5, stop_criteria="distance", stop_value=90)
        print(result)
        ```
        """
        # Validate time_steps
        if time_steps is None:
            time_steps = range(len(self.rho_files))
        elif isinstance(time_steps, int):
            if time_steps < 0 or time_steps >= len(self.rho_files):
                raise ValueError(f"Invalid time step '{time_steps}'. Must be between 0 and {len(self.rho_files) - 1}.")
            time_steps = [time_steps]
        elif isinstance(time_steps, (list, tuple)) and len(time_steps) == 2:
            start, end = time_steps
            if start < 0 or end > len(self.rho_files) or start >= end:
                raise ValueError(
                    f"Invalid range for `time_steps`: [{start}, {end}]. Must be within [0, {len(self.rho_files)}] and start < end."
                )
            time_steps = range(start, end)

        # Validate outputs
        if outputs not in {'front', 'binary', 'both'}:
            raise ValueError(f"Invalid `outputs` value '{outputs}'. Must be one of 'front', 'binary', or 'both'.")

        # Validate stop criteria
        if stop_criteria not in {'distance', 'steps'}:
            raise ValueError(f"Invalid `stop_criteria` '{stop_criteria}'. Must be one of 'distance' or 'steps'.")
        if stop_criteria == 'distance' and not isinstance(stop_value, (int, float)):
            raise ValueError("`stop_value` must be a numerical distance value when `stop_criteria='distance'`.")
        if stop_criteria == 'steps' and not isinstance(stop_value, int):
            raise ValueError("`stop_value` must be an integer step count when `stop_criteria='steps'`.")

        step_count = 0

        output_dir = path.join(self.save_path, "front_tracking_results")
        makedirs(output_dir, exist_ok=True)
    
        # Loop through selected time steps
        for step in tqdm(time_steps, desc="Processing steps"):
            try:
            # Load density file and preprocess
                rho = tools.load_reshape_trim(self.rho_files[step], self.domain_size, self.trim)
                rho_bin = self.binarize(rho)
                rho_index = np.where(rho_bin == 1)

                # Determine front location in the specified flow direction
                if self.flow_direction == 'x':
                    u_flat = self.xx[rho_index].flatten()
                elif self.flow_direction == 'y':
                    u_flat = self.yy[rho_index].flatten()
                elif self.flow_direction == 'z':
                    u_flat = self.zz[rho_index].flatten()
                else:
                    raise ValueError(f"Invalid flow direction '{self.flow_direction}'.")

                front = self.find_front(u_flat, num_elems=num_elems)

                # Check stop criteria
                if stop_criteria == 'distance':
                    max_stop = {'x': self.res_x, 'y': self.res_y, 'z': self.res_z}[self.flow_direction]
                    if abs(front - max_stop) <= stop_value:
                        print(f"Stopping early: Front reached near maximum distance {front}.")
                        break
                elif stop_criteria == 'steps' and step_count >= stop_value:
                    print(f"Stopping early: Maximum step count reached ({step_count}).")
                    break

                # Save outputs
                if outputs in {'front', 'both'}:
                    front_file = path.join(output_dir, "front.dat")
                    with open(front_file, 'a') as f:
                        f.write(f"{step},{front:.5f}\n")
                if outputs in {'binary', 'both'}:
                    binary_file = path.join(output_dir, f"rho_binary_{step}.dat")
                    np.savetxt(
                        binary_file, rho_bin.flatten(), fmt='%d', delimiter=' ',
                        header=f"{self.res_x} {self.res_y} {self.res_z}", comments="#"
                    )
        
                step_count += 1

            except Exception as e:
                print(f"Error processing time step {step}: {str(e)}")
                continue

        return f"Final front location: {front:.5f} at step {step}"


# computing phase velocities          
class PhaseVelocities:
    """
    Processes and visualizes fluid velocities in multiphase simulations.

    This class includes tools to average velocities within specified slices, extract
    slice-specific velocities, and create flow visualizations through streamlines and quiver plots.

    Example
    -------
    ```python
    pv = PhaseVelocities(
        path_to_data="/path/to/data",
        domain_size=(100, 100, 100),
        trim=[2, 2, 2, 2, 1, 1]
    )
    fig, ax = pv.plot_flow(
        phase="f1",
        plot_type="streamlines",
        number=50,
        z_plane=25
    )
    ```
    """

    def __init__(
        self, 
        path_to_data: PathLike, 
        domain_size: Sequence[int], 
        save_path: Optional[PathLike] = None, 
        trim: Sequence[int] = [0, 0, 0, 0, 0, 0]
    ):
        """
        Initialize the PhaseVelocities class with simulation parameters.

        Parameters
        ----------
        path_to_data : PathLike
            Path to the directory containing phase velocity simulation files.
        domain_size : Sequence[int]
            Dimensions of the simulation domain as (nx, ny, nz).
        save_path : PathLike, optional
            Directory to save processed outputs. If None, a subdirectory named `velocity_processing`
            is created inside `path_to_data`. Default is None.
        trim : Sequence[int], optional
            Padding values to trim the simulation domain as [x_min, x_max, y_min, y_max, z_min, z_max].
            Default is [0, 0, 0, 0, 0, 0].

        Raises
        ------
        ValueError
            If `domain_size` or `trim` are invalid.
            If `path_to_data` does not exist or is not a valid directory.
        IOError
            If there is an issue while creating or writing to the save directory.
        """
        # Validate domain_size
        if not (isinstance(domain_size, (list, tuple)) and len(domain_size) == 3 and all(isinstance(n, int) and n > 0 for n in domain_size)):
            raise ValueError("'domain_size' must be a sequence of three positive integers (nx, ny, nz).")

        # Validate trim
        if not (isinstance(trim, (list, tuple)) and len(trim) == 6 and all(isinstance(n, int) and n >= 0 for n in trim)):
            raise ValueError("'trim' must be a sequence of six non-negative integers [x_min, x_max, y_min, y_max, z_min, z_max].")

        # Validate path_to_data
        if not path.exists(path_to_data) or not path.isdir(path_to_data):
            raise ValueError(f"'path_to_data' is not a valid directory: {path_to_data}")

        # Initialize attributes
        self.domain_size = domain_size
        self.trim = trim
        self.path_to_data = path_to_data

        # Ensure velocity files are referenced correctly
        self._list_files(path_to_data)

        # Save path configuration
        if save_path is None:
            save_path = path.join(path_to_data, 'velocity_processing')
        try:
            self.save_path = tools.make_dir(save_path)
        except Exception as e:
            raise IOError(f"Failed to create save directory '{save_path}'. Error: {str(e)}")

        # Compute meshgrid arrays for the trimmed domain
        self.res_x = domain_size[0] - (trim[0] + trim[1])
        self.res_y = domain_size[1] - (trim[2] + trim[3])
        self.res_z = domain_size[2] - (trim[4] + trim[5])
        if self.res_x <= 0 or self.res_y <= 0 or self.res_z <= 0:
            raise ValueError("Trim values result in an invalid domain size. Ensure trimming does not exceed domain dimensions.")

        self.xx, self.yy, self.zz = np.meshgrid(
            np.arange(self.res_x),
            np.arange(self.res_y),
            np.arange(self.res_z),
            indexing='ij'
        )

        # Define slice axes for easy manipulation
        self.x_line = np.arange(self.res_x)
        self.y_line = np.arange(self.res_y)
        self.z_line = np.arange(self.res_z)

        # Validate file structure and log file count
        self.num_files = len(self.f1_files['rho'])  # Example usage of validated files
        print(f"Initialized PhaseVelocities with {self.num_files} velocity distribution files.")

    
    def method_counter(method):
        """
        A wrapper to count how many times a method is called.

        Parameters
        ----------
        method : callable
            The function or method to wrap.

        Returns
        -------
        callable
            The wrapped method with added functionality to count calls.

        Notes
        -----
        - Tracks call count at the instance level rather than globally.
        - Thread-safe operation ensures correct counting in multithreaded environments.
        - Counter can be accessed or reset via the `method_name.counter` attribute.
        """
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            with threading.Lock():  # Ensure thread-safe updates
                wrapper.counter += 1
            print(f"[DEBUG] {method.__name__} was called {wrapper.counter} times for {type(self).__name__}.")
            return method(self, *args, **kwargs)

        # Initialize counter
        wrapper.counter = 0
        def reset_counter():
            with threading.Lock():
                wrapper.counter = 0
        wrapper.reset_counter = reset_counter
        return wrapper

    @method_counter
    def average_velocity(self, planes: Dict[str, int], times: List[int]) -> None:
        """
        Compute and save average velocities for specified slices and times.

        Parameters
        ----------
        planes : dict
            Dictionary specifying slices to process (direction -> coordinate).
            Example: {'x': 10, 'y': 20}.
        times : list of int
            List of time steps to process.

        Returns
        -------
        None
            Saves the calculated average velocities to CSV files.
        """
        if not planes:
            raise ValueError("Invalid input: 'planes' cannot be empty.")
        if not times:
            raise ValueError("Invalid input: 'times' cannot be empty.")
    
        for time in times:
            for direction, coordinate in planes.items():
                print(f"[DEBUG] Processing time={time}, direction={direction}, coordinate={coordinate}.")

                # Extract slice
                slice_indices = self._get_slice(direction=direction, coordinate=coordinate)
                xx_c = np.squeeze(self.xx[slice_indices])
                yy_c = np.squeeze(self.yy[slice_indices])
                zz_c = np.squeeze(self.zz[slice_indices])

                # Get velocity components
                v1_slice = partial(self._get_slice_value, comp='f1', time=time)
                v2_slice = partial(self._get_slice_value, comp='f2', time=time)

                vx1 = v1_slice(key='vx', slice=slice_indices)
                vy1 = v1_slice(key='vy', slice=slice_indices)
                vz1 = v1_slice(key='vz', slice=slice_indices)

                vx2 = v2_slice(key='vx', slice=slice_indices)
                vy2 = v2_slice(key='vy', slice=slice_indices)
                vz2 = v2_slice(key='vz', slice=slice_indices)

                # Compute magnitude
                v1_mag = np.sqrt(vx1**2 + vy1**2 + vz1**2)
                v2_mag = np.sqrt(vx2**2 + vy2**2 + vz2**2)

                # Combine results
                slice_velocities = np.c_[
                    xx_c.flatten(), yy_c.flatten(), zz_c.flatten(),
                    vx1.flatten(), vy1.flatten(), vz1.flatten(),
                    vx2.flatten(), vy2.flatten(), vz2.flatten(),
                    v1_mag.flatten(), v2_mag.flatten()
                ]

                # Save to CSV
                slice_df = pd.DataFrame(
                    slice_velocities, 
                    columns=['xx', 'yy', 'zz', 'vx1', 'vy1', 'vz1', 'vx2', 'vy2', 'vz2', 'vmag1', 'vmag2']
                )
                output_dir = path.join(self.save_path, f"{direction}_slice_results")
                makedirs(output_dir, exist_ok=True)
                file_name = path.join(output_dir, f"velocity_slice_{coordinate}_time_{time}.csv")
                slice_df.to_csv(file_name, header=True, index=False, sep=',')


    def _list_files(self, path_to_data: PathLike) -> None:
        """
        Discover and organize velocity-related data files from the specified directory.

        Parameters
        ----------
        path_to_data : PathLike
            Path to the directory containing velocity data files.

        Returns
        -------
        None
            Updates `self.f1_files` and `self.f2_files` attributes with discovered file paths.

        Raises
        ------
        ValueError
            If required files are missing or naming conventions do not match expectations.
        """
        def validate_and_get_files(data_path: PathLike, stems: List[str]) -> Dict[str, List[str]]:
            """
            Discover files for each stem and return a dictionary containing file paths.

            Parameters
            ----------
            data_path : PathLike
                Path to the directory to search for files.
            stems : List[str]
                List of naming stems to search for files.

            Returns
            -------
            Dict[str, List[str]]
                A dictionary where keys are stems and values are lists of corresponding file paths.
        
            Raises
            ------
            ValueError
                If no files matching a given stem are found in the directory.
            """
            files_dict = {}
            for stem in stems:
                files = get_files(data_path, stem)
                if not files:
                    raise ValueError(f"No files matching stem '{stem}' were found in directory: {data_path}")
                files_dict[stem] = files
            return files_dict

        # Define file stems for velocity components and density data
        f1_velocity_stems = [f'f1_v{ax}_step_' for ax in 'xyz']
        f2_velocity_stems = [f'f2_v{ax}_step_' for ax in 'xyz']
        density_stems = ["f1_rho_dist__", "f2_rho_dist__"]

        try:
            self.f1_files = validate_and_get_files(path_to_data, f1_velocity_stems + [density_stems[0]])
            self.f2_files = validate_and_get_files(path_to_data, f2_velocity_stems + [density_stems[1]])
        except ValueError as e:
            print(f"[ERROR] File discovery failed: {str(e)}")
            raise
    
    def _get_slice(self, direction = None, coordinate = None):
        """
        Wrapper for the module-level `get_slice` function in utils.tool for compatibility with class attributes.
        """
        return tools.get_slice(self.domain_size, direction, coordinate)

    
    def get_slice_arrays(
        self, 
        comp: Literal['f1', 'f2'], 
        direction: Literal['x', 'y', 'z'], 
        coordinate: int, 
        time: int
    ) -> Dict[str, np.ndarray]:
        """
        Retrieve velocity components (`vx`, `vy`, `vz`) for a specific slice, direction, and time step.

        Parameters
        ----------
        comp : {'f1', 'f2'}
            Phase (fluid component) whose velocity data is being retrieved.
        direction : {'x', 'y', 'z'}
            Direction of the slice (`x`, `y`, or `z`).
        coordinate : int
            Coordinate along the given direction to slice the domain.
        time : int
            Time step from which velocity data is retrieved.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary with keys as velocity components ('vx', 'vy', 'vz') and values as NumPy arrays 
            corresponding to the requested slice.

        Raises
        ------
        ValueError
            If `direction`, `coordinate`, `comp`, or `time` is invalid.
        """
        if comp not in {'f1', 'f2'}:
            raise ValueError(f"Invalid component '{comp}'. Must be 'f1' or 'f2'.")
        if direction not in {'x', 'y', 'z'}:
            raise ValueError(f"Invalid direction '{direction}'. Must be 'x', 'y', or 'z'.")
        if coordinate < 0 or coordinate >= {'x': self.res_x, 'y': self.res_y, 'z': self.res_z}[direction]:
            raise ValueError(
                f"Invalid coordinate '{coordinate}' for direction '{direction}'. "
                f"Valid range: 0 <= coordinate < {self.domain_size['xyz'.index(direction)]}."
            )
        if time < 0 or time >= len(self.f1_files['vx']):  # Assume file lengths are consistent
            raise ValueError(f"Invalid time step '{time}'. Must be between 0 and {len(self.f1_files['vx']) - 1}.")

        print(f"[INFO] Retrieving velocity components for direction '{direction}', coordinate {coordinate}, and time step {time}.")

        slice_indices = tools.get_slice(self.domain_size, direction, coordinate)

        velocity_components = ['vx', 'vy', 'vz']

        slice_data = {
            v_comp: tools.load_reshape_trim(self.f1_files[v_comp][time] if comp == 'f1' else self.f2_files[v_comp][time], self.domain_size, trim=self.trim)[slice_indices]
            for v_comp in velocity_components
        }
        return slice_data
        

    def plot_flow(
        self, 
        plot_type: Literal['streamlines', 'quiver'] = 'streamlines',
        phase: Literal['f1', 'f2'] = 'f1',
        number: int = None,
        z_plane: int = None,
        y_start_ranges: Optional[List[int]] = None,
        start_density: int = 5,
        x_start: float = 0.1,
        density: float = 20,
        overlay_phase: Optional[Literal['f1', 'f2']] = None,
        figsize: Tuple[int, int] = (12, 6),
        colormap: str = 'jet'
    ) -> Tuple:
        """
        Visualize velocity patterns of a selected phase using streamlines or quiver plots.

        Parameters
        ----------
        plot_type : {'streamlines', 'quiver'}, optional
            Type of plot to generate:
            - 'streamlines': Flow visualization using streamlines (default).
            - 'quiver': Flow visualization using arrows (quiver plot).
        phase : {'f1', 'f2'}, optional
            Fluid phase whose velocity data will be visualized. Default is 'f1'.
        number : int, optional
            Time step index. Required for fetching velocity data.
        z_plane : int, optional
            Plane in the depth direction (`z`) to visualize. Must be valid within the domain dimensions.
        y_start_ranges : list of int, optional
            Start ranges for streamlines in the `y` direction, specified as `[y1_min, y1_max, y2_min, y2_max, ...]`.
            For multi-group streamline generation.
        start_density : int, optional
            Density of streamline starting points within each range. Default is 5.
        x_start : float, optional
            Starting point in the `x` direction for streamlines. Default is 0.1.
        density : float, optional
            Density of plotted streamlines or quiver arrows. Default is 20.
        overlay_phase : {'f1', 'f2'}, optional
            Phase density overlay for regions affected by the fluid phase. Default is None (no overlay).
        figsize : tuple of int, optional
            Figure size for plotting. Default is (12, 6).
        colormap : str, optional
            Colormap to use for the visualization. Default is 'jet'.

        Returns
        -------
        Tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
            The generated figure and axes objects for the plot.

        Raises
        ------
        ValueError
            If required inputs like `number`, `z_plane` are invalid or outside allowed ranges.
        IOError
            If velocity files for specified `phase` and `number` are inaccessible.

        Example
        -------
        ```python
        fig, ax = pv.plot_flow(
            plot_type='streamlines',
            phase='f1',
            number=10,
            z_plane=50,
            y_start_ranges=[20, 40, 60, 80],
            overlay_phase='f2',
            density=25
        )
        ```
        """
        # Validation
        if plot_type not in {'streamlines', 'quiver'}:
            raise ValueError(f"Invalid plot type '{plot_type}'. Must be 'streamlines' or 'quiver'.")
        if phase not in {'f1', 'f2'}:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'f1' or 'f2'.")
        if number is None or number < 0 or number >= len(self.f1_files['vx']):
            raise ValueError(f"Invalid time step '{number}'. Must be between 0 and {len(self.f1_files['vx']) - 1}.")
        if z_plane is None or z_plane < 0 or z_plane >= self.domain_size[2]:
            raise ValueError(f"Invalid z-plane '{z_plane}'. Must be between 0 and {self.domain_size[2] - 1}.")

        # Log the action
        print(f"[INFO] Plotting flow for phase '{phase}', time step {number}, z-plane {z_plane}.")

        # Prepare velocity data for the selected phase
        try:
            vx = np.loadtxt(self.f1_files['vx'][number] if phase == 'f1' else self.f2_files['vx'][number]).reshape(self.domain_size)
            vy = np.loadtxt(self.f1_files['vy'][number] if phase == 'f1' else self.f2_files['vy'][number]).reshape(self.domain_size)
            vz = np.loadtxt(self.f1_files['vz'][number] if phase == 'f1' else self.f2_files['vz'][number]).reshape(self.domain_size)

            vx = np.squeeze(vx[:, :, z_plane])
            vy = np.squeeze(vy[:, :, z_plane])
            vz = np.squeeze(vz[:, :, z_plane])  # In case `z_plane` is selected
            magnitude = np.sqrt(vx**2 + vy**2 + vz**2)  # Magnitude of velocity
        except Exception as e:
            raise IOError(f"Failed to load velocity data for phase '{phase}', time step {number}. Error: {str(e)}")

        # Overlay density data (optional)
        if overlay_phase:
            try:
                overlay_density = np.loadtxt(self.f1_files['rho'][number] if overlay_phase == 'f1' else self.f2_files['rho'][number]).reshape(self.domain_size)
                overlay_density = np.squeeze(overlay_density[:, :, z_plane])
            except Exception as e:
                raise IOError(f"Failed to load overlay density data for phase '{overlay_phase}', time step {number}. Error: {str(e)}")
        else:
            overlay_density = None

        # Prepare figure
        plt.rcParams.update({'font.size': 20, 'font.family': 'Times New Roman'})
        fig, ax = plt.subplots(figsize=figsize)

        # Generate plots
        if plot_type == 'streamlines':
            # Configure starting points for streamlines
            if y_start_ranges is not None:
                start_points = []
                for i in range(len(y_start_ranges) // 2):
                    y_min, y_max = y_start_ranges[2 * i], y_start_ranges[2 * i + 1]
                    y_starts = np.linspace(y_min, y_max, start_density)
                    start_points.append(np.c_[np.full_like(y_starts, x_start), y_starts])
                start_points = np.vstack(start_points)

            # Plot streamlines
            ax.streamplot(
                self.x_line, self.y_line, vx.T, vy.T, density=density, color='w',
                start_points=start_points if y_start_ranges is not None else None
            )
            img = ax.contourf(self.xx[:, :, z_plane], self.yy[:, :, z_plane], magnitude, levels=50, cmap=colormap)
        elif plot_type == 'quiver':
            # Quiver plot
            ax.quiver(self.xx[:, :, z_plane], self.yy[:, :, z_plane], vx, vy, scale=1.5, headwidth=3, headlength=5)
            img = ax.contourf(self.xx[:, :, z_plane], self.yy[:, :, z_plane], magnitude, levels=50, cmap=colormap)

        # Format axes
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title(f"Velocity Flow (Phase '{phase}', z-plane {z_plane})")
        cbar = fig.colorbar(img, ax=ax, orientation="vertical")
        cbar.set_label("Velocity Magnitude")

        return fig, ax

    








            




    
