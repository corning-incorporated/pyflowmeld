
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
from os import path, makedirs, PathLike
from tqdm import tqdm 
from datetime import datetime  
from functools import partial 
from .. utils import tools 
from typing import Sequence, Optional, Tuple, List, Union, Literal, Dict  


def read_phases(f1_file: PathLike,
                f2_file: PathLike, 
                domain_size: Sequence, 
                trim: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads and processes density files for two phases.

    Parameters:
    -----------
    f1_file : PathLike
        Path to the file representing phase 1 density distribution.
    f2_file : PathLike
        Path to the file representing phase 2 density distribution.
    domain_size : Sequence
        Shape of the domain, typically as [x_size, y_size, z_size], used for reshaping.
    trim : Sequence
        Padding/trimming values [x0, x1, y0, y1, z0, z1], specifying the regions to be trimmed.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing reshaped and trimmed density arrays for both phases.

    Raises:
    -------
    FileNotFoundError:
        If `f1_file` or `f2_file` does not exist.
    ValueError:
        If `domain_size` or `trim` is malformed or incompatible.
    """
    if not path.exists(f1_file):
        raise FileNotFoundError(f"File not found: {f1_file}")
    if not path.exists(f2_file):
        raise FileNotFoundError(f"File not found: {f2_file}")
    
    if len(domain_size) != 3:
        raise ValueError("domain_size must contain exactly three dimensions: [x_size, y_size, z_size].")
    if len(trim) != 6:
        raise ValueError("trim must contain six values: [x0, x1, y0, y1, z0, z1].")
    
    try:
        rho_f1 = tools.load_reshape_trim(f1_file, domain_size, trim)
        rho_f2 = tools.load_reshape_trim(f2_file, domain_size, trim)
    except Exception as e:
        raise ValueError(f"Error processing files: {e}")
    
    return rho_f1, rho_f2

def phase_fraction(f1_file: PathLike, 
                   f2_file: PathLike,
                   domain_size: Optional[Sequence] = None,
                   trim: Optional[Sequence] = [0]*6) -> np.ndarray:
    """
    Calculates the fraction of phase 1 relative to the total density distribution.

    Parameters:
    -----------
    f1_file : PathLike
        Path to the file representing phase 1 density distribution.
    f2_file : PathLike
        Path to the file representing phase 2 density distribution.
    domain_size : Optional[Sequence]
        Shape of the domain, typically as [x_size, y_size, z_size], used for reshaping.
        If None, domain_size must be handled by the `tools.load_reshape_trim` utility.
    trim : Optional[Sequence]
        Padding/trimming values [x0, x1, y0, y1, z0, z1], specifying the regions to be trimmed.

    Returns:
    --------
    np.ndarray
        The computed phase fraction array: phase 1 density divided by total density.

    Raises:
    -------
    FileNotFoundError:
        If `f1_file` or `f2_file` does not exist.
    ValueError:
        If densities cannot be computed due to division by zero or invalid input.
    """
    if not path.exists(f1_file):
        raise FileNotFoundError(f"File not found: {f1_file}")
    if not path.exists(f2_file):
        raise FileNotFoundError(f"File not found: {f2_file}")
    
    if domain_size and len(domain_size) != 3:
        raise ValueError("domain_size must contain exactly three dimensions: [x_size, y_size, z_size].")
    if trim and len(trim) != 6:
        raise ValueError("trim must contain six values: [x0, x1, y0, y1, z0, z1].")
    
    try:
        rho_f1, rho_f2 = read_phases(f1_file, f2_file, domain_size, trim)
    except Exception as e:
        raise ValueError(f"Error reading phase densities: {e}")
    
    try:
        total_density = rho_f1 + rho_f2
        if np.any(total_density == 0):
            raise ValueError("Division by zero encountered when calculating total density.")
        
        phase_fraction_result = tools.true_division(rho_f1, total_density)
    except Exception as e:
        raise ValueError(f"Error computing phase fraction: {e}")
    
    return phase_fraction_result

def compute_drying_phase(rho_fraction: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Computes the binary drying phase map based on a threshold.

    Parameters:
    -----------
    rho_fraction : np.ndarray
        Fraction of phase density calculated as Y = rho_f1 / (rho_f1 + rho_f2).
        Expected to be a 3D array.
    threshold : float, optional (default=0.1)
        Fractional threshold for determining the drying domain. Must be in the range [0, 1].

    Returns:
    --------
    np.ndarray
        Binary array representing the drying domain: 1 for values above the threshold, 0 otherwise.

    Raises:
    -------
    ValueError:
        If `rho_fraction` is empty or if `threshold` is invalid (not in the range [0, 1]).
    """
    if not isinstance(rho_fraction, np.ndarray) or rho_fraction.size == 0:
        raise ValueError("rho_fraction must be a non-empty NumPy array.")
    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be a float between 0 and 1.")
    
    rho_max = rho_fraction.max()
    normalized_threshold = rho_max * (1 - threshold)
    drying_domain = (rho_fraction > normalized_threshold).astype(np.uint8)

    return drying_domain

def generate_coordinate_arrays(domain_size: Sequence, padding: Optional[Sequence] = None) -> Tuple[np.ndarray, ...]:
    """
    Generates coordinate arrays for plotting or other analyses.

    Parameters:
    -----------
    domain_size : Sequence
        Shape of the domain, typically as [x_size, y_size, z_size], specifying spatial dimensions.
    padding : Sequence, optional (default=[0, 0, 0, 0, 0, 0])
        Padding or trimming values [x0, x1, y0, y1, z0, z1], specifying regions to exclude.
        Must contain six non-negative integers.

    Returns:
    --------
    Tuple[np.ndarray, ...]
        Coordinate arrays (`res_x`, `res_y`, `res_z`, `xx`, `yy`, `zz`) where:
        - `res_x`, `res_y`, `res_z`: sizes of the domain after applying padding.
        - `xx`, `yy`, `zz`: coordinate grids for spatial dimensions.

    Raises:
    -------
    ValueError:
        If `domain_size` or `padding` is invalid (wrong length or contains invalid values).
    """
    if len(domain_size) != 3:
        raise ValueError("domain_size must contain exactly three positive integers: [x_size, y_size, z_size].")
    if padding and len(padding) != 6 :
        padding = [0]*6
        print("0 padding will be applied for coordinate array generation")
        
    res_x = max(0, domain_size[0] - (padding[0] + padding[1]))
    res_y = max(0, domain_size[1] - (padding[2] + padding[3]))
    res_z = max(0, domain_size[2] - (padding[4] + padding[5]))

    if res_x == 0 or res_y == 0 or res_z == 0:
        raise ValueError("Padding exceeds domain size, resulting in zero or negative dimensions.")

    xx, yy, zz = np.meshgrid(np.arange(0, res_x),
                              np.arange(0, res_y),
                              np.arange(0, res_z), 
                              indexing="ij")

    return res_x, res_y, res_z, xx, yy, zz


# ################################################################### #
# a separate drying class for sequential processing of multiple files #
# ################################################################### #
class Drying:
    """
    A class for post-processing drying phenomena simulations.

    This class reads density files for two phases, calculates drying-related metrics, 
    and outputs results such as receding contact lines, drying rates, or phase maps. 
    It is equipped to handle sequential processing of multiple files over time steps.

    Attributes:
    -----------
    path_to_data : PathLike
        Path to the directory containing the density files.
    padding : Sequence
        Padding/trimming values [x0, x1, y0, y1, z0, z1], specifying regions to exclude.
    domain_size : Sequence
        Shape of the computational domain as [x_size, y_size, z_size].
    f1_rho_files : List[str]
        List of file paths for phase 1 density files matching the provided stem.
    f2_rho_files : List[str]
        List of file paths for phase 2 density files matching the provided stem.
    save_path : PathLike
        Directory where post-processing results will be saved.
    threshold_tol : float
        Fractional threshold used for characterizing the drying domain.
    thresh_index : Optional[int]
        Internal index for threshold tracking, initialized as None.
    res_x, res_y, res_z : int
        Dimensions of the domain after applying padding.
    xx, yy, zz : np.ndarray
        Coordinate arrays for x-, y-, and z-dimensions.
    void_volume : Optional[int]
        Placeholder attribute; can be used for future volume-related computations.
    _keep_going : bool
        Internal flag controlling the flow of sequential computation.

    Methods:
    --------
    _get_slice(direction, coordinate):
        Private method to retrieve a specific slice from the domain.
    _get_slice_df(direction, coordinate, rho_array, column_name):
        Private method to generate a pandas DataFrame for a given slice.
    _generate_drying_rate(rho_fraction, count):
        Private method to compute and save drying rates to a file.
    _generate_slice_phase_map(rho_fraction, count, slices):
        Private method to generate and save phase maps for specific slices.
    _validate_time_range(time_range):
        Validates the provided time range and returns a tuple of start and end indices.
    __call__(time_range, slices):
        Main method to execute the drying simulation post-processing workflow.
    """
    def __init__(self, path_to_data: PathLike, 
                 padding: Sequence, 
                 domain_size: Sequence, 
                 f1_rho_stem: Literal['f1_rho_dist'] = "f1_rho_dist",
                 f2_rho_stem: Literal['f2_rho_dist'] = "f2_rho_dist",
                 save_path: Optional[PathLike] = None, 
                 threshold_tol: float = 0.1):
        """
        Initializes the Drying class with paths, domain settings, file stems, and optional parameters.

        Parameters:
        -----------
        path_to_data : PathLike
            Path to the directory containing density files.
        padding : Sequence
            Padding/trimming values [x0, x1, y0, y1, z0, z1], specifying regions to exclude.
        domain_size : Sequence
            Shape of the computational domain as [x_size, y_size, z_size].
        f1_rho_stem : Literal['f1_rho_dist'], optional (default="f1_rho_dist")
            Stem for phase 1 density files.
        f2_rho_stem : Literal['f2_rho_dist'], optional (default="f2_rho_dist")
            Stem for phase 2 density files.
        save_path : Optional[PathLike], optional (default=None)
            Directory where results will be saved. If None, a default directory is created based on the current date-time.
        threshold_tol : float, optional (default=0.1)
            Fractional threshold used for identifying the drying domain.

        Raises:
        -------
        ValueError:
            If domain_size or padding is invalid (wrong length or contains invalid values).
        """
        if len(domain_size) != 3 or not all(isinstance(dim, int) and dim > 0 for dim in domain_size):
            raise ValueError("domain_size must contain exactly three positive integers: [x_size, y_size, z_size].")
        if len(padding) != 6 or not all(isinstance(pad, int) and pad >= 0 for pad in padding):
            raise ValueError("padding must contain six non-negative integers: [x0, x1, y0, y1, z0, z1].")
        if threshold_tol <= 0 or threshold_tol > 1:
            raise ValueError("threshold_tol must be a float between 0 and 1.")

        self.f1_rho_files = tools.list_files(path_to_data, f1_rho_stem)
        self.f2_rho_files = tools.list_files(path_to_data, f2_rho_stem)
        self.path_to_data = path_to_data
        self.domain_size = domain_size
        self.padding = padding
        self.threshold_tol = threshold_tol
        self.thresh_index = None
        self._keep_going = True

        self.res_x, self.res_y, self.res_z, self.xx, self.yy, self.zz = generate_coordinate_arrays(domain_size, padding)

        if save_path is None:
            save_path = path.join(path_to_data, 'postprocess_on_' + datetime.today().strftime('%Y-%m-%d-%H-%M'))
            if not path.exists(save_path):
                makedirs(save_path)
        self.save_path = save_path

        self.void_volume = None
       
    def _get_slice(self, direction: Literal['x', 'y', 'z'], coordinate: int) -> Sequence:
        """ wrapper for tools.get_slice """
        return tools.get_slice(self.domain_size, direction, coordinate) 

    def _get_slice_df(self, direction: Literal['x', 'y', 'z'], 
                      coordinate: int, 
                      rho_array: np.ndarray, 
                      column_name: str = 'rho') -> pd.DataFrame:
        """
        Generates a pandas DataFrame for a specific spatial slice of the density array.

        Parameters:
        -----------
        direction : Literal['x', 'y', 'z']
            Direction of the slice (x, y, or z).
        coordinate : int
            Coordinate index for the slice in the specified direction.
        rho_array : np.ndarray
            Density array from which the slice will be extracted.
        column_name : str, optional (default='rho')
            Name of the density column in the resulting dataframe.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing spatial coordinates and density values for the selected slice.
            Includes `x`, `y`, `z`, and the specified column_name, filtered to exclude zero density.

        Raises:
        -------
        ValueError:
            If `direction` is not one of 'x', 'y', 'z' or if `coordinate` exceeds the bounds of the domain.
        """
        if direction not in ['x', 'y', 'z']:
            raise ValueError("Invalid direction. Must be one of 'x', 'y', 'z'.")
        if not (0 <= coordinate < self.res_x if direction == 'x' else
                0 <= coordinate < self.res_y if direction == 'y' else
                0 <= coordinate < self.res_z):
            raise ValueError(f"Coordinate {coordinate} exceeds valid bounds for direction '{direction}'.")

        slice_index = self._get_slice(direction, coordinate)

        xx, yy, zz = self.xx[slice_index], self.yy[slice_index], self.zz[slice_index]
        rho = rho_array[slice_index]

        valid_mask = rho != 0.0
        slice_df = pd.DataFrame({
            'x': xx.flatten()[valid_mask],
            'y': yy.flatten()[valid_mask],
            'z': zz.flatten()[valid_mask],
            column_name: rho.flatten()[valid_mask]
        })

        return slice_df

    def _generate_drying_rate(self, rho_fraction: np.ndarray, count: int) -> None:
        """
        Computes the drying rate and saves it to a CSV file.

        This method generates a binary map of the drying domain based on the threshold,
        calculates the phase volume and volume fraction, and appends the results
        into a CSV file.

        Parameters:
        -----------
        rho_fraction : np.ndarray
            Array representing the liquid fraction calculated from densities.
        count : int
            The current cycle count or time step index.

        Raises:
        -------
        ValueError:
            If `rho_fraction` contains invalid or empty values.
        """
        if not isinstance(rho_fraction, np.ndarray) or rho_fraction.size == 0:
            raise ValueError("rho_fraction must be a non-empty NumPy array.")
        
        drying_domain = compute_drying_phase(rho_fraction, self.threshold_tol)

        phase_volume = np.sum(drying_domain)
        total_volume = np.sum(np.where(rho_fraction > 0, 1, 0))

        if total_volume == 0:
            raise ValueError("Total liquid volume is zero; cannot compute volume fraction.")
        
        volume_fraction = phase_volume / total_volume
        print(f"Volume fraction for cycle {count}: {volume_fraction:.5f}")

        file_name = path.join(self.save_path, 'drying_rate.csv')

        if not path.exists(file_name):
            with open(file_name, 'w') as _file:
                _file.write("cycle,volume_fraction\n")
                _file.write(f"{count},{volume_fraction:.5f}\n")
        else:
            with open(file_name, 'a') as _file:
                _file.write(f"{count},{volume_fraction:.5f}\n")

        if phase_volume == 0:
            self._keep_going = False
            print(f"Evaporation is complete at cycle {count}. Stopping further computation.")
    
    # ###################################### #
    # generating phase map
    def _generate_slice_phase_map(self, rho_fraction: np.ndarray,
                                  count: int,
                                  slices: Union[Dict, Literal['xy', 'xz', 'yz']]) -> None:
        """
        Generates phase maps for specific spatial slices or averaged dimensions and saves them in CSV format.

        Parameters:
        -----------
        rho_fraction : np.ndarray
            Array representing the liquid fraction calculated from densities.
        count : int
            The current cycle count or time step index.
        slices : Union[Dict, Literal['xy', 'xz', 'yz']]
            Specifies the slices to process:
            - If a string ('xy', 'xz', 'yz'), averages the liquid fraction along the specified axis.
            - If a dictionary, specifies directions ('x', 'y', 'z') as keys and coordinates as values.

        Raises:
        -------
        ValueError:
            If `rho_fraction` is not a valid NumPy array or `slices` is invalid.
        """
        if not isinstance(rho_fraction, np.ndarray) or rho_fraction.size == 0:
            raise ValueError("rho_fraction must be a non-empty NumPy array.")

        if isinstance(slices, str):
            if slices not in ['xy', 'xz', 'yz']:
                raise ValueError("Invalid slice string. Must be one of 'xy', 'xz', 'yz'.")

            avg_axis = {'xy': 2, 'xz': 1, 'yz': 0}[slices]
            rho_mean = np.mean(rho_fraction, axis=avg_axis)

            res_u, res_v = rho_mean.shape
            uu, vv = np.meshgrid(np.arange(0, res_u), 
                                 np.arange(0, res_v), 
                                 indexing='ij')

            out_df = pd.DataFrame({
                'u': uu.flatten(),
                'v': vv.flatten(),
                'rho': rho_mean.flatten()
            })
            file_name = f'average_liquid_fraction_on_{slices}_cycle_{count}.csv'
            out_df.to_csv(path.join(self.save_path, file_name), 
                          sep=',', header=True, index=False, float_format='%.5f')

        elif isinstance(slices, dict):
            for direction, coordinates in slices.items():
                if direction not in ['x', 'y', 'z']:
                    raise ValueError(f"Invalid slice direction '{direction}'. Must be 'x', 'y', or 'z'.")

                if isinstance(coordinates, int):
                    coordinates = [coordinates]
                elif not all(isinstance(coord, int) for coord in coordinates):
                    raise ValueError("Coordinates must be integers.")

                for coord in coordinates:
                    slice_df = self._get_slice_df(direction, coord, rho_fraction, column_name='liquid_fraction')

                    file_name = f'liquid_fraction_{direction}_{coord}_cycle_{count}.csv'
                    slice_df.to_csv(path.join(self.save_path, file_name), 
                                    sep=',', header=True, index=False, float_format='%.5f')

        else:
            raise ValueError("Invalid `slices` parameter. Use a string ('xy', 'xz', 'yz') or a dictionary.")

    def _validate_time_range(self, time_range: Optional[Union[List, Tuple, int, str]] = None) -> Tuple[int, int]:
        """
        Validates and determines the range of time steps for processing density files.

        Parameters:
        -----------
        time_range : Optional[Union[List, Tuple, int, str]], optional (default=None)
            Specifies the range of time steps:
            - List or Tuple: Start and end indices as [min_time, max_time].
            - int: Specific time step (used as max_time with min_time = time_step - 1).
            - str: "last" to indicate the last available time step.
            - None: Processes all available time steps.

        Returns:
        --------
        Tuple[int, int]
            Validated time range as (min_time, max_time).

        Raises:
        -------
        ValueError:
            If `time_range` is invalid or exceeds available file indices.
        """
        total_files = len(self.f1_rho_files)
        if total_files == 0:
            raise ValueError("No files available in the directory to process.")

        if time_range is None:
            return 0, total_files  

        if isinstance(time_range, (list, tuple)):
            if len(time_range) != 2 or not all(isinstance(i, int) for i in time_range):
                raise ValueError("time_range list or tuple must contain exactly two integer indices: [min_time, max_time].")
            min_time, max_time = time_range
            if min_time < 0 or max_time > total_files or min_time >= max_time:
                raise ValueError("time_range indices must be valid and within file bounds.")
        
        elif isinstance(time_range, int):
            if time_range < 1 or time_range > total_files:
                raise ValueError(f"Invalid time_range. Time step must be between 1 and {total_files}.")
            min_time = time_range - 1
            max_time = time_range

        elif isinstance(time_range, str) and time_range.lower() == 'last':
            if total_files < 1:
                raise ValueError("No time steps available to process 'last'.")
            min_time = total_files - 1
            max_time = total_files

        else:
            raise ValueError("Invalid time_range. Expected None, int, list, tuple, or 'last'.")

        return min_time, max_time
    
    def __call__(self, time_range: Optional[Union[List, Tuple, int, str]] = None,
                 slices: Optional[Union[Dict, Literal['xy', 'yz', 'xz']]] = None) -> None:
        """
        Executes the drying post-processing workflow for the specified time range and slices.

        Parameters:
        -----------
        time_range : Optional[Union[List, Tuple, int, str]], optional (default=None)
            Specifies the range of time steps to process (same as `_validate_time_range`).
            - List or Tuple: Start and end indices as [min_time, max_time].
            - int: Specific time step (only processes that single step).
            - str: "last" to process the last available time step.
            - None: Processes all available time steps.

        slices : Optional[Union[Dict, Literal['xy', 'yz', 'xz']]], optional (default=None)
            Specifies the slices to generate phase maps for:
            - If a string ('xy', 'xz', 'yz'), averages the liquid fraction along the specified axis.
            - If a dictionary, keys specify directions ('x', 'y', 'z') and values specify coordinate indices.

        Raises:
        -------
        ValueError:
            If `time_range` or `slices` is invalid.
        RuntimeError:
            If no valid time steps are found for processing.
        """
        try:
            time_min, time_max = self._validate_time_range(time_range)
        except ValueError as e:
            raise ValueError(f"Invalid `time_range`: {e}")
        
        if time_max <= time_min or time_min < 0 or time_max > len(self.f1_rho_files):
            raise RuntimeError(f"No valid time steps found in the range {time_min}-{time_max}.")

        print(f"Processing time steps from {time_min} to {time_max - 1}.")
        
        _phase_fraction = partial(phase_fraction, domain_size=self.domain_size, trim=self.padding)

        for count, (f1_file, f2_file) in tqdm(enumerate(zip(self.f1_rho_files[time_min:time_max],
                                                            self.f2_rho_files[time_min:time_max])),
                                              desc="Processing drying steps"):
            if self._keep_going:
                print(f"Performing drying post-process on density file #{count + time_min}:")
                
                try:
                    rho_fraction = _phase_fraction(f1_file, f2_file)
                except Exception as e:
                    raise RuntimeError(f"Error calculating phase fraction for file #{count + time_min}: {e}")
                
                if slices:
                    try:
                        self._generate_slice_phase_map(rho_fraction=rho_fraction, count=count + time_min, slices=slices)
                    except ValueError as e:
                        raise ValueError(f"Invalid slices configuration: {e}")
                
                try:
                    self._generate_drying_rate(rho_fraction=rho_fraction, count=count + time_min)
                except Exception as e:
                    raise RuntimeError(f"Error computing drying rate for file #{count + time_min}: {e}")
            else:
                print(f"Evaporation is complete at file #{count + time_min}. Stopping further computation.")
                break
    
