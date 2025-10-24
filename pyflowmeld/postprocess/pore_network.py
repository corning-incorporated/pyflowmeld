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
from os import path, listdir, makedirs, PathLike  
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import binned_statistic 
import pandas as pd 
import porespy as ps 
import openpnm as op
from .. utils import geometry, tools 
from typing import Optional, Tuple, Iterable, Any, Literal, Sequence, Dict    
from natsort import natsorted  
from datetime import datetime 
from abc import ABCMeta, abstractmethod  
from tqdm import tqdm 
import re 

# ################################# #
# Methods for pore-network analysis #
# ################################  #  
class PoreNetwork3D:
    """
    input must be a binary image
    then it will be converted to openpnm image format [True, False]
    this class is for pure pore-network analysis
    """
    def __init__(self, binary_image: np.ndarray, save_path: Optional[PathLike] = None,
                     save_name: Optional[str] = None, method: Literal['snow', 'snow2'] = 'snow2'):
        
        pn_image = np.where(binary_image == 1, True, False)
        self.res_x, self.res_y, self.res_z = binary_image.shape 
        self.domain_volume = self.res_x*self.res_y*self.res_z
        print(f"will process a domain of {self.res_x}, {self.res_y} and {self.res_z}")
        
        self.porosity = np.sum(binary_image)/(self.res_x*self.res_y*self.res_z)
        self.regions = None
        self.region_ids = None  
        
        if method == 'snow2':
            snow = self._build_snow_2(pn_image)
            self.network = op.io.network_from_porespy(snow.network)
        elif method == 'snow':
            snow = self._build_snow(pn_image)
            net = ps.networks.regions_to_network(snow.regions)
            self.network = op.io.network_from_porespy(net)
        
        self.pore_id = np.arange(0, len(self.network['pore']['all']))
        
        self.regions = snow.regions 
        # note that 0 refers to solid regions and is not considered
        self.region_ids = np.unique(self.regions)[1:]
                
        self.num_pores = len(self.network['pore']['coords'])
        if save_path is not None and not path.exists(save_path):
            makedirs(save_path)
        self.save_path = save_path 
        self.save_name = {True: save_name, False: 'pore_network_' }[save_name is not None]
        if self.save_path is not None:
            self._domain_to_vtk(binary_image)
        
        self.connected_df = None
        self.connected_net = None  
    
    # ### vtk and other helper methods ### #
    @staticmethod 
    def _build_snow_2(pn_image: np.ndarray) -> Any:
       snow_output = ps.networks.snow2(pn_image, boundary_width = 0)
       return snow_output 

    @staticmethod 
    def _build_snow(pn_image: np.ndarray) -> Any:
        """
        build a network without inserting boundary padding using snow
        """
        snow = ps.filters.snow_partitioning(pn_image)
        return snow 

    @staticmethod 
    def _convert_pore_scale(scale_value: float, key: str) -> Tuple:
        # volume to radius conversion 
        if key == 'volume':
            return ((scale_value)*3/(4*np.pi))**(1/3), 'radius'
        else:
            return scale_value, key 
    
    def _domain_to_vtk(self, binary_image: np.ndarray) -> None:
        """
        generates a vtk file of the initial domain 
        """
        res_x, res_y, res_z = binary_image.shape 
        xx, yy, zz = np.meshgrid(np.arange(0, res_x),
                    np.arange(0, res_y),
                np.arange(0, res_z), indexing = 'ij')
        domain_df = pd.DataFrame(np.c_[xx.flatten()[:, None], yy.flatten()[:, None], 
                                        zz.flatten()[:, None], binary_image.flatten()[:, None]],
                            columns = ['x', 'y', 'z', 'domain'])
        domain_df = domain_df[domain_df['domain'] == 0]
        tools.to_vtk(file_name = path.join(self.save_path, 'initial_domain.vtk'), data_frame = domain_df)
    

    def to_csv(self, time_index: int, scale_pores: Optional[Dict] = None) -> None:
        pore_coordinates = self.network['pore']['coords']
        columns = ['x', 'y', 'z']
        scales = {'diameter': self.network['pore']['inscribed_diameter'],
                             'surface_area': self.network['pore']['surface_area']}
        if scale_pores is not None or len(scale_pores) > 0:
            scales.update(scale_pores)
        arrays = np.hstack([pore_coordinates] + [scale[:, None] for scale in scales.values()])
        pore_df = pd.DataFrame(arrays, columns = columns + list(scales.keys()))
        pore_df.to_csv(path.join(self.save_path, f'pore_network_csv_at_time_{time_index}.csv'), sep = ',', header = True, index = False)


    def array_to_vtk(self, arrays, name = 'phase'):
        print("will be generating phase coordinates and will store it in ", self.save_path)
        file_name = path.join(self.save_path, name + '.vtk')
        with open(file_name, 'w+') as f:
            f.write('# vtk DataFile Version 4.2 \n')
            f.write('paraview vtk output \n')
            f.write('ASCII \n')
            f.write('DATASET ' + 'POLYDATA' + '\n')
            f.write('POINTS ' + str(len(arrays)) + ' float' + '\n')
            np.savetxt(f, arrays, fmt = '%.5f', delimiter = ' ')
            f.write('\n'*3)

    def to_vtk(self, scale_pores = {'inscribed_diameter': None},
                        time_index = 'last'):
        """
        generates a vtk file containing sticks and balls 
        """
        
        pore_coordinates = self.network['pore']['coords']
        num_pores = len(pore_coordinates)

        throat_connections = self.network['throat']['conns']

        throat_name = path.join(self.save_path, self.save_name + f'_throats_{time_index}.vtk')
        pore_name = path.join(self.save_path, self.save_name + f'_pores_{time_index}.vtk')

        with open(throat_name, 'w+') as t, open(pore_name, 'w+') as p:
            t.write('# vtk DataFile Version 4.2 \n')
            t.write('paraview vtk output \n')
            t.write('ASCII \n')
            t.write('DATASET ' + 'POLYDATA' + '\n')
            t.write('POINTS ' + str(num_pores) + ' float' + '\n')
            np.savetxt(t, pore_coordinates, fmt = '%.5f', delimiter = ' ')
            t.write('\n'*3)

            all_pairs = np.c_[np.tile(2, (len(throat_connections), 1)), throat_connections]
            t.write('LINES 2 ' + str(len(all_pairs)*3) + '\n')
            np.savetxt(t, all_pairs, fmt = '%d', delimiter = ' ')
            t.write('\n'*3)

            # write pore attributes 
            p.write('# vtk DataFile Version 4.2 \n')
            p.write('paraview vtk output \n')
            p.write('ASCII \n')
            p.write('DATASET ' + 'POLYDATA' + '\n')
            p.write('POINTS ' + str(num_pores) + ' float' + '\n')
            np.savetxt(p, pore_coordinates, fmt = '%.5f', delimiter = ' ')
            p.write('\n'*3)

            p.write('POINT_DATA ' + str(num_pores) + '\n')
            for scale_name, scale_value in scale_pores.items():
                if scale_name in self.network['pore'].keys():
                    p.write(f'SCALARS {scale_name} float\n')
                    p.write('LOOKUP_TABLE default\n')
                    np.savetxt(p, self.network['pore'][scale_name], delimiter =' ', fmt = '%.5f')
                elif scale_name not in self.network['pore'].keys() and isinstance(scale_value, np.ndarray):
                    p.write(f'SCALARS {scale_name} float\n')
                    p.write('LOOKUP_TABLE default\n')
                    np.savetxt(p, scale_value, delimiter =' ', fmt = '%.5f')
                else:
                    print('unknown pore scale')
    
    def plot_network(self, connections = True, color_by = None, cmap = 'jet', axs = None, figsize = (8,8)):
        if axs is None:
            _, axs = plt.subplots(figsize = figsize)
        if color_by is None:
            color_by = self.network['pore']['inscribed_diameter']
        
        axs = op.visualization.plot_coordinates(network = self.network, ax = axs, color_by = color_by, 
                        size_by = self.network['pore']['inscribed_diameter'], markersize = 1000,
                    cmap = cmap)
        if connections:
            axs = op.visualization.plot_connections(network = self.network, color = 'k',
                                    size_by = self.network['throat']['inscribed_diameter'],
                                      ax = axs, linewidth = 1, alpha = 0.5)
        return axs 

        
    def histogram(self, attribute: str, what: str, 
                  bins: int =  40, axs: plt.Axes = None, connect = "connected") -> Any:
        
        if attribute == 'pore':
            quant = {'connected': self.connected_pore_diameters,
                         'disconnected': self.disconnected_pore_diameters}[connect]
        else:
            quant = self.network[attribute][what]
        hist, bins = self.compute_histogram(quant, bins = bins)
        if axs is not None:
            axs.plot(bins, hist/np.sum(hist))
            return axs
        else:
            return hist, bins 

    @staticmethod 
    def compute_histogram(array: np.ndarray, bins: Optional[int] = 40) -> Tuple[np.ndarray, np.ndarray]:
        """
        computes histogram of any numpy array
        """
        hist, bins = np.histogram(array, bins, density = False)
        bins = np.array([0.5*(bin_min + bin_max) for bin_min, bin_max in zip(bins[:-1], bins[1:])])
        return hist, bins 

    def mean_std(self, attribute: str, what: str) -> Tuple[float, float]:
        """
        computes mean and standard deviation 
        """
        quant = self.network[attribute][what]
        return np.mean(quant), np.std(quant)
    
    @property 
    def connectivity(self):
        adj_matrix = self.network.create_adjacency_matrix()
        return np.sum(adj_matrix.toarray(), axis = 0)

    @property
    def weighted_connectivity(self):
        adj_matrix = self.network.create_adjacency_matrix(self.network['throat']['inscribed_diameter'])
        return np.sum(adj_matrix.toarray(), axis = 0)
    
    @property   
    def disconnected_index(self):
        return np.where(self.connectivity == 0, 1, 0)
    
    @property
    def connected_index(self):
        return np.where(self.connectivity == 0, 0, 1)
    
    @property
    def disconnected_pore_ids(self):
        return self.pore_id[self.disconnected_index == 1]
    
    @property
    def connected_pore_ids(self):
        return self.pore_id[self.connected_index == 1]
    
    @property
    def total_pore_volume(self):
        return np.sum((4/3)*np.pi*(0.5*self.network['pore']['inscribed_diameter'])**3)
    
    @property
    def total_connected_pore_volume(self):
        return np.sum((4/3)*np.pi*(self.connected_pore_diameters)**3)
    
    @property
    def total_disconnected_pore_volume(self):
        return np.sum((4/3)*np.pi*(self.disconnected_pore_diameters)**3)
    
    @property
    def total_disconnected_voxel_volume(self):
        return np.sum(self.disconnected_pore_diameters**3)

    @property
    def disconnected_pore_diameters(self):
        return self.network['pore']['inscribed_diameter'][self.disconnected_pore_ids]
    
    @property
    def connected_pore_diameters(self):
        return self.network['pore']['inscribed_diameter'][self.connected_pore_ids]
    
    @property
    def connected_pore_coordinates(self):
        return self.network['pore']['coords'][self.connected_pore_ids]
    
    @property 
    def disconnected_pore_coordinates(self):
        return self.netwrok['pore']['coords'][self.disconnected_pore_ids]
    
    @property
    def connected_network_df(self):
        if self.connected_df is None and self.total_disconnected_voxel_volume > 0:
            network_df = op.io.network_to_pandas(self.network, join = False)
            pore_df = network_df['pore']
            throat_df = network_df['throat']
            connected_pores = pore_df[pore_df.index.isin(self.connected_pore_ids)]
            connected_throats = throat_df[(throat_df['throat.conns[0]'].isin(self.connected_pore_ids)) &
                                           (throat_df['throat.conns[1]'].isin(self.connected_pore_ids))]
            old_index = connected_pores.index.values 
            new_index = np.arange(0, len(old_index))
            old_new = {k:v for k,v in zip(old_index, new_index)}
            connected_pores.index = new_index 
            connected_pores["pore.region_label"] = new_index 

            connected_throats["throat.conns[0]"] = connected_throats["throat.conns[0]"].apply(lambda x: old_new[x])
            connected_throats["throat.conns[1]"] = connected_throats["throat.conns[1]"].apply(lambda x: old_new[x])
            network_df['pore'] = connected_pores
            network_df['throat'] = connected_throats 
            print('network keys are ', network_df.keys())
            self.connected_df = network_df['throat'].join(network_df['pore'])
        return self.connected_df 
    

    def connected_to_csv(self):
        if self.connected_network_df > 0:
            file_name = path.join(self.save_path, 'connected_network.csv')
            self.connected_network_df.to_csv(file_name, index = False)
    
    @property 
    def connected_network(self):
        if self.connected_net is None:
            net = self.connected_network_df
            dct = {}
            keys = sorted(list(net.keys()))
            for item in keys:
                m = re.search(r'\[.\]', item)  
                if m:  
                    pname = re.split(r'\[.\]', item)[0]  
                    merge_keys = [k for k in net.keys() if k.startswith(pname)]
                    merge_cols = [net.pop(k) for k in merge_keys]
                    dct[pname] = np.vstack(merge_cols).T
                    for k in keys:
                        if k.startswith(pname):
                            keys.pop(keys.index(k))
                else:
                    dct[item] = np.array(net.pop(item))
            try:
                Np = np.where(np.isnan(dct['pore.coords'][:, 0]))[0][0]
            except IndexError:
                Np = dct['pore.coords'][:, 0].shape[0]
            try:
                Nt = np.where(np.isnan(dct['throat.conns'][:, 0]))[0][0]
            except IndexError:
                Nt = dct['throat.conns'][:, 0].shape[0]
            for k, v in dct.items():
                if k.startswith('pore.'):
                    dct[k] = v[:Np, ...]
                if k.startswith('throat.'):
                    dct[k] = v[:Nt, ...]
            self.connected_net = op.network.Network()
            self.connected_net.update(dct)
        return self.connected_net

    @property 
    def info(self):
        print(self.network) 

    # padding removal method 
    @staticmethod  
    def _remove_padding(domain, padding):
        if padding[0] != 0:
            domain = domain[padding[0]:,:,:]
        if padding[1] != 0:
            domain = domain[:-padding[1], :,:]
        if padding[2] != 0:
            domain = domain[:,padding[2]:,:]
        if padding[3] != 0:
            domain = domain[:,:-padding[3],:]
        if padding[4] != 0:
            domain = domain[:,:,padding[4]:]
        if padding[5] != 0:
            domain = domain[:,:,:-padding[5]]
        return domain 

    @classmethod
    def from_dat(cls, file_info = None, domain_size = None, slice_size = 0,
                        min_slice = None, save_path = None,
                             save_name = None, file_format = 'lbm', 
                                padding = None, method = 'snow2'):
        """
        reads a file: file format can be 
        domain_size = [size_x, size_y, size_z]: required for palabos format 
        slice_size and min_slice: if more slicing is needed
        slice_size: integer
        min_slice = [min_x, min_y, min_z]
        padding must be an array of 6 elements. 
        standard file is formatted as:
        res_x res_y, res_z
        0
        1
        ...

        LBM files are just 0 and 1s
        """
        if file_format == 'standard':
            domain = geometry.read_dat_file(file_info)
        elif file_format == 'lbm':
            domain = np.loadtxt(file_info).reshape(domain_size)
        else:
            raise ValueError("file format unknown; use lbm or standard or develop your own format routine")

        # note that pores are 1 and solid zones are 0 in the porespy
        domain = np.where(np.logical_or(domain == 1, domain == 2), 0, 1)
        
        if slice_size is not None and slice_size != 0 and not all(slice_size == 0):
            if isinstance(slice_size, int):
                slice_x, slice_y, slice_z = [slice_size]*3 
            elif isinstance(slice_size, (list, tuple)):
                slice_x, slice_y, slice_z = slice_size 
            min_x, min_y, min_z = {True: (0,0,0), False: min_slice}[min_slice is None]
            domain = domain[min_x: min_x + slice_x, 
                                min_y: min_y + slice_y, 
                                    min_z: min_z + slice_z]
        if padding is not None:
            domain = PoreNetwork3D._remove_padding(domain, padding)
        return cls(domain, save_path = save_path, save_name = save_name, method = method)


# ########################################### #
# Phase distribution network                  #
# ########################################### #
class PhaseDistNetwork(metaclass = ABCMeta):
    """
    generates distribution of the wetting phase on a porenetwork
    note that ms_file is a dat file with 0,1,2,3 tags corresponding to palabos inputs
    in the ms_file: all 1 and 2 tags are converted to 1 
    """
    def __init__(self, path_to_data: PathLike, domain_size: Sequence, physics: Literal['multiphase', 'drying'] = 'multiphase',
                  phase: Literal['wetting', 'defending'] = 'wetting',
                    padding: Optional[Sequence] = None, 
                        f1_rho_stem: str = 'f1_rho_dist', f2_rho_stem: str = 'f2_rho_dist',
                          save_path: Optional[PathLike] = None, 
                        threshold_tol: float  = 0.1):
        
        self.f1_rho_files = natsorted([path.join(path_to_data, _file) for _file in listdir(path_to_data) if f1_rho_stem in _file and '.dat' in _file],
                        key = lambda y: y.lower())
        
        self.f2_rho_files = natsorted([path.join(path_to_data, _file) for _file in listdir(path_to_data) if f2_rho_stem in _file and '.dat' in _file], 
                        key = lambda y: y.lower())

        self.domain_size = domain_size  
        self.padding = padding 
        self._generate_domain_size_arrays()
        
        if save_path is None:
            save_path = path.join(path_to_data, 'phase_network_analysis_on_' + datetime.today().strftime('%Y-%m-%d-%H'))
            if not path.exists(save_path):
                makedirs(save_path)
        self.save_path = save_path 

        # threshold_tol paramneter is only applicable to drying 
        self.threshold_tol = threshold_tol 
        self.ms_domain = None 
        self.physics = physics
        self.phase = phase  

    # determine domain size 
    def _generate_domain_size_arrays(self):
        if self.padding is None:
            self.res_x, self.res_y, self.res_z = self.domain_size 
        elif isinstance(self.padding, Iterable):
            res_x, res_y, res_z = self.domain_size 
            res_x -= (self.padding[0] + self.padding[1])
            res_y -= (self.padding[2] + self.padding[3])
            res_z -= (self.padding[4] + self.padding[5])
            self.res_x = res_x 
            self.res_y = res_y 
            self.res_z = res_z 
        self.xx, self.yy, self.zz = np.meshgrid(np.arange(0, self.res_x),
                    np.arange(0, self.res_y), np.arange(0, self.res_z), indexing = 'ij') 
    
    @abstractmethod
    def run(self):
        ...
    
    median = staticmethod(lambda data_array: 0.5*(data_array.min() + data_array.max()))
    nonzero_min = staticmethod(lambda data_array: data_array[data_array > 0].min())

    def _remove_padding(self, rho):
        if self.padding[0] != 0:
            rho = rho[self.padding[0]:, :,:]
        if self.padding[1] != 0:
            rho = rho[:-self.padding[1],:,:]
        if self.padding[2] != 0:
            rho = rho[:,self.padding[2]:,:]
        if self.padding[3] != 0:
            rho = rho[:,:-self.padding[3],:]
        if len(self.padding) > 4:
            if self.padding[4] != 0:
                rho = rho[:, :, self.padding[4]:]
            if self.padding[5] != 0:
                rho = rho[:,:,:-self.padding[5]]
        return rho
        
    def _load_reshape_trim(self, f_file):
        rho_f = np.loadtxt(f_file).reshape(self.domain_size) 
        if self.padding is not None:
            rho_f = self._remove_padding(rho_f)
        return rho_f   

    @staticmethod
    def _find_wet_pore_index_center(arr1, arr2):
        arr_index = np.zeros(len(arr1))
        overlap_idx = np.where((arr1 == arr2[:, None]).all(-1))[1]
        arr_index[overlap_idx] = 1 
        return arr_index 

    @staticmethod 
    def _have_overlaps(arr1, arr2):
        return (arr1[:, None] == arr2).all(-1).any(-1).any(-1) 
    
    _get_bin = staticmethod(lambda bin: np.array([0.5*(b_min + b_max) for b_min, b_max in zip(bin[:-1], bin[1:])]))

    def get_slice(self, direction = None, coordinate = None, thickness = 1):
        slice_index = {'x': (slice(coordinate, coordinate + thickness), slice(0, self.res_y), slice(0, self.res_z)), 
                'y': (slice(0, self.res_x), slice(coordinate, coordinate + thickness), slice(0, self.res_z)), 
            'z': (slice(0, self.res_x), slice(0, self.res_y), slice(coordinate, coordinate + thickness))}[direction]
        return slice_index  

# ########################################### #
# Phase distribution network 3D version       #
# ########################################### #
class PhaseDistNetwork3D(PhaseDistNetwork):
    """
    generates distribution of the wetting phase on a porenetwork
    note that ms_file is a dat file with 0,1,2,3 tags corresponding to palabos inputs
    in the ms_file: all 1 and 2 tags are converted to 1 
    """
    def __init__(self, path_to_data: PathLike, domain_size: Sequence,
                    ms_file: PathLike,  physics: Literal['multiphase', 'drying'] = 'multiphase',
                  phase: Literal['invading', 'defending'] = 'invading',
                    padding: Optional[Sequence] = None, 
                        f1_rho_stem: str = 'f1_rho_dist', f2_rho_stem: str = 'f2_rho_dist',
                          save_path: Optional[PathLike] = None, 
                        threshold_tol: float  = 0.1):
        
        super(PhaseDistNetwork3D, self).__init__(**{key:value for key,value in locals().items()
                        if key not in ['self', '__class__', 'ms_file']})
        
        self.porenet = PoreNetwork3D.from_dat(file_info = ms_file, save_path = self.save_path, 
                                save_name = 'network', file_format = 'lbm',
                                    padding = padding, slice_size = 0,
                                      domain_size = self.domain_size)
        # if True only connected pores are considered in this class
        self.pore_coordinates = np.rint(self.porenet.connected_pore_coordinates)
        self.pore_sizes = self.porenet.connected_pore_diameters 
        
        self.num_pores = len(self.pore_coordinates)
        self.pore_ids = np.arange(0, self.num_pores)

        # storage dictionaries
        # pore fill and pore volumes are calculated using snow.region
        self.pore_fills = {}
        self.pore_volumes = {}

        self.phase_indices = {}
        self.saturations = {}
        self.phase_hists = {}
        self.phase_bins = {}
        self.all_hists = {}
        self.all_bins = {}
        self.saturations = {}
        self.mip_hists = {}
        self.wet_bins = {}
        self.wet_hists = {}

    @staticmethod 
    def _have_overlaps(arr1, arr2):
        """ function returns three lengths """
        arr1_set = set(tuple(row) for row in arr1)
        arr2_set = set(tuple(row) for row in arr2)
        overlap_set = arr1_set.intersection(arr2_set)
        return len(arr1_set), len(overlap_set)
        
    # methods that determine coordinates occupied by the phase of interest
    def _phase_index_drying(self, rho1_file = None, rho2_file = None):
        rho1 = self._load_reshape_trim(rho1_file)
        rho2 = self._load_reshape_trim(rho2_file)
        rho_fraction = tools.true_division(rho1, (rho1 + rho2))
        rho_fraction_max = rho_fraction.max()
        threshold = rho_fraction_max - self.threshold_tol*rho_fraction_max 
        phase_index = np.where(rho_fraction > threshold)
        return phase_index 

    def _phase_index_multiphase(self, rho_file = None):
        rho = self._load_reshape_trim(rho_file)
        thresh_rho = self.median(rho)
        phase_index =  np.where(rho > thresh_rho)
        return phase_index 
         
    def _generate_phase_coordinates(self, time_step = None):

        rho1_file = self.f1_rho_files[time_step]
        rho2_file = self.f2_rho_files[time_step]

        if self.physics == "drying":
            phase_index = self._phase_index_drying(rho1_file = rho1_file, rho2_file = rho2_file) 
        elif self.physics == "multiphase":
            rho_file = {"invading": rho1_file, "defending": rho2_file}[self.phase]
            phase_index = self._phase_index_multiphase(rho_file = rho_file)
   
        x_wet = self.xx[phase_index]
        y_wet = self.yy[phase_index]
        z_wet = self.zz[phase_index]
        wet_coords = np.c_[x_wet[:,None], y_wet[:, None], z_wet[:, None]]
        return wet_coords 
    
    # pore coordinates of the microstructure 
    def _generate_pore_coordinates_pores(self, n_pore: int):
        pore_center = self.pore_coordinates[n_pore]
        pore_radius = self.pore_sizes[n_pore]/2 
        pore_index = np.where((self.xx - pore_center[0])**2 + (self.yy - pore_center[1])**2 +
                                     (self.zz - pore_center[2])**2 <= pore_radius**2)
        x_pore = self.xx[pore_index]
        y_pore = self.yy[pore_index]
        z_pore = self.zz[pore_index]
        pore_points = np.c_[x_pore[:,None], y_pore[:, None], z_pore[:, None]]
       
        return pore_points 

    def _find_phase_pore_index(self, phase_coordinates: np.ndarray):
        """
        compute for size range limits teh calculation for a specific size range
         for example (0,10) limits the calculation to pores of 0 to 10 microns
        """
        if self._run_ids is not None:
            pore_ids = self._run_ids
            len_ids = len(pore_ids)
            phase_index = np.zeros(len_ids)
            pore_volumes = np.zeros(len_ids)
            pore_fills = np.zeros(len_ids)
        else:
            phase_index = np.zeros(self.num_pores)
            pore_volumes = np.zeros(self.num_pores)
            pore_fills = np.zeros(self.num_pores)
            pore_ids = self.pore_ids
        
        for count, n_pore in enumerate(tqdm(pore_ids)):
            pore_points = self._generate_pore_coordinates_pores(n_pore)
            pore_volume, overlap_volume = self._have_overlaps(pore_points, phase_coordinates)
            pore_volumes[count] = pore_volume 
            pore_fills[count] = overlap_volume/pore_volume
            if overlap_volume > 0:
                phase_index[count] = 1

        return pore_volumes, pore_fills, phase_index 
    
    def compute_phase_pore_properties(self, time_step: int):
        """ for a standalone phase_index calculation 
            the phase index can be used in plotting the wet/dry pores using 
                openpnm mpl functions
        """
        phase_coordinates = self._generate_phase_coordinates(time_step = time_step)
        pore_volumes, pore_fills, phase_index =  self._find_phase_pore_index(phase_coordinates=phase_coordinates)
        return pore_volumes, pore_fills, phase_index, phase_coordinates 

    def run(self, time_step: int, outputs: str = 'vtk', output_phase: bool = True):
        pore_volumes, pore_fills, phase_index, phase_coordinates = self.compute_phase_pore_properties(time_step)
        if outputs is not None and self._run_ids is None:
            if 'vtk' in outputs:
                self.porenet.to_vtk(scale_pores = {'inscribed_diameter': None, 'phase_index': phase_index},
                                 time_index = time_step)
            if 'csv' in outputs:
                self.porenet.to_csv(time_step, scale_pores = {'pore_index': phase_index})
        
        if output_phase:
            self.porenet.array_to_vtk(phase_coordinates, name = f'phase_distribution_at_{time_step}.vtk')    

        return pore_volumes, pore_fills, phase_index 

    def __call__(self, time_step: int, bins: int = 10,
                  save_hist: bool = True, save_network: bool = True, output_phase: bool = False, stat: Literal['mean'] = 'mean', 
                    compute_for_size_range: Optional[Sequence] = None, 
                    outputs: Optional[str] = 'vtk'):
        
        self._run_ids = None 
        if compute_for_size_range:
            append_name = f"_for_size_range_{compute_for_size_range[0]}_to_{compute_for_size_range[1]}"
            size_index = (self.pore_sizes >= compute_for_size_range[0]) & (self.pore_sizes <= compute_for_size_range[1])
            self._run_ids = self.pore_ids[size_index]
            run_pore_sizes = self.pore_sizes[size_index]
        else:
            run_pore_sizes = self.pore_sizes

        pore_volumes, pore_fills, phase_index = self.run(time_step, outputs = outputs, output_phase = output_phase)           
        
        self.phase_indices[time_step] = phase_index  
        self.pore_volumes[time_step] = pore_volumes 
        self.pore_fills[time_step] = pore_fills    

        
        wet_size = run_pore_sizes*phase_index  
        wet_size = wet_size[wet_size > 0]
        # generate histogram of pore sizes 
        wet_hist, wet_edges = np.histogram(wet_size, bins = bins,
                                              range = (wet_size.min(), wet_size.max()))
        wet_bins = self._get_bin(wet_edges)
        
        # note that histograms are presented as fractions 
        all_fill = np.ones_like(run_pore_sizes)
        phase_fill = all_fill*pore_fills
        size_min, size_max = run_pore_sizes.min(), run_pore_sizes.max()

        phase_means, phase_edges,_ = binned_statistic(run_pore_sizes, phase_fill, statistic = stat, bins = bins, 
                                                      range = (size_min, size_max))
        all_means, all_edges, _ = binned_statistic(run_pore_sizes, all_fill, statistic = stat, bins = bins, 
                                                        range = (size_min, size_max))
        
        phase_bin = self._get_bin(phase_edges)
        all_bin = self._get_bin(all_edges)
        phase_means = np.nan_to_num(phase_means, copy = False, nan = 0.0)
        all_means = np.nan_to_num(all_means, copy = False, nan = 0.0)

        saturation = np.mean(phase_means)
        print(f'calculated saturation is {saturation}')
        self.saturations[time_step] = saturation 
        self.all_bins[time_step] = all_bin 
        self.phase_bins[time_step] = phase_bin 
        self.phase_hists[time_step] = phase_means 
        self.all_hists[time_step] = all_means 
        # wet hists and bins are not weighted stats
        self.wet_bins[time_step] = wet_bins 
        self.wet_hists[time_step] = wet_hist 

        if save_hist:
            self.save_hist(time_step=time_step, append_name = append_name)
        if save_network:
            self.save_network(time_step = time_step)
            
        return all_bin, all_means, phase_bin, phase_means  

    
    def _set_time_step(self, time_step):
        if isinstance(time_step, int):
            time_step = [time_step]
        elif time_step is None:
            time_step = list(self.all_bins.keys())
        elif isinstance(time_step, (list, tuple)):
            time_step = time_step
        return time_step 

    def save_hist(self, time_step = None, append_name = ""):
        save_hist_path = path.join(self.save_path, 'histograms' + append_name)
        if not path.exists(save_hist_path):
            makedirs(save_hist_path)
        
        time_step = self._set_time_step(time_step)
                
        for step in time_step:
            file_name = path.join(save_hist_path, f'hist_at_{step}.csv')
            hist_df = pd.DataFrame(np.c_[self.all_bins[step][:, None], self.all_hists[step][:, None], 
                                            self.phase_bins[step][:,None], self.phase_hists[step][:,None]], 
                                                columns = ['all_bin', 'all_hist', 'phase_bin', 'phase_hist'])
            hist_df.to_csv(file_name, sep = ',', header = True, index = False)
            # wet historagm that shows histogram of wet pores
            wet_name = path.join(save_hist_path, f'wet_pores_size_{step}.csv')
            wet_df = pd.DataFrame(np.c_[self.wet_bins[step][:,None], self.wet_hists[step][:,None]], 
                                   columns = ['wet_size', 'wet_hist']) 
            wet_df.to_csv(wet_name, sep = ',', header = True, index = False)

    
    def save_network(self, time_step = None):
        """ adds pore index and pore filling and saves the data into a csv file """
        save_path = path.join(self.save_path, 'networks')
        if not path.exists(save_path):
            makedirs(save_path)
        
        time_step = self._set_time_step(time_step)

        for step in time_step:
            file_name = path.join(save_path, f'network_at_step_{step}.csv')
            if isinstance(self.porenet.connected_network_df, pd.DataFrame):
                network = self.porenet.connected_network_df
                fill_net = pd.DataFrame(np.c_[self.pore_fills[step].reshape(-1,1), 
                                            self.phase_indices[step].reshape(-1,1)], columns = ['pore.fills', 'pore.fill_index'])
                all_network = network.join(fill_net)
                #final_net = network['throat'].join(network['pore'])
            else:
                network = op.io.network_to_pandas(self.porenet.network)
                network['pore']['pore.fills'] = self.pore_fills[step].reshape(-1,1)
                network['pore']['pore.fill_index'] = self.phase_indices[step].reshape(-1,1)
                all_network = network['throat'].join(network['pore'])
            all_network.to_csv(file_name, index = False)

    
    def mip_histograms(self, pore_size_min = None, pore_size_max = None, bins = None, time_step = None):
        """
        generates a histogram similar to mercury porosimetry
        """
        if len(self.all_bins) == 0:
            raise ValueError("calculation not possible with no histogram")

        all_bins = self.all_bins[time_step]
        phase_hist = self.phase_hists[time_step]
        all_hist = self.all_hists[time_step]

        mip_bins = np.linspace(pore_size_min, pore_size_max, bins)
        # dictionary of pore size range and fill %
        mip_phase = {}
        for bin_min, bin_max in zip(mip_bins[:-1], mip_bins[1:]):
            index = np.where((all_bins >= bin_min) & (all_bins <= bin_max))
            all_pores = np.sum(all_hist[index])
            wet_pores = np.sum(phase_hist[index])
            mip_phase[f'{round(bin_min,2)} to {round(bin_max, 2)}'] = wet_pores/all_pores 
        
        return mip_phase 

    @staticmethod
    def _compute_saturation_from_histogram(phase_hist, phase_bin, all_hist, all_bin):
        total_volume = np.sum(all_bin*all_hist)
        phase_volume = np.sum(phase_hist*(4/3)*np.pi*(phase_bin**3))
        return phase_volume/total_volume 

            
        

