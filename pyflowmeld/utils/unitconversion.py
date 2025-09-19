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

# ###### Performs Units Conversions ####### #
class Converter:
    """
    nu: fluid viscosity in physical units
    tau: lattice relaxation used in LBM simulation
    dx: lattice size in physical units
    rho: fluid density in physical units
    """
    _cs = np.sqrt(1/3)
    _g = 9.8
    def __init__(self, nu: float = 2*10**-5,
                    tau: float = 0.8,
                         dx: float = 10**-6,
                           rho: float = 1.225, lattice_rho: float =  1):
        """
        nu: kinematic viscosity in physical units 
        tau: lattice relaxation time in BGK
        rho: density in phusical units
        lattice_rho: lattice density in lattice units. 
        dx: conversion factor of lattice unit to the physical units : x(lattice) = x(physical)/dx
        """
        self.dx = dx
        self.tau = tau
        self.nu = nu
        self.rho = rho
        self.lattice_rho = lattice_rho 

    @property
    def dt(self) -> float:
        return (self.nu/((self._cs**2)*(self.tau - 0.5)*(self.dx**2)))**-1
    
    @property
    def dv(self) -> float:
        return (self.dx/self.dt)
    
    @property
    def dnu(self) -> float:
        return ((self.dx**2)/self.dt)
    
    @property
    def dp(self) -> float:
        return (self.rho)*(self.dx/self.dt)**2
    
    @property 
    def drho(self) -> float:
        return self.rho/self.lattice_rho 
    
    @property 
    def df(self) -> float:
        return self.drho*self.dx/(self.dt**2)

    # acceleration 
    @property
    def da(self) -> float:
        return self.dx/(self.dt**2) 

    @property 
    def lattice_gravity_force(self) -> float:
        return self.rho*self._g/self.df

    def set_units(self, nu: float,
                     tau: float,
                         dx: float,
                             rho: float,
                                 lattice_rho: float) -> None:
        self.nu = nu
        self.tau = tau
        self.dx = dx
        self.rho = rho
        self.lattice_rho = lattice_rho   
    # converters 
    def time_2_real(self, time_steps: float) -> float:
        return time_steps*self.dt 

    def velocity_2_lattice(self, physical_velocity: float) -> float:
        return physical_velocity/self.dv 
    
    def velocity_2_real(self, lattice_velocity: float) -> float:
        return lattice_velocity*self.dv 
    
    def pressure_2_real(self, lattice_pressure: float) -> float:
        return lattice_pressure*self.dp 
    
    def pressure_2_lattice(self, physical_pressure: float) -> float:
        return physical_pressure/self.dp 
    
    def viscosity_2_real(self, lattice_nu: float) -> float:
        return lattice_nu*self.dnu
    
    def viscosity_2_lattice(self, physical_nu: float) -> float:
        return physical_nu/self.dnu

    def force_2_lattice(self, physical_force: float) -> float:
        return physical_force/self.df 
    
    def force_2_physical(self, lattice_force: float) -> float:
        return lattice_force*self.df
    
    
    

    
    