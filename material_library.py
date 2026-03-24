from pathlib import Path
from typing import Dict
import h5py
from attr import dataclass
from dolfinx.io import XDMFFile
from mpi4py import MPI


@dataclass(frozen=True)
class ThermalMaterial:
    name: str
    k: float    # Thermal Conductivity [W / (m * K)]
    cp: float   # Specific Heat Capacity [J / (kg * K)]
    rho: float  # Density [kg / m^3]
    
    @property
    def alpha(self):
        """Thermal diffusivity [m²/s]"""
        return self.k / (self.rho * self.cp)


@dataclass(frozen=True)
class LinearElasticMaterial:
    name: str
    E: float    # Young's Modulus [N / m^2]
    nu: float   # Poisson's Ratio [-]
    rho: float  # Density [[kg / m^3]]

    @property
    def lmbda(self):
        """First Lamé parameter"""
        return (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def mu(self):
        """Second Lamé parameter (Shear modulus)"""
        return self.E / (2 * (1 + self.nu))


class ElasticMaterialLibrary:
    _registry = {
        "steel": LinearElasticMaterial(name="Steel", E=210e9, nu=0.3),
        "aluminum": LinearElasticMaterial(name="Aluminum", E=70e9, nu=0.33),
    }

    @classmethod
    def get(cls, name: str) -> LinearElasticMaterial:
        return cls._registry[name.lower()]
    
    
class ThermalMaterialLibrary:
    _registry = {
        "steel": ThermalMaterial(name="Steel", k=210e9, cp=0.3),
        "aluminum": ThermalMaterial(name="Aluminum", k=70e9, cp=0.33),
    }

    @classmethod
    def get(cls, name: str) -> LinearElasticMaterial:
        return cls._registry[name.lower()]

if __name__ == "__main__":
    
