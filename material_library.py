from pathlib import Path
from typing import Dict
import h5py
from attr import dataclass
from dolfinx.io import XDMFFile
from matplotlib.dates import TH
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
    # data source: https://thermtest.com/thermal-resources/materials-database
    _registry = {
        "aluminum-6061": ThermalMaterial(name="Aluminum-6061", k=167, cp=896, rho=2700),
        "air": ThermalMaterial(name="Air", k=0.025, cp=1004, rho=1.29),
        "copper": ThermalMaterial(name="Copper", k=397, cp=385, rho=8940),
        "ss-304": ThermalMaterial(name="Stainless Steel 304", k=14.6, cp=502, rho=7920),
        "ss-316": ThermalMaterial(name="Stainless Steel 316", k=13.5, cp=470, rho=8230)
    }

    @classmethod
    def get(cls, name: str) -> LinearElasticMaterial:
        return cls._registry[name.lower()]

if __name__ == "__main__":
    pass
