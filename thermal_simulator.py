from pathlib import Path
from mesh_loader import Mesh3DLoader

class Thermal_simulator:
    def __init__(self, geometry_path:Path):
        self.geometry_path = geometry_path
        mesh_loader = Mesh3DLoader(self.geometry_path)
        self.mesh, self.volume_tag, self.face_tags = mesh_loader.load_mesh()

if __name__ == '__main__':
    simulation = Thermal_simulator()