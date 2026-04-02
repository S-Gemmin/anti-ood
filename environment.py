import numpy as np
from scipy.spatial import KDTree
from config import CENTER, SIGMA, SAFE_RADIUS

class SafeRegion:
    def __init__(self, n_points=1000):
        self.points = np.random.multivariate_normal(CENTER, SIGMA**2 * np.eye(2), size=n_points)
        self.centroid = np.mean(self.points, axis=0)
        self.kdtree = KDTree(self.points)
    
    def distance(self, z):
        dist, _ = self.kdtree.query(z.reshape(1, -1))
        return dist[0]
    
    def distance_to_centroid(self, z):
        return np.linalg.norm(z - self.centroid)
    
    def gradient(self, z):
        to_center = self.centroid - z
        norm = np.linalg.norm(to_center)
        if norm > 1e-6:
            return to_center / norm
        return np.zeros_like(to_center)
    
    def is_safe(self, z):
        return self.distance(z) < SAFE_RADIUS

class Particle:
    def __init__(self, pos, vel):
        self.z = np.array(pos, dtype=float)
        self.v = np.array(vel, dtype=float)
        self.crossed = False
    
    def step(self, acc, dt=0.1):
        self.v += acc * dt
        self.z += self.v * dt
