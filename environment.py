import numpy as np
from scipy.spatial import KDTree
from config import CENTER, SIGMA, SAFE_RADIUS

class SafeRegion:
    def __init__(self, n_points=1000):
        self.points = np.random.multivariate_normal(CENTER, SIGMA**2 * np.eye(2), size=n_points)
        self.kdtree = KDTree(self.points)
    
    def distance(self, z):
        dist, _ = self.kdtree.query(z.reshape(1, -1))
        return dist[0]
    
    def nearest(self, z):
        _, idx = self.kdtree.query(z.reshape(1, -1))
        return self.points[idx[0]]
    
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
