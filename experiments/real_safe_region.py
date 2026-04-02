import numpy as np
import json
from scipy.spatial import KDTree

class RealSafeRegion:
    def __init__(self, path='results/safe_embeddings.json', threshold=None):
        with open(path, 'r') as f:
            data = json.load(f)
        
        all_emb = np.array(data['embeddings'])
        self.centroid = np.mean(all_emb, axis=0)
        
        if threshold is None:
            dists = [np.linalg.norm(e - self.centroid) for e in all_emb]
            self.threshold = np.percentile(dists, 80)
        else:
            self.threshold = threshold
        
        self.embeddings = np.array([e for e in all_emb 
                                    if np.linalg.norm(e - self.centroid) < self.threshold])
        self.kdtree = KDTree(self.embeddings)
    
    def distance_to_centroid(self, z):
        return np.linalg.norm(z - self.centroid)
    
    def gradient(self, z):
        z = np.array(z).reshape(-1)
        direction = self.centroid - z
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            return direction / norm
        return np.zeros_like(direction)
    
    def is_safe(self, z):
        return self.distance_to_centroid(z) < self.threshold
