import numpy as np

class Reactive:
    def __init__(self, safe_region, tau=0.8, strength=0.8):
        self.safe = safe_region
        self.tau = tau
        self.strength = strength
    
    def act(self, particle):
        d = self.safe.distance_to_centroid(particle.z)
        if d > self.tau:
            to_center = self.safe.centroid - particle.z
            norm = np.linalg.norm(to_center)
            if norm > 1e-6:
                to_center = to_center / norm
            return to_center * self.strength
        return np.array([0.0] * len(particle.z))

class Anticipatory:
    def __init__(self, safe_region, tau=0.8, beta=3.0, strength=0.8):
        self.safe = safe_region
        self.tau = tau
        self.beta = beta
        self.strength = strength
    
    def act(self, particle):
        d = self.safe.distance_to_centroid(particle.z)
        grad = self.safe.gradient(particle.z)
        
        v_dot_grad = np.dot(particle.v, grad)
        
        risk = d + self.beta * max(0, -v_dot_grad)
        
        if risk > self.tau:
            to_center = self.safe.centroid - particle.z
            norm = np.linalg.norm(to_center)
            if norm > 1e-6:
                to_center = to_center / norm
            return to_center * self.strength
        return np.array([0.0] * len(particle.z))

class NoControl:
    def act(self, particle):
        return np.array([0.0] * len(particle.z))