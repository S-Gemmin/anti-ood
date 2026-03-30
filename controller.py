import numpy as np

class Reactive:
    def __init__(self, safe_region, tau=0.8, strength=0.8):
        self.safe = safe_region
        self.tau = tau
        self.strength = strength
    
    def act(self, particle):
        d = self.safe.distance(particle.z)
        if d < self.tau:
            nearest = self.safe.nearest(particle.z)
            away = particle.z - nearest
            norm = np.linalg.norm(away)
            if norm > 1e-6:
                away = away / norm
            return -away * self.strength
        return np.array([0.0, 0.0])

class Anticipatory:
    def __init__(self, safe_region, tau=0.8, beta=3.0, strength=0.8):
        self.safe = safe_region
        self.tau = tau
        self.beta = beta
        self.strength = strength
    
    def act(self, particle):
        d = self.safe.distance(particle.z)
        nearest = self.safe.nearest(particle.z)
        r = particle.z - nearest
        r_norm = np.linalg.norm(r)
        
        if r_norm < 1e-6:
            return np.array([0.0, 0.0])
        
        r_hat = r / r_norm
        v_radial = np.dot(particle.v, r_hat)
        
        risk = d - self.beta * min(0, v_radial)
        
        if risk < self.tau:
            return -r_hat * self.strength
        return np.array([0.0, 0.0])

class NoControl:
    def act(self, particle):
        return np.array([0.0, 0.0])
