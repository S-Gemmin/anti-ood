import numpy as np
from config import SEED, MAX_STEPS, DT
from environment import Particle
from controller import Reactive, Anticipatory, NoControl
from experiments.real_safe_region import RealSafeRegion

np.random.seed(SEED)

def run_trial(controller, safe, start_pos, start_vel):
    particle = Particle(start_pos, start_vel)
    for _ in range(MAX_STEPS):
        acc = controller.act(particle)
        particle.step(acc, DT)
        if not safe.is_safe(particle.z):
            return True
    return False

def run_distance_sweep(start_distances, n_trials=100, tau=0.65, beta=0.3, display=None):
    safe = RealSafeRegion()
    center = safe.centroid
    
    avg_direction = np.mean(safe.embeddings - center, axis=0)
    avg_direction = avg_direction / np.linalg.norm(avg_direction)
    
    print(f"Threshold: {safe.threshold:.4f}")
    print(f"{'Start Dist':>10} {'Reactive':>10} {'Anticipatory':>12} {'None':>10}")
    print("-" * 45)
    
    results_summary = []
    for target_dist in start_distances:
        start_pos = center + avg_direction * target_dist
        start_vel = avg_direction * 0.015
        
        noise_pos = 0.002
        noise_vel = 0.001
        
        controllers = {
            'reactive': Reactive(safe, tau=tau, strength=1.0),
            'anticipatory': Anticipatory(safe, tau=tau, beta=beta, strength=1.0),
            'none': NoControl()
        }
        
        results = {name: [] for name in controllers}
        
        for trial in range(n_trials):
            pos = start_pos + np.random.randn(len(start_pos)) * noise_pos
            vel = start_vel + np.random.randn(len(start_vel)) * noise_vel
            
            for name, ctrl in controllers.items():
                crossed = run_trial(ctrl, safe, pos, vel)
                results[name].append(crossed)
        
        r = np.mean(results['reactive']) * 100
        a = np.mean(results['anticipatory']) * 100
        n = np.mean(results['none']) * 100
        
        # only print if in display list (or print all if display not specified)
        if display is None or target_dist in display:
            print(f"{target_dist:>10.2f} {r:>10.1f} {a:>12.1f} {n:>10.1f}")
        
        results_summary.append((target_dist, r, a, n))
    
    return results_summary

if __name__ == '__main__':
    run_distance_sweep([0.70, 0.72, 0.74, 0.76, 0.78])
