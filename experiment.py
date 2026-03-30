import numpy as np
from config import N_TRIALS, MAX_STEPS, DT, NOISE_POS, NOISE_VEL, SEED, TAU, BETA, STRENGTH
from environment import SafeRegion, Particle
from controller import Reactive, Anticipatory, NoControl

np.random.seed(SEED)

def run_trial(controller, safe, start_pos, start_vel):
    particle = Particle(start_pos, start_vel)
    for _ in range(MAX_STEPS):
        acc = controller.act(particle)
        particle.step(acc, DT)
        if not safe.is_safe(particle.z):
            return True
    return False

def run_experiment(scenarios):
    safe = SafeRegion()
    controllers = {
        'reactive': Reactive(safe, tau=TAU, strength=STRENGTH),
        'anticipatory': Anticipatory(safe, tau=TAU, beta=BETA, strength=STRENGTH),
        'none': NoControl()
    }
    
    results = {}
    for name, (pos, vel) in scenarios.items():
        results[name] = {c: [] for c in controllers}
        for trial in range(N_TRIALS):
            if trial % 20 == 0:
                print(f"  {name}: trial {trial}/{N_TRIALS}")
            pos_noisy = pos + np.random.randn(2) * NOISE_POS
            vel_noisy = vel + np.random.randn(2) * NOISE_VEL
            for cname, ctrl in controllers.items():
                crossed = run_trial(ctrl, safe, pos_noisy, vel_noisy)
                results[name][cname].append(crossed)
    
    return results

scenarios = {
    'stationary': (np.array([1.2, 0.0]), np.array([0.0, 0.0])),
    'inward': (np.array([1.2, 0.0]), np.array([-0.3, 0.0])),
    'outward': (np.array([1.2, 0.0]), np.array([0.3, 0.0])),
    'tangent': (np.array([1.2, 0.0]), np.array([0.0, 0.3])),
}

print("Running experiment...")
results = run_experiment(scenarios)

print("CROSSING RATES (%)")
print(f"{'Scenario':<12} {'Reactive':>12} {'Anticipatory':>14} {'None':>8}")

for name in scenarios:
    r = np.mean(results[name]['reactive']) * 100
    a = np.mean(results[name]['anticipatory']) * 100
    n = np.mean(results[name]['none']) * 100
    print(f"{name:<12} {r:>12.1f} {a:>14.1f} {n:>8.1f}")

out_r = np.mean(results['outward']['reactive']) * 100
out_a = np.mean(results['outward']['anticipatory']) * 100
if out_r > 0:
    improvement = (out_r - out_a) / out_r * 100
    print(f"\nOutward scenario improvement: {improvement:.1f}%")
else:
    print("\nOutward scenario improvement: N/A (no reactive crossings)")
