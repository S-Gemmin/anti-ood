import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import SEED, MAX_STEPS, DT, TAU, BETA, STRENGTH, NOISE_POS, NOISE_VEL, PLT_STYLE
from environment import SafeRegion, Particle
from controller import Reactive, Anticipatory

np.random.seed(SEED)
plt.rcParams.update(PLT_STYLE)


def _run_outward_crossing_rate(controller, safe, n_trials=100):
    start_pos = np.array([1.2, 0.0])
    start_vel = np.array([0.3, 0.0])
    crossings = 0
    for _ in range(n_trials):
        pos = start_pos + np.random.randn(2) * NOISE_POS
        vel = start_vel + np.random.randn(2) * NOISE_VEL
        particle = Particle(pos, vel)
        for _ in range(MAX_STEPS):
            acc = controller.act(particle)
            particle.step(acc, DT)
            if not safe.is_safe(particle.z):
                crossings += 1
                break
    return crossings / n_trials * 100


def plot_beta_sweep(safe, output_path='results/beta_sensitivity_sweep.png'):
    betas = np.arange(0, 16.5, 1.0)
    rates = []

    for b in betas:
        ctrl = Anticipatory(safe, tau=TAU, beta=b, strength=STRENGTH)
        rate = _run_outward_crossing_rate(ctrl, safe)
        rates.append(rate)

    reactive_rate = _run_outward_crossing_rate(
        Reactive(safe, tau=TAU, strength=STRENGTH), safe)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    ax.fill_between(betas, rates, alpha=0.15, color='#3fb950')
    ax.plot(betas, rates, color='#3fb950', lw=2.5,
           marker='o', markersize=5, label='Anticipatory crossing rate')
    ax.axhline(reactive_rate, color='#f78166', lw=1.5, ls='--',
              label=f'Reactive baseline ({reactive_rate:.0f}%)')

    chosen_idx = np.argmin(np.abs(betas - BETA))
    ax.scatter([betas[chosen_idx]], [rates[chosen_idx]],
               color='#ffa657', s=120, zorder=5,
               edgecolors='white', linewidths=1,
               label=f'Chosen β={BETA} ({rates[chosen_idx]:.0f}%)')

    target = reactive_rate * 0.25
    ax.axhline(target, color='#a5d6ff', lw=1, ls=':',
              label=f'75% improvement target ({target:.0f}%)')

    ax.set_xlabel('Beta  (β)', fontsize=11)
    ax.set_ylabel('Crossing Rate (%)', fontsize=11)
    ax.set_title('Beta Sensitivity — Anticipatory Controller\nOutward Scenario, 100 trials',
                fontsize=12, fontweight='bold')
    ax.set_xlim(betas[0] - 0.3, betas[-1] + 0.3)
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9, facecolor='#161b22',
             edgecolor='#30363d', labelcolor='#e6edf3')

    fig.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    safe = SafeRegion(n_points=2000)
    plot_beta_sweep(safe)