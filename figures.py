"""
generate_figures.py — figure add-on for experiment.py
Run directly:  python generate_figures.py
Produces three PNGs in the current directory:
  - trajectory_visualization.png   (Ticket 1)
  - beta_sensitivity_sweep.png     (Ticket 2)
  - risk_landscape_heatmap.png     (Ticket 3)
All parameters are read from config.py.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from config import SEED, MAX_STEPS, DT, TAU, BETA, STRENGTH, SAFE_RADIUS, NOISE_POS, NOISE_VEL
from environment import SafeRegion, Particle
from controller import Reactive, Anticipatory, NoControl

np.random.seed(SEED)

#shared plot style
PLT_STYLE = {
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#0d1117',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'grid.color':       '#21262d',
    'grid.linewidth':   0.6,
    'text.color':       '#e6edf3',
    'font.family':      'monospace',
}
plt.rcParams.update(PLT_STYLE)


# TICKET 1 — Trajectory Visualization
def _record_trajectory(controller, safe, start_pos, start_vel, extra_steps=60):
    """Run simulation without stopping at crossing so the escape path is visible."""
    particle = Particle(start_pos, start_vel)
    traj = [particle.z.copy()]
    crossed, cross_point = False, None
    for _ in range(MAX_STEPS + extra_steps):
        acc = controller.act(particle)
        particle.step(acc, DT)
        traj.append(particle.z.copy())
        if not safe.is_safe(particle.z) and not crossed:
            crossed = True
            cross_point = particle.z.copy()
    return np.array(traj), crossed, cross_point


def _draw_safe_region(ax, safe):
    """Draw point cloud and compute the real boundary contour via KD-tree distances."""
    ax.scatter(safe.points[:, 0], safe.points[:, 1],
               s=2, alpha=0.18, color='#388bfd', zorder=1,
               label='Safe region (point cloud)')
    res = 180
    xs = np.linspace(-7, 8, res)
    ys = np.linspace(-7, 7, res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    dists, _ = safe.kdtree.query(pts)
    ZZ = dists.reshape(res, res)
    ax.contour(XX, YY, ZZ, levels=[SAFE_RADIUS],
               colors=['#388bfd'], linewidths=1.5, linestyles='--',
               zorder=2, alpha=0.85)
    ax.plot([], [], color='#388bfd', lw=1.5, ls='--',
            label='Safe boundary (actual)')


def plot_trajectories(safe, output_path='trajectory_visualization.png'):
    start_pos = np.array([1.2, 0.0])
    start_vel = np.array([0.3, 0.0])   # outward scenario

    controllers = {
        'Reactive':     Reactive(safe, tau=TAU, strength=STRENGTH),
        'Anticipatory': Anticipatory(safe, tau=TAU, beta=BETA, strength=STRENGTH),
        'No Control':   NoControl(),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    fig.suptitle('Outward Scenario — Particle Trajectories',
                 fontsize=14, fontweight='bold', color='#e6edf3')

    for ax, (name, ctrl) in zip(axes, controllers.items()):
        traj, crossed, cross_point = _record_trajectory(
            ctrl, safe, start_pos.copy(), start_vel.copy())

        _draw_safe_region(ax, safe)

        # start marker drawn first so it's first in the legend
        ax.scatter(*start_pos, color='#ffa657', s=120, zorder=6,
                   marker='o', edgecolors='white', linewidths=1.2, label='Start')

        if crossed and cross_point is not None:
            cross_idx = next(
                (i for i, p in enumerate(traj) if not safe.is_safe(p)), len(traj))
            inside  = traj[:cross_idx + 1]
            outside = traj[cross_idx:]

            if len(inside) > 1:
                pts_in = inside.reshape(-1, 1, 2)
                segs_in = np.concatenate([pts_in[:-1], pts_in[1:]], axis=1)
                lc = LineCollection(segs_in, cmap='plasma', linewidth=2.5,
                                    alpha=0.95, zorder=3)
                lc.set_array(np.linspace(0, 0.6, len(segs_in)))
                ax.add_collection(lc)

            if len(outside) > 1:
                pts_out = outside.reshape(-1, 1, 2)
                segs_out = np.concatenate([pts_out[:-1], pts_out[1:]], axis=1)
                lc2 = LineCollection(segs_out, color='#ff3333', linewidth=2.5,
                                     alpha=0.9, zorder=4, linestyle='--')
                ax.add_collection(lc2)

            ax.scatter(*cross_point, color='#ff0000', s=120, zorder=7,
                       marker='X', edgecolors='white', linewidths=1.0,
                       label='Boundary crossed')
            ax.annotate('CROSSED', xy=cross_point,
                        xytext=cross_point + np.array([0.5, 0.7]),
                        color='#ff6b6b', fontsize=8, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=1.5),
                        zorder=8)
            if len(outside) > 5:
                ax.annotate('', xy=outside[-1], xytext=outside[-6],
                            arrowprops=dict(arrowstyle='->', color='#ff3333',
                                            lw=2.0), zorder=5)
        else:
            points = traj.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='plasma', linewidth=2.5,
                                alpha=0.95, zorder=3)
            lc.set_array(np.linspace(0, 1, len(segments)))
            ax.add_collection(lc)

            ax.scatter(*traj[-1], color='#3fb950', s=150, zorder=6,
                       marker='*', edgecolors='white', linewidths=1,
                       label='End (safe)')
            ax.annotate('SAFE', xy=traj[-1],
                        xytext=traj[-1] + np.array([0.4, 0.5]),
                        color='#3fb950', fontsize=8, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#3fb950', lw=1.5),
                        zorder=8)

        status = '✗ CROSSED' if crossed else '✓ SAFE'
        status_color = '#f78166' if crossed else '#3fb950'
        ax.set_title(f'{name}\n{status}', fontsize=12,
                     color=status_color, fontweight='bold', pad=10)
        ax.set_xlim(-6.5, 7.5)
        ax.set_ylim(-6.5, 6.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.legend(fontsize=7.5, loc='upper left',
                  facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')

    sm = plt.cm.ScalarMappable(cmap='plasma',
                                norm=plt.Normalize(0, MAX_STEPS * DT))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.65, pad=0.02)
    cbar.set_label('Time (s)', color='#8b949e')
    cbar.ax.yaxis.set_tick_params(color='#8b949e')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e')

    fig.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    print(f"Saved: {output_path}")



# TICKET 2 — Beta Sensitivity Sweep
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


def plot_beta_sweep(safe, output_path='beta_sensitivity_sweep.png'):
    betas = np.arange(0, 16.5, 1.0)
    rates = []

    print("Running beta sweep...")
    for b in betas:
        ctrl = Anticipatory(safe, tau=TAU, beta=b, strength=STRENGTH)
        rate = _run_outward_crossing_rate(ctrl, safe)
        rates.append(rate)
        print(f"  beta={b:.1f} → {rate:.1f}%")

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



# TICKET 3 — Risk Landscape Heatmap
def plot_risk_heatmap(safe, output_path='risk_landscape_heatmap.png'):
    res = 300
    xs = np.linspace(-6.5, 7.5, res)
    ys = np.linspace(-6.5, 6.5, res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    dists, _ = safe.kdtree.query(pts)
    dists = dists.reshape(res, res)

    # fixed outward velocity for anticipatory risk computation
    v_radial = -0.3   # negative = moving away from safe region

    reactive_risk     = dists.copy()
    anticipatory_risk = dists - BETA * v_radial   # = dists + BETA * 0.3

    colors_heat = ['#0d2137', '#0a3d5c', '#0e7490',
                   '#22c55e', '#eab308', '#ef4444', '#7f1d1d']
    cmap_heat = LinearSegmentedColormap.from_list('risk', colors_heat, N=256)

    vmin = 0
    vmax = max(reactive_risk.max(), anticipatory_risk.max()) * 0.6

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle('Risk Landscape — Reactive vs Anticipatory\n'
                 '(fixed outward radial velocity v = 0.3)',
                 fontsize=13, fontweight='bold', color='#e6edf3')

    panels = [
        ('Reactive Risk\n(distance to boundary only)',          reactive_risk),
        (f'Anticipatory Risk\n(distance − β·v_radial,  β={BETA})', anticipatory_risk),
    ]

    for ax, (title, risk) in zip(axes, panels):
        im = ax.pcolormesh(XX, YY, risk, cmap=cmap_heat,
                           vmin=vmin, vmax=vmax, shading='auto', zorder=1)

        cs_tau = ax.contour(XX, YY, risk, levels=[TAU],
                            colors=['#ffffff'], linewidths=1.8,
                            linestyles='-', zorder=3)
        ax.clabel(cs_tau, fmt=f'risk = τ ({TAU})', fontsize=7.5,
                  colors='#ffffff', inline=True)

        cs_safe = ax.contour(XX, YY, dists, levels=[SAFE_RADIUS],
                             colors=['#38bdf8'], linewidths=1.4,
                             linestyles='--', zorder=3)
        ax.clabel(cs_safe, fmt='safe boundary', fontsize=7.5,
                  colors='#38bdf8', inline=True)

        ax.scatter(safe.points[:, 0], safe.points[:, 1],
                   s=1, alpha=0.08, color='#e2e8f0', zorder=2)

        ax.scatter(1.2, 0.0, color='#ffa657', s=100, zorder=5,
                   marker='o', edgecolors='white', linewidths=1.0,
                   label='Start pos')
        ax.annotate('', xy=(2.2, 0.0), xytext=(1.2, 0.0),
                    arrowprops=dict(arrowstyle='->', color='#ffa657',
                                   lw=2.0, mutation_scale=15), zorder=6)
        ax.text(2.3, 0.15, 'v', color='#ffa657', fontsize=9, fontweight='bold')

        ax.set_title(title, fontsize=11, color='#e6edf3', pad=8)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0], ys[-1])
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.legend(fontsize=8, loc='lower right',
                  facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Risk value', color='#8b949e', fontsize=9)
        cbar.ax.yaxis.set_tick_params(color='#8b949e')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e')
        cbar.ax.axhline(TAU, color='white', lw=1.5, ls='-')
        cbar.ax.text(1.1, TAU / (vmax - vmin), f'τ={TAU}',
                     transform=cbar.ax.transAxes,
                     color='white', fontsize=7.5, va='center')

    fig.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    print(f"Saved: {output_path}")


# Entry point
if __name__ == '__main__':
    print("Initialising safe region...")
    safe = SafeRegion(n_points=2000)

    print("\n── Ticket 1: Trajectory Visualization ──")
    plot_trajectories(safe)

    print("\n── Ticket 2: Beta Sensitivity Sweep ──")
    plot_beta_sweep(safe)

    print("\n── Ticket 3: Risk Landscape Heatmap ──")
    plot_risk_heatmap(safe)

    print("\nAll figures saved.")