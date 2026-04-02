import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from config import SEED, MAX_STEPS, DT, TAU, BETA, STRENGTH, SAFE_RADIUS, PLT_STYLE
from environment import SafeRegion, Particle
from controller import Reactive, Anticipatory, NoControl

np.random.seed(SEED)
plt.rcParams.update(PLT_STYLE)


def _record_trajectory(controller, safe, start_pos, start_vel, extra_steps=60):
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
    ax.scatter(safe.points[:, 0], safe.points[:, 1],
               s=2, alpha=0.18, color='#388bfd', zorder=1)
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


def plot_trajectories(safe, output_path='results/trajectory_visualization.png'):
    start_pos = np.array([1.2, 0.0])
    start_vel = np.array([0.3, 0.0])

    controllers = {
        'Reactive': Reactive(safe, tau=TAU, strength=STRENGTH),
        'Anticipatory': Anticipatory(safe, tau=TAU, beta=BETA, strength=STRENGTH),
        'No Control': NoControl(),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    fig.suptitle('Outward Scenario — Particle Trajectories',
                 fontsize=14, fontweight='bold', color='#e6edf3')

    for ax, (name, ctrl) in zip(axes, controllers.items()):
        traj, crossed, cross_point = _record_trajectory(
            ctrl, safe, start_pos.copy(), start_vel.copy())

        _draw_safe_region(ax, safe)

        ax.scatter(*start_pos, color='#ffa657', s=120, zorder=6,
                   marker='o', edgecolors='white', linewidths=1.2, label='Start')

        if crossed and cross_point is not None:
            cross_idx = next(
                (i for i, p in enumerate(traj) if not safe.is_safe(p)), len(traj))
            inside = traj[:cross_idx + 1]
            outside = traj[cross_idx:]

            if len(inside) > 1:
                pts_in = inside.reshape(-1, 1, 2)
                segs_in = np.concatenate([pts_in[:-1], pts_in[1:]], axis=1)
                lc = LineCollection(segs_in, cmap='plasma', linewidths=2.5,
                                   alpha=0.95, zorder=3)
                lc.set_array(np.linspace(0, 0.6, len(segs_in)))
                ax.add_collection(lc)

            if len(outside) > 1:
                pts_out = outside.reshape(-1, 1, 2)
                segs_out = np.concatenate([pts_out[:-1], pts_out[1:]], axis=1)
                lc2 = LineCollection(segs_out, color='#ff3333', linewidths=2.5,
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
            lc = LineCollection(segments, cmap='plasma', linewidths=2.5,
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


def main():
    safe = SafeRegion(n_points=2000)
    plot_trajectories(safe)