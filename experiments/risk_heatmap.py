import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from config import SEED, DT, TAU, BETA, SAFE_RADIUS, PLT_STYLE
from environment import SafeRegion

np.random.seed(SEED)
plt.rcParams.update(PLT_STYLE)


def plot_risk_heatmap(safe, output_path='results/risk_landscape_heatmap.png'):
    res = 300
    xs = np.linspace(-6.5, 7.5, res)
    ys = np.linspace(-6.5, 6.5, res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    dists, _ = safe.kdtree.query(pts)
    dists = dists.reshape(res, res)

    v_radial = -0.3

    reactive_risk = dists.copy()
    anticipatory_risk = dists - BETA * v_radial

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
        ('Reactive Risk\n(distance to boundary only)', reactive_risk),
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


def main():
    safe = SafeRegion(n_points=2000)
    plot_risk_heatmap(safe)