import numpy as np

SEED = 42
N_TRIALS = 100
MAX_STEPS = 200
DT = 0.1
TAU = 0.3
BETA = 8.0
STRENGTH = 0.056
CENTER = np.array([0.0, 0.0])
SIGMA = 1.0
SAFE_RADIUS = 2.0
NOISE_POS = 0.05
NOISE_VEL = 0.02

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
