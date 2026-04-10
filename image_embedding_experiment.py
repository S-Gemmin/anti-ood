import os
import sys
import argparse
import numpy as np
from pathlib import Path

SEED = 42
np.random.seed(SEED)

N_TRIALS = 100
MAX_STEPS = 200
DT = 0.1
TAU_PCT = 80 
BETA = 1.5
VEL = 0.001
STRENGTH = 0.00083


def extract_embeddings(image_dir):
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError:
        print("please install: pip install open_clip_torch torch torchvision Pillow")
        sys.exit(1)

    print("loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model.eval()

    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(Path(image_dir).glob(ext))

    if len(image_paths) == 0:
        print("no images found in", image_dir)
        sys.exit(1)

    print(f"found {len(image_paths)} images, extracting embeddings...")

    embeddings = []
    for i, path in enumerate(image_paths):
        try:
            img = preprocess(Image.open(path).convert('RGB')).unsqueeze(0)
            with torch.no_grad():
                emb = model.encode_image(img)
                # normalize to unit sphere (standard for CLIP)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.squeeze().numpy())
            if (i + 1) % 10 == 0:
                print(f"  done {i+1}/{len(image_paths)}")
        except Exception as e:
            print(f"  skipping {path.name}: {e}")

    return np.array(embeddings)


def compute_safe_manifold(embeddings):
    centroid = embeddings.mean(axis=0)
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    tau = np.percentile(dists, TAU_PCT)

    print(f"\nsafe manifold stats:")
    print(f"  num embeddings: {len(embeddings)}")
    print(f"  embedding dim: {len(centroid)}")
    print(f"  min dist to centroid: {dists.min():.4f}")
    print(f"  max dist to centroid: {dists.max():.4f}")
    print(f"  mean dist to centroid: {dists.mean():.4f}")
    print(f"  tau (80th pct): {tau:.4f}")

    return centroid, tau


def distance_to_centroid(z, centroid):
    return float(np.linalg.norm(z - centroid))


def grad_f(z, centroid):
    diff = centroid - z
    norm = np.linalg.norm(diff)
    if norm < 1e-8:
        return np.zeros_like(z)
    return diff / norm


def reactive_control(z, v, centroid, tau, strength):
    d = distance_to_centroid(z, centroid)
    if d > tau:
        toward = centroid - z
        norm = np.linalg.norm(toward)
        if norm > 1e-8:
            return (toward / norm) * strength
    return np.zeros_like(z)


def anticipatory_control(z, v, centroid, tau, beta, strength):
    d = distance_to_centroid(z, centroid)
    gf = grad_f(z, centroid)
    v_radial = np.dot(v, gf)
    risk = d + beta * max(0, -v_radial)

    if risk > tau:
        toward = centroid - z
        norm = np.linalg.norm(toward)
        if norm > 1e-8:
            return (toward / norm) * strength
    return np.zeros_like(z)


def run_trial(controller_type, start_pos, start_vel, centroid, tau, beta, strength):
    z = start_pos.copy()
    v = start_vel.copy()
    for _ in range(MAX_STEPS):
        if controller_type == 'reactive':
            acc = reactive_control(z, v, centroid, tau, strength)
        elif controller_type == 'anticipatory':
            acc = anticipatory_control(z, v, centroid, tau, beta, strength)
        else:
            acc = np.zeros_like(z)

        v = v + acc * DT
        z = z + v * DT

        if distance_to_centroid(z, centroid) > tau:
            return True

    return False


def run_all_experiments(centroid, tau, start_distances, beta, strength, vel):
    dim = len(centroid)
    noise = tau * 0.005
    rng = np.random.RandomState(0)
    outward_dir = rng.randn(dim)
    outward_dir = outward_dir / np.linalg.norm(outward_dir)

    results = {}

    for dist in start_distances:
        results[dist] = {'reactive': [], 'anticipatory': [], 'none': []}
        base_pos = centroid + outward_dir * dist
        base_vel = outward_dir * vel

        for trial in range(N_TRIALS):
            pos = base_pos + np.random.randn(dim) * noise
            vel_noisy = base_vel + np.random.randn(dim) * (noise * 0.1)

            for ctrl in ['reactive', 'anticipatory', 'none']:
                crossed = run_trial(ctrl, pos, vel_noisy, centroid, tau, beta, strength)
                results[dist][ctrl].append(crossed)

        r = np.mean(results[dist]['reactive']) * 100
        a = np.mean(results[dist]['anticipatory']) * 100
        n = np.mean(results[dist]['none']) * 100
        print(f"  dist={dist:.4f}  reactive={r:5.1f}%  anticipatory={a:5.1f}%  no_control={n:5.1f}%")

    return results, outward_dir


def beta_sensitivity_sweep(centroid, tau, dist, strength, vel, outward_dir):
    dim = len(centroid)
    noise = tau * 0.005
    base_pos = centroid + outward_dir * dist
    base_vel = outward_dir * vel
    print(f"\nbeta sensitivity sweep (dist={dist:.4f}):")
    for b in [0, 1, 2, 3, 4, 6, 8, 10, 12, 16]:
        crossings = []
        for _ in range(N_TRIALS):
            pos = base_pos + np.random.randn(dim) * noise
            vel_noisy = base_vel + np.random.randn(dim) * (noise * 0.1)
            crossed = run_trial('anticipatory', pos, vel_noisy, centroid, tau, b, strength)
            crossings.append(crossed)
        print(f"  beta={b:2d}  crossing rate={np.mean(crossings)*100:5.1f}%")


def print_results_table(results, start_distances, tau):
    display_distances = start_distances[:3]

    print("\n" + "="*65)
    print("CROSSING RATES (%) - IMAGE EMBEDDING EXPERIMENT")
    print("="*65)
    print(f"{'start dist':>12} {'no control':>12} {'reactive':>12} {'anticipatory':>14}")

    reactive_vals = []
    anticipatory_vals = []

    for dist in display_distances:
        r = np.mean(results[dist]['reactive']) * 100
        a = np.mean(results[dist]['anticipatory']) * 100
        n = np.mean(results[dist]['none']) * 100
        reactive_vals.append(r)
        anticipatory_vals.append(a)
        print(f"{dist:>12.4f} {n:>12.1f} {r:>12.1f} {a:>14.1f}")

    print(f"\n  tau = {tau:.4f}")
    print("="*65)

    pp_reductions = [r - a for r, a in zip(reactive_vals, anticipatory_vals)]
    avg_pp = np.mean(pp_reductions)

    print(f"\navg improvement: {avg_pp:.1f} pp reduction across {len(display_distances)} start distances")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images',     type=str,   default=None,
                        help='folder with safe hallway images (needed on first run)')
    parser.add_argument('--embeddings', type=str,   default=None,
                        help='path to save/load embeddings (.npy file)')
    parser.add_argument('--beta',       type=float, default=BETA)
    parser.add_argument('--strength',   type=float, default=STRENGTH)
    parser.add_argument('--vel',        type=float, default=VEL)
    args = parser.parse_args()

    if args.embeddings and os.path.exists(args.embeddings):
        print(f"loading cached embeddings from {args.embeddings}")
        embeddings = np.load(args.embeddings)
        print(f"loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")
    elif args.images:
        embeddings = extract_embeddings(args.images)
        if args.embeddings:
            np.save(args.embeddings, embeddings)
            print(f"saved embeddings to {args.embeddings}")
    else:
        print("error: need --images (first run) or --embeddings (cached)")
        sys.exit(1)

    centroid, tau = compute_safe_manifold(embeddings)
    start_distances = np.round(np.linspace(tau * 0.90, tau * 1.02, 8), 4)[:3]

    print(f"\nstart distances: {start_distances}")
    print(f"tau (boundary): {tau:.4f}")
    print(f"vel: {args.vel}  strength: {args.strength}  beta: {args.beta}")
    print(f"\nrunning experiment ({N_TRIALS} trials per distance)...")

    results, outward_dir = run_all_experiments(
        centroid, tau, start_distances, args.beta, args.strength, args.vel)

    print_results_table(results, start_distances, tau)
    print("\ndone.")


if __name__ == '__main__':
    main()