"""
Inspect action distribution logs saved by workers.
Run after killing training after batch 1:

    python tools/inspect_action_dist.py

Reads output/action_dist_log/pid*_ep*.npz and prints per-dim statistics.
"""

import os
import glob
import numpy as np

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "action_dist_log")
LABELS  = ["Roll", "Pitch", "Throttle", "Water"]

def load_all():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "pid*_ep*.npz")))
    if not files:
        print(f"No files found in {LOG_DIR}")
        return None, None, None, None
    all_means   = []
    all_stds    = []
    all_samples = []
    lifespans   = []
    for f in files:
        d = np.load(f)
        all_means.append(d["means"])
        all_stds.append(d["stds"])
        all_samples.append(d["samples"])
        lifespans.append(int(d["lifespan"][0]))
    print(f"Loaded {len(files)} files from {LOG_DIR}")
    print(f"  Files: {[os.path.basename(f) for f in files[:5]]}{'...' if len(files)>5 else ''}")
    print(f"  Lifespans: min={min(lifespans)}, max={max(lifespans)}, avg={np.mean(lifespans):.0f}\n")
    return (np.vstack(all_means),
            np.vstack(all_stds),
            np.vstack(all_samples),
            lifespans)

def print_stats(means, stds, samples):
    print("=" * 70)
    print(f"{'Dim':<10} {'Mean(mean)':>12} {'Std(mean)':>12} {'Mean(std)':>12} {'Min(sample)':>13} {'Max(sample)':>13}")
    print("-" * 70)
    for i, lbl in enumerate(LABELS):
        m     = means[:, i]
        s     = stds[:, i]
        smp   = samples[:, i]
        print(f"{lbl:<10} {m.mean():>12.4f} {m.std():>12.4f} {s.mean():>12.4f} {smp.min():>13.4f} {smp.max():>13.4f}")
    print("=" * 70)

def print_saturated(samples, threshold=0.90):
    """Report fraction of steps where |sample| > threshold (saturated output)."""
    print(f"\nSaturation (|sample| > {threshold}):")
    for i, lbl in enumerate(LABELS):
        frac = np.mean(np.abs(samples[:, i]) > threshold)
        bar  = "#" * int(frac * 40)
        print(f"  {lbl:<10} {frac*100:5.1f}%  |{bar:<40}|")

def print_water_trigger(samples):
    """Water triggers when mapped value > 0.5, i.e. sample > 0 (before (x+1)/2 mapping)."""
    raw   = samples[:, 3]
    mapped = (raw + 1.0) / 2.0
    frac_on = np.mean(mapped > 0.5)
    print(f"\nWater valve ON (mapped > 0.5): {frac_on*100:.1f}% of steps")
    print(f"  Raw sample range: [{raw.min():.3f}, {raw.max():.3f}],  mean={raw.mean():.3f}")

def print_throttle(samples):
    raw = samples[:, 2]
    mapped = np.maximum(0.5, (raw + 1.0) / 2.0)  # same as env_core throttle_physical
    speed  = mapped * 30.0
    print(f"\nThrottle (Va_c):")
    print(f"  Raw sample range: [{raw.min():.3f}, {raw.max():.3f}],  mean={raw.mean():.3f}")
    print(f"  Mapped [0,1]:     [{mapped.min():.3f}, {mapped.max():.3f}],  mean={mapped.mean():.3f}")
    print(f"  Va_c [m/s]:       [{speed.min():.1f}, {speed.max():.1f}],  mean={speed.mean():.1f}")
    dead_frac = np.mean(raw < 0.0)
    print(f"  Dead-zone (raw<0, throttle floored at 0.5): {dead_frac*100:.1f}% of steps")

def print_per_step_trend(means, n_steps=20):
    """Show how mean Roll and Pitch evolve over first n_steps (avg across all episodes)."""
    if means.shape[0] < n_steps:
        return
    # Group by step position — only works if all episodes have same length
    # Fall back to just printing overall mean per first n steps
    print(f"\nFirst {n_steps} steps — mean of distribution mean (Roll, Pitch):")
    print(f"  {'step':>5}  {'Roll':>8}  {'Pitch':>8}")
    for t in range(min(n_steps, means.shape[0])):
        print(f"  {t:>5}  {means[t, 0]:>8.4f}  {means[t, 1]:>8.4f}")

if __name__ == "__main__":
    means, stds, samples, lifespans = load_all()
    if means is None:
        exit(1)

    print_stats(means, stds, samples)
    print_saturated(samples)
    print_water_trigger(samples)
    print_throttle(samples)
    print_per_step_trend(means)
