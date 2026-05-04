"""
eval_multifire_progress.py
==========================
Compare scout checkpoints across multi-fire fine-tuning progress.

Runs N episodes per checkpoint and reports:
  - fire_disc   : fraction of episodes where fire was found
  - coverage%   : mean fraction of steps with fire in camera
  - avg_seen    : mean fire intensity under camera
  - survived%   : mean fraction of scouts that survived
  - R/scout     : mean reward per scout

Usage:
  python tools/eval_multifire_progress.py [--episodes 20] [--scouts 3] [--fires 3]
"""
import sys, os, argparse, glob, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import random

from env_core import DroneFireEnv
from models import ScoutActor

# ── helpers ─────────────────────────────────────────────────────────────────

def run_episode(env, actor, seed, n_quads, device, hidden_dim=128,
                max_steps=1000, n_fires=3):
    """Run one episode, return dict of metrics."""
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    obs, _ = env.reset(epizode_number=seed + 50000)  # full-difficulty episodes

    # Place extra fires (same as demo_scout.py logic)
    b = env.map_bounds * 0.65
    fire_positions = list(env.fire_positions)  # already has 1+ from reset
    attempts = 0
    while len(fire_positions) < n_fires and attempts < 200:
        attempts += 1
        fx = rng.uniform(-b, b)
        fy = rng.uniform(-b, b)
        if any(np.hypot(fx - px, fy - py) < 150.0 for px, py in fire_positions):
            continue
        fire_positions.append((fx, fy))
        env.sim.start_fire([fx, fy], intensity=0.5)
    env.fire_positions = fire_positions

    quad_agents = env.quad_agents
    scout_h = {q: torch.zeros(1, 1, hidden_dim) for q in quad_agents}
    scout_alive = {q: True for q in quad_agents}
    total_rewards = [0.0] * n_quads
    fire_hist = {qi: [] for qi in range(n_quads)}
    fire_discovered = False

    for step in range(max_steps):
        actions = {}
        for qi, q in enumerate(quad_agents):
            if scout_alive[q] and q in env.agents:
                with torch.no_grad():
                    lm = torch.FloatTensor(obs[q]["local_map"]).unsqueeze(0)
                    ss = torch.FloatTensor(obs[q]["self_state"]).unsqueeze(0)
                    ns = torch.FloatTensor(obs[q]["neighbor_states"]).unsqueeze(0)
                    nm = torch.BoolTensor(obs[q]["neighbor_mask"]).unsqueeze(0)
                    dist, _, h_out = actor(lm, ss, ns, nm, scout_h[q])
                    act = dist.mean  # deterministic for eval
                scout_h[q] = h_out
                actions[q] = act.squeeze(0).cpu().numpy()
            else:
                scout_alive[q] = False

        if env.agents:
            obs, rewards, terms, truncs, _ = env.step(actions)
            for qi, q in enumerate(quad_agents):
                total_rewards[qi] += rewards.get(q, 0.0)
                if scout_alive[q] and q in env.agents:
                    fire_val = float(np.sum(obs[q]["local_map"]))
                    fire_hist[qi].append(fire_val)
                    if fire_val > 0.1:
                        fire_discovered = True
                else:
                    fire_hist[qi].append(0.0)
                if terms.get(q, False) or truncs.get(q, False):
                    scout_alive[q] = False
        else:
            break
        if not any(scout_alive.values()):
            break

    survived = sum(1 for q in quad_agents if q in env.sim.drones)
    all_vals = [v for qi in range(n_quads) for v in fire_hist[qi] if v > 0]
    avg_seen = float(np.mean(all_vals)) if all_vals else 0.0
    steps_with_fire = sum(
        1 for s in range(len(fire_hist[0]))
        if any(fire_hist[qi][s] > 0.1
               for qi in range(n_quads) if s < len(fire_hist[qi])))
    total_steps = len(fire_hist[0]) or 1
    coverage_pct = steps_with_fire / total_steps * 100.0

    return {
        "fire_discovered": fire_discovered,
        "coverage_pct":    coverage_pct,
        "avg_seen":        avg_seen,
        "survived":        survived / n_quads * 100.0,
        "avg_r":           sum(total_rewards) / n_quads,
    }


def eval_checkpoint(ckpt_path, n_quads, n_fires, seeds, device, hidden_dim=128, max_steps=500):
    """Evaluate a checkpoint over `seeds` and return aggregate metrics."""
    env = DroneFireEnv(num_quads=n_quads, num_fixed=0,
                       grid_size_m=1000.0, max_steps=max_steps,
                       n_fires_range=(n_fires, n_fires))
    obs_space = env.observation_space("quad_0")
    actor = ScoutActor(self_state_dim=obs_space["self_state"].shape[0],
                       msg_dim=5, hidden_dim=hidden_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    actor.load_state_dict(ckpt, strict=False)
    actor.eval()

    results = []
    for seed in seeds:
        r = run_episode(env, actor, seed, n_quads, device,
                        hidden_dim=hidden_dim, n_fires=n_fires,
                        max_steps=max_steps)
        results.append(r)

    env.sim.stop_simulation()

    return {
        "ckpt":       os.path.basename(ckpt_path),
        "disc%":      np.mean([r["fire_discovered"] for r in results]) * 100,
        "coverage%":  np.mean([r["coverage_pct"]    for r in results]),
        "avg_seen":   np.mean([r["avg_seen"]         for r in results]),
        "survived%":  np.mean([r["survived"]         for r in results]),
        "R/scout":    np.mean([r["avg_r"]            for r in results]),
    }


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int, default=10)
    parser.add_argument("--scouts",     type=int, default=3)
    parser.add_argument("--fires",      type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=500)
    parser.add_argument("--max-steps",  type=int, default=500,
                        help="Episode length (default 500 — faster eval)")
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(2)

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    root  = os.path.join(os.path.dirname(__file__), '..')

    # Collect checkpoints to compare — only key milestones to keep eval fast
    checkpoints = []

    # Original pre-multifire scout (baseline)
    base = os.path.join(root, "saved_models", "multi", "scout_best.pt")
    if os.path.isfile(base):
        checkpoints.append(("baseline (pre-multifire)", base))

    # Sample every other multifire checkpoint to keep it manageable
    mf_dir = os.path.join(root, "saved_models", "scout_multifire")
    all_mf = sorted(glob.glob(os.path.join(mf_dir, "scout_b*.pt")))
    # Always include first, last, and a few in between
    if all_mf:
        indices = sorted(set([0, len(all_mf)//3, 2*len(all_mf)//3, len(all_mf)-1]))
        for i in indices:
            bn = os.path.basename(all_mf[i])
            checkpoints.append((bn, all_mf[i]))

    # Best multifire
    best_mf = os.path.join(mf_dir, "scout_best.pt")
    if os.path.isfile(best_mf):
        checkpoints.append(("multifire_best", best_mf))

    print(f"Evaluating {len(checkpoints)} checkpoints × {args.episodes} episodes "
          f"({args.scouts} scouts, {args.fires} fires)")
    print(f"Seeds: {seeds[0]}–{seeds[-1]}\n")

    hdr = f"{'Checkpoint':<30}  {'disc%':>6}  {'cover%':>7}  {'seen':>6}  {'surv%':>6}  {'R/scout':>8}"
    print(hdr)
    print("─" * len(hdr))

    all_rows = []
    for label, path in checkpoints:
        metrics = eval_checkpoint(path, args.scouts, args.fires, seeds, device,
                                   hidden_dim=128, max_steps=args.max_steps)
        row = {"label": label, **metrics}
        all_rows.append(row)
        print(f"{label:<30}  {metrics['disc%']:>5.1f}%  {metrics['coverage%']:>6.1f}%  "
              f"{metrics['avg_seen']:>6.3f}  {metrics['survived%']:>5.1f}%  "
              f"{metrics['R/scout']:>+8.1f}")

    print("─" * len(hdr))

    # Save CSV
    csv_path = os.path.join(root, "eval_multifire_progress.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV → {csv_path}")
