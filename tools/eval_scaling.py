"""
eval_scaling.py — Final scaling evaluation
==========================================
Evaluates trained scout+commander team across all combinations of:
  - number of scouts:  1, 3, 5, 7, 10
  - number of FW:      1, 3, 5, 7, 10
  - number of fires:   1, 3, 5, 7
  - map sizes:         700, 1100, 2000, 4000, 6000 m
  - 10 seeds per configuration

Total: 5 × 5 × 4 × 5 × 10 = 5000 episodes

Outputs:
  results/eval_scaling/results.csv       — full CSV with all metrics
  results/eval_scaling/summary.csv       — aggregated stats per config
  results/eval_scaling/plots/            — generated plots

Usage:
  python tools/eval_scaling.py \
      --scout-model saved_models/v10_finetune/scout_best.pt \
      --cmdr-model  saved_models/v10_finetune/cmdr_best.pt \
      --workers 15

  # Partial run (e.g. only 3 scouts, 5 FW):
  python tools/eval_scaling.py \
      --scout-model saved_models/v10_finetune/scout_best.pt \
      --cmdr-model  saved_models/v10_finetune/cmdr_best.pt \
      --filter-scouts 3 --filter-fw 5
"""

import sys, os, argparse, csv, random, time, json
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "src"))

# ── Configuration ────────────────────────────────────────────────────────────
SCOUT_COUNTS = [1, 3, 5, 7, 10]
FW_COUNTS    = [1, 3, 5, 7, 10]
FIRE_COUNTS  = [1, 3, 5, 7]
MAP_SIZES    = [700.0, 1100.0, 2000.0, 4000.0, 6000.0]
N_SEEDS      = 10
MAX_STEPS    = 1200

HIDDEN_DIM_SCOUT = 128
HIDDEN_DIM_CMDR  = 64
WAYPOINT_STEPS   = 30
WAYPOINT_RANGE   = 200.0
WP_REACHED_DIST  = 30.0
WATER_CAPACITY   = 200.0


# ── Metrics dataclass ────────────────────────────────────────────────────────
@dataclass
class EpisodeMetrics:
    # Configuration
    n_scouts: int = 0
    n_fw: int = 0
    n_fires: int = 0
    map_size_m: float = 0.0
    seed: int = 0

    # Timing [steps]
    steps_total: int = 0
    time_first_discovery: int = -1          # step when first fire found (-1 = never)
    time_all_fires_discovered: int = -1     # step when all fires seen (-1 = never)
    time_first_water_drop: int = -1         # step when first water hits fire
    time_full_suppression: int = -1         # step when all fire extinguished (-1 = never)

    # Fire
    peak_fire_cells: int = 0                # max burning cells at any point
    final_burning_cells: int = 0            # cells still burning at episode end
    total_burned_cells: int = 0             # cells that ever burned (fuel depleted or burned)
    suppression_rate_pct: float = 0.0       # (peak - final_burning) / peak × 100

    # Water
    water_drops_total: int = 0
    water_drops_hit: int = 0
    water_drops_miss: int = 0
    water_accuracy_pct: float = 0.0
    water_consumed_L: float = 0.0
    refill_count: int = 0

    # Scout behavior
    scout_time_over_fire_steps: float = 0.0  # mean steps where scout sees fire
    scout_time_over_fire_pct: float = 0.0
    scout_mean_altitude_m: float = 0.0
    scout_mean_separation_m: float = 0.0
    scout_deaths: int = 0

    # Commander behavior
    fw_mean_drop_distance_m: float = 0.0     # mean dist to fire at water drop
    fw_mean_drop_altitude_m: float = 0.0
    fw_deaths: int = 0

    # Per-fire metrics (JSON-serialized lists)
    per_fire_discovery_step: str = "[]"
    per_fire_suppression_step: str = "[]"

    # Reward
    scout_avg_reward: float = 0.0
    cmdr_avg_reward: float = 0.0

    # Runtime
    wallclock_s: float = 0.0
    error: str = ""


# ── Worker function (runs in subprocess) ─────────────────────────────────────

def run_single_episode(args_tuple):
    """Run one episode. Called via ProcessPoolExecutor."""
    (scout_path, cmdr_path, n_scouts, n_fw, n_fires, map_size, seed) = args_tuple

    # Imports inside worker (each process needs its own)
    import numpy as np
    import torch
    import random as rnd

    os.chdir(os.path.join(PROJECT, "src"))
    sys.path.insert(0, os.path.join(PROJECT, "src"))

    from env_core import DroneFireEnv
    from models import ScoutActor, CommanderActor
    from commander_control import CommanderController

    device = torch.device("cpu")  # CPU for parallel workers
    t0 = time.time()

    metrics = EpisodeMetrics(
        n_scouts=n_scouts, n_fw=n_fw, n_fires=n_fires,
        map_size_m=map_size, seed=seed
    )

    try:
        # ── Setup ────────────────────────────────────────────────────────
        rnd.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = DroneFireEnv(
            num_quads=n_scouts, num_fixed=n_fw,
            grid_size_m=map_size, max_steps=MAX_STEPS,
            n_fires_range=(n_fires, n_fires),
        )
        obs, _ = env.reset(epizode_number=seed)

        quad_agents = env.quad_agents
        fixed_agents = env.fixed_agents

        # Load models
        tmp_env_for_dims = env
        scout_self_dim = tmp_env_for_dims.observation_space("quad_0")["self_state"].shape[0]
        fixed_self_dim = tmp_env_for_dims.observation_space("fixed_0")["self_state"].shape[0]

        scout_actor = ScoutActor(self_state_dim=scout_self_dim,
                                 msg_dim=5, hidden_dim=HIDDEN_DIM_SCOUT).to(device)
        cmdr_actor = CommanderActor(self_state_dim=fixed_self_dim,
                                    msg_input_dim=5,
                                    hidden_dim=HIDDEN_DIM_CMDR).to(device)

        if os.path.exists(scout_path):
            scout_actor.load_state_dict(torch.load(scout_path, map_location=device))
        if os.path.exists(cmdr_path):
            cmdr_actor.load_state_dict(torch.load(cmdr_path, map_location=device))
        scout_actor.eval()
        cmdr_actor.eval()

        # ── State tracking ───────────────────────────────────────────────
        h_scout = {q: torch.zeros(1, 1, HIDDEN_DIM_SCOUT) for q in quad_agents}
        h_cmdr = {f: torch.zeros(1, 1, HIDDEN_DIM_CMDR) for f in fixed_agents}

        scout_alive = {q: True for q in quad_agents}
        fw_alive = {f: True for f in fixed_agents}
        last_scout_msgs = {q: np.zeros(5) for q in quad_agents}

        # Commander controllers
        cmdr_ctrl = {}
        for f in fixed_agents:
            ctrl = CommanderController(WAYPOINT_RANGE, WAYPOINT_STEPS, WP_REACHED_DIST)
            ctrl.reset(env.map_bounds)
            cmdr_ctrl[f] = ctrl

        # Per-fire tracking
        fire_positions = list(env.fire_positions)
        n_actual_fires = len(fire_positions)
        fire_discovered = [False] * n_actual_fires
        fire_discovery_step = [-1] * n_actual_fires
        fire_suppressed = [False] * n_actual_fires
        fire_suppression_step = [-1] * n_actual_fires

        # Episode-level tracking
        peak_burning = 0
        ever_burned_mask = np.zeros_like(env.sim.environment.fire_grid.B, dtype=bool)
        first_discovery_step = -1
        all_discovered_step = -1
        first_water_hit_step = -1
        full_suppression_step = -1

        # Scout metrics
        scout_fire_steps = {q: 0 for q in quad_agents}
        scout_alt_sum = {q: 0.0 for q in quad_agents}
        scout_alt_count = {q: 0 for q in quad_agents}
        scout_deaths = 0

        # FW metrics
        fw_drop_distances = []
        fw_drop_altitudes = []
        fw_deaths = 0
        fw_prev_water = {}
        fw_refill_count = 0
        fw_water_start = {}
        fw_water_end = {}

        # Water tracking
        water_total = 0
        water_hit = 0
        water_miss = 0

        # Rewards
        scout_rewards = {q: 0.0 for q in quad_agents}
        fw_rewards = {f: 0.0 for f in fixed_agents}

        for f in fixed_agents:
            d = env.sim.drones.get(f)
            if d:
                fw_water_start[f] = d.current_water
                fw_prev_water[f] = d.current_water

        # ── Main loop ────────────────────────────────────────────────────
        for step in range(MAX_STEPS):
            if not env.agents:
                break

            actions = {}
            fg = env.sim.environment.fire_grid

            # Track burning state
            if fg is not None:
                current_burning = int(fg.B.sum())
                peak_burning = max(peak_burning, current_burning)
                ever_burned_mask |= fg.B

                # Per-fire discovery check (via proximity of scouts to fire positions)
                for fi, (fx, fy) in enumerate(fire_positions):
                    if fire_suppressed[fi]:
                        continue
                    # Check if fire at this position is still burning
                    gm = env.sim.environment.grid_mapper
                    ci, cj = gm.world_to_cell([fx, fy])
                    # Check 5×5 area around ignition point
                    r_min = max(0, ci - 5)
                    r_max = min(fg.H, ci + 6)
                    c_min = max(0, cj - 5)
                    c_max = min(fg.W, cj + 6)
                    local_burning = fg.B[r_min:r_max, c_min:c_max].sum()
                    if local_burning == 0 and step > 30:
                        # This fire zone is suppressed
                        fire_suppressed[fi] = True
                        fire_suppression_step[fi] = step

                # Full suppression check
                if current_burning == 0 and full_suppression_step == -1 and step > 30:
                    full_suppression_step = step

            # --- Scouts ---
            for q in quad_agents:
                if not scout_alive[q] or q not in env.agents:
                    scout_alive[q] = False
                    continue

                drone = env.sim.drones.get(q)
                if drone:
                    pos = drone.get_position()
                    scout_alt_sum[q] += pos[2]
                    scout_alt_count[q] += 1

                with torch.no_grad():
                    lm = torch.FloatTensor(obs[q]["local_map"]).unsqueeze(0)
                    ss = torch.FloatTensor(obs[q]["self_state"]).unsqueeze(0)
                    ns = torch.FloatTensor(obs[q]["neighbor_states"]).unsqueeze(0)
                    nm = torch.BoolTensor(obs[q]["neighbor_mask"]).unsqueeze(0)
                    dist, msg, h_out = scout_actor(lm, ss, ns, nm, h_scout[q])
                    act = dist.mean
                h_scout[q] = h_out
                actions[q] = act.squeeze(0).cpu().numpy()
                last_scout_msgs[q] = msg.squeeze(0).cpu().numpy()

                # Fire visibility check
                fire_in_cam = float(np.sum(obs[q]["local_map"]))
                if fire_in_cam > 0.1:
                    scout_fire_steps[q] += 1

                    # Per-fire discovery (scout sees fire → mark discovered)
                    if drone:
                        pos = drone.get_position()
                        for fi, (fx, fy) in enumerate(fire_positions):
                            if not fire_discovered[fi]:
                                d = np.hypot(pos[0] - fx, pos[1] - fy)
                                # Scout FOV ~ alt*1.5, so if dist < FOV scout likely sees it
                                fov = max(10.0, pos[2] * 1.5)
                                if d < fov:
                                    fire_discovered[fi] = True
                                    fire_discovery_step[fi] = step
                                    if first_discovery_step == -1:
                                        first_discovery_step = step
                                    if all(fire_discovered) and all_discovered_step == -1:
                                        all_discovered_step = step

            # --- Commanders ---
            for f in fixed_agents:
                if not fw_alive[f] or f not in env.agents:
                    fw_alive[f] = False
                    continue

                drone = env.sim.drones.get(f)
                if drone is None:
                    fw_alive[f] = False
                    continue

                # Build scout message tensor
                msg_list = [torch.FloatTensor(last_scout_msgs.get(q, np.zeros(5)))
                            for q in quad_agents]
                msgs_t = torch.stack(msg_list).unsqueeze(0)
                mask_t = torch.BoolTensor(
                    [[not scout_alive.get(q, False) for q in quad_agents]])

                # FW neighbor tensor
                pos_f = drone.get_position()
                fw_neigh_list = []
                fw_mask_list = []
                for f2 in fixed_agents:
                    if f2 == f:
                        continue
                    d2 = env.sim.drones.get(f2)
                    if d2 is not None and fw_alive.get(f2, False):
                        p2 = d2.get_position()
                        fw_neigh_list.append([
                            (p2[0] - pos_f[0]) / max(env.map_bounds, 1.0),
                            (p2[1] - pos_f[1]) / max(env.map_bounds, 1.0),
                            p2[2] / 100.0
                        ])
                        fw_mask_list.append(False)
                    else:
                        fw_neigh_list.append([0.0, 0.0, 0.0])
                        fw_mask_list.append(True)

                fw_neigh_t = torch.FloatTensor([fw_neigh_list]) if fw_neigh_list else None
                fw_mask_t = torch.BoolTensor([fw_mask_list]) if fw_mask_list else None

                action, h_cmdr[f], ctrl_info = cmdr_ctrl[f].step(
                    drone, obs[f]["self_state"], env,
                    cmdr_actor, h_cmdr[f], msgs_t, mask_t,
                    deterministic=True,
                    fw_neighbor_states=fw_neigh_t, fw_neighbor_mask=fw_mask_t)
                actions[f] = action

                # Track water valve
                if len(action) > 2 and action[2] > 0.5:
                    water_total += 1
                    # Check if water hit fire (will be reflected in extinguish stats)
                    # We track drop position for distance metric
                    fw_drop_altitudes.append(pos_f[2])
                    # Distance to nearest fire
                    min_d = float('inf')
                    for fx, fy in fire_positions:
                        d = np.hypot(pos_f[0] - fx, pos_f[1] - fy)
                        min_d = min(min_d, d)
                    fw_drop_distances.append(min_d)

            # Step environment
            obs, rewards, terms, truncs, infos = env.step(actions)
            metrics.steps_total = step + 1

            # Accumulate rewards
            for q in quad_agents:
                scout_rewards[q] += rewards.get(q, 0.0)
            for f in fixed_agents:
                fw_rewards[f] += rewards.get(f, 0.0)

            # Track water hits via extinguish stats
            for f in fixed_agents:
                eff = env.sim.drone_extinguish_stats.get(f, 0.0)
                if eff > 0.0:
                    water_hit += 1
                    if first_water_hit_step == -1:
                        first_water_hit_step = step

            # Detect refills
            for f in fixed_agents:
                d = env.sim.drones.get(f)
                if d and d.water_capacity > 0:
                    cur_w = d.current_water
                    if cur_w > fw_prev_water.get(f, 0) + 1.0:
                        fw_refill_count += 1
                    fw_prev_water[f] = cur_w
                    fw_water_end[f] = cur_w

            # Deaths
            for q in quad_agents:
                if terms.get(q, False) and scout_alive.get(q, True):
                    scout_alive[q] = False
                    scout_deaths += 1
            for f in fixed_agents:
                if terms.get(f, False) and fw_alive.get(f, True):
                    fw_alive[f] = False
                    fw_deaths += 1

        # ── Compute final metrics ────────────────────────────────────────
        fg = env.sim.environment.fire_grid
        final_burning = int(fg.B.sum()) if fg is not None else 0
        total_burned = int(ever_burned_mask.sum())
        water_miss = water_total - water_hit

        metrics.time_first_discovery = first_discovery_step
        metrics.time_all_fires_discovered = all_discovered_step
        metrics.time_first_water_drop = first_water_hit_step
        metrics.time_full_suppression = full_suppression_step

        metrics.peak_fire_cells = peak_burning
        metrics.final_burning_cells = final_burning
        metrics.total_burned_cells = total_burned
        metrics.suppression_rate_pct = round(
            (peak_burning - final_burning) / max(1, peak_burning) * 100.0, 1)

        metrics.water_drops_total = water_total
        metrics.water_drops_hit = water_hit
        metrics.water_drops_miss = water_miss
        metrics.water_accuracy_pct = round(
            water_hit / max(1, water_total) * 100.0, 1)

        # Water consumed
        total_consumed = 0.0
        for f in fixed_agents:
            start_w = fw_water_start.get(f, WATER_CAPACITY)
            end_w = fw_water_end.get(f, 0.0)
            refills_est = fw_refill_count  # approximate
            total_consumed += (start_w - end_w) + (fw_refill_count / max(1, n_fw)) * WATER_CAPACITY
        metrics.water_consumed_L = round(total_consumed, 1)
        metrics.refill_count = fw_refill_count

        # Scout metrics
        total_scout_fire = sum(scout_fire_steps.values())
        total_scout_steps = sum(scout_alt_count.values())
        metrics.scout_time_over_fire_steps = round(total_scout_fire / max(1, n_scouts), 1)
        metrics.scout_time_over_fire_pct = round(
            total_scout_fire / max(1, total_scout_steps) * 100.0, 1)

        alt_vals = [scout_alt_sum[q] / max(1, scout_alt_count[q])
                    for q in quad_agents if scout_alt_count[q] > 0]
        metrics.scout_mean_altitude_m = round(float(np.mean(alt_vals)) if alt_vals else 0.0, 1)

        # Scout separation (mean pairwise distance over episode — approximate from final positions)
        sep_vals = []
        for i, q1 in enumerate(quad_agents):
            d1 = env.sim.drones.get(q1)
            if d1 is None:
                continue
            p1 = d1.get_position()
            for q2 in quad_agents[i+1:]:
                d2 = env.sim.drones.get(q2)
                if d2 is None:
                    continue
                p2 = d2.get_position()
                sep_vals.append(np.hypot(p1[0]-p2[0], p1[1]-p2[1]))
        metrics.scout_mean_separation_m = round(float(np.mean(sep_vals)) if sep_vals else 0.0, 1)
        metrics.scout_deaths = scout_deaths

        # FW metrics
        metrics.fw_mean_drop_distance_m = round(
            float(np.mean(fw_drop_distances)) if fw_drop_distances else 0.0, 1)
        metrics.fw_mean_drop_altitude_m = round(
            float(np.mean(fw_drop_altitudes)) if fw_drop_altitudes else 0.0, 1)
        metrics.fw_deaths = fw_deaths

        # Per-fire
        metrics.per_fire_discovery_step = json.dumps(fire_discovery_step)
        metrics.per_fire_suppression_step = json.dumps(fire_suppression_step)

        # Rewards
        metrics.scout_avg_reward = round(
            sum(scout_rewards.values()) / max(1, n_scouts), 2)
        metrics.cmdr_avg_reward = round(
            sum(fw_rewards.values()) / max(1, n_fw), 2)

        env.sim.stop_simulation()

    except Exception as e:
        metrics.error = str(e)

    metrics.wallclock_s = round(time.time() - t0, 1)
    return asdict(metrics)


# ── Main ─────────────────────────────────────────────────────────────────────

def generate_configs(filter_scouts=None, filter_fw=None,
                     filter_fires=None, filter_maps=None):
    """Generate all (n_scouts, n_fw, n_fires, map_size, seed) tuples."""
    scouts = [filter_scouts] if filter_scouts else SCOUT_COUNTS
    fws = [filter_fw] if filter_fw else FW_COUNTS
    fires = [filter_fires] if filter_fires else FIRE_COUNTS
    maps = [filter_maps] if filter_maps else MAP_SIZES

    configs = []
    for ns in scouts:
        for nf in fws:
            for nfire in fires:
                for ms in maps:
                    for seed in range(N_SEEDS):
                        configs.append((ns, nf, nfire, ms, seed))
    return configs


def main():
    ap = argparse.ArgumentParser(description="Final scaling evaluation")
    ap.add_argument("--scout-model", required=True)
    ap.add_argument("--cmdr-model", required=True)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out-dir", default=os.path.join(PROJECT, "results", "eval_scaling"))
    ap.add_argument("--filter-scouts", type=int, default=None)
    ap.add_argument("--filter-fw", type=int, default=None)
    ap.add_argument("--filter-fires", type=int, default=None)
    ap.add_argument("--filter-maps", type=float, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Skip configs already in results.csv")
    args = ap.parse_args()

    scout_path = os.path.join(PROJECT, args.scout_model) if not os.path.isabs(args.scout_model) else args.scout_model
    cmdr_path = os.path.join(PROJECT, args.cmdr_model) if not os.path.isabs(args.cmdr_model) else args.cmdr_model

    assert os.path.exists(scout_path), f"Scout model not found: {scout_path}"
    assert os.path.exists(cmdr_path), f"Commander model not found: {cmdr_path}"

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "plots"), exist_ok=True)

    csv_path = os.path.join(args.out_dir, "results.csv")

    # Generate configs
    configs = generate_configs(
        filter_scouts=args.filter_scouts,
        filter_fw=args.filter_fw,
        filter_fires=args.filter_fires,
        filter_maps=args.filter_maps,
    )

    # Resume: skip already-done configs
    done_keys = set()
    if args.resume and os.path.exists(csv_path):
        import csv as csv_mod
        with open(csv_path, "r") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                key = (int(row["n_scouts"]), int(row["n_fw"]),
                       int(row["n_fires"]), float(row["map_size_m"]), int(row["seed"]))
                done_keys.add(key)
        print(f"[RESUME] Found {len(done_keys)} completed configs, skipping them.")

    configs = [c for c in configs if c not in done_keys]

    print(f"{'='*70}")
    print(f"  FINAL SCALING EVALUATION")
    print(f"  Scout model:  {scout_path}")
    print(f"  Cmdr model:   {cmdr_path}")
    print(f"  Configs:      {len(configs)} episodes to run")
    print(f"  Workers:      {args.workers}")
    print(f"  Max steps:    {MAX_STEPS}")
    print(f"  Output:       {csv_path}")
    print(f"{'='*70}")

    # Prepare worker args
    worker_args = [
        (scout_path, cmdr_path, ns, nf, nfire, ms, seed)
        for ns, nf, nfire, ms, seed in configs
    ]

    # CSV fields from dataclass
    sample = EpisodeMetrics()
    csv_fields = list(asdict(sample).keys())

    # Open CSV (append if resuming)
    write_header = not (args.resume and os.path.exists(csv_path))
    csvf = open(csv_path, "a" if args.resume else "w", newline="")
    writer = csv.DictWriter(csvf, fieldnames=csv_fields, extrasaction='ignore')
    if write_header:
        writer.writeheader()

    # Run episodes
    completed = 0
    errors = 0
    t_start = time.time()

    # Process in chunks to survive worker crashes (PyBullet segfaults)
    CHUNK_SIZE = args.workers * 4  # small enough to limit blast radius

    try:
        for chunk_start in range(0, len(worker_args), CHUNK_SIZE):
            chunk = worker_args[chunk_start:chunk_start + CHUNK_SIZE]

            with ProcessPoolExecutor(max_workers=args.workers,
                                     max_tasks_per_child=1) as executor:
                futures = {executor.submit(run_single_episode, wa): wa for wa in chunk}

                for future in as_completed(futures):
                    completed += 1
                    wa = futures[future]
                    try:
                        result = future.result()
                        writer.writerow(result)
                        csvf.flush()

                        if result.get("error"):
                            errors += 1
                            print(f"  [{completed}/{len(configs)}] ERROR: sc={wa[2]} fw={wa[3]} "
                                  f"fires={wa[4]} map={wa[5]} seed={wa[6]} — {result['error']}")
                        else:
                            # Progress print every 50 episodes
                            if completed % 50 == 0 or completed <= 5:
                                elapsed = time.time() - t_start
                                eta = elapsed / completed * (len(configs) - completed)
                                supp = result['suppression_rate_pct']
                                t_disc = result['time_first_discovery']
                                t_supp = result['time_full_suppression']
                                print(f"  [{completed:>5}/{len(configs)}] "
                                      f"sc={result['n_scouts']} fw={result['n_fw']} "
                                      f"fires={result['n_fires']} map={int(result['map_size_m'])}m "
                                      f"| supp={supp}% disc@{t_disc} full@{t_supp} "
                                      f"| {result['wallclock_s']:.0f}s "
                                      f"| ETA: {eta/60:.0f}min")
                    except Exception as e:
                        errors += 1
                        # Write a row with error so --resume skips it
                        err_metrics = EpisodeMetrics(
                            n_scouts=wa[2], n_fw=wa[3], n_fires=wa[4],
                            map_size_m=wa[5], seed=wa[6],
                            error=str(e)[:200])
                        writer.writerow(asdict(err_metrics))
                        csvf.flush()
                        if completed % 50 == 0 or completed <= 5:
                            print(f"  [{completed}/{len(configs)}] EXCEPTION: {e}")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Partial results saved.")
    finally:
        csvf.close()

    elapsed_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  DONE: {completed} episodes in {elapsed_total/60:.1f} min")
    print(f"  Errors: {errors}")
    print(f"  Results: {csv_path}")
    print(f"{'='*70}")

    # Generate summary
    if completed > 0:
        generate_summary(csv_path, args.out_dir)


def generate_summary(csv_path, out_dir):
    """Aggregate results into summary statistics per configuration."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    # Group by configuration
    group_cols = ["n_scouts", "n_fw", "n_fires", "map_size_m"]
    agg_dict = {
        "time_first_discovery": ["mean", "std"],
        "time_all_fires_discovered": ["mean", "std"],
        "time_first_water_drop": ["mean", "std"],
        "time_full_suppression": ["mean", "std"],
        "peak_fire_cells": ["mean", "std"],
        "final_burning_cells": ["mean", "std"],
        "total_burned_cells": ["mean", "std"],
        "suppression_rate_pct": ["mean", "std", "min"],
        "water_drops_total": ["mean"],
        "water_drops_hit": ["mean"],
        "water_accuracy_pct": ["mean", "std"],
        "water_consumed_L": ["mean"],
        "refill_count": ["mean"],
        "scout_time_over_fire_pct": ["mean"],
        "scout_mean_altitude_m": ["mean"],
        "scout_mean_separation_m": ["mean"],
        "scout_deaths": ["sum", "mean"],
        "fw_mean_drop_distance_m": ["mean"],
        "fw_mean_drop_altitude_m": ["mean"],
        "fw_deaths": ["sum", "mean"],
        "scout_avg_reward": ["mean"],
        "cmdr_avg_reward": ["mean"],
        "wallclock_s": ["mean"],
    }

    summary = df.groupby(group_cols).agg(agg_dict).round(2)
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns]

    # Add success rate (% episodes where fire was fully suppressed)
    success = df.copy()
    success["suppressed"] = (success["time_full_suppression"] > 0).astype(int)
    success_rate = success.groupby(group_cols)["suppressed"].mean().round(3) * 100
    summary["success_rate_pct"] = success_rate

    summary_path = os.path.join(out_dir, "summary.csv")
    summary.to_csv(summary_path)
    print(f"  Summary: {summary_path}")

    # Print top-level stats
    print(f"\n  Overall success rate: {success['suppressed'].mean()*100:.1f}%")
    print(f"  Mean time to suppression: {df[df['time_full_suppression']>0]['time_full_suppression'].mean():.0f} steps")
    print(f"  Mean water accuracy: {df['water_accuracy_pct'].mean():.1f}%")


if __name__ == "__main__":
    main()
