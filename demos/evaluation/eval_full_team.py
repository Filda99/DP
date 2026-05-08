"""
eval_full_team.py
─────────────────
Quantitative evaluation of the heterogeneous MAPPO team across multiple
configurations (map sizes, scout counts). No GIFs — pure metrics + CSV.

Produces the thesis result table:
  Config | Map  | Scouts | Discovery% | Dwell% | Supp% | Refills | R_cmdr | Failures
  S1C1   | 800m |   1    |    X±σ    |  X±σ  |  X±σ  |  X±σ   |  X±σ  |  K/N

Usage (from project root):
  python demos/evaluation/eval_full_team.py \\
      --scout  saved_models/finetune_07/scout_best.pt \\
      --cmdr   saved_models/finetune_07/cmdr_best.pt \\
      --episodes 50
"""

import os, sys, csv, argparse
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.env_core import DroneFireEnv
from src.models import ScoutActor, CommanderActor

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_SCOUT = os.path.join(project_root, "saved_models", "finetune_07", "scout_best.pt")
DEFAULT_CMDR  = os.path.join(project_root, "saved_models", "finetune_07", "cmdr_best.pt")

# Commander waypoint params — must match training
WAYPOINT_RANGE  = 200.0
WAYPOINT_STEPS  = 30
WP_REACHED_DIST = 30.0
MAX_STEPS       = 1000

def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _load_models(scout_path, cmdr_path, n_quads, device):
    env_tmp = DroneFireEnv(num_quads=n_quads, num_fixed=1, grid_size_m=1200.0, max_steps=100)
    scout_dim = env_tmp.observation_space("quad_0")["self_state"].shape[0]
    fixed_dim = env_tmp.observation_space(env_tmp.fixed_agents[0])["self_state"].shape[0]
    env_tmp.sim.stop_simulation()

    scout = ScoutActor(self_state_dim=scout_dim, msg_dim=5, hidden_dim=128).to(device)
    cmdr  = CommanderActor(self_state_dim=fixed_dim, msg_input_dim=5,
                           action_dim=4, hidden_dim=64).to(device)

    scout.load_state_dict(torch.load(scout_path, map_location=device), strict=False)
    cmdr.load_state_dict(torch.load(cmdr_path,  map_location=device), strict=False)
    scout.eval(); cmdr.eval()
    return scout, cmdr


def run_episode(env, scout_actor, cmdr_actor, seed, ep_num, n_quads, device):
    """Run one headless episode. Returns metrics dict."""
    quad_names = [f"quad_{i}" for i in range(n_quads)]
    hidden_scout = 128
    hidden_cmdr  = 64

    obs, _ = env.reset(seed=seed, epizode_number=ep_num)

    h_scout = {q: torch.zeros(1, 1, hidden_scout).to(device) for q in quad_names}
    h_cmdr  = torch.zeros(1, 1, hidden_cmdr).to(device)

    # Commander waypoint state
    need_new_wp   = True
    target_x, target_y = 0.0, 0.0
    target_alt_raw, water_raw = 0.0, -0.5
    steps_in_seg  = 0
    scout_msgs    = {q: {"latest": torch.zeros(1, 5).to(device), "valid": False}
                     for q in quad_names}

    # Metric accumulators
    total_rf          = 0.0
    fire_steps        = 0          # steps where ≥1 scout saw fire
    dwell_steps       = 0          # steps where FW was over fire (extinguish_stats > 0)
    refill_count      = 0
    water_prev        = 1.0
    fire_cells_peak   = 0
    safe_limit        = max(50.0, env.map_bounds * 0.7)

    for step in range(MAX_STEPS):
        if not env.agents:
            break

        if env.sim.environment.fire_grid is not None:
            fire_cells_peak = max(fire_cells_peak,
                                  int(np.sum(env.sim.environment.fire_grid.B)))

        actions = {}

        # ── Scout forward pass ─────────────────────────────────────────
        for q in quad_names:
            if q in env.agents:
                l_map = torch.FloatTensor(obs[q]["local_map"]).to(device).unsqueeze(0)
                s_st  = torch.FloatTensor(obs[q]["self_state"]).to(device).unsqueeze(0)
                n_st  = torch.FloatTensor(obs[q]["neighbor_states"]).to(device).unsqueeze(0)
                n_m   = torch.BoolTensor(obs[q]["neighbor_mask"]).to(device).unsqueeze(0)
                with torch.no_grad():
                    dist, scout_msg, h_scout[q] = scout_actor(l_map, s_st, n_st, n_m, h_scout[q])
                actions[q] = dist.sample().squeeze(0).cpu().numpy()
                scout_msgs[q]["latest"] = scout_msg
                scout_msgs[q]["valid"]  = True
                # Fire visibility metric
                if float(scout_msg[0, 2]) > 0.01:
                    fire_steps = step  # at least one scout saw fire this step

        # Fire coverage: fraction of steps with any scout seeing fire
        any_fire_seen = any(
            float(scout_msgs[q]["latest"][0, 2]) > 0.01
            for q in quad_names if scout_msgs[q]["valid"]
        )

        # ── Commander forward pass ─────────────────────────────────────
        if "fixed_0" in env.agents:
            drone = env.sim.drones.get("fixed_0")
            if drone is not None:
                pos = drone.get_position()

                # Boundary emergency
                in_emergency = (abs(pos[0]) > env.map_bounds * 0.6 or
                                abs(pos[1]) > env.map_bounds * 0.6)
                if in_emergency:
                    target_x, target_y = 0.0, 0.0

                if not in_emergency:
                    dx = target_x - pos[0]; dy = target_y - pos[1]
                    if np.hypot(dx, dy) < WP_REACHED_DIST or steps_in_seg >= WAYPOINT_STEPS:
                        need_new_wp = True

                if need_new_wp and not in_emergency:
                    s_st_f = torch.FloatTensor(obs["fixed_0"]["self_state"]).to(device).unsqueeze(0)
                    msgs_t = torch.stack([scout_msgs[q]["latest"] for q in quad_names], dim=1).to(device)
                    msgs_m = torch.tensor([[not scout_msgs[q]["valid"] for q in quad_names]],
                                          dtype=torch.bool).to(device)
                    with torch.no_grad():
                        dist_c, _, h_cmdr = cmdr_actor(s_st_f, msgs_t, msgs_m, h_cmdr)
                    act = dist_c.mean.squeeze(0).cpu().numpy()
                    dx_raw = float(act[0]); dy_raw = float(act[1])
                    target_alt_raw = float(act[2]); water_raw = float(act[3])
                    target_x = float(np.clip(pos[0] + dx_raw * WAYPOINT_RANGE, -safe_limit, safe_limit))
                    target_y = float(np.clip(pos[1] + dy_raw * WAYPOINT_RANGE, -safe_limit, safe_limit))
                    steps_in_seg = 0; need_new_wp = False

                # Heading controller
                dx_to = target_x - pos[0]; dy_to = target_y - pos[1]
                dist_to = np.hypot(dx_to, dy_to)
                if dist_to > 1.0:
                    desired = np.arctan2(dy_to, dx_to)
                    cur_yaw = drone.get_orientation_rpy()[2]
                    heading_cmd = float(np.clip(_wrap_angle(desired - cur_yaw) / np.pi, -1, 1))
                else:
                    heading_cmd = 0.0
                actions["fixed_0"] = np.array([heading_cmd, target_alt_raw, water_raw], np.float32)
                steps_in_seg += 1

        obs, rewards, _, _, _ = env.step(actions)
        total_rf += rewards.get("fixed_0", 0.0)

        # Dwell: FW dropped water on burning cells
        if env.sim.drone_extinguish_stats.get("fixed_0", 0.0) > 0.0:
            dwell_steps += 1

        # Refill detection
        fw_drone = env.sim.drones.get("fixed_0")
        if fw_drone and fw_drone.water_capacity > 0:
            water_now = fw_drone.current_water / fw_drone.water_capacity
            if water_now > water_prev + 0.05:
                refill_count += 1
            water_prev = water_now

    # ── Final metrics ──────────────────────────────────────────────────
    end_cells = (int(np.sum(env.sim.environment.fire_grid.B))
                 if env.sim.environment.fire_grid is not None else 0)
    supp_pct = (1.0 - end_cells / fire_cells_peak) * 100.0 if fire_cells_peak > 0 else 100.0

    fw_survived    = "fixed_0" in env.sim.drones
    scouts_survived = sum(1 for q in quad_names if q in env.sim.drones)

    # Discovery: did any scout ever see fire?
    fire_discovered = any(scout_msgs[q]["latest"][0, 2].item() > 0.01 for q in quad_names)

    # Dwell% = fraction of steps FW spent actively suppressing
    dwell_pct = dwell_steps / MAX_STEPS * 100.0

    # Water used
    fw_d = env.sim.drones.get("fixed_0")
    water_used_pct = 0.0
    if fw_d and fw_d.water_capacity > 0:
        water_used_pct = (1.0 - fw_d.current_water / fw_d.water_capacity) * 100.0

    # Failure: FW died early (before step 900) OR scouts never found fire
    failure = (not fw_survived) or (not fire_discovered)

    return {
        "seed":             seed,
        "fw_survived":      fw_survived,
        "scouts_survived":  scouts_survived,
        "fire_discovered":  fire_discovered,
        "dwell_pct":        round(dwell_pct, 2),
        "supp_pct":         round(supp_pct, 2),
        "refill_count":     refill_count,
        "total_rf":         round(total_rf, 2),
        "water_used_pct":   round(water_used_pct, 1),
        "failure":          failure,
        "peak_cells":       fire_cells_peak,
        "end_cells":        end_cells,
    }


def evaluate_config(scout_path, cmdr_path, n_quads, grid_size, episodes, seed_start, device):
    """Evaluate one (n_quads, grid_size) configuration over N episodes."""
    tag = f"S{n_quads}C1_{int(grid_size)}m"
    print(f"\n{'='*60}")
    print(f"  Config: {tag}  ({episodes} episodes, seed {seed_start}–{seed_start+episodes-1})")
    print(f"{'='*60}")

    scout_actor, cmdr_actor = _load_models(scout_path, cmdr_path, n_quads, device)

    env = DroneFireEnv(num_quads=n_quads, num_fixed=1, grid_size_m=grid_size,
                       max_steps=MAX_STEPS)

    results = []
    for i in range(episodes):
        seed = seed_start + i
        ep_num = 30000  # full difficulty
        r = run_episode(env, scout_actor, cmdr_actor, seed, ep_num, n_quads, device)
        r["config"] = tag
        r["n_quads"] = n_quads
        r["grid_size"] = grid_size
        results.append(r)

        fw_icon = "✓" if r["fw_survived"] else "✗"
        disc    = "✓" if r["fire_discovered"] else "✗"
        print(f"  [{i+1:3d}/{episodes}] seed={seed:4d} | "
              f"FW={fw_icon} disc={disc} | "
              f"dwell={r['dwell_pct']:5.1f}% supp={r['supp_pct']:5.1f}% "
              f"refill={r['refill_count']} R={r['total_rf']:+7.1f}")

    env.sim.stop_simulation()
    return results


def _print_summary_table(all_results):
    """Print mean±std table grouped by config."""
    configs = sorted(set(r["config"] for r in all_results))
    print(f"\n{'─'*100}")
    print(f"{'Config':>10} {'Map':>6} {'N':>2} {'Disc%':>6} {'Dwell%':>9} {'Supp%':>9} "
          f"{'Refills':>9} {'R_cmdr':>10} {'FW_surv%':>9} {'Failures':>9}")
    print(f"{'─'*100}")
    for cfg in configs:
        sub = [r for r in all_results if r["config"] == cfg]
        n   = len(sub)
        disc_pct   = np.mean([r["fire_discovered"]  for r in sub]) * 100
        dwell      = [r["dwell_pct"]   for r in sub]
        supp       = [r["supp_pct"]    for r in sub]
        refills    = [r["refill_count"] for r in sub]
        rewards    = [r["total_rf"]    for r in sub]
        fw_surv    = np.mean([r["fw_survived"] for r in sub]) * 100
        failures   = sum(1 for r in sub if r["failure"])
        gs         = sub[0]["grid_size"]
        nq         = sub[0]["n_quads"]
        print(f"{cfg:>10} {int(gs):>5}m {nq:>2} "
              f"{disc_pct:>5.1f}% "
              f"{np.mean(dwell):>5.1f}±{np.std(dwell):>4.1f}% "
              f"{np.mean(supp):>5.1f}±{np.std(supp):>4.1f}% "
              f"{np.mean(refills):>4.1f}±{np.std(refills):>3.1f}  "
              f"{np.mean(rewards):>+7.1f}±{np.std(rewards):>5.1f}  "
              f"{fw_surv:>6.1f}%  "
              f"{failures:>3d}/{n}")
    print(f"{'─'*100}")


def _save_csv(all_results, out_path):
    if not all_results:
        return
    fieldnames = list(all_results[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_results)
    print(f"\nCSV → {out_path}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-team quantitative evaluation")
    parser.add_argument("--scout",       type=str, default=DEFAULT_SCOUT)
    parser.add_argument("--cmdr",        type=str, default=DEFAULT_CMDR)
    parser.add_argument("--episodes",    type=int, default=50,
                        help="Episodes per configuration")
    parser.add_argument("--seed-start",  type=int, default=1000,
                        help="First seed (each config uses seeds seed_start…seed_start+episodes-1)")
    parser.add_argument("--scouts",      type=int, nargs="+", default=[1, 2],
                        help="Scout counts to evaluate (default: 1 2)")
    parser.add_argument("--map-sizes",   type=float, nargs="+", default=[800.0, 1200.0, 2000.0],
                        help="Map sizes in metres (default: 800 1200 2000)")
    parser.add_argument("--csv",         type=str,
                        default=os.path.join(project_root, "results", "eval_full_team.csv"),
                        help="Output CSV path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Scout:  {args.scout}")
    print(f"Cmdr:   {args.cmdr}")
    print(f"Configs: scouts={args.scouts}  maps={args.map_sizes}  eps={args.episodes}")

    all_results = []
    for n_q in args.scouts:
        for gs in args.map_sizes:
            res = evaluate_config(
                scout_path=args.scout,
                cmdr_path=args.cmdr,
                n_quads=n_q,
                grid_size=gs,
                episodes=args.episodes,
                seed_start=args.seed_start,
                device=device,
            )
            all_results.extend(res)

    _print_summary_table(all_results)

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    _save_csv(all_results, args.csv)
