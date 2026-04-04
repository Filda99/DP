"""
demo_both_training.py
─────────────────────
Vizualizační demo pro heterogenní MAPPO tým (Scout + Commander).

Layout každého snímku GIFu:
┌──────────────────────────┬──────────────────┐
│                          │  Scout kamera    │
│   Globální mapa + oheň   │  (co 32×32 vidí) │
│   Trajektorie obou       ├──────────────────┤
│   FoV rectangle scoutye  │  Stats           │
└──────────────────────────┴──────────────────┘

Výstup: demo_training.gif  +  demo_training_analysis.png
"""

import torch
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio
import io, tqdm
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.env_core import DroneFireEnv
from src.models import ScoutActor, CommanderActor

# ============================================================
# KONFIGURACE
# ============================================================
MODEL_SCOUT     = os.path.join(project_root, "saved_models", "multi", "scout_best.pt")
MODEL_COMMANDER = os.path.join(project_root, "saved_models", "multi", "cmdr_best.pt")

N_QUADS    = 1
N_FIXED    = 1
MAX_STEPS  = 2000
GRID_SIZE  = 1000.0
GIF_EVERY  = 3
GIF_FPS    = 15
EPISODE_SEED = 101

# Commander waypoint parameters (must match training)
WAYPOINT_RANGE  = 100.0   # metres per unit of dx/dy
WAYPOINT_STEPS  = 50      # physics steps per waypoint segment
WP_REACHED_DIST = 30.0    # metres
SAFE_LIMIT_BUF  = 250.0   # boundary buffer
# ============================================================


def _wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _load_models(device):
    env_tmp = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED,
                           grid_size_m=GRID_SIZE, max_steps=MAX_STEPS)
    scout_self_dim = env_tmp.observation_space("quad_0")["self_state"].shape[0]
    fixed_self_dim = env_tmp.observation_space(env_tmp.fixed_agents[0])["self_state"].shape[0]

    scout = ScoutActor(self_state_dim=scout_self_dim, msg_dim=5, hidden_dim=128).to(device)
    cmdr  = CommanderActor(self_state_dim=fixed_self_dim, msg_input_dim=5,
                           action_dim=4, hidden_dim=64).to(device)

    for path, name, model in [(MODEL_SCOUT, "Scout", scout),
                               (MODEL_COMMANDER, "Commander", cmdr)]:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"✅ {name}: {path}")
        else:
            print(f"⚠️ {name}: model nenalezen ({path})")
    
    scout.eval()
    cmdr.eval()
    return scout, cmdr, scout_self_dim, fixed_self_dim

def _render_frame(step, fire_map, b,
                  q_path_x, q_path_y, q_pos, q_fov,
                  f_path_x, f_path_y, f_pos, f_water_pct,
                  refill_pos, refill_size,
                  local_map_np,
                  total_reward_q, total_reward_f,
                  fire_seen_sum, q_alive, f_alive):

    fig = plt.figure(figsize=(12, 6), facecolor='#1a1a1a')
    gs  = gridspec.GridSpec(2, 2, width_ratios=[2, 1], hspace=0.35, wspace=0.25,
                            left=0.06, right=0.97, top=0.93, bottom=0.07)

    # ── Panel 1: Globální mapa ──────────────────────────────────────────────
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor('#111111')

    # Oheň
    extent = [-b, b, -b, b]
    fire_masked = np.ma.masked_where(fire_map < 0.01, fire_map)
    ax_map.imshow(fire_masked, extent=extent, origin='lower',
                  cmap='YlOrRd', vmin=0, vmax=1.0, alpha=0.9)

    # Refill Zóna (Visualizace)
    if refill_pos is not None:
        # Vykreslíme Refill zónu jako azurový kruh (zvětšený pro viditelnost)
        circle = plt.Circle((refill_pos[0], refill_pos[1]), 150,  # 150m = skutečný detekční rádius

                            color='cyan', fill=False, linestyle='--', linewidth=1.5, alpha=0.6)
        ax_map.add_patch(circle)
        ax_map.text(refill_pos[0], refill_pos[1] + 60, "REFILL", color='cyan', 
                    fontsize=8, ha='center', fontweight='bold', alpha=0.8)

    # Trajektorie
    if len(q_path_x) > 1:
        ax_map.plot(q_path_x, q_path_y, color='cyan',  alpha=0.4, linewidth=1.2, linestyle=':')
    if len(f_path_x) > 1:
        ax_map.plot(f_path_x, f_path_y, color='tomato', alpha=0.4, linewidth=1.2, linestyle=':')

    # Pozice agentů
    if q_alive and q_pos is not None:
        rect = plt.Rectangle((q_pos[0] - q_fov/2, q_pos[1] - q_fov/2), q_fov, q_fov,
                              fill=False, edgecolor='cyan', linewidth=1.0, alpha=0.7)
        ax_map.add_patch(rect)
        ax_map.scatter(q_pos[0], q_pos[1], c='cyan', s=100, marker='^', edgecolors='white', zorder=5)

    if f_alive and f_pos is not None:
        ax_map.scatter(f_pos[0], f_pos[1], c='tomato', s=130, marker='>', edgecolors='white', zorder=5)

    ax_map.set_xlim(-b, b); ax_map.set_ylim(-b, b)
    ax_map.set_title(f"Krok {step:04d} | Globální přehled", color='white', fontsize=11)
    ax_map.tick_params(colors='#888888', labelsize=7)

    # ── Panel 2: Scout kamera ───────────────────────────────────────────────
    ax_cam = fig.add_subplot(gs[0, 1])
    ax_cam.set_facecolor('#0a0a0a')
    if local_map_np is not None:
        ax_cam.imshow(local_map_np, origin='lower', cmap='YlOrRd', vmin=0, vmax=1.0)
    ax_cam.set_title("Scout kamera (32×32)", color='white', fontsize=9)
    ax_cam.set_xticks([]); ax_cam.set_yticks([])

    # ── Panel 3: Stats + Water Bar ──────────────────────────────────────────
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.set_facecolor('#111111')
    ax_stats.set_xticks([]); ax_stats.set_yticks([])

    # Vizuální bar pro vodu
    ax_stats.add_patch(plt.Rectangle((0.1, 0.15), 0.8, 0.08, color='#333333', transform=ax_stats.transAxes))
    ax_stats.add_patch(plt.Rectangle((0.1, 0.15), 0.8 * f_water_pct, 0.08, color='cyan', transform=ax_stats.transAxes))
    
    stats_text = (
        f"Scout reward:      {total_reward_q:+.1f}\n"
        f"Commander reward:  {total_reward_f:+.1f}\n\n"
        f"Fire viditelnost:  {fire_seen_sum:.2f}\n"
        f"Voda v nádrži:     {f_water_pct*100:3.1f}%\n\n"
        f"Stav: {'NÁDRŽ PLNÁ' if f_water_pct > 0.9 else 'REFILL POTŘEBNÝ' if f_water_pct < 0.2 else 'OPERATIVNÍ'}"
    )
    ax_stats.text(0.08, 0.9, stats_text, color='#dddddd', fontsize=8.5, va='top', transform=ax_stats.transAxes, fontfamily='monospace')
    ax_stats.set_title("Statistiky mise", color='white', fontsize=9)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor())
    buf.seek(0)
    arr = np.array(Image.open(buf))
    plt.close(fig)
    return arr

def run_demo():
    print("🎬 Demo: Heterogenní MAPPO tým (Scout + Commander)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scout_actor, commander_actor, _, _ = _load_models(device)

    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=GRID_SIZE, max_steps=MAX_STEPS)
    obs, _ = env.reset(seed=EPISODE_SEED, epizode_number=10000)
    
    refill_info = env.sim.environment.refill_zone
    refill_pos = refill_info['position'] if refill_info else None
    refill_size = refill_info['size'] if refill_info else 20.0

    safe_limit = env.map_bounds - SAFE_LIMIT_BUF
    boundary_emergency = env.map_bounds - 100.0

    h_scout = torch.zeros(1, 1, 128).to(device)
    h_cmdr  = torch.zeros(1, 1, 64).to(device)

    hist = { "q_r": [], "f_r": [], "q_alt": [], "f_alt": [], "fire": [], "water": [] }
    frames, total_rq, total_rf = [], 0.0, 0.0
    q_path_x, q_path_y, f_path_x, f_path_y = [], [], [], []
    last_local_map = None

    # Commander waypoint state
    need_new_waypoint = True
    target_x, target_y = 0.0, 0.0
    target_alt_raw, water_raw = 0.0, -0.5
    steps_in_segment = 0
    last_scout_msg = torch.zeros(1, 1, 5).to(device)
    scout_msg_valid = False

    print("🚀 Mise začíná...")
    for step in tqdm.tqdm(range(MAX_STEPS)):
        if not env.agents: break
        actions = {}

        # ── Scout ────────────────────────────────────────────────────────
        if "quad_0" in env.agents:
            l_map = torch.FloatTensor(obs["quad_0"]["local_map"]).to(device).unsqueeze(0)
            s_st  = torch.FloatTensor(obs["quad_0"]["self_state"]).to(device).unsqueeze(0)
            n_st  = torch.FloatTensor(obs["quad_0"]["neighbor_states"]).to(device).unsqueeze(0)
            n_m   = torch.BoolTensor(obs["quad_0"]["neighbor_mask"]).to(device).unsqueeze(0)
            with torch.no_grad():
                dist, scout_msg, h_scout = scout_actor(l_map, s_st, n_st, n_m, h_scout)
            actions["quad_0"] = dist.mean.squeeze(0).cpu().numpy()
            last_local_map = obs["quad_0"]["local_map"][0]
            last_scout_msg = scout_msg.unsqueeze(1)  # [1, 5] → [1, 1, 5]
            scout_msg_valid = True
        else:
            scout_msg_valid = False

        # ── Commander (waypoint mode) ────────────────────────────────────
        if "fixed_0" in env.agents:
            # Check if waypoint reached or timeout → new decision
            drone = env.sim.drones.get("fixed_0")
            if drone is not None:
                pos = drone.get_position()
                dx_to = target_x - pos[0]
                dy_to = target_y - pos[1]
                dist_to = np.sqrt(dx_to**2 + dy_to**2)
                if dist_to < WP_REACHED_DIST:
                    need_new_waypoint = True

            if steps_in_segment >= WAYPOINT_STEPS:
                need_new_waypoint = True

            if need_new_waypoint:
                s_st_f = torch.FloatTensor(obs["fixed_0"]["self_state"]).to(device).unsqueeze(0)
                # Build message tensor for commander (latest + best-fire slots)
                msgs_t = torch.cat([last_scout_msg, last_scout_msg], dim=1)  # [1, 2, 5]
                msgs_m = torch.tensor([[not scout_msg_valid, not scout_msg_valid]])  # [1, 2]
                with torch.no_grad():
                    dist_c, _, h_cmdr = commander_actor(s_st_f, msgs_t, msgs_m, h_cmdr)
                act_np = dist_c.mean.squeeze(0).cpu().numpy()
                dx_raw = float(act_np[0])
                dy_raw = float(act_np[1])
                target_alt_raw = float(act_np[2])
                water_raw = float(act_np[3])

                cur_pos = drone.get_position() if drone else np.zeros(3)
                target_x = np.clip(cur_pos[0] + dx_raw * WAYPOINT_RANGE, -safe_limit, safe_limit)
                target_y = np.clip(cur_pos[1] + dy_raw * WAYPOINT_RANGE, -safe_limit, safe_limit)
                steps_in_segment = 0
                need_new_waypoint = False

            # Heading controller → physical action
            if drone is not None:
                pos = drone.get_position()
                # Emergency boundary override
                if abs(pos[0]) > boundary_emergency or abs(pos[1]) > boundary_emergency:
                    target_x = 0.0
                    target_y = 0.0
                    need_new_waypoint = True
                dx_to = target_x - pos[0]
                dy_to = target_y - pos[1]
                dist_to = np.sqrt(dx_to**2 + dy_to**2)
                if dist_to > 1.0:
                    desired_heading = np.arctan2(dy_to, dx_to)
                    cur_yaw = drone.get_orientation_rpy()[2]
                    heading_error = _wrap_angle(desired_heading - cur_yaw)
                    heading_cmd = np.clip(heading_error / np.pi, -1.0, 1.0)
                else:
                    heading_cmd = 0.0
                actions["fixed_0"] = np.array(
                    [heading_cmd, target_alt_raw, water_raw], dtype=np.float32)

            steps_in_segment += 1

        obs, rewards, _, _, _ = env.step(actions)
        total_rq += rewards.get("quad_0", 0.0)
        total_rf += rewards.get("fixed_0", 0.0)

        # Sběr dat pro vizualizaci
        f_water_pct = 0.0
        q_alive = "quad_0" in env.sim.drones
        f_alive = "fixed_0" in env.sim.drones
        q_pos, f_pos = None, None

        if q_alive:
            q_pos = env.sim.drones["quad_0"].get_position()
            q_path_x.append(q_pos[0]); q_path_y.append(q_pos[1])
        if f_alive:
            f_pos = env.sim.drones["fixed_0"].get_position()
            f_path_x.append(f_pos[0]); f_path_y.append(f_pos[1])
            f_water_pct = env.sim.drones["fixed_0"].current_water / env.sim.drones["fixed_0"].water_capacity

        fire_seen = float(np.sum(last_local_map)) if last_local_map is not None else 0.0
        hist["q_r"].append(rewards.get("quad_0", 0.0))
        hist["f_r"].append(rewards.get("fixed_0", 0.0))
        hist["fire"].append(fire_seen)
        hist["water"].append(f_water_pct)
        if q_alive: hist["q_alt"].append(q_pos[2])
        if f_alive: hist["f_alt"].append(f_pos[2])

        if step % GIF_EVERY == 0:
            frame = _render_frame(
                step, env.sim.environment.fire_grid.I.copy(), env.map_bounds,
                list(q_path_x), list(q_path_y), q_pos, 40.0,
                list(f_path_x), list(f_path_y), f_pos, f_water_pct,
                refill_pos, refill_size,
                last_local_map.copy() if last_local_map is not None else None,
                total_rq, total_rf, fire_seen, q_alive, f_alive
            )
            frames.append(frame)

    imageio.mimsave(os.path.join(project_root, "demo_training.gif"), frames, fps=GIF_FPS, loop=0)
    _save_analysis(hist, project_root)

def _save_analysis(hist, project_root):
    fig, axes = plt.subplots(6, 1, figsize=(12, 16)) # Zvětšeno na 6 grafů
    axes[0].plot(np.cumsum(hist["q_r"]), label="Scout", color='cyan')
    axes[0].plot(np.cumsum(hist["f_r"]), label="Commander", color='tomato')
    axes[0].set_title("Kumulativní odměna"); axes[0].legend()
    
    axes[1].plot(hist["q_alt"], color='cyan', label="Scout Alt")
    axes[1].plot(hist["f_alt"], color='tomato', label="Commander Alt")
    axes[1].set_title("Výška (m)"); axes[1].legend()

    axes[2].fill_between(range(len(hist["fire"])), hist["fire"], color='orange', alpha=0.5)
    axes[2].set_title("Intenzita ohně pod Scoutem")

    # NOVÝ GRAF: Hladina vody v čase
    axes[3].plot(hist["water"], color='deepskyblue', linewidth=2)
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_title("Hladina vody v nádrži (0.0 - 1.0)")
    axes[3].grid(alpha=0.3)

    axes[4].plot(hist["q_r"], color='cyan', alpha=0.4, label="Scout")
    axes[4].set_title("Odměna Scout per krok")

    axes[5].plot(hist["f_r"], color='tomato', alpha=0.4, label="Commander")
    axes[5].set_title("Odměna Commander per krok")

    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "demo_training_analysis.png"), dpi=120)
    print("✅ Analýza uložena!")

if __name__ == "__main__":
    run_demo()