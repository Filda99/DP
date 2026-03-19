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
import io
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.env_core import DroneFireEnv
from src.models import ScoutActor, CommanderActor

# ============================================================
# KONFIGURACE  — uprav cesty k modelům
# ============================================================
MODEL_SCOUT     = os.path.join(project_root, "saved_models", "scout_ep6000.pt")
MODEL_COMMANDER = os.path.join(project_root, "saved_models", "commander_ep6000.pt")

N_QUADS    = 1
N_FIXED    = 1
MAX_STEPS  = 1000
GRID_SIZE  = 2000.0    # musí odpovídat grid_size_m z worker_config v train.py
GIF_EVERY  = 3        # každý N-tý krok uložit snímek (menší = hladší gif, větší soubor)
GIF_FPS    = 15
EPISODE_SEED = 12
# ============================================================


def _load_models(device):
    """Načtení sítí ze souborů."""
    env_tmp = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED,
                           grid_size_m=GRID_SIZE, max_steps=MAX_STEPS)
    scout_self_dim = env_tmp.observation_space("quad_0")["self_state"].shape[0]
    fixed_self_dim = env_tmp.observation_space(env_tmp.fixed_agents[0])["self_state"].shape[0]
    # env_tmp.sim.stop_simulation()

    scout = ScoutActor(self_state_dim=scout_self_dim, msg_dim=5, hidden_dim=128).to(device)
    cmdr  = CommanderActor(self_state_dim=fixed_self_dim, msg_input_dim=5).to(device)

    for path, name, model in [(MODEL_SCOUT, "Scout", scout),
                               (MODEL_COMMANDER, "Commander", cmdr)]:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"✅  {name}: {path}")
        else:
            print(f"⚠️   {name}: model nenalezen ({path}) — spouštím s náhodnou politikou")
    scout.eval()
    cmdr.eval()
    return scout, cmdr, scout_self_dim, fixed_self_dim


def _render_frame(step, fire_map, b,
                  q_path_x, q_path_y, q_pos, q_fov,
                  f_path_x, f_path_y, f_pos,
                  local_map_np,
                  total_reward_q, total_reward_f,
                  fire_seen_sum, q_alive, f_alive):
    """Sestaví jeden snímek a vrátí ho jako numpy array."""

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

    # Trajektorie
    if len(q_path_x) > 1:
        ax_map.plot(q_path_x, q_path_y, color='cyan',  alpha=0.5, linewidth=1.5, linestyle=':')
    if len(f_path_x) > 1:
        ax_map.plot(f_path_x, f_path_y, color='tomato', alpha=0.5, linewidth=1.5, linestyle=':')

    # FoV rectangle scoutye
    if q_alive and q_pos is not None:
        fov = q_fov
        rect = plt.Rectangle((q_pos[0] - fov/2, q_pos[1] - fov/2), fov, fov,
                              fill=False, edgecolor='cyan', linewidth=1.2, linestyle='-', alpha=0.8)
        ax_map.add_patch(rect)
        ax_map.scatter(q_pos[0], q_pos[1], c='cyan',  s=120, marker='^',
                       edgecolors='white', linewidths=0.8, zorder=5, label='Scout')

    if f_alive and f_pos is not None:
        ax_map.scatter(f_pos[0], f_pos[1], c='tomato', s=150, marker='>',
                       edgecolors='white', linewidths=0.8, zorder=5, label='Commander')

    ax_map.set_xlim(-b, b)
    ax_map.set_ylim(-b, b)
    ax_map.set_title(f"Krok {step:04d}  |  Oheň + Trajektorie",
                     color='white', fontsize=11)
    ax_map.tick_params(colors='#888888', labelsize=7)
    for spine in ax_map.spines.values():
        spine.set_edgecolor('#444444')
    legend = ax_map.legend(loc='upper right', fontsize=8,
                            facecolor='#333333', labelcolor='white', edgecolor='#666666')

    # ── Panel 2: Scout kamera (lokální mapa ohně) ───────────────────────────
    ax_cam = fig.add_subplot(gs[0, 1])
    ax_cam.set_facecolor('#0a0a0a')
    if local_map_np is not None:
        cam_masked = np.ma.masked_where(local_map_np < 0.001, local_map_np)
        ax_cam.imshow(cam_masked, origin='lower', cmap='YlOrRd', vmin=0, vmax=1.0,
                      interpolation='nearest')
        ax_cam.imshow(np.zeros_like(local_map_np), origin='lower', cmap='Greys',
                      alpha=0.15)  # tmavé pozadí pod maskou
    else:
        ax_cam.text(0.5, 0.5, 'Scout\nmartev', color='red',
                    ha='center', va='center', transform=ax_cam.transAxes)
    ax_cam.set_title("Scout kamera  (32×32)", color='white', fontsize=9)
    ax_cam.set_xticks([]); ax_cam.set_yticks([])
    # rámeček
    fire_intensity = float(np.mean(local_map_np)) if local_map_np is not None else 0.0
    edge_color = 'orange' if fire_intensity > 0.005 else 'cyan'
    for spine in ax_cam.spines.values():
        spine.set_edgecolor(edge_color)
        spine.set_linewidth(2.0)

    # ── Panel 3: Stats ──────────────────────────────────────────────────────
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.set_facecolor('#111111')
    ax_stats.set_xticks([]); ax_stats.set_yticks([])
    for spine in ax_stats.spines.values():
        spine.set_edgecolor('#444444')

    stats_text = (
        f"Scout reward:      {total_reward_q:+.1f}\n"
        f"Commander reward:  {total_reward_f:+.1f}\n"
        f"\n"
        f"Fire viditelnost:  {fire_seen_sum:.4f}\n"
        f"\n"
        f"Scout alive:       {'✓' if q_alive else '✗'}\n"
        f"Commander alive:   {'✓' if f_alive else '✗'}"
    )
    ax_stats.text(0.08, 0.88, stats_text, color='#dddddd', fontsize=8.5,
                  va='top', ha='left', transform=ax_stats.transAxes,
                  fontfamily='monospace')
    ax_stats.set_title("Stats", color='white', fontsize=9)

    # Titul celého obrázku
    fig.suptitle("Heterogenní MAPPO  —  Wildfire Suppression Demo",
                 color='white', fontsize=12, y=0.98)

    # Převod na numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor())
    buf.seek(0)
    arr = np.array(Image.open(buf))
    plt.close(fig)
    return arr


def run_demo():
    print("🎬 Demo: Heterogenní MAPPO tým (Scout + Commander)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scout_actor, commander_actor, scout_self_dim, fixed_self_dim = _load_models(device)

    # ── Prostředí ───────────────────────────────────────────────────────────
    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED,
                       grid_size_m=GRID_SIZE, max_steps=MAX_STEPS)
    obs, _ = env.reset(seed=EPISODE_SEED, epizode_number=5000)

    b = env.map_bounds   # polovina velikosti mapy

    # ── Paměti GRU ──────────────────────────────────────────────────────────
    h_scout = torch.zeros(1, 1, 128).to(device)
    h_cmdr  = torch.zeros(1, 1, 128).to(device)

    # ── Historie pro analýzu ─────────────────────────────────────────────────
    hist = {
        "q_reward": [], "f_reward": [],
        "q_vel": [],    "f_vel": [],
        "q_alt": [],    "f_alt": [],
        "q_actions": [], "f_actions": [],
        "fire_seen": [],
    }

    frames         = []
    total_reward_q = 0.0
    total_reward_f = 0.0
    q_path_x, q_path_y = [], []
    f_path_x, f_path_y = [], []
    last_local_map = None

    Q_ID = "quad_0"
    F_ID = "fixed_0"

    print("🚀 Mise začíná...")

    for step in range(MAX_STEPS):
        if not env.agents:
            print("💥  Oba agenti jsou mrtví — mise ukončena.")
            break

        actions_to_env = {}

        # ── Scout forward pass ───────────────────────────────────────────────
        msg_tensor = torch.zeros(1, 1, 5).to(device)
        msg_mask   = torch.BoolTensor([[True]]).to(device)   # default: scout mrtvý

        if Q_ID in env.agents:
            l_map   = torch.FloatTensor(obs[Q_ID]["local_map"]).to(device).unsqueeze(0)
            s_state = torch.FloatTensor(obs[Q_ID]["self_state"]).to(device).unsqueeze(0)
            n_state = torch.FloatTensor(obs[Q_ID]["neighbor_states"]).to(device).unsqueeze(0)
            n_mask  = torch.BoolTensor(obs[Q_ID]["neighbor_mask"]).to(device).unsqueeze(0)

            with torch.no_grad():
                dist_q, message_q, h_scout = scout_actor(l_map, s_state, n_state, n_mask, h_scout)
                action_q = dist_q.mean

            actions_to_env[Q_ID] = action_q.squeeze(0).cpu().numpy()
            msg_tensor = message_q.unsqueeze(1)   # [1, 1, 5]
            msg_mask   = torch.BoolTensor([[False]]).to(device)

            # Uložení lokální mapy pro vykreslení
            last_local_map = obs[Q_ID]["local_map"][0]   # [32, 32]

        # ── Commander forward pass ───────────────────────────────────────────
        if F_ID in env.agents:
            s_state_f = torch.FloatTensor(obs[F_ID]["self_state"]).to(device).unsqueeze(0)

            with torch.no_grad():
                dist_f, _, h_cmdr = commander_actor(s_state_f, msg_tensor, msg_mask, h_cmdr)
                action_f = dist_f.mean

            actions_to_env[F_ID] = action_f.squeeze(0).cpu().numpy()

        # ── Krok prostředí ───────────────────────────────────────────────────
        obs, rewards, terminations, truncations, infos = env.step(actions_to_env)

        rq = rewards.get(Q_ID, 0.0)
        rf = rewards.get(F_ID, 0.0)
        total_reward_q += rq
        total_reward_f += rf

        # Pozice a logy
        q_alive = Q_ID in env.sim.drones
        f_alive = F_ID in env.sim.drones
        q_pos, f_pos = None, None
        q_fov = 30.0

        if q_alive:
            q_pos = env.sim.drones[Q_ID].get_position()
            q_path_x.append(q_pos[0]);  q_path_y.append(q_pos[1])
            q_fov = max(10.0, q_pos[2] * 1.5)
            hist["q_vel"].append(np.linalg.norm(env.sim.drones[Q_ID].get_velocity()))
            hist["q_alt"].append(q_pos[2])
            hist["q_actions"].append(actions_to_env.get(Q_ID, np.zeros(4)))

        if f_alive:
            f_pos = env.sim.drones[F_ID].get_position()
            f_path_x.append(f_pos[0]);  f_path_y.append(f_pos[1])
            hist["f_vel"].append(np.linalg.norm(env.sim.drones[F_ID].get_velocity()))
            hist["f_alt"].append(f_pos[2])
            hist["f_actions"].append(actions_to_env.get(F_ID, np.zeros(4)))

        fire_seen = float(np.sum(last_local_map)) if last_local_map is not None else 0.0
        hist["q_reward"].append(rq)
        hist["f_reward"].append(rf)
        hist["fire_seen"].append(fire_seen)

        if step % 20 == 0:
            q_str = f"[{q_pos[0]:5.0f},{q_pos[1]:5.0f},{q_pos[2]:4.0f}]" if q_pos is not None else "DEAD"
            f_str = f"[{f_pos[0]:5.0f},{f_pos[1]:5.0f},{f_pos[2]:4.0f}]" if f_pos is not None else "DEAD"
            print(f"  Krok {step:04d} | Scout {q_str} | Cmdr {f_str} "
                  f"| Fire {fire_seen:.3f} | Rq {total_reward_q:+.0f} Rf {total_reward_f:+.0f}")

        # ── Snímek GIFu ──────────────────────────────────────────────────────
        if step % GIF_EVERY == 0:
            fire_map = env.sim.environment.fire_grid.I.copy()
            frame = _render_frame(
                step, fire_map, b,
                list(q_path_x), list(q_path_y), q_pos, q_fov,
                list(f_path_x), list(f_path_y), f_pos,
                last_local_map.copy() if last_local_map is not None else None,
                total_reward_q, total_reward_f,
                fire_seen, q_alive, f_alive
            )
            frames.append(frame)

    print(f"\n🏁 Hotovo! Scout reward: {total_reward_q:.1f} | Commander reward: {total_reward_f:.1f}")

    # ── Uložení GIFu ─────────────────────────────────────────────────────────
    out_gif = os.path.join(project_root, "demo_training.gif")
    print(f"💾  Ukládám GIF ({len(frames)} snímků) → {out_gif} ...")
    imageio.mimsave(out_gif, frames, fps=GIF_FPS, loop=0)
    print(f"✅  GIF uložen!")

    # ── Analytický graf ───────────────────────────────────────────────────────
    _save_analysis(hist, project_root)


def _save_analysis(hist, project_root):
    steps = range(len(hist["q_reward"]))

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=False)
    fig.suptitle("Demo analýza — Scout & Commander", fontsize=14)

    # 1. Kumulativní odměna
    ax = axes[0]
    ax.plot(np.cumsum(hist["q_reward"]),  label="Scout (kum.)",     color='cyan')
    ax.plot(np.cumsum(hist["f_reward"]),  label="Commander (kum.)", color='tomato')
    ax.set_title("Kumulativní odměna"); ax.legend(); ax.grid(alpha=0.3)

    # 2. Odměna per krok
    ax = axes[1]
    ax.bar(steps, hist["q_reward"],  label="Scout reward",     color='cyan',  alpha=0.5)
    ax.bar(steps, hist["f_reward"],  label="Commander reward", color='tomato',alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_title("Odměna per krok"); ax.legend(); ax.grid(alpha=0.3)

    # 3. Rychlosti
    ax = axes[2]
    if hist["q_vel"]: ax.plot(hist["q_vel"], label="Scout m/s",     color='cyan')
    if hist["f_vel"]: ax.plot(hist["f_vel"], label="Commander m/s", color='tomato')
    ax.set_title("Rychlost (m/s)"); ax.legend(); ax.grid(alpha=0.3)

    # 4. Nadmořská výška
    ax = axes[3]
    if hist["q_alt"]: ax.plot(hist["q_alt"], label="Scout výška (m)",     color='cyan')
    if hist["f_alt"]: ax.plot(hist["f_alt"], label="Commander výška (m)", color='tomato')
    ax.set_title("Výška (m)"); ax.legend(); ax.grid(alpha=0.3)

    # 5. Viditelnost ohně scoutyem
    ax = axes[4]
    ax.fill_between(range(len(hist["fire_seen"])), hist["fire_seen"],
                    color='orange', alpha=0.6, label="Fire intensity (kamera)")
    ax.set_title("Viditelnost ohně — scout kamera"); ax.set_xlabel("Krok")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(project_root, "demo_training_analysis.png")
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"📊  Analýza uložena → {out_png}")


if __name__ == "__main__":
    run_demo()
