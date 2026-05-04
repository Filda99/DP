"""
demo_commander.py — Demo pro Fixed-Wing Commander
==================================================
Spustí 2000 kroků s natrénovaným commanderem (bez scouta) a uloží:
  - commander_demo.gif      : animace letu nad mapou
  - commander_analysis.png  : 7-panelová analýza chování
"""

import torch
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio
import io
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.env_core import DroneFireEnv
from src.models import CommanderActor

# ─── Konfigurace ──────────────────────────────────────────────────────────────
MODEL_PATH  = "/homes/eva/xj/xjahnf00/tmp/DP/saved_models/commander_ep32400.pt"
MAX_STEPS   = 2000
GIF_EVERY   = 3          # zapiš každý N-tý krok do GIFu (menší soubor)
GIF_FPS     = 20
GRID_SIZE   = 2000.0
SEED        = 42

# ─── Barvy pro stavový automat ─────────────────────────────────────────────────
STATE_COLORS = {
    "MISSION": "#FF6B35",
    "REFILL":  "#4ECDC4",
    "PATROL":  "#95A5A6",
}

# ─── Pomocné funkce ────────────────────────────────────────────────────────────

def get_state_label(water_lvl, fire_visible):
    if fire_visible:
        return "MISSION"
    elif water_lvl < 0.1:
        return "REFILL"
    else:
        return "PATROL"


def render_frame(env, pos, path_x, path_y, step, total_reward,
                 state_label, water_lvl, altitude):
    """Vykreslí jeden snímek pro GIF."""
    b = env.map_bounds
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#1a1a2e')

    # ── Levý panel: mapa + let ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('#1a1a2e')

    # Fire grid
    if env.sim.environment.fire_grid is not None:
        fire_map = env.sim.environment.fire_grid.I
        fire_masked = np.ma.masked_where(fire_map < 0.01, fire_map)
        ax.imshow(fire_masked, extent=[-b, b, -b, b], origin='lower',
                  cmap='YlOrRd', vmin=0, vmax=1.0, alpha=0.85)

    # Refill zóna
    if env.sim.environment.refill_zone:
        rp = env.sim.environment.refill_zone['position']
        refill_circle = plt.Circle((rp[0], rp[1]), 80,
                                   color='#4ECDC4', alpha=0.3, fill=True)
        ax.add_patch(refill_circle)
        ax.scatter(rp[0], rp[1], c='#4ECDC4', s=120, marker='s',
                   edgecolors='white', linewidths=1.5, label='Refill', zorder=5)

    # Trajektorie
    if len(path_x) > 1:
        ax.plot(path_x, path_y, color='white', alpha=0.25,
                linestyle=':', linewidth=1.2)

    # Barevná stopa podle stavu (posledních 80 kroků)
    seg_x = path_x[-80:]
    seg_y = path_y[-80:]
    color = STATE_COLORS[state_label]
    ax.plot(seg_x, seg_y, color=color, alpha=0.7, linewidth=2.0)

    # Pozice commandera
    ax.scatter(pos[0], pos[1], c=color, s=200, marker='^',
               edgecolors='white', linewidths=1.5, zorder=10)

    # Altitude indikátor (vertikální čára v rohu)
    alt_norm = np.clip(altitude / 450.0, 0, 1)
    ax.barh(-b * 0.92, b * 0.06 * alt_norm * 2, left=b * 0.82,
            height=b * 0.04, color=color, alpha=0.8)
    ax.text(b * 0.85, -b * 0.85, f"Alt: {altitude:.0f}m",
            color='white', fontsize=8, ha='left')

    # Stavové info
    state_patch = mpatches.Patch(color=color, label=f"Stav: {state_label}")
    ax.legend(handles=[state_patch], loc='upper left',
              facecolor='#1a1a2e', edgecolor='white',
              labelcolor='white', fontsize=9)

    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    ax.set_title(
        f"Krok: {step:04d} | Reward: {total_reward:.1f} | "
        f"Voda: {water_lvl*100:.0f}%",
        color='white', fontsize=11
    )
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    ax.grid(color='white', alpha=0.05)

    # ── Pravý panel: mini-dashboard ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')
    ax2.axis('off')

    dashboard = (
        f"  Commander Dashboard\n"
        f"  {'─'*28}\n"
        f"  Krok:       {step:>6d} / {MAX_STEPS}\n"
        f"  Pozice X:   {pos[0]:>8.1f} m\n"
        f"  Pozice Y:   {pos[1]:>8.1f} m\n"
        f"  Výška:      {altitude:>8.1f} m\n"
        f"  Voda:       {water_lvl*100:>7.1f} %\n"
        f"  Stav:       {state_label:>8s}\n"
        f"  Reward:     {total_reward:>8.1f}\n"
    )
    ax2.text(0.05, 0.55, dashboard,
             transform=ax2.transAxes,
             color='white', fontsize=10,
             fontfamily='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='#0d0d1a',
                       edgecolor=color, linewidth=2))

    # Legenda stavů
    for i, (s, c) in enumerate(STATE_COLORS.items()):
        ax2.text(0.1, 0.25 - i * 0.08, f"  ● {s}",
                 transform=ax2.transAxes,
                 color=c, fontsize=10, fontfamily='monospace')

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=80,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = np.array(Image.open(buf))
    plt.close(fig)
    return img


# ─── Hlavní demo funkce ────────────────────────────────────────────────────────

def run_demo():
    print("🛩️  Spouštím Demo: Commander (Fixed-Wing)")

    # Prostředí
    env = DroneFireEnv(num_quads=0, num_fixed=1,
                       grid_size_m=GRID_SIZE, max_steps=MAX_STEPS)

    fixed_self_dim  = env.observation_space(env.fixed_agents[0])["self_state"].shape[0]
    scout_msg_dim   = 5

    # Prázdné zprávy (žádný scout)
    msgs_t = torch.zeros(1, 1, scout_msg_dim)
    msgs_m = torch.ones(1, 1, dtype=torch.bool)   # mask = ignore

    # Síť
    commander = CommanderActor(self_state_dim=fixed_self_dim,
                               msg_input_dim=scout_msg_dim)

    if os.path.exists(MODEL_PATH):
        commander.load_state_dict(
            torch.load(MODEL_PATH, map_location='cpu'), strict=False)
        print(f"✅ Model načten: {MODEL_PATH}")
    else:
        print(f"⚠️  Model nenalezen, spouštím s náhodnou sítí: {MODEL_PATH}")

    commander.eval()

    # Reset
    obs, _ = env.reset(seed=SEED, epizode_number=5000)
    hidden = torch.zeros(1, 1, 128)

    # ── Logování ───────────────────────────────────────────────────────────
    log = {
        "pos":          [],   # [x, y, z]
        "vel":          [],   # scalar speed
        "actions":      [],   # [roll, pitch, throttle, water]
        "reward":       [],   # per-step reward
        "water":        [],   # water level [0,1]
        "state":        [],   # "MISSION" / "REFILL" / "PATROL"
        "altitude":     [],
        "path_x":       [],
        "path_y":       [],
    }
    total_reward = 0.0
    frames       = []

    agent = env.fixed_agents[0]

    print(f"✈️  Letím {MAX_STEPS} kroků...")

    for step in range(MAX_STEPS):

        if agent not in env.sim.drones:
            print(f"💥 Commander havaroval v kroku {step}!")
            break

        drone = env.sim.drones[agent]
        pos   = drone.get_position()
        vel   = np.linalg.norm(drone.get_velocity())
        water = drone.current_water / drone.water_capacity \
                if drone.water_capacity > 0 else 0.0

        # Určení stavu (bez scouta je vždy PATROL nebo REFILL)
        fire_visible = False   # bez scouta nevidíme oheň
        state = get_state_label(water, fire_visible)

        # Forward pass
        s_st = torch.FloatTensor(obs[agent]["self_state"]).unsqueeze(0)
        with torch.no_grad():
            dist, _, hidden = commander(s_st, msgs_t, msgs_m, hidden)
            action = dist.mean   # deterministické demo

        act_np = action.squeeze(0).numpy()
        actions_dict = {agent: act_np}

        # Krok prostředí
        obs, rewards, terminations, truncations, _ = env.step(actions_dict)
        r = rewards.get(agent, 0.0)
        total_reward += r

        # Logování
        log["pos"].append(pos.copy())
        log["vel"].append(vel)
        log["actions"].append(act_np.copy())
        log["reward"].append(r)
        log["water"].append(water)
        log["state"].append(state)
        log["altitude"].append(pos[2])
        log["path_x"].append(pos[0])
        log["path_y"].append(pos[1])

        # Konzolový výpis každých 50 kroků
        if step % 50 == 0:
            print(f"  Krok {step:04d} | Pos [{pos[0]:6.0f},{pos[1]:6.0f},{pos[2]:5.0f}m] "
                  f"| Voda {water*100:4.0f}% | Stav {state:7s} "
                  f"| R {r:+6.2f} | ΣR {total_reward:7.1f}")

        # GIF frame
        if step % GIF_EVERY == 0:
            frame = render_frame(env, pos,
                                 log["path_x"], log["path_y"],
                                 step, total_reward,
                                 state, water, pos[2])
            frames.append(frame)

        if terminations.get(agent, False):
            print(f"⚠️  Agent terminoval v kroku {step}")
            break

    print(f"\n🏁 Demo dokončeno! Celkový Reward: {total_reward:.2f}")

    # ── Uložení GIF ────────────────────────────────────────────────────────
    gif_path = "commander_demo.gif"
    print(f"💾 Ukládám GIF ({len(frames)} snímků) → {gif_path} ...")
    imageio.mimsave(gif_path, frames, fps=GIF_FPS, loop=0)
    print(f"✅ GIF uložen!")

    # ── Analytický graf ────────────────────────────────────────────────────
    steps_range = range(len(log["actions"]))
    acts = np.array(log["actions"])
    pos_arr = np.array(log["pos"])

    # Barvy pro stavovou metriku
    state_color_seq = [STATE_COLORS[s] for s in log["state"]]

    fig, axes = plt.subplots(7, 1, figsize=(14, 24))
    fig.suptitle("Commander — Analýza Demo Epizody", fontsize=16, fontweight='bold')

    # 1. Akce sítě
    ax = axes[0]
    labels = ['Roll', 'Pitch', 'Throttle', 'Water Trigger']
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
    for i in range(4):
        ax.plot(steps_range, acts[:, i], label=labels[i], color=colors[i], alpha=0.85)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title("Výstupy sítě (Actions)")
    ax.set_ylabel("Hodnota [-1, 1]")
    ax.legend(loc='upper right', ncol=4)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    # 2. Výška (altitude)
    ax = axes[1]
    ax.plot(steps_range, log["altitude"], color='#9B59B6', linewidth=1.5)
    ax.axhline(80,  color='green',  linestyle='--', alpha=0.6, label='Ideál 80m')
    ax.axhline(150, color='orange', linestyle='--', alpha=0.6, label='Max 150m')
    ax.axhline(450, color='red',    linestyle='--', alpha=0.6, label='Ceiling 450m')
    ax.fill_between(steps_range, 40, 150,
                    color='green', alpha=0.07, label='Sweet spot')
    ax.set_title("Výška (Altitude)")
    ax.set_ylabel("Metrů")
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3)

    # 3. Pozice XY + Z
    ax = axes[2]
    ax.plot(steps_range, pos_arr[:, 0], label='X (East)',  color='#3498DB')
    ax.plot(steps_range, pos_arr[:, 1], label='Y (North)', color='#E74C3C')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    b = env.map_bounds
    ax.axhline(b, color='red', linewidth=0.8, linestyle=':', alpha=0.5, label='Hranice mapy')
    ax.axhline(-b, color='red', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.set_title("Horizontální pozice")
    ax.set_ylabel("Metry")
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3)

    # 4. Rychlost
    ax = axes[3]
    ax.plot(steps_range, log["vel"], color='#1ABC9C', linewidth=1.2)
    ax.axhline(15, color='orange', linestyle='--', alpha=0.7, label='Nominální 15 m/s')
    ax.set_title("Rychlost (Speed)")
    ax.set_ylabel("m/s")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Hladina vody
    ax = axes[4]
    ax.fill_between(steps_range, log["water"], alpha=0.6,
                    color='#4ECDC4', label='Voda')
    ax.axhline(0.1, color='red', linestyle='--', alpha=0.7,
               label='Refill práh (10%)')
    ax.set_title("Hladina vody v nádrži")
    ax.set_ylabel("Úroveň [0-1]")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Stav (MISSION / REFILL / PATROL)
    ax = axes[5]
    state_map = {"MISSION": 2, "REFILL": 1, "PATROL": 0}
    state_num = [state_map[s] for s in log["state"]]
    ax.scatter(steps_range, state_num, c=state_color_seq,
               s=4, alpha=0.8)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["PATROL", "REFILL", "MISSION"])
    ax.set_title("Stav commandera")
    for s, c in STATE_COLORS.items():
        ax.scatter([], [], c=c, label=s, s=40)
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3)

    # 7. Reward per step
    ax = axes[6]
    pos_r = [max(0, r) for r in log["reward"]]
    neg_r = [min(0, r) for r in log["reward"]]
    ax.fill_between(steps_range, pos_r, alpha=0.6, color='#2ECC71', label='Kladný reward')
    ax.fill_between(steps_range, neg_r, alpha=0.6, color='#E74C3C', label='Záporný reward')
    ax.axhline(0, color='gray', linewidth=0.8)

    # Kumulativní reward (pravá osa)
    ax2_r = ax.twinx()
    cumsum = np.cumsum(log["reward"])
    ax2_r.plot(steps_range, cumsum, color='white', linewidth=1.5,
               linestyle='--', alpha=0.8, label='Kumulativní')
    ax2_r.set_ylabel("Kumulativní reward", color='white')
    ax2_r.tick_params(colors='white')

    ax.set_title("Reward per krok + Kumulativní reward")
    ax.set_ylabel("Reward")
    ax.set_xlabel("Krok epizody")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("commander_analysis.png", dpi=120, bbox_inches='tight')
    print("📊 Analýza uložena: commander_analysis.png")
    plt.close()

    # ── Shrnutí do konzole ─────────────────────────────────────────────────
    total_steps = len(log["actions"])
    mission_steps = log["state"].count("MISSION")
    refill_steps  = log["state"].count("REFILL")
    patrol_steps  = log["state"].count("PATROL")
    avg_alt = np.mean(log["altitude"])
    min_alt = np.min(log["altitude"])
    max_alt = np.max(log["altitude"])
    avg_speed = np.mean(log["vel"])
    water_refills = sum(
        1 for i in range(1, len(log["water"]))
        if log["water"][i] > log["water"][i-1] + 0.5
    )

    print(f"""
┌─────────────────────────────────────────┐
│         COMMANDER DEMO SHRNUTÍ          │
├─────────────────────────────────────────┤
│ Celkový reward:   {total_reward:>10.2f}           │
│ Kroků přežito:    {total_steps:>10d} / {MAX_STEPS}       │
├─────────────────────────────────────────┤
│ MISSION kroků:    {mission_steps:>10d} ({100*mission_steps//max(1,total_steps):>2d}%)       │
│ REFILL kroků:     {refill_steps:>10d} ({100*refill_steps//max(1,total_steps):>2d}%)       │
│ PATROL kroků:     {patrol_steps:>10d} ({100*patrol_steps//max(1,total_steps):>2d}%)       │
├─────────────────────────────────────────┤
│ Průměrná výška:   {avg_alt:>10.1f} m          │
│ Min / Max výška:  {min_alt:>5.1f} / {max_alt:<5.1f} m       │
│ Průměrná rychlost:{avg_speed:>9.1f} m/s         │
│ Doplnění vody:    {water_refills:>10d}×              │
└─────────────────────────────────────────┘
""")


if __name__ == "__main__":
    run_demo()