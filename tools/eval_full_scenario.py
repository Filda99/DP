"""
eval_full_scenario.py
=====================
Souhrná evaluace obou agentů (scout quadkoptéry + FW commander) přes
sadu náhodných seedů, různé velikosti map a různé lokality.

Výstup:
  results/eval_full_scenario.csv   — CSV tabulka, otevřitelná v Excelu/Pandas
  results/eval_full_scenario.txt   — textová tabulka pro rychlý přehled

Použití:
  python tools/eval_full_scenario.py \
      --scout-model  saved_models/multi/scout_best.pt \
      --cmdr-model   saved_models/multi/cmdr_b0780.pt \
      --runs 50 \
      --max-steps 500

Volitelné přepínače:
  --scouts N              pevný počet scoutů (jinak náhodně 1–4)
  --fw N                  pevný počet FW (jinak náhodně 1–2)
  --seed-start N          první seed (default 0)
  --out results/eval_full_scenario.csv
"""

import sys, os, argparse, csv, random, time
import numpy as np
import torch

# ── cesta k projektu ─────────────────────────────────────────────────────────
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "src"))
os.chdir(os.path.join(PROJECT, "src"))

from env_core import DroneFireEnv
from models import ScoutActor, CommanderActor
from commander_control import CommanderController

# ── Lokality ─────────────────────────────────────────────────────────────────
# Každá lokalita = jméno + (lat, lon). Pokud use_osm=False, jméno se použije
# jen jako popisek; pokud use_osm=True a data jsou v cache, načte se terén.
LOCATIONS = [
    ("Procedural-A",  49.35, 16.42, False),
    ("Procedural-B",  49.20, 16.60, False),
    ("Procedural-C",  49.50, 16.20, False),
    ("Brno-okolí",    49.35, 16.42, True),   # vyžaduje cache v data/
]

MAP_SIZES = [800.0, 1200.0, 2000.0]   # metry

HIDDEN_DIM_SCOUT = 128
HIDDEN_DIM_CMDR  = 64   # musí souhlasit s train_multi.py hidden_dim_cmdr
WAYPOINT_STEPS   = 30     # FW dostane nový waypoint každých N kroků
WAYPOINT_RANGE   = 200.0
WP_REACHED_DIST  = 30.0
NORM_DIST        = 1000.0
WATER_CAPACITY   = 200.0  # litrů (musí souhlasit s FixedWing)

# ── Utility ──────────────────────────────────────────────────────────────────


def _fire_stats(env) -> dict:
    """Vrátí aktuální statistiky fire_gridu."""
    fg = env.sim.environment.fire_grid
    if fg is None:
        return {"burning_cells": 0, "intensity_sum": 0.0, "burned_fraction": 0.0}
    B = fg.B
    I = fg.I
    total_cells = B.size
    burning = int(B.sum())
    i_sum   = float(I.sum())
    # "spálená plocha" = buňky kde intensity > 0.01 (hoří nebo dohořely)
    burned = int((I > 0.01).sum())
    return {
        "burning_cells":  burning,
        "intensity_sum":  round(i_sum, 3),
        "burned_fraction": round(burned / total_cells * 100.0, 2),
    }


# ── Jedno spuštění epizody ───────────────────────────────────────────────────

def run_episode(scout_actor, cmdr_actor, device,
                n_quads, n_fw, seed, grid_size_m, max_steps,
                loc_name, osm_lat, osm_lon, use_osm) -> dict:
    """Spustí jednu epizodu a vrátí slovník metrik."""

    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Pokus o OSM; pokud selže, fallback na procedurální
    env_kwargs = dict(
        num_quads=n_quads, num_fixed=n_fw,
        grid_size_m=grid_size_m, max_steps=max_steps,
    )
    if use_osm:
        env_kwargs.update(use_osm=True, osm_lat=osm_lat, osm_lon=osm_lon,
                          osm_cache_dir=os.path.join(PROJECT, "data"))
    try:
        env = DroneFireEnv(**env_kwargs)
        obs, _ = env.reset(epizode_number=seed)
    except Exception as e:
        # Fallback: procedurální mapa
        env_kwargs.pop("use_osm", None)
        env_kwargs.pop("osm_lat", None)
        env_kwargs.pop("osm_lon", None)
        env_kwargs.pop("osm_cache_dir", None)
        env = DroneFireEnv(**env_kwargs)
        obs, _ = env.reset(epizode_number=seed)
        loc_name = loc_name + "(proc.)"

    quad_agents  = env.quad_agents
    fixed_agents = env.fixed_agents

    # ── Inicializace pamětí GRU ──
    h_scout = {q: torch.zeros(1, 1, HIDDEN_DIM_SCOUT).to(device) for q in quad_agents}
    h_cmdr  = {f: torch.zeros(1, 1, HIDDEN_DIM_CMDR).to(device)  for f in fixed_agents}

    # ── Sledovací proměnné ──
    scout_alive   = {q: True for q in quad_agents}
    fw_alive      = {f: True for f in fixed_agents}
    scout_rewards = {q: 0.0 for q in quad_agents}
    fw_rewards    = {f: 0.0 for f in fixed_agents}
    scout_fire_steps = {q: 0 for q in quad_agents}   # kroky kde měl fire v kameře
    scout_total_steps= {q: 0 for q in quad_agents}
    fw_water_start   = {}
    fw_water_end     = {}
    fw_refill_count  = {f: 0 for f in fixed_agents}
    fw_prev_water    = {}
    fw_valve_steps   = {f: 0 for f in fixed_agents}
    fw_total_steps   = {f: 0 for f in fixed_agents}
    last_scout_msgs  = {q: np.zeros(5) for q in quad_agents}  # zprávy pro FW compass
    fire_discovered  = False
    steps_done       = 0
    fw_mode          = {f: 'nn' for f in fixed_agents}  # current mode: nn/scripted/emergency
    fw_death_info    = {}  # f -> {step, mode, pos}

    # Per-FW CommanderController (waypoint + PD heading + scripted refill)
    cmdr_ctrl = {}
    for f in fixed_agents:
        ctrl = CommanderController(WAYPOINT_RANGE, WAYPOINT_STEPS, WP_REACHED_DIST)
        ctrl.reset(env.map_bounds)
        cmdr_ctrl[f] = ctrl

    # Počáteční stav ohně
    fs_init = _fire_stats(env)

    for f in fixed_agents:
        d = env.sim.drones.get(f)
        if d and d.water_capacity > 0:
            fw_water_start[f] = d.current_water
            fw_prev_water[f]  = d.current_water
        else:
            fw_water_start[f] = WATER_CAPACITY
            fw_prev_water[f]  = WATER_CAPACITY

    # ── Hlavní smyčka ──────────────────────────────────────────────────────
    for step in range(max_steps):
        if not env.agents:
            break

        actions = {}

        # --- Scouti ---
        for q in quad_agents:
            if not scout_alive[q] or q not in env.agents:
                scout_alive[q] = False
                continue
            scout_total_steps[q] += 1
            with torch.no_grad():
                lm = torch.FloatTensor(obs[q]["local_map"]).unsqueeze(0).to(device)
                ss = torch.FloatTensor(obs[q]["self_state"]).unsqueeze(0).to(device)
                ns = torch.FloatTensor(obs[q]["neighbor_states"]).unsqueeze(0).to(device)
                nm = torch.BoolTensor(obs[q]["neighbor_mask"]).unsqueeze(0).to(device)
                dist, msg, h_out = scout_actor(lm, ss, ns, nm, h_scout[q])
                act = dist.mean
            h_scout[q] = h_out
            actions[q] = act.squeeze(0).cpu().numpy()
            last_scout_msgs[q] = msg.squeeze(0).cpu().numpy()

            # Dwell: byl oheň v kameře?
            fire_in_cam = float(np.sum(obs[q]["local_map"]))
            if fire_in_cam > 0.1:
                scout_fire_steps[q] += 1
                fire_discovered = True

        # --- Commander (FW) ---
        for f in fixed_agents:
            if not fw_alive[f] or f not in env.agents:
                fw_alive[f] = False
                continue
            fw_total_steps[f] += 1

            drone = env.sim.drones.get(f)
            if drone is None:
                fw_alive[f] = False
                continue

            # Build scout message tensor
            msg_list = [torch.FloatTensor(last_scout_msgs.get(q, np.zeros(5)))
                        for q in quad_agents]
            msgs_t = torch.stack(msg_list).unsqueeze(0).to(device)
            mask_t = torch.BoolTensor(
                [[not scout_alive[q] for q in quad_agents]]
            ).to(device)

            # Build FW neighbor tensor
            pos_f = drone.get_position()
            map_half = env.map_bounds
            fw_neigh_list = []
            fw_mask_list = []
            for f2 in fixed_agents:
                if f2 == f:
                    continue
                d2 = env.sim.drones.get(f2)
                if d2 is not None and fw_alive.get(f2, False):
                    p2 = d2.get_position()
                    fw_neigh_list.append([
                        (p2[0] - pos_f[0]) / max(map_half, 1.0),
                        (p2[1] - pos_f[1]) / max(map_half, 1.0),
                        p2[2] / 100.0
                    ])
                    fw_mask_list.append(False)
                else:
                    fw_neigh_list.append([0.0, 0.0, 0.0])
                    fw_mask_list.append(True)
            fw_neigh_t = torch.FloatTensor([fw_neigh_list]).to(device) if fw_neigh_list else None
            fw_mask_t = torch.BoolTensor([fw_mask_list]).to(device) if fw_mask_list else None

            # CommanderController handles: boundary emergency, scripted
            # refill, NN waypoint, PD heading — exactly as in training
            action, h_cmdr[f], ctrl_info = cmdr_ctrl[f].step(
                drone, obs[f]["self_state"], env,
                cmdr_actor, h_cmdr[f], msgs_t, mask_t,
                deterministic=True,
                fw_neighbor_states=fw_neigh_t, fw_neighbor_mask=fw_mask_t)
            actions[f] = action

            # Track current FW mode for crash diagnostics
            if ctrl_info['in_emergency']:
                fw_mode[f] = 'emergency'
            elif ctrl_info['scripted']:
                fw_mode[f] = 'scripted'
            elif ctrl_info.get('new_waypoint'):
                fw_mode[f] = 'nn'
            # else: keep previous mode (PD heading between waypoints)

            # Valve open: action[2] is water_raw from heading_action
            if len(action) > 2 and action[2] > 0.5:
                fw_valve_steps[f] += 1

        # Krok prostředí
        obs, rewards, terms, truncs, _ = env.step(actions)
        steps_done += 1

        # Odměny
        for q in quad_agents:
            scout_rewards[q] += rewards.get(q, 0.0)
        for f in fixed_agents:
            fw_rewards[f] += rewards.get(f, 0.0)

        # Detekce refillů FW (když voda vzroste)
        for f in fixed_agents:
            d = env.sim.drones.get(f)
            if d and d.water_capacity > 0:
                cur_w = d.current_water
                if cur_w > fw_prev_water[f] + 1.0:
                    fw_refill_count[f] += 1
                fw_prev_water[f] = cur_w
                fw_water_end[f] = cur_w

        # Konec agentů — only crash (terminated) counts as death,
        # truncation (time limit) means the agent survived
        for q in quad_agents:
            if terms.get(q, False):
                scout_alive[q] = False
        for f in fixed_agents:
            if terms.get(f, False):
                fw_alive[f] = False
                if f not in fw_death_info:
                    d = env.sim.drones.get(f)
                    pos = d.get_position() if d else [0, 0, 0]
                    fw_death_info[f] = {
                        'step': step, 'mode': fw_mode[f],
                        'pos': [round(p, 1) for p in pos]
                    }

    # Finální stav ohně
    fs_final = _fire_stats(env)

    # Dočteme water_end pro ty, kteří přežili
    for f in fixed_agents:
        if f not in fw_water_end:
            d = env.sim.drones.get(f)
            fw_water_end[f] = d.current_water if d else 0.0

    # ── Agregace metrik ──────────────────────────────────────────────────
    scouts_survived = sum(1 for q in quad_agents
                          if q in env.sim.drones or scout_alive[q])
    fw_survived     = sum(1 for f in fixed_agents
                          if f in env.sim.drones or fw_alive[f])

    dwell_vals = []
    for q in quad_agents:
        tot = scout_total_steps[q]
        dwell_vals.append(scout_fire_steps[q] / tot * 100.0 if tot > 0 else 0.0)
    scout_dwell_mean = round(float(np.mean(dwell_vals)) if dwell_vals else 0.0, 1)

    total_water_dropped = 0.0
    for f in fixed_agents:
        dropped = (fw_water_start[f]
                   + fw_refill_count[f] * WATER_CAPACITY
                   - fw_water_end.get(f, 0.0))
        total_water_dropped += max(0.0, dropped)

    fw_valve_pct = 0.0
    if fixed_agents:
        tot_fw_steps = sum(fw_total_steps[f] for f in fixed_agents)
        tot_valve    = sum(fw_valve_steps[f]  for f in fixed_agents)
        fw_valve_pct = round(tot_valve / tot_fw_steps * 100.0 if tot_fw_steps > 0 else 0.0, 1)

    avg_scout_r = round(
        sum(scout_rewards.values()) / n_quads if n_quads > 0 else 0.0, 2)
    avg_fw_r    = round(
        sum(fw_rewards.values()) / n_fw if n_fw > 0 else 0.0, 2)

    fire_reduction = round(fs_init["intensity_sum"] - fs_final["intensity_sum"], 3)
    fire_extinguished = (fs_final["intensity_sum"] < 0.1 * fs_init["intensity_sum"]
                         if fs_init["intensity_sum"] > 0 else False)

    env.sim.stop_simulation()

    return {
        "loc_name":          loc_name,
        "seed":              seed,
        "map_size_m":        int(grid_size_m),
        "max_steps":         max_steps,
        "steps_done":        steps_done,
        "n_scouts":          n_quads,
        "n_fw":              n_fw,
        # Scout metriky
        "scouts_survived":   scouts_survived,
        "scouts_surv_pct":   round(scouts_survived / n_quads * 100.0, 1),
        "fire_discovered":   int(fire_discovered),
        "scout_dwell_pct":   scout_dwell_mean,
        "scout_avg_reward":  avg_scout_r,
        # FW metriky
        "fw_survived":       fw_survived,
        "fw_surv_pct":       round(fw_survived / n_fw * 100.0, 1) if n_fw > 0 else 0.0,
        "fw_water_dropped_L": round(total_water_dropped, 1),
        "fw_refills":        sum(fw_refill_count.values()),
        "fw_valve_open_pct": fw_valve_pct,
        "fw_avg_reward":     avg_fw_r,
        # Oheň
        "fire_cells_init":   fs_init["burning_cells"],
        "fire_cells_final":  fs_final["burning_cells"],
        "fire_intensity_init": fs_init["intensity_sum"],
        "fire_intensity_final": fs_final["intensity_sum"],
        "fire_intensity_reduction": fire_reduction,
        "burned_frac_pct":   fs_final["burned_fraction"],
        "fire_extinguished": int(fire_extinguished),
        # FW crash diagnostics
        "fw_deaths":         fw_death_info,
    }


# ── Načtení modelů ───────────────────────────────────────────────────────────

def load_models(scout_path, cmdr_path, device, n_quads_max=4):
    """Načte scout a cmdr modely. Vrátí (scout_actor, cmdr_actor) nebo (None, None)."""
    # Zjistíme dimenze přímo z prostorů definovaných v __init__ (bez reset)
    tmp_env = DroneFireEnv(num_quads=n_quads_max, num_fixed=1, grid_size_m=1000.0, max_steps=5)
    obs_sp  = tmp_env.observation_space("quad_0")
    scout_self_dim = obs_sp["self_state"].shape[0]
    fixed_self_dim = tmp_env.observation_space("fixed_0")["self_state"].shape[0]
    # sim neexistuje před reset() — nic nečistíme

    scout_actor = ScoutActor(self_state_dim=scout_self_dim,
                             msg_dim=5, hidden_dim=HIDDEN_DIM_SCOUT).to(device)
    cmdr_actor  = CommanderActor(self_state_dim=fixed_self_dim,
                                 msg_input_dim=5,
                                 hidden_dim=HIDDEN_DIM_CMDR).to(device)

    if not os.path.exists(scout_path):
        print(f"[WARN] Scout model nenalezen: {scout_path}")
        print("       Spouštím s náhodnou politikou (výsledky nebudou reprezentativní).")
    else:
        scout_actor.load_state_dict(torch.load(scout_path, map_location=device))
        print(f"[OK]  Scout model: {scout_path}")

    if not os.path.exists(cmdr_path):
        print(f"[WARN] Commander model nenalezen: {cmdr_path}")
        print("       Spouštím s náhodnou politikou.")
    else:
        cmdr_actor.load_state_dict(torch.load(cmdr_path, map_location=device))
        print(f"[OK]  Commander model: {cmdr_path}")

    scout_actor.eval()
    cmdr_actor.eval()
    return scout_actor, cmdr_actor


# ── Tisk tabulky ─────────────────────────────────────────────────────────────
COLS_PRINT = [
    ("run",     "run_id"),
    ("seed",    "seed"),
    ("lokalita","loc_name"),
    ("mapa[m]", "map_size_m"),
    ("sc",      "n_scouts"),
    ("fw",      "n_fw"),
    ("sc_surv%","scouts_surv_pct"),
    ("disc",    "fire_discovered"),
    ("dwell%",  "scout_dwell_pct"),
    ("H2O[L]",  "fw_water_dropped_L"),
    ("refills",  "fw_refills"),
    ("val%",    "fw_valve_open_pct"),
    ("brn%",    "burned_frac_pct"),
    ("extin",   "fire_extinguished"),
    ("sc_R",    "scout_avg_reward"),
    ("fw_surv%","fw_surv_pct"),
    ("fw_R",    "fw_avg_reward"),
]

def print_table(rows):
    header = [h for h, _ in COLS_PRINT]
    widths = [max(len(h), 6) for h in header]
    for row in rows:
        for i, (_, key) in enumerate(COLS_PRINT):
            widths[i] = max(widths[i], len(str(row.get(key, ""))))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    print(sep)
    print(fmt.format(*header))
    print(sep)
    for row in rows:
        vals = [str(row.get(key, "")) for _, key in COLS_PRINT]
        print(fmt.format(*vals))
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scout-model", default="saved_models/multi/scout_best.pt")
    ap.add_argument("--cmdr-model",  default="saved_models/multi/cmdr_b0780.pt")
    ap.add_argument("--runs",        type=int, default=50,
                    help="Celkový počet spuštění (seed × lokality × velikosti)")
    ap.add_argument("--max-steps",   type=int, default=500)
    ap.add_argument("--scouts",      type=int, default=None,
                    help="Pevný počet scoutů (jinak náhodně 1-4)")
    ap.add_argument("--fw",          type=int, default=None,
                    help="Pevný počet FW (jinak náhodně 1-2)")
    ap.add_argument("--seed-start",  type=int, default=0)
    ap.add_argument("--out",         default=os.path.join(PROJECT, "results",
                                                           "eval_full_scenario.csv"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Zařízení: {device}")

    scout_actor, cmdr_actor = load_models(
        os.path.join(PROJECT, args.scout_model),
        os.path.join(PROJECT, args.cmdr_model),
        device
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Vygenerujeme konfiguraci runů
    rng_meta = random.Random(args.seed_start)
    configs  = []
    for i in range(args.runs):
        seed      = args.seed_start + i
        loc       = LOCATIONS[i % len(LOCATIONS)]
        map_size  = MAP_SIZES[i % len(MAP_SIZES)]
        n_scouts  = args.scouts if args.scouts else rng_meta.randint(1, 4)
        n_fw      = args.fw     if args.fw     else rng_meta.randint(1, 2)
        configs.append((i + 1, seed, loc, map_size, n_scouts, n_fw))

    CSV_FIELDS = [
        "run_id", "seed", "loc_name", "map_size_m", "max_steps", "steps_done",
        "n_scouts", "n_fw",
        "scouts_survived", "scouts_surv_pct", "fire_discovered",
        "scout_dwell_pct", "scout_avg_reward",
        "fw_survived", "fw_surv_pct",
        "fw_water_dropped_L", "fw_refills", "fw_valve_open_pct", "fw_avg_reward",
        "fire_cells_init", "fire_cells_final",
        "fire_intensity_init", "fire_intensity_final",
        "fire_intensity_reduction", "burned_frac_pct", "fire_extinguished",
    ]

    rows = []
    with open(args.out, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()

        for run_id, seed, loc, map_size, n_scouts, n_fw in configs:
            loc_name, osm_lat, osm_lon, use_osm = loc
            print(f"\n[{run_id:>3}/{args.runs}] seed={seed}  loc={loc_name:<18}"
                  f"  mapa={int(map_size)}m  sc={n_scouts}  fw={n_fw}", flush=True)
            t0 = time.time()
            try:
                row = run_episode(
                    scout_actor, cmdr_actor, device,
                    n_quads=n_scouts, n_fw=n_fw, seed=seed,
                    grid_size_m=map_size, max_steps=args.max_steps,
                    loc_name=loc_name, osm_lat=osm_lat,
                    osm_lon=osm_lon, use_osm=use_osm,
                )
                row["run_id"] = run_id
                elapsed = time.time() - t0
                print(f"    surv={row['scouts_surv_pct']}%  disc={row['fire_discovered']}"
                      f"  dwell={row['scout_dwell_pct']}%  H2O={row['fw_water_dropped_L']}L"
                      f"  brn={row['burned_frac_pct']}%  t={elapsed:.1f}s", flush=True)
                # Print FW crash details
                fw_deaths = row.get("fw_deaths", {})
                for fname, dinfo in fw_deaths.items():
                    print(f"    ⚠ {fname} CRASHED step={dinfo['step']} "
                          f"mode={dinfo['mode']} pos={dinfo['pos']}", flush=True)
            except Exception as exc:
                print(f"    [CHYBA] {exc}")
                row = {f: "" for f in CSV_FIELDS}
                row["run_id"] = run_id
                row["seed"]   = seed
                row["loc_name"] = loc_name
                row["map_size_m"] = int(map_size)
                row["n_scouts"] = n_scouts
                row["n_fw"] = n_fw
            rows.append(row)
            writer.writerow(row)
            csvf.flush()

    # Souhrnný výpis
    print("\n" + "=" * 80)
    print("VÝSLEDKY")
    print("=" * 80)
    print_table(rows)

    # Textová verze tabulky
    txt_path = args.out.replace(".csv", ".txt")
    import io as _io
    buf = _io.StringIO()
    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = buf
    print_table(rows)
    _sys.stdout = old_stdout
    with open(txt_path, "w") as f:
        f.write(buf.getvalue())

    print(f"\nCSV uložen: {args.out}")
    print(f"TXT uložen: {txt_path}")

    # Agregované statistiky
    valid = [r for r in rows if r.get("scouts_surv_pct") != ""]
    if valid:
        print("\nAgregované průměry přes všechny runy:")
        for key, label in [
            ("scouts_surv_pct",        "Scout přežití [%]"),
            ("fire_discovered",        "Oheň nalezen [%]"),
            ("scout_dwell_pct",        "Dwell u ohně [%]"),
            ("fw_water_dropped_L",     "Voda svržena [L]"),
            ("fw_valve_open_pct",      "Ventil otevřen [%]"),
            ("burned_frac_pct",        "Spálená plocha [%]"),
            ("fire_intensity_reduction","Redukce intenzity"),
        ]:
            vals = [float(r[key]) for r in valid if r.get(key) not in ("", None)]
            if vals:
                print(f"  {label:<30}: {np.mean(vals):.2f}  ±{np.std(vals):.2f}")

        # FW crash analysis by mode
        all_deaths = []
        for r in valid:
            for fname, dinfo in r.get("fw_deaths", {}).items():
                all_deaths.append(dinfo)
        if all_deaths:
            total_fw = sum(int(r.get("n_fw", 0)) for r in valid)
            print(f"\n  FW crashes: {len(all_deaths)}/{total_fw}"
                  f" ({len(all_deaths)/total_fw*100:.0f}%)")
            mode_counts = {}
            mode_steps = {}
            for d in all_deaths:
                m = d['mode']
                mode_counts[m] = mode_counts.get(m, 0) + 1
                mode_steps.setdefault(m, []).append(d['step'])
            for m in sorted(mode_counts.keys()):
                avg_step = np.mean(mode_steps[m])
                print(f"    {m:>12}: {mode_counts[m]} crashes"
                      f"  (avg step={avg_step:.0f})")
        else:
            print(f"\n  FW crashes: 0 — all survived!")


if __name__ == "__main__":
    main()
