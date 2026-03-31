"""
train_fw_flight.py — Curriculum training: fixed-wing flight ONLY (no mission)
==============================================================================

Proč tenhle skript existuje:
  Commander (fixed-wing) se naučil jít přímo do země ("napálil do zeme").
  Scout se naučil jen viset někde poblíž a čekat na oheň.
  Obě chování jsou hluboce zakořeněná — nestačí fine-tuning.

  Řešení: nejdřív nauč commandera ZÁKLADNĚ LÉTAT, pak ho dotrenuj na misii.

Curriculum:
  Phase 1 (0–3 000 ep)  : Přežití  — max_steps=500,  autopilot=0.90
                           Cíl: nesesypat se, udržet výšku
  Phase 2 (3 000–8 000) : Patrol   — max_steps=1 200, autopilot=0.65
                           Cíl: kroužit kolem středu mapy
  Phase 3 (8 000+ ep)   : Volný let — max_steps=2 000, autopilot klesá 0.50→0.15
                           Cíl: létat bez pomoci autopilota

Výstupy:
  saved_models/fw_flight/commander_best.pt  ← načti do train.py jako path_to_commander
  saved_models/fw_flight/critic_best.pt

Reward (env_core.py s N_QUADS=0):
  Commander dostává pouze:
    • _apply_physics_shaping  (survival +0.05/krok, altitude, boundary penalty)
    • _get_fixed_reward_patrol (tah ke středu: max 0.05/krok)
  Žádná mise, žádný oheň, žádný scout.

Použití po vytrénování:
  V train.py nastav:
    path_to_commander = ".../saved_models/fw_flight/commander_best.pt"
    path_to_critic    = ""   # kritik má jiný global_state_dim (273 vs 288)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import time
from concurrent.futures import ProcessPoolExecutor

# ─────────────────────────────────────────────────────────────────────────────
# 1. CURRICULUM SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def get_curriculum_params(episodes_played: int) -> dict:
    """Vrátí parametry curriculumu podle počtu odehraných epizod."""
    if episodes_played < 3_000:
        return dict(
            phase=1, name="Phase 1 — Basic Survival",
            max_steps=500, autopilot=0.90, num_workers=20,
            disable_fire=True,   # BEZ OHNE — zabrání náhodným spike rewardům (ZASAH!).
                                 # Autopilot dává water=-1, ale policy může náhodně
                                 # zmáčknout water=+1 nad centrem mapy (kde je oheň)
                                 # a dostat +18 reward za náhodu — to korupuje učení.
        )
    elif episodes_played < 8_000:
        return dict(
            phase=2, name="Phase 2 — Patrol Orbit",
            max_steps=1_200, autopilot=0.65, num_workers=14,  # méně workerů — OOM fix pro delší epizody
            disable_fire=True,   # Stále bez ohne — fokus na patrol, ne hašení
        )
    else:
        # Phase 3: autopilot klesá z 0.50 na 0.15 přes 15 000 epizod
        decay = min(1.0, (episodes_played - 8_000) / 15_000.0)
        ap    = max(0.15, 0.50 - 0.35 * decay)
        return dict(
            phase=3, name="Phase 3 — Free Flight",
            max_steps=2_000, autopilot=round(ap, 3), num_workers=10,  # méně workerů — 2000 kroků = velké buffery
            disable_fire=True,   # BEZ OHNE ve všech fázích flight curriculumu.
                                 # Oheň způsobuje náhodné ZASAH! spike rewardy (až +744!)
                                 # kdykoli policy náhodně stiskne water=+1 nad centrem mapy.
                                 # Commander se má naučit LÉTAT, ne hasit — to přijde v mission fine-tuning.
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. WORKER FUNCTION  (běží na CPU, žádný scout)
# ─────────────────────────────────────────────────────────────────────────────

def collect_fw_flight_worker(num_eps, cmdr_w, critic_w, config, batch_start_idx):
    """
    Flight-curriculum worker.  Žádný scout, vylepšený autopilot.

    Autopilot (jen pro demonstraci):
      • Pitch: PD regulátor — drží výšku 60 m
      • Roll: otočí se ke středu mapy pokud je daleko, jinak mírný oblet
      • Throttle: konstantní cestovní rychlost (0.30 → fyzicky ~55 % max)
      • Water: vždy OFF (-1.0)

    Returns
    -------
    out_cmdr   : dict  — fixní_states / msgs / masks / actions / logprobs / returns
    out_crit   : dict  — g_states / values / returns
    out_init_h : dict  — počáteční GRU hidden states (cmdr, crit_cmdr)
    rewards    : list[float]
    lifespans  : list[float]
    """
    # ---- worker-side imports (spawn-safe) -----------------------------------
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    # Ujisti se, že pracovní adresář je src/ (aby URDF cesty fungovaly)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import torch
    torch.set_num_threads(1)
    import numpy as np

    from env_core import DroneFireEnv
    from models import CommanderActor, MAPPOCritic

    N_QUADS   = 0
    N_FIXED   = 1
    max_steps = config['max_steps']

    # ── Lokální sítě ─────────────────────────────────────────────────────────
    local_cmdr = CommanderActor(
        self_state_dim=config['fixed_self_dim'],
        msg_input_dim=config['scout_msg_dim']
    )
    local_cmdr.load_state_dict(cmdr_w)
    local_cmdr.eval()

    local_critic = MAPPOCritic(config['global_state_dim'])
    local_critic.load_state_dict(critic_w)
    local_critic.eval()

    # ── Dummy scout-message tensory (all-zero, all-masked) ───────────────────
    # Commander dostane "ticho" od scoutů — jako kdyby žádný neexistoval.
    scout_msg_dim = config['scout_msg_dim']
    d_msgs  = torch.zeros(1, 1, scout_msg_dim)  # [1, 1, 5]
    d_msg_m = torch.ones(1, 1, dtype=torch.bool)  # [1, 1] — vše maskováno

    # ── Buffery ───────────────────────────────────────────────────────────────
    cmdr_buf = {k: [] for k in ["fixed_states", "incoming_msgs", "msg_masks",
                                 "actions", "logprobs", "returns"]}
    crit_buf = {k: [] for k in ["g_states", "values", "returns"]}

    cmdr_h0_list = []   # počáteční hidden state commandera (per episode)
    crit_h0_list = []   # počáteční hidden state kritika   (per episode)

    all_rewards   = []
    all_lifespans = []

    # ── Prostředí (jedno na workera) ─────────────────────────────────────────
    local_env = DroneFireEnv(
        num_quads=N_QUADS, num_fixed=N_FIXED,
        grid_size_m=2000.0, max_steps=max_steps
    )

    # ── Sběr epizod ──────────────────────────────────────────────────────────
    disable_fire = config.get('disable_fire', True)
    for ep_off in range(num_eps):
        obs, _ = local_env.reset(epizode_number=batch_start_idx + ep_off)

        # Zhaš oheň ihned po resetu pokud je disable_fire=True.
        # env.reset() vytvoří fire grid a rozní oheň; tady ho okamžitě vynulujeme.
        # Tím el eliminujeme náhodné ZASAH! spike rewardů (+18) které korupují
        # učení letu — policy náhodně zmáčkne water a dostane jackpot za náhodu.
        if disable_fire and local_env.sim.environment.fire_grid is not None:
            local_env.sim.environment.fire_grid.B[:] = False   # žádné hořící buňky
            local_env.sim.environment.fire_grid.I[:] = 0.0     # nulová intenzita
            local_env.sim.drone_extinguish_stats = {}           # nulové hašení

        f_agent = local_env.fixed_agents[0]

        # GRU hidden states na začátku epizody = nuly
        cmdr_h = torch.zeros(1, 1, 128)
        crit_h = torch.zeros(1, 1, 128)

        # Ulož počáteční h_0 (pro PPO re-run na hlavním procesu)
        cmdr_h0_list.append(cmdr_h.clone())
        crit_h0_list.append(crit_h.clone())

        # Per-epizodní data [list-of-dicts, jeden za krok]
        ep_data   = []
        lifespan  = max_steps   # přepíšeme při první terminaci
        ep_reward = 0.0

        # Hod kostkou jednou na celou epizodu (konzistentní GRU trajektorie)
        use_autopilot = (np.random.random() < config.get('autopilot_prob', 0.5))

        for step in range(max_steps):
            g_state  = local_env.state()
            g_tensor = torch.FloatTensor(g_state).unsqueeze(0)  # [1, global_dim]

            if f_agent not in local_env.agents:
                # Commander je mrtvý — dummy krok (nulová odměna, nulové akce)
                ep_data.append({
                    "dead": True,
                    "gs":   g_tensor,
                    "val":  torch.tensor([[0.0]]),
                    "ret":  0.0,
                    "reward": 0.0,
                })
                continue

            # ── Inference (bez gradientů) ─────────────────────────────────
            with torch.no_grad():
                s_st = torch.FloatTensor(obs[f_agent]["self_state"]).unsqueeze(0)  # [1, 17]
                dist_out, _, h_out   = local_cmdr(s_st, d_msgs, d_msg_m, cmdr_h)
                val,       c_h_out   = local_critic(g_tensor, crit_h)
                act = dist_out.sample()  # [1, 4]

            # ── Vylepšený autopilot (flight mode) ────────────────────────
            if use_autopilot and f_agent in local_env.sim.drones:
                f_pos = local_env.sim.drones[f_agent].get_position()
                f_vel = local_env.sim.drones[f_agent].get_velocity()

                # PD výškový regulátor — cíl 60 m
                alt_err  = 60.0 - f_pos[2]
                vz       = f_vel[2]
                pitch_pd = float(np.clip(0.007 * alt_err - 0.015 * vz, -0.25, 0.25))

                # Roll: otočit se ke středu pokud je dál než 300 m, jinak mírný oblet
                dist_c = float(np.sqrt(f_pos[0]**2 + f_pos[1]**2))
                if dist_c > 300.0:
                    # Vypočítej chybu headingu ke středu mapy
                    ang_to_c = float(np.arctan2(-f_pos[1], -f_pos[0]))
                    chi      = float(local_env.sim.drones[f_agent].state_chi)
                    h_err    = (ang_to_c - chi + np.pi) % (2.0 * np.pi) - np.pi
                    roll_cmd = float(np.clip(h_err / np.pi * 0.8, -0.8, 0.8))
                else:
                    # Blízko středu — mírný pravotočivý oblet
                    roll_cmd = 0.25

                # 80% šance aplikovat autopilot na tento krok (zbylých 20% — policy volná)
                if np.random.random() < 0.80:
                    act[0, 0] = roll_cmd    # roll
                    act[0, 1] = pitch_pd    # pitch (výška)
                    act[0, 2] = 0.30        # throttle → ~55 % fyzické rychlosti
                    act[0, 3] = -1.0        # water OFF

            # Ulož krok
            ep_data.append({
                "dead":  False,
                "self":  s_st,
                "msgs":  d_msgs,
                "mmask": d_msg_m,
                "h":     cmdr_h,
                "act":   act,
                "lp":    dist_out.log_prob(act).sum(1),  # [1]
                "val":   val,                             # [1, 1]
                "gs":    g_tensor,
                "reward": 0.0,   # doplněno po env.step()
            })

            # Posuň hidden states
            cmdr_h, crit_h = h_out, c_h_out

            # ── Environment step ─────────────────────────────────────────
            actions = {f_agent: act.squeeze(0).numpy()}
            if local_env.agents:
                obs, rewards_env, terms, _, _ = local_env.step(actions)
                r = rewards_env.get(f_agent, 0.0)
                ep_data[-1]["reward"] = r
                ep_reward += r
                if terms.get(f_agent, False) and lifespan == max_steps:
                    lifespan = step
            # Pokud jsou všichni mrtví, reward zůstane 0.0

        # ── GAE (Generalized Advantage Estimation) ────────────────────────────
        gamma   = config.get('gamma', 0.99)
        gae_lam = config.get('gae_lambda', 0.95)
        z_val   = torch.tensor([[0.0]])
        gae     = 0.0

        for i in reversed(range(max_steps)):
            d    = ep_data[i]
            v_t  = d.get("val", z_val).item()
            r_t  = d.get("reward", 0.0)
            v_np = ep_data[i + 1].get("val", z_val).item() if i < max_steps - 1 else 0.0
            delta = r_t + gamma * v_np - v_t
            gae   = delta + gamma * gae_lam * gae
            d["ret"] = gae + v_t

        # ── Napakuj buffery (step-by-step, konzistentní pořadí) ──────────────
        d_fixed_zero = torch.zeros(1, config['fixed_self_dim'])
        z_val_t      = torch.tensor([[0.0]])

        for d in ep_data:
            # Kritik — vždy jeden záznam za krok (num_agents=1)
            crit_buf["g_states"].append(d["gs"])
            crit_buf["values"].append(d.get("val", z_val_t))
            crit_buf["returns"].append(d.get("ret", 0.0))

            # Commander
            if d["dead"]:
                cmdr_buf["fixed_states"].append(d_fixed_zero)
                cmdr_buf["incoming_msgs"].append(d_msgs)
                cmdr_buf["msg_masks"].append(d_msg_m)
                cmdr_buf["actions"].append(torch.zeros(1, 4))
                cmdr_buf["logprobs"].append(torch.tensor([0.0]))
            else:
                cmdr_buf["fixed_states"].append(d["self"])
                cmdr_buf["incoming_msgs"].append(d["msgs"])
                cmdr_buf["msg_masks"].append(d["mmask"])
                cmdr_buf["actions"].append(d["act"])
                cmdr_buf["logprobs"].append(d["lp"])
            cmdr_buf["returns"].append(d.get("ret", 0.0))

        all_rewards.append(ep_reward)
        all_lifespans.append(float(lifespan))

    # ── Finální torch.cat ─────────────────────────────────────────────────────
    def _cat(lst):
        return torch.cat(lst, dim=0)

    out_cmdr = {
        "fixed_states":  _cat(cmdr_buf["fixed_states"]),            # [T, 17]
        "incoming_msgs": _cat(cmdr_buf["incoming_msgs"]),            # [T, 1, 5]
        "msg_masks":     _cat(cmdr_buf["msg_masks"]),                # [T, 1]
        "actions":       _cat(cmdr_buf["actions"]),                  # [T, 4]
        "logprobs":      _cat(cmdr_buf["logprobs"]),                 # [T]
        "returns":       torch.tensor(cmdr_buf["returns"], dtype=torch.float32),  # [T]
    }
    out_crit = {
        "g_states": _cat(crit_buf["g_states"]),                      # [T, global_dim]
        "values":   _cat(crit_buf["values"]),                        # [T, 1]
        "returns":  torch.tensor(crit_buf["returns"], dtype=torch.float32),  # [T]
    }
    # Per-epizodní h_0: [num_eps, 1, 128] → squeeze dim 1 → [num_eps, 128]
    out_init_h = {
        "cmdr":      torch.cat(cmdr_h0_list, dim=0),  # [num_eps, 1, 128]
        "crit_cmdr": torch.cat(crit_h0_list, dim=0),  # [num_eps, 1, 128]
    }

    return out_cmdr, out_crit, out_init_h, all_rewards, all_lifespans


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRÉNOVACÍ FUNKCE
# ─────────────────────────────────────────────────────────────────────────────

def train_fw_flight(resume_episodes: int = 0,
                    resume_commander: str = "",
                    resume_critic: str = ""):
    """
    Parameters
    ----------
    resume_episodes  : int  -- počet již odehraných epizod; curriculum začne ve správné fázi.
    resume_commander : str  -- cesta k checkpointu commandera (prázdná = od nuly).
    resume_critic    : str  -- cesta k checkpointu kritika   (prázdná = od nuly).
    """
    print("=" * 70)
    print("  Fixed-Wing FLIGHT Curriculum Training")
    print("  Naučíme commandera létat PŘED tím, než ho pustíme na misii.")
    if resume_episodes > 0:
        print(f"  RESUME: začínáme od epizody {resume_episodes}")
    print("=" * 70)

    # Ujisti se, že cwd = src/ (URDF cesty, importy)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from env_core import DroneFireEnv
    from models import CommanderActor, MAPPOCritic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(4)
    print(f"Device: {device}\n")

    # ── Hyperparametry ───────────────────────────────────────────────────────
    N_QUADS = 0   # žádný scout — izolujeme problém letu
    N_FIXED = 1

    num_episodes   = 50_000   # celkový počet epizod curriculumu
    gamma          = 0.99
    gae_lambda     = 0.95
    clip_coef      = 0.2
    update_epochs  = 4
    eps_per_worker = 2        # num_workers se mění s curriculum fází

    lr_commander = 1e-4   # sníženo z 3e-4 — loss spiky přepisují naučené chování
    lr_critic    = 1e-4

    # Načtení checkpointu (prázdné = od nuly, jinak resume)
    path_to_commander = resume_commander
    path_to_critic    = resume_critic

    # ── Dimenze sítí (dotáž se env) ─────────────────────────────────────────
    # Poznámka: global_state_dim = 256 + N_QUADS*15 + N_FIXED*17
    #           S N_QUADS=0 → 256+0+17 = 273.
    #           To se liší od mission trénování (N_QUADS=1 → 288).
    #           CommanderActor váhy jsou ale přenositelné (17→64 MLP nezáleží).
    #           Kritik se musí reinicializovat při startu mission tréninku.
    temp_env = DroneFireEnv(
        num_quads=N_QUADS, num_fixed=N_FIXED,
        grid_size_m=2000.0, max_steps=500
    )
    fixed_self_dim   = temp_env.observation_space(temp_env.fixed_agents[0])["self_state"].shape[0]
    global_state_dim = temp_env.state_space.shape[0]
    scout_msg_dim    = 5  # odpovídá dummy d_msgs ve workeru

    # Uklid temp_env (aby nedošlo ke konfliktu portu s PyBulletem)
    if hasattr(temp_env, 'sim') and temp_env.sim is not None:
        temp_env.sim.stop_simulation()

    print(f"fixed_self_dim   = {fixed_self_dim}")
    print(f"global_state_dim = {global_state_dim}  (N_QUADS=0; bude jiné v mission trénování!)")
    print(f"scout_msg_dim    = {scout_msg_dim}  (dummy vstupy)")

    worker_config = {
        'N_QUADS': N_QUADS, 'N_FIXED': N_FIXED,
        'grid_size_m': 2000.0,
        'max_steps': 500,           # aktualizuje se per-batch z curriculumu
        'fixed_self_dim': fixed_self_dim,
        'scout_msg_dim': scout_msg_dim,
        'global_state_dim': global_state_dim,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'autopilot_prob': 0.90,     # aktualizuje se per-batch z curriculumu
        'disable_fire': True,       # aktualizuje se per-batch z curriculumu
    }

    # ── Sítě ─────────────────────────────────────────────────────────────────
    commander_actor = CommanderActor(
        self_state_dim=fixed_self_dim,
        msg_input_dim=scout_msg_dim
    ).to(device)

    if path_to_commander and os.path.exists(path_to_commander):
        ckpt    = torch.load(path_to_commander, map_location=device)
        shapes  = {k: v.shape for k, v in commander_actor.state_dict().items()}
        filtered = {k: v for k, v in ckpt.items()
                    if k in shapes and v.shape == shapes[k]}
        skipped  = [k for k in ckpt if k not in filtered]
        commander_actor.load_state_dict(filtered, strict=False)
        print(f"📥 Commander načten z {path_to_commander}")
        if skipped:
            print(f"   ↳ Přeskočeno: {skipped}")
    else:
        print("⚡ Commander: inicializace od nuly")

    critic = MAPPOCritic(global_state_dim).to(device)
    if path_to_critic and os.path.exists(path_to_critic):
        critic.load_state_dict(torch.load(path_to_critic, map_location=device))
        print(f"📥 Kritik načten z {path_to_critic}")
    else:
        print("⚡ Kritik: inicializace od nuly")

    # ── Optimalizátor ────────────────────────────────────────────────────────
    optimizer = optim.Adam([
        {"params": commander_actor.parameters(), "lr": lr_commander},
        {"params": critic.parameters(),          "lr": lr_critic},
    ])
    # episodes_per_batch se mění dynamicky s fází — použijeme konzervativní odhad
    # (Phase 1: 20*2=40, Phase 2: 14*2=28, Phase 3: 10*2=20); round up aby smyčka nepřeskočila
    num_batches = num_episodes // 20   # 20 = nejmenší možná batch (Phase 3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-5
    )

    # ── Tracking ─────────────────────────────────────────────────────────────
    reward_history   = []
    loss_history     = []
    lifespan_history = []

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "saved_models", "fw_flight")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpointy ukládám do: {save_dir}\n")

    best_avg        = -1e9
    episodes_played = resume_episodes   # curriculum začné ve správné fázi
    last_phase      = -1
    # Počet batchí již odehraných (pro správný offset batch_idx a scheduler)
    batches_already = resume_episodes // (eps_per_worker * 14)  # odhad; neovlivňuje PPO, jen log

    # ── Hlavní trénovací smyčka ───────────────────────────────────────────────
    # Pool se vytváří PER BATCH (ne jednou pro celý trénink).
    # Důvod: při přechodu fází (max_steps 500→1200→2000) se dramaticky mění
    # velikost bufferů. Dlouhý pool si pamatuje workers z předchozí fáze
    # a nový větší buffer je OOM → OS worker zabije → BrokenProcessPool.
    # Overhead: ~0.5s spawn při každém batchi vs ~10s rollout = zanedbatelné.
    for batch_idx in range(1, num_batches + 1):

        # -- Aktualizuj curriculum --------------------------------------------
        cur = get_curriculum_params(episodes_played)
        if cur["phase"] != last_phase:
            print(f"\n{'='*60}")
            print(f"  CURRICULUM PŘECHOD → {cur['name']}")
            print(f"  ep={episodes_played}, max_steps={cur['max_steps']}, "
                  f"autopilot={cur['autopilot']:.2f}, workers={cur['num_workers']}")
            print(f"{'='*60}\n")
            last_phase = cur["phase"]

        num_workers_now = cur['num_workers']
        worker_config['max_steps']      = cur['max_steps']
        worker_config['autopilot_prob'] = cur['autopilot']
        worker_config['disable_fire']   = cur['disable_fire']
        max_steps = cur['max_steps']
        episodes_per_batch = num_workers_now * eps_per_worker

        # Entropy annealing: 0.02 → 0.005 přes 500 batchí
        _ann = 500
        entropy_coef = max(
            0.005,
            0.02 - (0.02 - 0.005) * min(batch_idx - 1, _ann) / _ann
        )

        # -- Snapshot vah pro workery -----------------------------------------
        cmdr_w   = {k: v.cpu() for k, v in commander_actor.state_dict().items()}
        critic_w = {k: v.cpu() for k, v in critic.state_dict().items()}

        # -- Rollout: nový pool pro každý batch (OOM-safe) --------------------
        rollout_start = time.time()
        batch_cmdr = {k: [] for k in ["fixed_states", "incoming_msgs",
                                       "msg_masks", "actions",
                                       "logprobs", "returns"]}
        batch_crit = {k: [] for k in ["g_states", "values", "returns"]}
        batch_h_cmdr = []
        batch_h_crit = []
        batch_rewards = []

        with ProcessPoolExecutor(max_workers=num_workers_now) as executor:
            futures = [
                executor.submit(
                    collect_fw_flight_worker,
                    eps_per_worker, cmdr_w, critic_w,
                    worker_config,
                    episodes_played + i * eps_per_worker
                )
                for i in range(num_workers_now)
            ]
            for fut in futures:
                w_cmdr, w_crit, w_h, w_rew, w_life = fut.result()
                batch_rewards.extend(w_rew)
                lifespan_history.extend(w_life)
                reward_history.extend(w_rew)
                episodes_played += len(w_rew)

                for k in batch_cmdr: batch_cmdr[k].append(w_cmdr[k])
                for k in batch_crit: batch_crit[k].append(w_crit[k])
                batch_h_cmdr.append(w_h["cmdr"])      # [eps_per_worker, 1, 128]
                batch_h_crit.append(w_h["crit_cmdr"])

        rollout_time = time.time() - rollout_start

        avg_batch = float(np.mean(batch_rewards))
        win = min(40, len(reward_history))
        avg_roll  = float(np.mean(reward_history[-win:]))
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} | "
              f"Batch {batch_idx:04d} (Ep {episodes_played:05d}) | "
              f"[{cur['name']}] "
              f"Batch: {avg_batch:+6.1f}  Roll{win}: {avg_roll:+6.1f}  "
              f"AP={cur['autopilot']:.2f}  Rollout: {rollout_time:.1f}s")

        # -- Uložení nejlepšího checkpointu -------------------------------
        if episodes_played >= 40 and avg_roll > best_avg:
            best_avg = avg_roll
            torch.save(commander_actor.state_dict(),
                       os.path.join(save_dir, "commander_best.pt"))
            torch.save(critic.state_dict(),
                       os.path.join(save_dir, "critic_best.pt"))
            print(f"   ⭐ Nový nejlepší model! Rolling avg = {best_avg:.2f}")

        # Periodický checkpoint každých 50 batchí
        if batch_idx % 50 == 0:
            torch.save(commander_actor.state_dict(),
                       os.path.join(save_dir, f"commander_b{batch_idx:04d}.pt"))

        # ── PPO UPDATE (GPU) ─────────────────────────────────────────────
        num_agents = 1                    # jen commander
        # episodes_per_batch se mění s fází (14*2=28 nebo 10*2=20), ne fixin na 40

        # Cat worker výstupy do jednoho tensoru per batch
        c_fixed    = torch.cat(batch_cmdr["fixed_states"]).to(device)   # [40*T, 17]
        c_msgs     = torch.cat(batch_cmdr["incoming_msgs"]).to(device)   # [40*T, 1, 5]
        c_msg_m    = torch.cat(batch_cmdr["msg_masks"]).to(device)       # [40*T, 1]
        c_actions  = torch.cat(batch_cmdr["actions"]).to(device)         # [40*T, 4]
        c_logprobs = torch.cat(batch_cmdr["logprobs"]).to(device)        # [40*T]
        c_returns  = torch.cat(batch_cmdr["returns"]).to(device)         # [40*T]

        cr_g     = torch.cat(batch_crit["g_states"]).to(device)          # [40*T, global_dim]
        cr_vals  = torch.cat(batch_crit["values"]).to(device)            # [40*T, 1]
        cr_rets  = torch.cat(batch_crit["returns"]).to(device)           # [40*T]

        # Počáteční hidden states:
        #   každý worker vrátí [eps_per_worker, 1, 128]
        #   po cat: [40, 1, 128] → squeeze(1) → [40, 128] → unsqueeze(0) → [1, 40, 128]
        def _mk_h(lst):
            return (torch.cat(lst, dim=0)          # [40, 1, 128]
                    .squeeze(1)                     # [40, 128]
                    .unsqueeze(0)                   # [1, 40, 128]
                    .to(device))

        h_cmdr = _mk_h(batch_h_cmdr)  # [1, 40, 128]
        h_crit = _mk_h(batch_h_crit)  # [1, 40, 128]

        # Výhoda: A(s,a) = G - V(s)
        cr_adv = cr_rets.unsqueeze(1) - cr_vals.detach()  # [40*T, 1]
        cr_adv = (cr_adv - cr_adv.mean()) / (cr_adv.std() + 1e-8)

        # Reshape na sekvence [episodes, max_steps, ...]
        cr_adv_seq  = cr_adv.view(episodes_per_batch, max_steps, num_agents)   # [N, T, 1]
        c_adv_seq   = cr_adv_seq[:, :, 0]                                       # [N, T]

        c_fixed_seq    = c_fixed.view(episodes_per_batch, max_steps, -1)        # [N, T, 17]
        c_msgs_seq     = c_msgs.view(episodes_per_batch, max_steps,
                                     c_msgs.size(-2), c_msgs.size(-1))          # [N, T, 1, 5]
        c_msg_m_seq    = c_msg_m.view(episodes_per_batch, max_steps, -1)        # [N, T, 1]
        c_actions_seq  = c_actions.view(episodes_per_batch, max_steps, -1)      # [N, T, 4]
        c_logprobs_seq = c_logprobs.view(episodes_per_batch, max_steps)         # [N, T]

        # h_cmdr: [1, N, 128] → transpose → h_cmdr_seq: [N, 1, 128]
        h_cmdr_seq = h_cmdr.transpose(0, 1)   # [N, 1, 128]

        # Kritik: [N*T, global_dim] → [N, T, 1, global_dim] → [N, 1, T, global_dim]
        cr_g_seq   = cr_g.view(episodes_per_batch, max_steps, num_agents, -1).transpose(1, 2)   # [N, 1, T, dim]
        cr_ret_seq = cr_rets.view(episodes_per_batch, max_steps, num_agents).transpose(1, 2)    # [N, 1, T]

        h_crit_seq = h_crit.transpose(0, 1)   # [N, 1, 128]

        # ── Gradient update loop ──────────────────────────────────────────
        num_minibatches = 4
        mb_size   = max(1, episodes_per_batch // num_minibatches)   # epizody na minibatch
        batch_loss = 0.0

        for epoch in range(update_epochs):
            b_inds  = np.random.permutation(episodes_per_batch)
            ep_loss = 0.0

            for start in range(0, episodes_per_batch, mb_size):
                mb     = b_inds[start:start + mb_size]
                curr_mb = len(mb)

                # ── Commander policy loss (PPO-clip) ──────────────────
                mb_fixed  = c_fixed_seq[mb]                            # [mb, T, 17]
                mb_msgs   = c_msgs_seq[mb]                             # [mb, T, 1, 5]
                mb_mmask  = c_msg_m_seq[mb]                            # [mb, T, 1]
                mb_acts   = c_actions_seq[mb]                          # [mb, T, 4]
                mb_old_lp = c_logprobs_seq[mb].view(-1)               # [mb*T]
                mb_adv    = c_adv_seq[mb].reshape(-1)                  # [mb*T]
                mb_adv    = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                mb_h_c    = h_cmdr_seq[mb].transpose(0, 1)            # [1, mb, 128]

                dist_c, _, _ = commander_actor(mb_fixed, mb_msgs, mb_mmask, mb_h_c)
                flat_acts    = mb_acts.view(-1, 4)                     # [mb*T, 4]
                new_lp       = dist_c.log_prob(flat_acts).sum(1)       # [mb*T]
                entropy      = dist_c.entropy().sum(1).mean()

                log_ratio = (new_lp - mb_old_lp).clamp(-10.0, 10.0)
                ratio     = torch.exp(log_ratio)
                pg1       = -mb_adv * ratio
                pg2       = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                policy_loss = torch.max(pg1, pg2).mean()

                # ── Critic value loss ─────────────────────────────────
                # mb_cr_g: [mb, 1, T, dim] → reshape → [mb*1, T, dim]
                mb_cr_g   = cr_g_seq[mb].reshape(curr_mb * num_agents, max_steps, -1)
                mb_cr_ret = cr_ret_seq[mb].reshape(-1, 1)              # [mb*T, 1]

                # hidden state pro kritik: [mb, 1, 128] → [1, mb*1, 128]
                mb_h_cr = h_crit_seq[mb].reshape(1, curr_mb * num_agents, -1)

                new_vals, _ = critic(mb_cr_g, mb_h_cr)                # [mb*T, 1]
                value_loss  = nn.MSELoss()(new_vals, mb_cr_ret)

                # Celková ztráta
                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

                if not torch.isfinite(loss):
                    print(f"⚠️  Non-finite loss {loss.item():.4e} — přeskakuji minibatch")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()
                params = list(commander_actor.parameters()) + list(critic.parameters())
                nn.utils.clip_grad_norm_(params, max_norm=0.3)   # sníženo z 0.5 — loss spiky

                if any(torch.isnan(p.data).any() for p in params):
                    raise RuntimeError(
                        "NaN váhy po optimizer.step() — trénink nestabilní. "
                        "Zkus snížit lr nebo zkontrolovat reward scale."
                    )

                optimizer.step()
                ep_loss += loss.item()

            batch_loss += ep_loss / num_minibatches

        scheduler.step()
        loss_history.append(batch_loss / update_epochs)

        # Periodický plot každých 100 batchí
        if batch_idx % 100 == 0:
            _save_flight_plot(reward_history, loss_history, lifespan_history,
                              save_dir, batch_idx)

    print("\n✅ Flight curriculum dokončen!")
    print(f"   Nejlepší commander: {save_dir}/commander_best.pt")
    print(f"\nProveditelné kroky po dokončení:")
    print("  1. Zkopíruj commander_best.pt do saved_models/")
    print("  2. V train.py nastav:")
    print("       path_to_commander = 'saved_models/fw_flight/commander_best.pt'")
    print("       path_to_critic    = ''   # kritik má jiný global_state_dim (273 vs 288)")
    print("       N_QUADS = 1, N_FIXED = 1")
    print("  3. Spusť train.py pro mission fine-tuning")

    _save_flight_plot(reward_history, loss_history, lifespan_history, save_dir, batch_idx)


# ─────────────────────────────────────────────────────────────────────────────
# 4. HELPER: výstupní plot
# ─────────────────────────────────────────────────────────────────────────────

def _save_flight_plot(rewards, losses, lifespans, save_dir, batch_idx):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"FW Flight Curriculum — Batch {batch_idx}", fontsize=12)

    # Odměna
    axes[0].plot(rewards, alpha=0.3, color='steelblue', linewidth=0.5)
    if len(rewards) >= 20:
        ma = np.convolve(rewards, np.ones(20) / 20, mode='valid')
        axes[0].plot(range(19, len(rewards)), ma, color='navy', linewidth=1.5, label='MA 20')
        axes[0].legend()
    axes[0].set_title("Reward per Episode")
    axes[0].set_xlabel("Episodes")
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(losses, color='tomato', linewidth=1)
    axes[1].set_title("PPO Loss (per batch)")
    axes[1].set_xlabel("Batches")
    axes[1].grid(True, alpha=0.3)

    # Lifespan
    axes[2].plot(lifespans, alpha=0.4, color='orange', linewidth=0.5)
    if len(lifespans) >= 20:
        ma2 = np.convolve(lifespans, np.ones(20) / 20, mode='valid')
        axes[2].plot(range(19, len(lifespans)), ma2, color='darkorange', linewidth=1.5)
    axes[2].set_title("Avg Lifespan (steps)")
    axes[2].set_xlabel("Episodes")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_b{batch_idx:04d}.png"), dpi=100)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FW Flight Curriculum Training")
    parser.add_argument(
        "--resume-episodes", type=int, default=0,
        help="Počet již odehraných epizod (pro správný curriculum start). "
             "Např. 6000 pro restart po paťdu při batchi 150."
    )
    parser.add_argument(
        "--resume-commander", type=str,
        default="../saved_models/fw_flight/commander_b0150.pt",
        help="Cesta k checkpointu commandera."
    )
    parser.add_argument(
        "--resume-critic", type=str,
        default="../saved_models/fw_flight/critic_best.pt",
        help="Cesta k checkpointu kritika."
    )
    args = parser.parse_args()

    train_fw_flight(
        resume_episodes=args.resume_episodes,
        resume_commander=args.resume_commander,
        resume_critic=args.resume_critic,
    )
