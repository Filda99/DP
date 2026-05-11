"""Quick evaluation with water-drop diagnostics."""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'src'))

from env_core import DroneFireEnv
from models import ScoutActor, CommanderActor
from commander_control import CommanderController

scout_path = sys.argv[1] if len(sys.argv) > 1 else '../saved_models/finetune/scout_best.pt'
cmdr_path  = sys.argv[2] if len(sys.argv) > 2 else '../saved_models/finetune/cmdr_best.pt'

N_QUADS, N_FIXED = 3, 2

# Create temp env to get dims
_tenv = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=1000.0,
                     max_steps=10, use_osm=False)
_tenv.reset(seed=999)
scout_self_dim = _tenv.observation_space(_tenv.quad_agents[0])["self_state"].shape[0]
cmdr_self_dim  = _tenv.observation_space(_tenv.fixed_agents[0])["self_state"].shape[0]
scout_msg_dim  = 5
del _tenv

scout_actor = ScoutActor(self_state_dim=scout_self_dim, msg_dim=scout_msg_dim)
scout_actor.load_state_dict(torch.load(scout_path, map_location='cpu', weights_only=True))
scout_actor.eval()

cmdr_actor = CommanderActor(self_state_dim=cmdr_self_dim, msg_input_dim=scout_msg_dim)
cmdr_actor.load_state_dict(torch.load(cmdr_path, map_location='cpu', weights_only=True))
cmdr_actor.eval()

results = []
all_alts, all_dists = [], []

for ep in range(10):
    env = DroneFireEnv(num_quads=N_QUADS, num_fixed=N_FIXED, grid_size_m=1000.0,
                       max_steps=1000, use_osm=False)
    obs, _ = env.reset(seed=ep)
    map_half = env.map_bounds

    scout_h = {q: torch.zeros(1, 1, 128) for q in env.quad_agents}
    cmdr_h  = {f: torch.zeros(1, 1, 64)  for f in env.fixed_agents}
    scout_msg = {q: torch.zeros(1, scout_msg_dim) for q in env.quad_agents}
    cmdr_ctrl = {}
    for f in env.fixed_agents:
        c = CommanderController(); c.reset(map_half)
        cmdr_ctrl[f] = c

    total_ext, water_drops, water_hits = 0.0, 0, 0
    fw_alive = {f: True for f in env.fixed_agents}

    for step in range(1000):
        if not env.agents:
            break
        actions = {}

        for q in env.quad_agents:
            if q in env.agents:
                with torch.no_grad():
                    lm = torch.FloatTensor(obs[q]['local_map']).unsqueeze(0)
                    ss = torch.FloatTensor(obs[q]['self_state']).unsqueeze(0)
                    ns = torch.FloatTensor(obs[q]['neighbor_states']).unsqueeze(0)
                    nm = torch.BoolTensor(obs[q]['neighbor_mask']).unsqueeze(0)
                    dist_s, msg_s, h_s = scout_actor(lm, ss, ns, nm, scout_h[q])
                    actions[q] = dist_s.sample().squeeze(0).numpy()
                    scout_h[q] = h_s
                    scout_msg[q] = msg_s.detach()

        for f in env.fixed_agents:
            if f not in env.agents or not fw_alive[f]:
                continue
            fw_drone = env.sim.drones.get(f)
            if fw_drone is None:
                continue

            in_emerg = cmdr_ctrl[f].check_boundary_emergency(fw_drone.get_position())
            if in_emerg:
                cmdr_ctrl[f].need_new_waypoint = True

            if cmdr_ctrl[f].need_new_waypoint:
                msgs_t = torch.stack([scout_msg[q] for q in env.quad_agents], dim=1)
                msgs_m = torch.tensor([[False] * N_QUADS])

                fw_neigh, fw_mask = [], []
                my_pos = fw_drone.get_position()
                for of in env.fixed_agents:
                    if of == f:
                        continue
                    if of in env.sim.drones:
                        op = env.sim.drones[of].get_position()
                        fw_neigh.append([(op[0]-my_pos[0])/map_half,
                                         (op[1]-my_pos[1])/map_half,
                                         (op[2]-my_pos[2])/100.0])
                        fw_mask.append(False)
                    else:
                        fw_neigh.append([0, 0, 0])
                        fw_mask.append(True)
                if not fw_neigh:
                    fw_neigh, fw_mask = [[0, 0, 0]], [True]

                cmdr_h[f], _ = cmdr_ctrl[f].decide_waypoint(
                    fw_drone, obs[f]['self_state'], env,
                    cmdr_actor, cmdr_h[f], msgs_t, msgs_m,
                    deterministic=True,
                    fw_neighbor_states=torch.FloatTensor([fw_neigh]),
                    fw_neighbor_mask=torch.BoolTensor([fw_mask]),
                    in_emergency=in_emerg)

            actions[f] = cmdr_ctrl[f].heading_action(fw_drone, env=env)

        obs, rewards, terms, truncs, infos = env.step(actions)

        for f in env.fixed_agents:
            eff = env.sim.drone_extinguish_stats.get(f, 0.0)
            total_ext += eff
            fi = infos.get(f, {})
            if 'wd_alt' in fi:
                water_drops += 1
                all_alts.append(fi['wd_alt'])
                all_dists.append(fi['wd_dist'])
                if fi['wd_eff'] > 0:
                    water_hits += 1
            if terms.get(f, False):
                fw_alive[f] = False

    fg = env.sim.environment.fire_grid
    final_burn = int(np.sum(fg.B)) if fg is not None else 0
    acc = 100 * water_hits / water_drops if water_drops > 0 else 0
    scout_alive = sum(1 for q in env.quad_agents if q in env.agents)
    fw_alive_n  = sum(1 for f in env.fixed_agents if f in env.agents)

    print(f'Ep{ep:2d}: ext={total_ext:6.1f} final_burn={final_burn:3d} '
          f'drops={water_drops:5d} hits={water_hits:3d} acc={acc:4.1f}% '
          f'scout={scout_alive}/{N_QUADS} fw={fw_alive_n}/{N_FIXED}')
    results.append((total_ext, final_burn, water_drops, water_hits, acc,
                     scout_alive, fw_alive_n))

print()
print('=== SUMMARY ===')
print(f'Avg extinguish: {np.mean([r[0] for r in results]):.1f}')
print(f'Avg final burn: {np.mean([r[1] for r in results]):.0f} cells')
print(f'Avg drops/ep:   {np.mean([r[2] for r in results]):.0f}')
print(f'Avg hits/ep:    {np.mean([r[3] for r in results]):.0f}')
print(f'Avg accuracy:   {np.mean([r[4] for r in results]):.1f}%')
print(f'Scout survival: {np.mean([r[5] for r in results])/N_QUADS*100:.0f}%')
print(f'FW survival:    {np.mean([r[6] for r in results])/N_FIXED*100:.0f}%')
if all_alts:
    print(f'Water alt:      {np.mean(all_alts):.0f}m [{np.min(all_alts):.0f}-{np.max(all_alts):.0f}]')
    print(f'Water dist:     {np.mean(all_dists):.0f}m [{np.min(all_dists):.0f}-{np.max(all_dists):.0f}]')
    # Histogram of dist
    bins = [0, 25, 50, 100, 200, 500]
    counts, _ = np.histogram(all_dists, bins=bins)
    total = len(all_dists)
    print(f'Dist histogram: ' + '  '.join(f'{bins[i]}-{bins[i+1]}m: {counts[i]} ({100*counts[i]/total:.0f}%)' for i in range(len(counts))))
