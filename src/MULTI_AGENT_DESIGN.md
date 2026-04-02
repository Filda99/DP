# Multi-Agent Training Design (train_multi.py)

## Current State

### Scout (ScoutActor) — from train.py/worker.py
- **Network**: ScoutActor (~25K params)
  - Input: local_map [1,32,32] + self_state [15] + neighbor_states [N,3]
  - Architecture: CNN(128) + self_embed(64) + neighbor_attn(64) → LayerNorm(256) → GRU(128) → action(4) + message(5)
  - Output: 4D action (roll, pitch, yaw, throttle) + 5D message to commander
  - Message: [norm_pos_x, norm_pos_y, fire_intensity, learned_1, learned_2]
- **Control**: Frame-by-frame (NN called every physics step)
- **Training**: PPO, buffer = [episodes × max_steps]
- **Status**: Pre-trained model exists at `results/TrainingTogether/07_evaluation/models/scout_best.pt`

### Fixed-Wing Commander (SimpleFWActor) — from train_fw_survival.py
- **Network**: SimpleFWActor (~5K params)
  - Input: self_state [17]
  - Architecture: encoder(17→64→64) → GRU(64) → waypoint(4)
  - Output: 4D waypoint [dx, dy, target_alt, water_trigger]
- **Control**: Waypoint-based (NN called every 50 physics steps)
  - Between calls: heading-hold controller flies toward waypoint (atan2 → roll, PD → pitch, fixed throttle)
  - Segment ends early if waypoint reached (<50m) or at timeout (50 steps)
- **Training**: PPO, buffer = [episodes × num_decisions], no critic
- **Status**: Training in progress, boundary bug recently fixed

### CommanderActorV2 — upgrade path
- **Network**: CommanderActorV2 (~12K params)
  - Input: self_state [17] + scout_messages [N×5]
  - Architecture: encoder(17→64) + cross_attention(msgs→64) → GRU(128→64) → waypoint(4)
  - Output: 4D waypoint [dx, dy, target_alt, water_trigger] (same as SimpleFWActor)
- **Weight transfer**: `CommanderActorV2.from_simple_fw(simple_actor)`
  - Copies: encoder, action_mean, action_logstd, gru.weight_hh/bias_hh
  - Reinits: gru.weight_ih (input size 64→128, old weights in first 64 cols)
  - New: msg_embed, cross_attention (trained from scratch)
- **Key property**: With zero messages, output ≈ SimpleFWActor output (smooth transition)

---

## Multi-Agent Design

### Episode Structure (2000 physics steps)

```
step 0     step 50    step 100   ...  step 1950  step 2000
  |-----------|-----------|----   ----|-----------|
  Scout: NN call every step (2000 decisions)
  Cmdr:  NN call ─────────|───────────|──────────| (40 decisions max)
         controller flies toward waypoint between NN calls
```

- **max_steps = 2000** (matches original train.py for scouts)
- **waypoint_steps = 50** (commander decision frequency)
- **num_decisions_cmdr = 40** (2000 / 50, padded if episode ends early)
- **dt per step = 0.167s** → Episode = ~333s of sim time

### FW with 2000 steps — is it OK?
- **YES** — more decisions (40 vs 20) = finer navigation.
- With waypoint_range=200m, in 40 decisions the FW can cover most of the map.
- Each segment still has 50 steps (8.3s), so flight dynamics unchanged.
- The only change: more total budget → FW lives longer, more reward signal.

### Inner Loop (per physics step)

```python
for step in range(max_steps):  # 2000 steps
    
    # 1. Scout forward pass (EVERY step)
    scout_action, scout_message = scout_actor(obs_quad)
    
    # 2. Commander decision (every waypoint_steps OR when waypoint reached)
    if need_new_waypoint:
        cmdr_waypoint = cmdr_actor(obs_fixed, scout_messages)
        target_pos = cur_pos + waypoint_offset * range
        target_pos = clamp_to_safe_zone(target_pos)
    
    # 3. Heading controller for FW (deterministic, no NN)
    heading_cmd = atan2_to_target(drone_pos, target_pos)
    fw_inner_action = [heading_cmd, target_alt, water_trigger]
    
    # 4. env.step({quad: scout_action, fixed: fw_inner_action})
    
    # 5. Accumulate segment reward for commander
    # 6. Check waypoint reached / timeout → trigger new decision
```

### Buffer Structure

**Scout buffer** (per step, 2000 entries/episode):
- maps: [eps×2000, 1, 32, 32]
- self_states: [eps×2000, 15]
- neighbor_states: [eps×2000, N, 3]
- neighbor_masks: [eps×2000, N]
- actions: [eps×2000, 4]
- logprobs: [eps×2000]
- returns: [eps×2000]

**Commander buffer** (per decision, 40 entries/episode, padded):
- self_states: [eps×40, 17]
- messages: [eps×40, N_quads, 5]
- msg_masks: [eps×40, N_quads]
- actions: [eps×40, 4]
- logprobs: [eps×40]
- returns: [eps×40]
- alive: [eps×40] (mask for padding)

**No shared critic** — separate GAE for each agent type:
- Scout GAE: per step, γ=0.99, λ=0.95
- Commander GAE: per decision (segment rewards), γ=0.99, λ=0.95

### PPO Update

**Separate optimizers** (lesson learned from unstable training):
- optimizer_scout: Adam(scout_actor.parameters(), lr=1e-4)
- optimizer_cmdr: Adam(cmdr_actor.parameters(), lr=3e-4)

**Per minibatch**:
1. Scout policy loss (PPO-clip over 2000-step sequences)
2. Commander policy loss (PPO-clip over 40-decision sequences)
3. Separate backward + grad clip + step for each

**No critic** for now (raw returns as advantages, normalized).
Can add later if training is stable.

### Reward Flow

**Scout** (per physics step):
- survival_bonus: +0.3 /step
- boundary_penalty: -5.0 max /step (quadratic near edge)
- altitude penalty: for flying outside [40, 120]m
- fire_flat_bonus: +1.0 when over fire
- fire_intensity: up to +10.0 proportional to fire under FOV
- fire_approach: +3.0 × progress toward fire (potential-based)
- crash: -50 (terminal)

**Commander** (accumulated per 50-step segment):
- survival_bonus: +0.3 /step × ~50 = +15 /segment
- boundary_penalty: up to -5.0 /step
- altitude penalty: for flying outside [40, 150]m
- timeout_penalty: -1.0 if waypoint not reached in 50 steps
- crash: -50 (terminal)
- fire proximity: +0.5 to +1.0 /step when near fire

### Network Choice for Commander

**Phase 2a** (initial multi-agent): Use **SimpleFWActor** (no messages)
- Scout trains alongside but commander ignores messages
- Purpose: verify the hybrid loop works correctly

**Phase 2b** (full multi-agent): Upgrade to **CommanderActorV2**
- Load SimpleFWActor weights via `from_simple_fw()`
- Commander now receives scout messages for fire targeting
- Scout lr frozen (1e-5), commander lr active (3e-4)

### Scout → Commander Message Buffering

Scout sends a 5D message [pos_x, pos_y, fire_intensity, learned_1, learned_2]
every physics step. Commander decides every 50 steps. Solution:

**Buffer + max-intensity selection** per scout per segment:

```python
# Per commander decision (every 50 steps):
for each scout:
    msgs_buffer = last 50 messages from this scout
    latest_msg  = msgs_buffer[-1]           # current scout position
    best_msg    = max(msgs_buffer, key=fire_intensity)  # best fire sighting

# Feed to CommanderActorV2 cross-attention:
all_msgs = [latest_0, best_fire_0, latest_1, best_fire_1, ...]  # [2*N_quads, 5]
mask     = [alive_0,  alive_0,     alive_1,  alive_1,     ...]  # False=valid
```

This gives commander two views per scout:
| Slot | Content | Purpose |
|------|---------|---------|
| **latest** | Message from step 49 | Where scout IS now |
| **best_fire** | Message with max intensity | Where scout SAW fire |

Cross-attention in CommanderActorV2 handles variable key count natively —
no architecture change needed. With zero fire (intensity≈0), both slots
are nearly identical → attention stays neutral.

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_steps | 2000 | Match train.py, longer episodes for mission |
| waypoint_steps | 50 | ~8.3s per segment, reachable at 21 m/s |
| waypoint_range | 200m | Max offset per decision, within reach |
| safe_limit | map_bounds - 200m | Turning radius protection |
| wp_reached_dist | 50m | Close enough to count as reached |
| wp_timeout_penalty | -1.0 | Only penalty, no reached bonus |
| num_workers | 15 | Parallel CPU rollout collection |
| eps_per_worker | 2 | 30 episodes per batch |
| gamma | 0.99 | Standard discount |
| gae_lambda | 0.95 | Standard GAE |
| num_decisions_cmdr | 40 | 2000 / 50, padded with alive mask |

### File Structure

```
train_multi.py
├── collect_multi_worker()     — rollout: hybrid scout+cmdr loop
│   ├── Scout: NN every step, stores to scout_buf
│   ├── Cmdr: NN every 50 steps, heading controller between
│   ├── GAE scout: per step (2000 entries)
│   └── GAE cmdr: per decision (40 entries, padded)
├── train_multi()              — PPO loop
│   ├── Parallel workers via ProcessPoolExecutor
│   ├── Separate optimizer_scout, optimizer_cmdr
│   ├── Scout PPO update (reshape to eps×2000)
│   └── Cmdr PPO update (reshape to eps×40, alive mask)
└── _save_plot()               — training dashboard
```
