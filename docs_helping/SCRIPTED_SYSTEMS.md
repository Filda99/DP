# Scripted Systems — Design Simplifications for the Thesis

Two major subsystems of the commander pipeline are **not learned by the NN**
but implemented as deterministic, rule-based controllers. This document
describes what they do, *why* they exist, and how to frame them in the thesis.

---

## 1. Scripted Refill Autopilot

### What it does
When FW water level drops below **10%**, the waypoint controller switches
from NN-generated waypoints to a deterministic autopilot that flies to the
**nearest refill zone**. The valve is forced closed (`water_raw = -1`).
A dummy NN forward pass is still executed to keep the GRU hidden state fresh.

The NN resumes control once water is above 10% again.

### Where it lives
- `commander_control.py` → `decide_waypoint()`, lines checking `use_scripted`
- Triggered by: `water_frac <= 0.10 and refill_zones exist`

### Why it exists (motivation for the thesis)

Early experiments revealed a fundamental **reward hacking** exploit: the NN
learned that it could collect positive reward by loitering near the refill
zone, repeatedly dumping water and immediately refilling. The approach-to-fire
shaping was too weak to compete with the guaranteed refill proximity bonus,
causing the agent to abandon the fire entirely.

Several counter-measures were tried:
1. **Water-conditioned reward switching** (approach-fire when full,
   approach-refill when empty) — the NN learned to oscillate at the
   boundary (water ≈ 5%) to exploit both signals simultaneously.
2. **Stronger fire-approach reward** — caused boundary crashes (FW flew
   off the map chasing distant fires).
3. **Zone-based bonuses at 150m/300m from fire** — the NN parked at exactly
   the bonus threshold, never entering the fire zone.

The scripted autopilot eliminates the problem entirely: the NN *cannot*
control the drone during low-water flight, so it cannot learn to exploit
the refill cycle. Instead, the NN's training signal comes exclusively from
firefighting segments, producing a clean, exploitable-free learning signal.

### Multiple refill zones

A single refill zone placed at a fixed location caused dead-time of up to
500 m round trips. Three refill zones are placed 120° apart around the fire
centroid (clipped to 0.6× map boundary), ensuring the nearest zone is
always within ~150 m. This is a deterministic environment setup, not learned.

### Reward isolation

During scripted refill, the FW flies away from scouts, accumulating negative
fire-approach shaping. These rewards are excluded from `ep_reward_cmdr` in
the training loop so they don't pollute the NN's learning signal.

### How to frame in the thesis

> The refill cycle is decomposed into a scripted low-level controller and
> a learned high-level policy. When the water level drops below 10%, a
> deterministic autopilot navigates to the nearest of three pre-placed
> refill zones, while the neural network's GRU state is maintained via
> dummy forward passes. This decomposition was motivated by persistent
> reward-hacking during early experiments: the agent learned to exploit
> the refill proximity bonus by loitering near the zone instead of
> engaging the fire. By removing the NN from the refill loop entirely,
> the policy gradient signal is restricted to firefighting decisions,
> producing a cleaner and more stable learning process.

---

## 2. Rule-Based Valve (Water Release)

### What it does
The NN's 4th output (`water_raw`) is **completely ignored**. Instead, the
valve is controlled by a deterministic rule in `commander_control.py` →
`heading_action()`:

1. **Preconditions**: `altitude < 80 m`, `water > 0`, `env is not None`
2. **Scout fire estimate**: For each scout reporting fire (intensity > 0),
   reconstruct the fire's world position:
   ```
   fire_pos = scout_pos + dyn_offset × (FOV / 2)
   FOV = max(10, scout_altitude × 1.5)
   ```
3. **Trigger**: Open valve if FW is within **50 m** of the closest estimate
   AND either:
   - Heading toward it (velocity · to_fire > 0), OR
   - Very close (< 15 m regardless of heading)

### Where it lives
- `commander_control.py` → `heading_action()`, the "Rule-based valve" section

### Why it exists (motivation for the thesis)

The NN was unable to learn *when* to release water because:

1. **Sparse signal**: successful extinguish events are rare (maybe 5–10 per
   1000-step episode). The NN sees `water_raw` sampled from a Gaussian — by
   the time it happens to output > 0 while directly over fire, the credit
   assignment is too diluted across 30 waypoint decisions.

2. **Spatial precision**: water falls straight down. Even a 20 m horizontal
   offset means zero effectiveness. The NN's waypoint granularity (50 m
   range, 30-step segments) is too coarse for the required spatial precision.

3. **Temporal coupling**: the valve must open for multiple consecutive steps
   to drain meaningful water. The NN decides every 30 steps, but the valve
   needs per-step control.

The rule-based valve solves all three problems:
- It acts **every physics step** (not every 30 steps)
- It uses the scout's **real-time camera data** to estimate fire position
- It includes a **direction check** so water is released along the approach
  vector, creating a ~50 m water strip through the fire core

### How to frame in the thesis

> Water release is controlled by a deterministic rule that reconstructs
> the fire position from the scout's camera message and opens the valve
> when the aircraft is within 50 m and heading toward the estimated fire
> centre. This design choice was driven by the extreme spatial precision
> required for effective suppression — water falls vertically, so even
> a 20 m horizontal offset results in zero effectiveness. The neural
> network's coarse temporal resolution (one decision every 30 physics
> steps) and the rarity of successful drop events made end-to-end
> learning of the valve infeasible within the available training budget.
> The rule-based valve operates at the physics-step level, continuously
> integrating scout sensor data to maximise water-on-fire accuracy.

---

## 3. PD Heading Controller

### What it does
Between NN waypoint decisions (every 30 steps), a proportional-derivative
controller steers the FW toward the active waypoint. The NN outputs
`[dx, dy, alt_raw, water_raw]` every 30 steps; the PD controller converts
this to per-step heading commands.

### Why it exists
Fixed-wing aircraft cannot execute instantaneous heading changes. The FW
physics model (plane.urdf) has a minimum turning radius. Direct heading
control at the NN level (every step) produced unstable oscillations because
the NN couldn't learn the multi-step dynamics of turning. The waypoint
abstraction reduces the NN's action space from "which direction to fly
right now" to "where should I be in 30 steps", making the problem tractable.

### How to frame in the thesis

> The commander operates through a hierarchical control architecture.
> A neural network selects waypoints every 30 physics steps (≈0.625 s),
> while a PD heading controller executes the low-level steering at 240 Hz.
> This temporal abstraction reduces the action space dimensionality and
> shields the policy from the complex rotational dynamics of the fixed-wing
> platform, allowing the network to focus on strategic navigation decisions.

---

## Summary: What the NN actually controls vs. what is scripted

| Component | Controller | Frequency |
|-----------|-----------|-----------|
| **Waypoint selection** (dx, dy) | NN (CommanderActor) | Every 30 steps |
| **Target altitude** (alt_raw) | NN (CommanderActor) | Every 30 steps |
| **Heading toward waypoint** | PD controller | Every physics step |
| **Water valve** | Rule-based (scout estimate) | Every physics step |
| **Refill navigation** | Scripted autopilot | When water < 10% |
| **Throttle** | Fixed at 1.0 (always max) | Every physics step |

The NN's effective decision is: "Given the scout messages and my current
state, where should I fly next?" Everything else is handled by deterministic
controllers that were found to be intractable for end-to-end learning.
