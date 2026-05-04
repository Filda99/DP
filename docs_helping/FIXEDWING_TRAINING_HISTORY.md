# Fixed-Wing Commander — Training History & Milestones

Chronological account of what went wrong, what was fixed, and why the
commander is where it is now (batch ~780, trigger precision ~39%).

---

## Phase 0 — Scout first (background)

Before any fixed-wing training, the quadcopter scout had to work reliably
because the commander's only source of fire-position information are the
scout messages.

Key scout milestones (for context):
- Results `TrainingQuad/` — ~13 iterations to get a scout that reliably
  finds and hovers over fire.
- Commit `08_QuadTrainedWithDemo` (`e571a46`) is the baseline scout;
  subsequent runs fine-tuned it to the current architecture.
- Final frozen scout: `saved_models/multi/scout_b0030.pt`.

---

## Phase 1 — First fixed-wing training (`results/TrainingFixed/`)

### 01 – FixedWing_Phase1
First attempt: `SimpleFWActor` (encoder 17→64→64 + GRU + waypoint).
The FW had **no scout messages** — it was navigating purely from its own
`self_state` (position, velocity, edges, water level, refill compass).

**Problem:** Crashes dominated (boundary violations, altitude too low).
The agent never had time to learn navigation before dying.

### 02 – 50 k steps — still too many crashes around step 300
Longer training, same architecture. FW survived the first 300 steps but
then entered a cycling pattern near map centre instead of going anywhere.

**Root cause identified:** Boundary threshold was computed as
`map_bounds / 2 * frac` instead of `map_bounds * frac` (map_bounds is
already the half-width). This halved the effective boundary warning zone —
the agent got penalty only when nearly out of bounds, too late to react.

**Fix:** Corrected threshold formula.

### 03 – reward 120 — seems OK
With the boundary fix the FW started accumulating positive reward (~120
over 1000 steps). Basic survival established.

### 04 – FW 50 k with demo
Demo confirmed FW flew around the map but **never sought fire** — it had
no incentive to and no information about where fire was.

**Lesson:** A pure survival-trained FW is useless for the mission. Must
be trained together with scouts.

---

## Phase 2 — Joint training starts (`results/TrainingTogether/`)

### 01 – 2 500 steps
First multi-agent runs: FW + 1 scout. FW received scout messages via
cross-attention (`CommanderActorV2`, later renamed `CommanderActor`).

**Problem:** Scout messages were all zeros — the scout had not yet
learned to report fire position (`msg[2]` = fire intensity ≈ 0 always).
Cross-attention got no useful signal. FW behaviour was random.

### 02 / 03 – 6 k / 11 k steps
Continued training. FW reward slowly positive but **no fire extinguishing
events at all** — FW never flew close enough to fire to trigger the water
system.

**Problem identified:** Reward landscape had no gradient toward fire.
FW `self_state` contained no fire-position info; the cross-attention
over near-zero scout messages provided no directional signal.
FW essentially did random walk → survival reward only.

### 04 – scout from `08_Quad`, FW from scratch
Replaced the jointly-trained scout with the pre-trained `08_` scout that
already reported fire reliably. FW trained from zero.

**Result:** First time FW received non-zero fire compass from scout messages.
FW started moving toward fire, first extinguishing events observed.

**New problem:** FW still ignored the refill zone. After expending 200 L
on the first fire approach it ran out of water and kept flying near fire
with nothing to drop — wasting steps.

### 05 – scout from `TrainingQuad`, FW from 04
Swapped in better scout. FW now navigated toward fire in ~60 % of episodes.

**Problem:** FW sometimes flew **past** fire (overshot waypoint).
Waypoint range too large (200 m) + heading controller lag.
Also: FW would drop water even when 500 m from fire (`water_trigger`
threshold was raw NN output, not checked against proximity).

**Fixes:**
- Added `water_trigger_dist = 200 m` proximity gate — water only drops
  when FW is within 200 m of a burning cell.
- Reduced `waypoint_range` from 200 m to 120 m.

### 06 – fixed cycling in the middle
**Critical bug discovered:** FW was looping in a tight circle at map
centre. Root cause: the refill zone was being placed at `(-fire_x, -fire_y)`
but the FW's obs only showed `(refill_x - fw_x) / NORM_DIST` — when the
FW was exactly between fire and refill the two compass signals cancelled and
the heading controller spun.

Also: `+20 reward every step past ep_max_steps` — the env gave survival
bonus every step after truncation because agents were not removed on
truncation (only on termination). This artificially inflated rewards and
masked the real training signal.

**Fix:** Env now removes agents on truncation. Worker checks `truncs` dict.

### 07 – evaluation baseline
Quantitative baseline established (30 episodes, `demo_eval_2scouts.py`):
- OK (water hit fire): **3/30 (10 %)**
- Trigger precision: **3.5 %** (most drops missed fire entirely)
- FW survival: ~70 %

This confirmed the FW was doing something but mostly failing at the mission.

---

## Phase 3 — Architecture & reward redesign

### Problem: FW has no ground-truth fire position
The design principle is that the FW **must not** receive fire coordinates
directly — that would make scouts redundant. The FW should navigate solely
through scout messages. But the cross-attention output was weak and the
GRU was not retaining useful memory.

### Fire Compass injection (major milestone)

Instead of hoping cross-attention would implicitly encode fire direction,
a **fire compass** was injected into `self_state[19–22]` by the
training/evaluation loop — computed from scout messages before each
commander forward pass:

```
self_state[19] = compass_x       # unit vector FW → fire centroid (X)
self_state[20] = compass_y       # unit vector FW → fire centroid (Y)
self_state[21] = dist_norm       # distance / 1000 m  (capped at 2.0)
self_state[22] = max_intensity   # max fire intensity seen by any scout
```

This gave the FW a **dense, explicit navigation signal** from scouts
without exposing the raw fire grid coordinate.

**Result (batch ~110 after compass):**
- Trigger precision jumped from 3.5 % to ~25 % immediately.
- FW started flying directly toward the fire centroid reported by scouts.

### aux_loss weight: 0.5 → 0.05

An auxiliary prediction loss (FW predicts fire position from its internal
state) was added to encourage the GRU to maintain useful fire memory.
Initial weight `0.5` **dominated** the policy gradient — FW stopped
optimising for the mission and only improved its fire-position prediction.

**Fix:** `aux_loss` weight reduced to `0.05` (supplementary, not dominant).

### waypoint_steps: 50 → 30

The FW was slow to react because it committed to a waypoint for 50 steps
(~8 s sim time). Reducing to 30 steps made corrections faster, especially
important when fire was moving due to spread.

### lr_cmdr: 1e-4 → 2e-4

Commander critic loss was ~1.0 (undertrained value function). Increasing
learning rate accelerated convergence without instability.

---

## Phase 4 — Refill problem

### Problem: FW depletes water, never refills

The refill zone is ~1 000–1 200 m from the fire (placed at `-fire_pos`).
The reward gradient toward the refill zone was only active within 600 m
**and** only when `water < 25 %`. The FW spent most of its time near the
fire and never received a meaningful pull toward the refill zone.

**Observed behaviour:** FW dropped water, ran empty, then continued
circling near fire doing nothing. Refill rate: ~0.1 events per episode.

### Refill reward redesign

| Parameter | Before | After |
|---|---|---|
| Trigger threshold | `water < 0.25` | `water < 0.50` |
| Gradient radius | 600 m | 2 000 m (covers full map) |
| Max refill bonus | 0.5 | 2.0 (scaled by water deficit) |
| Refill success bonus | none | `+5 × filled_fraction` |
| Empty tank penalty | -0.3 (only when fire active) | -1.0 always |

**Result:** Average refill events per episode increased from ~0.1 to ~1.1.
FW started planning refill excursions mid-episode.

---

## Phase 5 — Current state (batch ~780)

Quantitative results (`demo_eval_2scouts.py`, 30 episodes, seed 0–29):

```
OK (water hit fire):   20/30  (67 %)        ← was 10 % at baseline
FW_MISSES_FIRE:         3/30
WATER_DEPLETED_EARLY:   2/30
SCOUTS_NOT_REPORTING:   2/30

Avg refill events:  1.07 / episode
Avg empty steps:  356 / 1000             (FW flies empty for ~36 % of ep)
Trigger precision:  39.1 %               ← was 3.5 % (10× improvement)
Min dist FW → fire:  7.3 m               (navigation is working)
```

### What works
- FW reliably navigates toward the fire centroid reported by scouts.
- Water is dropped within 200 m of fire in 67 % of episodes.
- Refill loop (fire → refill zone → fire) observed in most episodes.
- FW survives ~95 % of episodes (boundary management stable).

### What still needs improvement
- **Trigger precision 39 %**: FW drops water slightly too early / too often.
  Many drops hit the edge of the fire rather than the core.
- **Empty steps 356/1000**: FW spends ~36 % of the episode with an empty
  tank. Either it refills too slowly or drops water before the mission.
- **Scouts merge at fire 0**: With parameter-shared scouts and a single
  compass in obs, both scouts converge to the same fire location.
  → Being addressed in current scout fine-tuning (multi-fire training).

---

## Summary table

| Run / Phase | Key change | Outcome |
|---|---|---|
| TrainingFixed/01 | First FW, no scouts | Crashes dominate |
| TrainingFixed/02 | Longer training | Cycling, no mission |
| TrainingFixed/02 | **Fix boundary threshold formula** | Survival stable |
| TrainingFixed/04 | Added demo | FW survives, ignores fire |
| TrainingTogether/04 | **Pre-trained scout plugged in** | First fire approach |
| TrainingTogether/05 | `water_trigger_dist` proximity gate | No wasted drops far from fire |
| TrainingTogether/06 | **Fix +20 reward after truncation** | Real reward signal |
| TrainingTogether/07 | Baseline eval: 10 % OK, 3.5 % precision | Documented baseline |
| train_multi — b~50 | **Fire compass injected (indices 19-22)** | Precision → ~25 % |
| train_multi — b~110 | `aux_loss` 0.5 → 0.05 | Policy gradient restored |
| train_multi — b~200 | `waypoint_steps` 50 → 30 | Faster mid-episode correction |
| train_multi — b~300 | `lr_cmdr` 1e-4 → 2e-4 | Critic loss drops |
| train_multi — b~500 | **Refill radius 600 → 2000 m, threshold 25 → 50 %** | Refill events → 1.1/ep |
| train_multi — b~780 | Current state | 67 % OK, 39 % precision |
