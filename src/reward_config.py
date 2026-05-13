# =============================================================================
# reward_config.py — Central reward parameter configuration
# =============================================================================
# All thresholds, weights and limits in one place.
# Import: from reward_config import QUAD, FIXED, SHARED

# ─── Shared parameters ──────────────────────────────────────────────────────
# All values are FIXED — independent of map size.
# NORM_DIST = 1000 in env_core.py normalises positions and distances in obs.
SHARED = {
    "survival_bonus":       0.02,  # per-step survival reward (felt even without fire)
    "boundary_penalty":     1.5,
    "boundary_extra":       0.5,
    "crash_penalty":        -10,   # reduced from -50 — must not dominate the learning signal
    "reward_clip_min":       -3.0,
    "reward_clip_max":        3.0,
}

# ─── Quadcopter (Scout) ─────────────────────────────────────────────────────
QUAD = {
    # Map boundary — fixed, independent of map_bounds
    "boundary_threshold_m":  150.0,

    # Altitude — fixed range, map-independent
    "alt_ideal_min":    40.0,   # aligned with ground_danger_alt — penalty starts below this
    "alt_ideal_max":    80.0,   # force scout lower for better fire estimate
    "alt_ceiling":     300.0,   # hard kill
    "alt_sweet_min":    70.0,   # ideal operating band — lower bound
    "alt_sweet_max":   100.0,   # lowered from 150 m — reward lower flight
    "alt_sweet_bonus":   0.02,  # per-step bonus for flying in the sweet spot

    # Mission — original values that worked (1.0/3.0 caused dive-crashes)
    "fire_flat_bonus":   0.5,    # per-step bonus for fire visibility
    "fire_intensity_k":  2.0,    # proportional to intensity
    "fire_speed_pen":    0.01,   # penalty for speed over fire

    # Approach — potential-based shaping (dense gradient towards fire)
    # Must be strong enough for the agent to FEEL departure from fire.
    # At 10 m/s departure: 10 * 0.03 = -0.3/step penalty → -15.0 over 50 steps
    "approach_k":        0.03,

    # Compass follow — reward for heading towards fire (velocity · fire_dir)
    # Distance-independent, pure directional signal.
    "compass_follow_k":  0.25,

    # First discovery bonus
    "first_discovery_bonus": 1.0,

    # Ground proximity — exponential penalty prevents dive-crashes
    # ground_danger_alt must be >= alt_ideal_min for a consistent reward landscape
    # Max fire reward: flat(0.5) + intensity_k(2.0)*0.5 = 1.5/step
    # ground_danger_pen=5.0 at z=0: -5.0 >> +1.5 → diving never pays off
    # at z=35 m (half of danger zone): -5.0*(0.5)^2 = -1.25 → still > fire reward
    "ground_danger_alt":  70.0,   # same as alt_ideal_min
    "ground_danger_pen":  5.0,    # reduced from 8.0 — less volatile, still outweighs fire reward

    # Separation — bonus for spacing between scouts
    "separation_min_m":   30.0,   # below this distance: penalty (too close)
    "separation_bonus":   0.05,   # per-step bonus when farther than separation_min_m

    # Alt penalty
    "alt_penalty":       0.05,  # flat penalty for flying outside the ideal range

    # Exploration — bonus for visiting a new 50 m cell (scout motivation without fire)
    "exploration_bonus":  0.1,

    # Fire abandonment — penalty for flying away from previously seen fire.
    # Applied only on transition fire_seen > threshold → fire_seen < threshold.
    # Helps scouts stay near discovered fire (critical in multi-fire scenarios).
    "fire_abandon_penalty": 1.0,  # scaled by previous intensity (max ~1.0/step)
    "fire_abandon_threshold": 0.05,  # min intensity to count as "saw fire"
}

# ─── Fixed-wing (Commander) ──────────────────────────────────────────────────
FIXED = {
    # Map boundary
    "boundary_threshold_m":  200.0,
    "boundary_extra_frac":      0.35,

    # Altitude — fixed
    "alt_ideal_min":    30.0,
    "alt_ideal_max":    80.0,
    "alt_ceiling":     200.0,
    "alt_penalty":       0.02,   # per-metre above alt_ideal_max

    # Survival donut
    "donut_radius":    250.0,
    "donut_bonus":       0.05,
    "survival_base":     0.02,
    "rubber_band_k":     0.02,

    # Water trigger bonus — FW receives reward for dropping near fire (ground-truth dist)
    "water_trigger_dist":   50.0,      # radius [m] for fire-proximity bonus (200→50)
    "water_trigger_alt":    80.0,
    "water_trigger_bonus":    1.5,
    "water_trigger_thresh":   0.0,
    "communication_range_m": 400.0,   # cross-attention message range (used in team reward)
    "water_guidance_bonus": 1.0,       # max bonus for dropping directly over fire
    "water_waste_penalty": 0.5,        # per-step penalty for open valve without hitting fire

    # Fire approach — potential-based shaping towards nearest scout (fire proxy)
    # Scouts hover near fire, so approach-to-scout ≈ approach-to-fire.
    # At 15 m/s approach speed: 15 * 0.25 = +3.75/waypoint-step → strong gradient.
    "fire_approach_k":  0.25,

    # Refill bonus
    "refill_state_bonus":  0.0,
    "refill_proximity_dist":  100.0,
    "refill_proximity_bonus":   0.05,

    # Blending survival vs mission
    "survival_weight":  0.2,
    "mission_weight":   0.8,

    # Scale
    "reward_scale":     0.5,
}