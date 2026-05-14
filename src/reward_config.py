# =============================================================================
# reward_config.py — Central reward parameter configuration
# =============================================================================
# All thresholds, weights and limits in one place.
# Import: from reward_config import QUAD, FIXED, SHARED

# ─── Shared parameters (both scout and commander) ───────────────────────────
SHARED = {
    "survival_bonus":       0.02,  # per-step reward for staying alive
    "boundary_penalty":     1.5,   # quadratic penalty near map edge
    "boundary_extra":       0.5,   # additional penalty beyond boundary zone
    "crash_penalty":        -10,   # one-time penalty on crash/death
    "reward_clip_min":       -3.0, # per-step reward lower bound
    "reward_clip_max":        3.0, # per-step reward upper bound
}

# ─── Quadcopter (Scout) ─────────────────────────────────────────────────────
QUAD = {
    # Map boundary
    "boundary_threshold_m":  150.0,  # distance from edge where penalty starts [m]

    # Altitude band — penalty outside [alt_ideal_min, alt_ideal_max]
    "alt_ideal_min":    40.0,   # lower bound of penalty-free altitude [m]
    "alt_ideal_max":    80.0,   # upper bound of penalty-free altitude [m]
    "alt_ceiling":     300.0,   # instant kill above this altitude [m]
    "alt_sweet_min":    70.0,   # lower bound of bonus altitude band [m]
    "alt_sweet_max":   100.0,   # upper bound of bonus altitude band [m]
    "alt_sweet_bonus":   0.02,  # per-step bonus inside [sweet_min, sweet_max]

    # Fire observation rewards
    "fire_flat_bonus":   0.5,    # per-step bonus when fire is visible in local map
    "fire_intensity_k":  2.0,    # bonus proportional to mean fire intensity in FOV
    "fire_speed_pen":    0.01,   # per-step penalty for speed while over fire (encourages hovering)

    # Approach shaping — potential-based dense gradient towards nearest fire
    "approach_k":        0.03,   # reward = (prev_dist - curr_dist) × approach_k

    # Compass follow — reward for velocity aligned with fire direction
    "compass_follow_k":  0.25,   # reward = dot(vel_dir, fire_dir) × compass_follow_k

    # One-time bonus when any scout first discovers fire
    "first_discovery_bonus": 1.0,

    # Ground proximity — quadratic penalty below ground_danger_alt
    # penalty = -ground_danger_pen × (1 - alt/ground_danger_alt)²
    "ground_danger_alt":  20.0,   # altitude below which penalty is applied [m]
    "ground_danger_pen":  5.0,    # max penalty magnitude (at alt=0)

    # Separation — encourages spacing between multiple scouts
    "separation_min_m":   30.0,   # min distance for bonus; penalty below [m]
    "separation_bonus":   0.05,   # per-step per-pair bonus/penalty

    # Altitude penalty — linear penalty per metre outside ideal band
    "alt_penalty":       0.05,   # penalty = excess_metres × alt_penalty

    # Exploration — one-time bonus for visiting a new 50 m grid cell
    "exploration_bonus":  0.1,

    # Fire abandonment — penalty on transition from seeing fire to not seeing it
    "fire_abandon_penalty": 1.0,    # penalty = prev_intensity × fire_abandon_penalty
    "fire_abandon_threshold": 0.05, # min intensity to count as "seeing fire"
}

# ─── Fixed-wing (Commander) ──────────────────────────────────────────────────
FIXED = {
    # Map boundary
    "boundary_threshold_m":  200.0,    # distance from edge where penalty starts [m]
    "boundary_extra_frac":      0.35,  # extra penalty fraction beyond boundary zone

    # Altitude band — penalty outside [alt_ideal_min, alt_ideal_max]
    "alt_ideal_min":    30.0,   # lower bound of penalty-free altitude [m]
    "alt_ideal_max":    80.0,   # upper bound of penalty-free altitude [m]
    "alt_ceiling":     200.0,   # instant kill above this altitude [m]
    "alt_penalty":       0.02,  # linear penalty per metre outside ideal band

    # UNUSED in env_core.py — kept for backwards compatibility
    "donut_radius":    250.0,
    "donut_bonus":       0.05,
    "survival_base":     0.02,
    "rubber_band_k":     0.02,

    # UNUSED in env_core.py — kept for backwards compatibility
    "water_trigger_dist":   50.0,
    "water_trigger_alt":    80.0,
    "water_trigger_bonus":    1.5,
    "water_trigger_thresh":   0.0,
    "communication_range_m": 400.0,
    "water_guidance_bonus": 1.0,
    "water_waste_penalty": 0.5,        # per-step penalty for water drop that misses fire

    # Fire approach — potential-based shaping towards nearest scout (proxy for fire)
    "fire_approach_k":  0.25,          # reward = (prev_dist - curr_dist) × fire_approach_k

    # UNUSED in env_core.py — kept for backwards compatibility
    "refill_state_bonus":  0.0,
    "refill_proximity_dist":  100.0,
    "refill_proximity_bonus":   0.05,
    "survival_weight":  0.2,
    "mission_weight":   0.8,
    "reward_scale":     0.5,
}