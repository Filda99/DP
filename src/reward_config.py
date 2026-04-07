# =============================================================================
# reward_config.py — Centrální konfigurace všech reward parametrů
# =============================================================================
# Všechny prahové hodnoty, váhy a limity na jednom místě.
# Importuj: from reward_config import QUAD, FIXED, SHARED

# ─── Sdílené parametry ───────────────────────────────────────────────────────
# Všechny hodnoty jsou FIXNÍ — nezávislé na velikosti mapy.
# NORM_DIST = 1000 v env_core.py normalizuje pozice a vzdálenosti v observacích.
SHARED = {
    "survival_bonus":       0.02,  # malý per-krok bonus (fire reward musí dominovat)
    "boundary_penalty":     0.5,   # max penalizace při nárazu do hranice (quadratic)
    "boundary_extra":       0.3,   # extra penalizace pro fixed-wing blízko hranice
    "crash_penalty":        -50,   # same as old — significant but not overwhelming
    "reward_clip_min":      -10.0,
    "reward_clip_max":       10.0,
}

# ─── Quadkoptéra (Scout) ─────────────────────────────────────────────────────
QUAD = {
    # Hranice mapy — fixní, nezávislé na map_bounds
    "boundary_threshold_m":  150.0,

    # Výška — fixní rozsah, nezávislý na mapě
    "alt_ideal_min":    10.0,   # pod 10m → penalizace
    "alt_ideal_max":   200.0,   # nad 200m → penalizace
    "alt_ceiling":     300.0,   # tvrdá smrt

    # Mise — hover nad ohněm (SILNÉ, musí dominovat nad survival)
    "fire_flat_bonus":   0.5, 
    "fire_intensity_k":  5.0,   # násobitel intenzity (max +5.0 při intensity=1.0)
    "fire_speed_pen":    0.05,  # penalizace za rychlost nad ohněm

    # Alt penalty
    "alt_penalty":       0.05,  # flat penalizace za létání mimo rozsah
}

# ─── Fixed-wing (Commander) ──────────────────────────────────────────────────
FIXED = {
    # Hranice mapy — fixní
    "boundary_threshold_m":  300.0,
    "boundary_extra_frac":      0.35,

    # Výška — fixní
    "alt_ideal_min":    40.0,
    "alt_ideal_max":   250.0,
    "alt_ceiling":     450.0,

    # Survival donut
    "donut_radius":    250.0,
    "donut_bonus":       0.05,
    "survival_base":     0.02,
    "rubber_band_k":     0.02,

    # Water trigger bonus
    "water_trigger_dist":   150.0,
    "water_trigger_alt":    150.0,
    "water_trigger_bonus":    1.5,
    "water_trigger_thresh":   0.0,
    "water_waste_penalty": 8.0,  # Postih za vypouštění mimo cíl — zvýšeno z 5 → plýtvání musí bolet

    # Refill bonus
    "refill_state_bonus":  0.0,
    "refill_proximity_dist":  100.0,
    "refill_proximity_bonus":   0.05,

    # Blending survival vs mission
    "survival_weight":  0.2,    # bylo 0.3 — dáme commanderovi víc survival signálu
    "mission_weight":   0.8,    # bylo 0.7

    # Scale
    "reward_scale":     0.5,
}