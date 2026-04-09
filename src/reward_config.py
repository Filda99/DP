# =============================================================================
# reward_config.py — Centrální konfigurace všech reward parametrů
# =============================================================================
# Všechny prahové hodnoty, váhy a limity na jednom místě.
# Importuj: from reward_config import QUAD, FIXED, SHARED

# ─── Sdílené parametry ───────────────────────────────────────────────────────
# Všechny hodnoty jsou FIXNÍ — nezávislé na velikosti mapy.
# NORM_DIST = 1000 v env_core.py normalizuje pozice a vzdálenosti v observacích.
SHARED = {
    "survival_bonus":       0.01,  # starý design: 0.01 (před *0.1 = 0.001)
    "boundary_penalty":     0.3,   # starý design: 0.3 (před *0.1 = 0.03)
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
    "alt_ideal_min":    30.0,   # pod 10m → penalizace
    "alt_ideal_max":   150.0,   # starý design: 150m
    "alt_ceiling":     300.0,   # tvrdá smrt
    "alt_sweet_min":    50.0,   # ideální operační pásmo — spodní hranice
    "alt_sweet_max":    100.0,   # ideální operační pásmo — horní hranice
    "alt_sweet_bonus":   0.02,  # per-krok bonus za let ve sweet-spotu

    # Mise — starý jednoduchý design (hodnoty odpovídají originálu po *0.1)
    "fire_flat_bonus":   0.1,   # starý: 1.0 * 0.1
    "fire_intensity_k":  5.0,   # starý: 10.0 * 0.1
    "fire_speed_pen":    0.005, # starý: 0.05 * 0.1

    # Approach — kompas k ohni (nové, starý design to neměl)
    "approach_k":        0.005,

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