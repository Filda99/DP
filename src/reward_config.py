# =============================================================================
# reward_config.py — Centrální konfigurace všech reward parametrů
# =============================================================================
# Všechny prahové hodnoty, váhy a limity na jednom místě.
# Importuj: from reward_config import QUAD, FIXED, SHARED

# ─── Sdílené parametry ───────────────────────────────────────────────────────
# Všechny hodnoty jsou FIXNÍ — nezávislé na velikosti mapy.
# NORM_DIST = 1000 v env_core.py normalizuje pozice a vzdálenosti v observacích.
SHARED = {
    "survival_bonus":       0.02,  # zvýšeno — přežívání musí být cítit i bez ohně
    "boundary_penalty":     0.3,
    "boundary_extra":       0.3,
    "crash_penalty":        -10,   # sníženo z -50 — crash nesmí dominovat learning signál
    "reward_clip_min":      -5.0,
    "reward_clip_max":       5.0,
}

# ─── Quadkoptéra (Scout) ─────────────────────────────────────────────────────
QUAD = {
    # Hranice mapy — fixní, nezávislé na map_bounds
    "boundary_threshold_m":  150.0,

    # Výška — fixní rozsah, nezávislý na mapě
    "alt_ideal_min":    30.0,   # pod tímto → penalizace
    "alt_ideal_max":   120.0,   # zvýšeno z 80m — netrestáme létání výš (větší FOV)
    "alt_ceiling":     300.0,   # tvrdá smrt
    "alt_sweet_min":    50.0,   # ideální operační pásmo — spodní hranice
    "alt_sweet_max":   150.0,   # zvýšeno z 100m
    "alt_sweet_bonus":   0.02,  # per-krok bonus za let ve sweet-spotu

    # Mise — silnější signál, agent musí CÍTIT rozdíl mezi "nad ohněm" a "jinde"
    "fire_flat_bonus":   0.5,    # výrazný per-step bonus za viditelnost ohně
    "fire_intensity_k":  2.0,    # proporcionální k intenzitě
    "fire_speed_pen":    0.01,   # penalizace za rychlost nad ohněm

    # Approach — potential-based shaping (hustý gradient k ohni)
    # Musí být dostatečně silný aby agent CÍTIL odlet od ohně.
    # Při 10 m/s odletu: 10 * 0.03 = -0.3/step penalizace → za 50 kroků -15.0
    "approach_k":        0.03,

    # Compass follow — odměna za směr letu k ohni (velocity · fire_dir)
    # Nezávislé na vzdálenosti, čistý direction signal.
    # Při letu 5 m/s přímo k ohni: reward = +0.15/step → za 500 kroků +75
    # Při letu 5 m/s od ohně:       reward = -0.15/step → za 500 kroků -75
    # Spread 150 bodů — dominantní signál, agent NEMŮŽE ignorovat.
    "compass_follow_k":  0.15,

    # First discovery bonus
    "first_discovery_bonus": 1.0,

    # Ground proximity — exponenciální penalizace zabraňuje dive-crashům
    # Na 50m: 0, na 25m: -0.75, na 10m: -2.2, na 0m: -3.0
    "ground_danger_alt":  50.0,   # zvýšeno z 20m — aktivuje se dřív
    "ground_danger_pen":  3.0,    # zvýšeno z 0.5 — must hurt

    # Separation — bonus za rozestup mezi scouty
    "separation_min_m":   30.0,   # pod touto vzdáleností: penalizace (příliš blízko)
    "separation_bonus":   0.05,   # per-step bonus když jsou dál než separation_min_m

    # Alt penalty
    "alt_penalty":       0.05,  # flat penalizace za létání mimo rozsah

    # Exploration — bonus za návštěvu nové 50m buňky (motivace pro scouta bez ohně)
    "exploration_bonus":  0.1,
}

# ─── Fixed-wing (Commander) ──────────────────────────────────────────────────
FIXED = {
    # Hranice mapy — fixní
    "boundary_threshold_m":  300.0,
    "boundary_extra_frac":      0.35,

    # Výška — fixní
    "alt_ideal_min":    40.0,
    "alt_ideal_max":   150.0,
    "alt_ceiling":     450.0,
    "alt_penalty":       0.01,   # per-metre above alt_ideal_max

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