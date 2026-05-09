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
    "alt_ideal_min":    60.0,   # zarovnáno s ground_danger_alt — pod tímto penalty začíná
    "alt_ideal_max":   120.0,   # zvýšeno z 80m — netrestáme létání výš (větší FOV)
    "alt_ceiling":     300.0,   # tvrdá smrt
    "alt_sweet_min":    70.0,   # ideální operační pásmo — spodní hranice
    "alt_sweet_max":   150.0,   # zvýšeno z 100m
    "alt_sweet_bonus":   0.02,  # per-krok bonus za let ve sweet-spotu

    # Mise — originální hodnoty které fungovaly (zvýšení na 1.0/3.0 způsobilo dive-crash)
    "fire_flat_bonus":   0.5,    # per-step bonus za viditelnost ohně
    "fire_intensity_k":  2.0,    # proporcionální k intenzitě
    "fire_speed_pen":    0.01,   # penalizace za rychlost nad ohněm

    # Approach — potential-based shaping (hustý gradient k ohni)
    # Musí být dostatečně silný aby agent CÍTIL odlet od ohně.
    # Při 10 m/s odletu: 10 * 0.03 = -0.3/step penalizace → za 50 kroků -15.0
    "approach_k":        0.03,

    # Compass follow — odměna za směr letu k ohni (velocity · fire_dir)
    # Nezávislé na vzdálenosti, čistý direction signal.
    # Zvýšeno z 0.15 — dominantní signál k ohni
    "compass_follow_k":  0.25,

    # First discovery bonus
    "first_discovery_bonus": 1.0,

    # Ground proximity — exponenciální penalizace zabraňuje dive-crashům
    # ground_danger_alt musí být >= alt_ideal_min aby reward landscape byl konzistentní
    # Max fire reward: flat(0.5) + intensity_k(2.0)*0.5 = 1.5/krok
    # ground_danger_pen=5.0 na z=0: -5.0 >> +1.5 → dive se nevyplatí
    # na z=35m (polovina danger zone): -5.0*(0.5)^2 = -1.25 → stále > fire reward
    "ground_danger_alt":  70.0,   # shodné s alt_ideal_min
    "ground_danger_pen":  5.0,    # sníženo z 8.0 — méně volatilní, stále překryje fire reward

    # Separation — bonus za rozestup mezi scouty
    "separation_min_m":   30.0,   # pod touto vzdáleností: penalizace (příliš blízko)
    "separation_bonus":   0.05,   # per-step bonus když jsou dál než separation_min_m

    # Alt penalty
    "alt_penalty":       0.05,  # flat penalizace za létání mimo rozsah

    # Exploration — bonus za návštěvu nové 50m buňky (motivace pro scouta bez ohně)
    "exploration_bonus":  0.1,

    # Fire abandonment — penalizace za odlet od ohně který už byl vidět.
    # Aplikuje se pouze při přechodu fire_seen > threshold → fire_seen < threshold.
    # Pomáhá scoutům zůstat u ohně co najdou (klíčové při multi-fire scénářích).
    "fire_abandon_penalty": 1.0,  # škálováno podle předchozí intenzity (max ~1.0/step)
    "fire_abandon_threshold": 0.05,  # min intenzita aby se počítalo jako "viděl oheň"
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

    # Water trigger bonus — FW dostane reward za drop blízko ohně (ground-truth vzdálenost)
    "water_trigger_dist":   200.0,     # radius [m] pro fire-proximity bonus (bylo 150)
    "water_trigger_alt":    150.0,
    "water_trigger_bonus":    1.5,
    "water_trigger_thresh":   0.0,
    "communication_range_m": 400.0,   # dosah cross-attention zpráv (použito v team reward)
    "water_guidance_bonus": 1.0,       # max bonus za drop přímo nad ohněm (bylo 0.2 u scoutu)
    "water_waste_penalty": 1.0,        # zdvojnásobeno z 0.5 — 0.5 nestačí překonat 60-krokovou setrvačnost (bylo 0.3, zkusili 1.5 — příliš agresivní)

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