# =============================================================================
# reward_config.py — Centrální konfigurace všech reward parametrů
# =============================================================================
# Všechny prahové hodnoty, váhy a limity na jednom místě.
# Importuj: from reward_config import QUAD, FIXED, SHARED

# ─── Sdílené parametry ───────────────────────────────────────────────────────
SHARED = {
    "survival_bonus":       0.3,   # per-krok bonus za přežití (oba agenti)
    "boundary_penalty":     5.0,    # max penalizace při nárazu do hranice
    "boundary_extra":       3.0,    # extra penalizace pro fixed-wing blízko hranice
    "crash_penalty":        -200,   # must hurt relative to extinguish (+5/step)
    "reward_clip_min":      -15.0,
    "reward_clip_max":      15.0,   # extinguish max = 5/step, survival = 0.3
}

# ─── Quadkoptéra (Scout) ─────────────────────────────────────────────────────
QUAD = {
    # Hranice mapy
    "boundary_threshold_m":  150.0,   # fixní 150m od okraje — nezávislé na velikosti mapy

    # Výška — sweet spot kde scout vidí dobře ale není v updraftu
    "alt_ideal_min":    40.0,   # pod touto výškou je silný updraft → penalizace
    "alt_ideal_max":   200.0,   # nad touto výškou je příliš daleko od ohně → penalizace
    "alt_ceiling":     400.0,   # tvrdá smrt — 200m výš než ideal_max = víc prostoru

    # Mise — hover nad ohněm
    "fire_flat_bonus":   1.0,   # flat bonus za detekci ohně
    "fire_intensity_k": 10.0,   # násobitel intenzity
    "fire_speed_pen":    0.05,  # penalizace za rychlost nad ohněm

    # Scale
    "reward_scale":      0.3,
}

# ─── Fixed-wing (Commander) ──────────────────────────────────────────────────
FIXED = {
    # Hranice mapy
    "boundary_threshold_m":  300.0,   # fixní 300m od okraje — fixed-wing potřebuje víc prostoru na otáčení
    "boundary_extra_frac":      0.35,   # extra penalizace pod 35 % prahu = 105m od okraje

    # Výška — sweet spot pro hašení
    "alt_ideal_min":    40.0,   # pod touto výškou je nebezpečí srážky s terénem
    "alt_ideal_max":   250.0,   # nad touto výškou voda nedopadne přesně
    "alt_ceiling":     450.0,   # tvrdá smrt (zvýšeno z 400 — víc prostoru pro recovery)
    "alt_penalty_k":     0.15,   # per 10m nad ideal_max (silnější penalizace)
    "alt_ideal_target":  80.0,  # středová cílová výška (pro tah)

    # Survival donut
    "donut_radius":    250.0,   # uvnitř = flat bonus (scaled for 1km map)
    "donut_bonus":       0.05,  # bonus uvnitř donutu
    "survival_base":     0.02,   # base survival bonus (navíc k SHARED)
    "rubber_band_k":     0.02,  # síla tahu zpět do středu

    # Mise — orbital reward
    "mission_state_bonus": 0.1,
    "orbital_radius_fire":    120.0,   # ideální kroužení kolem ohně (bylo 150)
    "orbital_radius_refill":  80.0,   # ideální kroužení u refill zóny
    "orbital_radius_patrol": 150.0,   # patrol orbit kolem středu (scaled for 1km map)

    # Water trigger bonus
    "water_trigger_dist":   150.0,   # max vzdálenost od ohně pro bonus (scaled for 1km map)
    "water_trigger_alt":    150.0,   # max výška pro bonus
    "water_trigger_bonus":    1.5,   # bonus za aktivaci triggeru
    "water_trigger_thresh":   0.0,   # threshold last_action[3] pro "is_dropping"
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