# Demonstration Scripts

This directory contains demonstration scripts showcasing the improved physics model.

## Demos

### Demo 1: Fire Spread with Terrain Physics
**File:** `demo_01_fire_spread.py`

Demonstrates realistic fire behavior:
- **Terrain-dependent burn rates**
  - Forest (slow burn): 0.03
  - Open areas (fast burn): 0.08
  - Water/buildings (no burn): 0.0
- **Wind effects** on fire spread direction
- **Natural barriers** (lakes, buildings)

**Output:** `output/demo_01_fire_spread.png`

**Run:**
```bash
python demos/demo_01_fire_spread.py
```

---

### Demo 2: Water-Based Fire Suppression
**File:** `demo_02_water_suppression.py`

Demonstrates the moisture-based suppression model:
- **Water drops** increase terrain moisture
- **Moisture prevents re-ignition** (p_ignition *= (1 - M))
- **Moisture evaporates** over time (M -= 0.01 × dt)
- **Immediate effects**:
  - Intensity reduction: I *= (1 - water × 0.8)
  - Probabilistic extinguishment: P(out) = water_amount
- **Drone firefighting** patrol pattern

**Output:** 
- `output/demo_02_suppression.png` (visual progression)
- `output/demo_02_stats.png` (effectiveness graphs)

**Run:**
```bash
python demos/demo_02_water_suppression.py
```

---

### Demo 3: Advanced Physics Model
**File:** `demo_03_physics.py`

Demonstrates complete physics implementation:
- **3D Temperature grid** (20 height levels)
  - Fire heats air: T = T_base + intensity × 500K
  - Heat diffusion with vertical bias
- **Air density** temperature-dependent
  - ρ = ρ₀ × (T₀ / T)
  - Hot air above fire has lower density
- **Quadcopter physics**
  - Reduced lift in low-density air
  - Observable altitude drift
- **Fixed-wing aerodynamics**
  - Lift calculated from airspeed, not groundspeed
  - v_air = ||v_aircraft - v_wind||
  - Realistic wind effects

**Output:**
- `output/demo_03_temperature.png` (3D temperature visualization)
- `output/demo_03_physics.png` (physics metrics over time)

**Run:**
```bash
python demos/demo_03_physics.py
```

---

## Running All Demos

```bash
cd /home/filip/tmp/DP
python demos/demo_01_fire_spread.py
python demos/demo_02_water_suppression.py
python demos/demo_03_physics.py
```

All outputs will be saved to the `output/` directory.

## Requirements

- NumPy
- Matplotlib
- PyBullet (for full simulation)

See `requirements.txt` in the project root.
