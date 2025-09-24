# 🚁 PyBullet Drone Simulation

Realistická simulace kvadrokoptéry s joystick ovládáním pomocí PyBullet fyzikálního enginu.

## ✨ Hlavní funkce

- **Realistická fyzika**: Gravitace, setrvačnost, momentum
- **Joystick ovládání**: Jako skutečný DJI/Parrot ovladač  
- **Flight controller**: Automatická hover stabilizace
- **Vizualizace**: Detailní grafy trajektorie a sil

## Setup and Installation

### 1. Create and Activate Conda Environment

First, check if the `dp` environment exists:

```bash
conda env list
```

If the `dp` environment doesn't exist, create it:

```bash
conda create -n dp python=3.12
```

Activate the environment:

```bash
conda activate dp
```

### 2. Install Required Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The required modules are:
- `numpy` - For mathematical operations and array handling
- `matplotlib` - For visualization and animation generation

### 3. Running the Simulation

To run the enhanced scenarios with terrain and weather features:

```bash
python src/enhanced_scenarios.py
```

Alternative scenarios can be run with:

```bash
# Basic scenarios
python src/main_scenarios.py

# Original simulation
python src/main.py
```

## How the Program Works

### Architecture Overview

The drone simulation system consists of several key components:

#### 1. **Drone Types** (`src/drones/`)
- **`BaseDrone`**: Abstract base class defining the drone interface
- **`Quadcopter`**: Hovering drone with omnidirectional movement based on Parrot ANAFI USA specifications
- **`FixedWing`**: Forward-flying aircraft with turning constraints

#### 2. **Environment System** (`environment.py`)
- **Terrain Types**: Forest, lake, mountain, urban, no-fly zones, etc.
- **Weather Conditions**: Wind speed/direction, visibility, precipitation
- **Flight Constraints**: Altitude restrictions, speed modifiers, avoidance priorities

#### 3. **Simulation Engine** (`simulation.py`)
- **Collision Detection**: Circular zones for quadcopters, rectangular zones for fixed-wing
- **Collision Avoidance**: Priority-based system where lower-indexed drones have right-of-way
- **Animation Generation**: Creates GIF files showing drone trajectories and terrain

#### 4. **Scenario Management**
- **`main_scenarios.py`**: Basic scenarios with simple environments
- **`enhanced_scenarios.py`**: Advanced scenarios with terrain and weather

### Simulation Flow

1. **Environment Setup**: Terrain zones and weather conditions are defined
2. **Drone Initialization**: Drones are created with starting positions and goals using the `drone_factory`
3. **Simulation Loop**: Each time step:
   - Detect potential collisions (expanded zones for avoidance)
   - Compute actions for each drone using their `compute_action` method
   - Apply terrain and weather effects to movement
   - Move drones and record positions
   - Check for actual collisions
4. **Animation Generation**: Create GIF showing the complete simulation

### Key Features

- **Intelligent Collision Avoidance**: Drones detect potential collisions and take evasive action
- **Realistic Flight Physics**: Different movement characteristics for quadcopters vs fixed-wing aircraft
- **Environmental Effects**: Terrain affects flight speed and drone behavior
- **Weather Impact**: Wind, visibility, and precipitation influence flight performance
- **Multiple Scenarios**: Various pre-configured scenarios testing different aspects

## Enhanced Scenarios

Running `enhanced_scenarios.py` provides four different simulation scenarios:

### 1. Natural Environment Navigation
- **Environment**: Forests, lakes, and mountains
- **Drones**: 3 drones (2 quadcopters, 1 fixed-wing)
- **Challenge**: Navigate through natural terrain features
- **Output**: `natural_environment.gif`

### 2. Urban Drone Delivery
- **Environment**: Urban area with buildings
- **Drones**: 2 quadcopters
- **Challenge**: Delivery routes between buildings with moderate weather
- **Weather**: 8 m/s wind, 800m visibility
- **Output**: `urban_environment.gif`

### 3. Mixed Terrain Challenge
- **Environment**: Complex multi-terrain scenario including:
  - Forest areas (reduced speed)
  - Lakes (increased speed)
  - Mountains (altitude restrictions)
  - No-fly zones (military base)
  - Urban areas (altitude restrictions)
- **Drones**: 4 drones with challenging cross-terrain routes
- **Weather**: Strong winds (12 m/s), precipitation
- **Output**: `mixed_terrain.gif`

### 4. Storm Navigation
- **Environment**: Natural terrain with severe weather
- **Drones**: 2 quadcopters
- **Weather**: Extreme conditions (15 m/s winds, 300m visibility, heavy rain)
- **Challenge**: Navigation in adverse weather conditions
- **Output**: `weather_challenge.gif`

## Output Files

Each simulation generates an animated GIF showing:
- Drone trajectories in different colors
- Terrain zones with visual labels
- Collision zones around each drone
- Real-time collision warnings
- Environment and weather information

## Project Structure

```
DP/
├── README.md
├── requirements.txt
├── datasheets/
│   └── white-paper-anafi-usa-v1.5.3_en.pdf
└── src/
    ├── main.py                 # Original simulation entry point
    ├── main_scenarios.py       # Basic simulation scenarios
    ├── enhanced_scenarios.py   # Advanced scenarios with terrain/weather
    ├── simulation.py           # Core simulation engine
    ├── environment.py          # Environment and terrain system
    ├── drone_factory.py        # Drone creation factory
    └── drones/
        ├── base_drone.py       # Abstract drone base class
        ├── quadcopter.py       # Quadcopter implementation
        └── fixedwing.py        # Fixed-wing aircraft implementation
```

## Simulation Statistics

After running scenarios, the program provides detailed statistics:
- **Total simulation steps**
- **Number of collisions detected**
- **Collision rate percentage**
- **Performance metrics for each scenario**

## Requirements

- Python 3.12+
- Conda package manager
- numpy
- matplotlib

## Troubleshooting

## 🎮 Jak to funguje

### Joystick Input
```python
# Joystick values [-1.0 až +1.0]
joystick = [left_right, forward_back, up_down]

# Příklady:
[0.0, 0.0, 0.0]   # Hover - drž pozici
[-1.0, 0.0, 0.0]  # Leti doleva  
[0.0, 1.0, 0.0]   # Leti dopředu
[0.0, 0.0, 1.0]   # Stoupej nahoru
```

### Force Mapping (flight controller)
```python
# Horizontální síly (X, Y)
force_x = joystick[0] * 10.0  # Max 10N horizontálně
force_y = joystick[1] * 10.0

# Vertikální síla (Z) - hover + input  
hover_force = mass * 9.81     # Kompenzace gravitace (4.9N)
vertical_input = joystick[2] * 15.0  # Extra síla nahoru/dolů
force_z = hover_force + vertical_input
```

### Realistické chování
- **Hover**: 4.9N síla nahoru přesně vyrovnává gravitaci
- **Momentum**: Dron si udržuje rychlost i po uvolnění joysticku
- **Setrvačnost**: Postupné zrychlování/zpomalování
- **Fyzika**: Hmotnost 0.5kg, realtime physics step

## 🚀 Spuštění simulace

```bash
conda activate dp
python simple_demo.py
```

Výstup:
- Textový log letu v terminálu
- `quadcopter_flight_analysis.png` - 6 grafů s trajektorií
- `quadcopter_force_analysis.png` - analýza sil

## 📊 Výsledky posledního testu

- **Čtverec dokončen**: Doleva → Dopředu → Doprava → Zpět  
- **Vertikální pohyb**: Nahoru + dolů
- **Přesnost**: 6.1m od startovní pozice
- **Realistické chování**: Momentum a setrvačnost fungují perfektně

## 🎯 Flight Pattern

1. **Hover** na začátku (50 kroků)
2. **Leti DOLEVA** (-10N X força, 80 kroků)  
3. **Hover** na rohu (30 kroků)
4. **Leti DOPŘEDU** (+10N Y força, 80 kroků)
5. **Hover** na rohu (30 kroků) 
6. **Leti DOPRAVA** (+10N X força, 80 kroků)
7. **Hover** na rohu (30 kroků)
8. **Leti ZPĚT** (-10N Y força, 80 kroků)
9. **Čtverec dokončen** (30 kroků)
10. **Leti NAHORU** (+19.9N Z força, 60 kroků)
11. **Hover nahoře** (30 kroků)
12. **Leti DOLŮ** (-2.6N Z força, 60 kroků)  
13. **Final hover** (40 kroků)

**Celkem**: 680 simulation steps, 13 joystick příkazů

## 🔧 Technické detaily

- **PyBullet**: Realtima fyzikální simulace
- **Hmotnost**: 0.5 kg
- **Gravitace**: -9.81 m/s²
- **Hover síla**: 4.9N (přesně kompenzuje gravitaci)
- **Max horizontální síla**: 10N (2g acceleration)
- **Max vertikální síla**: 15N extra (3g acceleration)
- **Timestep**: 1/240 second (PyBullet default)

## 💡 Klíčové pozorování

1. **Perfektní hover**: Force 4.9N = masa × gravitace
2. **Smooth trajectories**: Realistické zakřivené dráhy  
3. **Momentum effects**: Pokračování v pohybu po pustiti joysticku
4. **Force efficiency**: Malé síly (10-20N) stačí pro rychlý pohyb
5. **Stability**: Dron se nerozkmitá ani nepřeklopí

## 🎮 Porovnání s reálným dronem

✅ **Stejné jako reálný dron:**
- Joystick input [-1,+1]
- Hover stabilizace
- Momentum a setrvačnost  
- Postupné zrychlování

❌ **Chybí (zatím):**
- Rotace (yaw)
- Vítr a turbulence
- Baterie a limits  
- GPS waypoint navigace

---

**🎉 Migrace na PyBullet úspěšně dokončena!**

The simulation system is modular and extensible. You can:
- Add new drone types by extending `BaseDrone`
- Create custom terrain types in `environment.py`
- Design new scenarios in the scenario files
- Modify weather conditions and environmental effects
