# Drone Simulation Project

This project simulates multiple drones navigating through different environments with terrain features, weather conditions, and collision avoidance capabilities.

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

If you encounter issues:

1. **Environment not found**: Make sure you've created the conda environment with `conda create -n dp python=3.12`
2. **Module not found**: Ensure you've activated the environment with `conda activate dp` and installed requirements
3. **Animation not generating**: Check that matplotlib is properly installed and you have write permissions in the output directory

## Contributing

The simulation system is modular and extensible. You can:
- Add new drone types by extending `BaseDrone`
- Create custom terrain types in `environment.py`
- Design new scenarios in the scenario files
- Modify weather conditions and environmental effects
