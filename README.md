# Drone Firefighting Simulation - Diploma Thesis# 🚁 PyBullet Drone Simulation - Modular Architecture



A PyBullet-based simulation environment for autonomous drone firefighting research.Realistic multi-drone simulation with quadcopters and fixed-wing aircraft using PyBullet physics engine.



## 🏗️ Project Structure## ✨ Key Features



This project follows a **3-layer architecture** as documented in the thesis:- **Modular Architecture**: Separate classes for quadcopters and fixed-wing aircraft

- **Multi-Drone Support**: Simulate multiple aircraft simultaneously

### **Layer 1: Core Components** (`core/`, `src/`)- **Environmental System**: Cities, forests, lakes with realistic obstacles

Fundamental physics, algorithms, and simulation infrastructure.- **Weather Effects**: Wind, turbulence, visibility conditions

- **Realistic Physics**: Gravity, inertia, momentum, aerodynamics

```- **Flight Controllers**: Hover stabilization for quads, forward flight for fixed-wing

core/- **Comprehensive Visualization**: Individual trajectory and force analysis per drone

├── fire_grid.py         # Fire spread physics and cellular automata

└── grid_mapper.py       # Coordinate transformation system## Setup and Installation



src/### 1. Create and Activate Conda Environment

├── environment.py       # Wind system and environmental physics

├── simulation.py        # Main simulation orchestrationFirst, check if the `dp` environment exists:

└── drones/

    ├── base_drone.py    # Abstract drone interface```bash

    ├── quadcopter.py    # Quadcopter implementationconda env list

    └── fixedwing.py     # Fixed-wing drone (experimental)```

```

If the `dp` environment doesn't exist, create it:

### **Layer 2: Configuration** (`config/`)

Tunable parameters and scenario definitions.```bash

conda create -n dp python=3.12

``````

config/

└── (Parameter configuration files)Activate the environment:

```

```bash

### **Layer 3: Demos & Experiments** (`demos/`)conda activate dp

Demonstration scripts and experimental scenarios.```



```### 2. Install Required Dependencies

demos/

├── demo_fire_timelapse.py    # Main fire spread visualization demoInstall the required Python packages:

├── demo_fire_fighting.py     # Multi-drone firefighting scenario

└── demo_pure_fire.py         # Pure fire spread (no drones)```bash

```pip install -r requirements.txt

```

### **Supporting Directories**

The required modules are:

```- `numpy` - For mathematical operations and array handling

tests/                   # Unit and integration tests- `matplotlib` - For visualization and animation generation

├── test_fire_grid.py

├── test_integrated_fire.py## 🏗️ Modular Architecture

└── test_scenarios.py

The simulation uses a modular object-oriented design:

docs/                    # Documentation (if any)

datasheets/              # Hardware specifications and datasheets### Drone Classes (`src/drones/`)

output/                  # Generated plots and visualizations- **`BaseDrone`**: Abstract base class with common physics interface

urdf/                    # Robot description files- **`Quadcopter`**: Hover-capable multirotor with omnidirectional movement

```- **`FixedWing`**: Forward-flight aircraft with aerodynamic properties



## 🚀 Quick Start### Environment System (`src/environment.py`)

- **Cities**: Building blocks with collision detection

### Installation- **Natural Areas**: Forests, lakes, terrain features

- **Weather**: Wind effects, turbulence, visibility conditions

```bash

# Create virtual environment### Simulation Manager (`src/simulation.py`)

python -m venv dp- Multi-drone coordination

source dp/bin/activate  # Linux/Mac- Physics stepping and collision detection

# or: dp\Scripts\activate  # Windows- Comprehensive flight data logging

- Individual drone visualization generation

# Install dependencies

pip install -r requirements.txt### 3. Running Simulations

```

#### Simple Single-Drone Demo

### Running Demos```bash

python simple_demo.py

```bash```

# Fire spread visualization

python demos/demo_fire_timelapse.py#### Advanced Modular Multi-Drone Demo

```bash

# Multi-drone firefightingpython modular_demo.py

python demos/demo_fire_fighting.py```



# Pure fire spread (validation)The modular demo will:

python demos/demo_pure_fire.py- Create mixed environment (cities + natural features)

```- Add multiple drones (quadcopters + fixed-wing)

- Apply weather effects (wind, turbulence)

### Running Tests- Generate individual analysis plots per drone

- Save comprehensive flight data

```bash

# All tests## How the System Works

python -m pytest tests/

### PyBullet Architecture

# Specific test file

python -m pytest tests/test_fire_grid.pyThe simulation uses direct PyBullet API calls for realistic physics:

```

#### 1. **Physics Engine**

## 📊 Key Features- **PyBullet**: Real-time physics simulation with gravity (-9.81 m/s²)

- **Mass**: 0.5kg quadcopter body

- **Realistic Fire Physics**: Cellular automata-based fire spread with wind interaction- **Forces**: Applied directly using `p.applyExternalForce()`

- **Wind System**: 3D turbulent wind field with configurable parameters- **Timestep**: 1/240 second (240 FPS physics)

- **Multi-Drone Support**: Coordinate multiple drones for firefighting

- **Real-Time Visualization**: PyBullet 3D rendering + Matplotlib analysis#### 2. **Joystick Control System**

- **Modular Architecture**: Clean separation of concerns for easy extension- **Input Range**: [-1.0, +1.0] for each axis (left/right, forward/back, up/down)

- **Force Mapping**: Joystick values mapped to Newton forces

## 🔬 Research Applications- **Flight Controller**: Automatic gravity compensation for hover



This simulation environment supports research in:#### 3. **Force Calculation**

- Autonomous firefighting strategies- **Horizontal Forces**: Max 10N (provides ~2g acceleration)

- Multi-agent coordination- **Hover Force**: 4.9N (exactly compensates 0.5kg × 9.81 gravity)

- Wind-affected fire behavior modeling- **Vertical Control**: ±15N additional force for climb/descent

- Drone swarm control algorithms

### Simulation Flow

## 📖 Documentation

1. **PyBullet Setup**: Initialize physics world with gravity

For complete architecture details, mathematical foundations, and validation results, see:2. **Drone Creation**: Create 0.5kg quadcopter body at [0,0,5] position  

- `docs/THESIS_DOCUMENTATION.md` (if available)3. **Joystick Commands**: Execute 13-command flight sequence

- `docs/ARCHITECTURE.md` (if available)4. **Physics Step**: Each command runs for specified steps with force application

5. **Data Collection**: Record position, velocity, and force data

## 📝 Requirements6. **Visualization**: Generate comprehensive flight analysis plots



- Python 3.8+### Key Features

- PyBullet

- NumPy- **Realistic Physics**: Full momentum and inertia effects

- Matplotlib- **Hover Stabilization**: Automatic gravity compensation (4.9N upward force)

- SciPy- **Smooth Trajectories**: Natural curved flight paths due to momentum

- **Force Feedback**: Real-time force and position monitoring

See `requirements.txt` for full dependency list.- **Flight Analysis**: Detailed 6-panel visualization of flight performance



## 🎓 Academic Context## 🎮 Joystick Control System



This project is part of a diploma thesis on autonomous drone firefighting simulation.### Input Mapping

For citation or academic use, please contact the author.```python

# Joystick values [-1.0 to +1.0]

## 📄 Licensejoystick = [left_right, forward_back, up_down]



[Specify license here]# Examples:

[0.0, 0.0, 0.0]   # Hover - maintain position
[-1.0, 0.0, 0.0]  # Fly LEFT  
[0.0, 1.0, 0.0]   # Fly FORWARD
[0.0, 0.0, 1.0]   # Climb UP
```

### Force Mapping (Flight Controller)
```python
# Horizontal forces (X, Y)
force_x = joystick[0] * 10.0  # Max 10N horizontally
force_y = joystick[1] * 10.0

# Vertical force (Z) - hover + input  
hover_force = mass * 9.81     # Gravity compensation (4.9N)
vertical_input = joystick[2] * 15.0  # Extra force up/down
force_z = hover_force + vertical_input
```

### Realistic Behavior
- **Hover**: 4.9N upward force exactly balances gravity
- **Momentum**: Drone maintains velocity after joystick release
- **Inertia**: Gradual acceleration/deceleration
- **Physics**: 0.5kg mass, real-time physics steps

## Output Files

The simulation generates two comprehensive visualizations:
- `quadcopter_flight_analysis.png` - 6-panel flight trajectory analysis
- `quadcopter_force_analysis.png` - Force relationships and patterns

## Project Structure

```
DP/
├── README.md                         # This documentation
├── requirements.txt                  # Python dependencies
├── simple_demo.py                    # 🔥 Main PyBullet simulation (square pattern)
├── test_scenarios.py                 # 🧪 All movement validation scenarios  
├── scenario_launcher.py              # 🚀 Interactive scenario selector
├── *.png                            # Generated visualization files
├── datasheets/
│   └── white-paper-anafi-usa-v1.5.3_en.pdf
└── urdf/
    └── quadcopter.urdf              # Drone 3D model definition
```

## 🧪 Test Scenarios

The project includes comprehensive movement validation with 9 focused test scenarios:

### Individual Movement Tests
1. **Hover Test** - Perfect stationary flight (0.0m displacement)
2. **Horizontal Left/Right** - Pure X-axis movement (±2.4m precision)
3. **Horizontal Forward/Backward** - Pure Y-axis movement (±2.4m precision) 
4. **Vertical Up/Down** - Z-axis movement with hover compensation
5. **Diagonal Movement** - Combined XY movement (45° angles)
6. **Complex 3D Movement** - Full 3D diagonal trajectories

### Performance Results
- **All movements validated** with realistic physics behavior
- **Consistent precision** across horizontal movements (2.4m displacement)
- **Perfect hover stability** (0.0m drift during stationary tests)
- **Smooth trajectories** demonstrating momentum and inertia effects

## Requirements

- Python 3.12+
- Conda package manager
- numpy
- matplotlib

## Troubleshooting

## � Flight Pattern Details

The simulation executes a 13-command sequence demonstrating various flight maneuvers:

1. **Hover** at start (50 steps)
2. **Fly LEFT** (-10N X force, 80 steps)  
3. **Hover** corner 1 (30 steps)
4. **Fly FORWARD** (+10N Y force, 80 steps)
5. **Hover** corner 2 (30 steps) 
6. **Fly RIGHT** (+10N X force, 80 steps)
7. **Hover** corner 3 (30 steps)
8. **Fly BACK** (-10N Y force, 80 steps)
9. **Square complete** (30 steps)
10. **Fly UP** (+19.9N Z force, 60 steps)
11. **Hover** at top (30 steps)
12. **Fly DOWN** (-2.6N Z force, 60 steps)  
13. **Final hover** (40 steps)

**Total**: 680 simulation steps, 13 joystick commands

## 🔧 Technical Specifications

- **Physics Engine**: PyBullet real-time simulation
- **Mass**: 0.5 kg quadcopter body
- **Gravity**: -9.81 m/s²
- **Hover Force**: 4.9N (exactly compensates gravity)
- **Max Horizontal Force**: 10N (≈2g acceleration)
- **Max Vertical Force**: 15N extra (≈3g acceleration)
- **Timestep**: 1/240 second (PyBullet default)

## 💡 Key Observations

1. **Perfect Hover**: 4.9N force = mass × gravity compensation
2. **Smooth Trajectories**: Realistic curved flight paths  
3. **Momentum Effects**: Continued motion after joystick release
4. **Force Efficiency**: Small forces (10-20N) achieve rapid movement
5. **Stability**: No oscillation or instability issues

## 🎮 Comparison with Real Drones

✅ **Same as Real Drone:**
- Joystick input range [-1,+1]
- Hover stabilization
- Momentum and inertia effects  
- Gradual acceleration/deceleration

❌ **Missing (for now):**
- Rotation (yaw control)
- Wind and turbulence
- Battery limitations  
- GPS waypoint navigation

---

**🎉 PyBullet Migration Successfully Completed!**

## 🚀 Running the Simulation

### Main Demo (Square Pattern + Vertical)
```bash
conda activate dp
python simple_demo.py
```

### Individual Movement Tests
```bash
python scenario_launcher.py
```
Interactive menu to test specific movements:
- Hover Test
- Horizontal Movement (Left/Right/Forward/Backward) 
- Vertical Movement (Up/Down)
- Diagonal Movement
- Complex 3D Movement

### All Test Scenarios
```bash
python test_scenarios.py
```
Runs all 9 test scenarios automatically.

**Output Files:**
- Real-time flight log in terminal
- Individual PNG analysis files for each scenario
- Comprehensive 6-panel trajectory and force analysis

## 🔄 Extensibility

The system is designed for easy modification:
- **Joystick Commands**: Edit the `joystick_commands` list in `simple_demo.py`
- **Force Mapping**: Modify force calculations in the simulation loop
- **Flight Patterns**: Create custom sequences of joystick inputs
- **Visualization**: Extend the plotting functions for additional analysis
