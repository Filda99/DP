"""
Configuration file for the Wildfire Drone Simulation
All hardcoded values have been extracted here for centralized configuration.
"""

# ==============================================================================
# MAIN SCRIPT CONFIGURATION
# ==============================================================================

class MainConfig:
    """Configuration for main training and demo"""
    # Training parameters
    MAX_EPISODES = 100
    MAX_STEPS = 150
    SAVE_EVERY = 25
    
    # Visualization parameters
    VISUALIZATION_FIGSIZE = (15, 6)
    VISUALIZATION_DPI = 300
    DEMO_VISUALIZATION_FIGSIZE = (12, 6)
    DEMO_VISUALIZATION_DPI = 150
    
    # Model parameters
    ACTOR_MESSAGE_DIM = 8
    ACTOR_HIDDEN_SIZE = 128
    
    # PPO Trainer parameters
    LEARNING_RATE = 3e-5
    GAMMA = 0.99
    EPS_CLIP = 0.1
    ENTROPY_COEF = 0.05
    MAX_GRAD_NORM = 1.0
    MIN_LOSS_THRESHOLD = 0.01
    
    # Gentle training parameters - INCREASED for more aggressive actions
    GENTLE_TRAINING_START_SCALE = 0.6  # Increased from 0.4
    GENTLE_TRAINING_MAX_SCALE = 1.2   # Increased from 0.8
    GENTLE_TRAINING_INCREMENT = 0.1   # Increased from 0.05
    GENTLE_TRAINING_SUCCESS_THRESHOLD = 5.0  # Increased from 2.0 for easier progression
    GENTLE_TRAINING_SUCCESS_EPISODES = 2     # Reduced from 3 for faster scaling
    GENTLE_TRAINING_STABLE_THRESHOLD = 15
    GENTLE_TRAINING_MIN_SCALE = 0.9   # Increased from 0.7
    
    # Visualization colors
    DRONE_COLORS = ['blue', 'red', 'green', 'purple']
    FIRE_POSITION_CENTER = [0.0, 0.0]
    
    # Terrain colors
    TERRAIN_COLORS = {
        'grass': [0.6, 0.7, 0.4],
        'forest': [0.1, 0.4, 0.1],
        'water': [0.2, 0.5, 0.9],
        'building': [0.5, 0.5, 0.5]
    }
    
    # Fire visualization colors  
    FIRE_OVERLAY_COLOR = [1.0, 0.2, 0.0, 0.8]
    
    # Wind arrow parameters
    WIND_ARROW_LENGTH = 8
    WIND_ARROW_HEAD_WIDTH = 2
    WIND_ARROW_HEAD_LENGTH = 2
    WIND_ARROW_COLOR = 'yellow'
    WIND_ARROW_EDGE_COLOR = 'black'
    WIND_ARROW_WIDTH = 1
    
    # Demo parameters
    DEMO_MAX_STEPS = 15000
    DEMO_RECENT_REWARDS_MAX = 50
    DEMO_FRAME_INTERVAL = 10
    DEMO_CONSERVATIVE_POLICY = False  # Model should use FULL learned strength!
    DEMO_FIXED_POSITIONS = True  # Use fixed positions for consistent demo testing
    DEMO_DRONE_POSITION = [0, 0, 10]  # Fixed demo drone spawn  
    DEMO_FIRE_POSITION = [25, 25]  # Fixed demo fire position (moderate distance)
    
    # Validation parameters
    VALIDATION_HOVER_ACTION = [0.0, 0.0, 0.0, 0.1]

# ==============================================================================
# WILDFIRE GYM WRAPPER CONFIGURATION
# ==============================================================================

class WildfireGymConfig:
    """Configuration for wildfire gym wrapper"""
    # Observation processor settings
    OBSERVATION_WINDOW_SIZE = 30.0
    OBSERVATION_RESOLUTION = 32
    
    # Map bounds - ENLARGED MAP!
    MAP_BOUNDS = 100.0  # ±100m = 200x200m total area
    
    # Environment settings - FINER RESOLUTION!
    ENVIRONMENT_GRID_WIDTH = 200  # 200 cells
    ENVIRONMENT_GRID_HEIGHT = 200  # 200 cells 
    ENVIRONMENT_CELL_SIZE = 1.0  # General cell size for compatibility
    ENVIRONMENT_DT = 0.5
    
    # Fire settings - will be randomized in reset()
    FIRE_CENTER_POSITION = [0, 0]  # Default, will be randomized
    FIRE_INTENSITY = 3.0
    
    # Drone spawn settings - will be randomized in reset()
    QUAD_SPAWN_POSITION = [0, 0, 10]  # Default, will be randomized
    
    # Action space bounds
    ACTION_LOW = -1.0
    ACTION_HIGH = 1.0
    ACTION_DIMENSIONS = 4
    
    # Fixed wing action mapping
    FIXED_WING_VELOCITY_MIN = 10
    FIXED_WING_VELOCITY_MAX = 25
    FIXED_WING_VELOCITY_SCALE = 7.5
    FIXED_WING_GAMMA_SCALE = 0.5
    FIXED_WING_PHI_SCALE = 0.8
    FIXED_WING_DROP_THRESHOLD = 0.5
    
    # Reward parameters
    REWARD_EXPLORATION_NEW = 0.5
    REWARD_EXPLORATION_OLD = 0.1
    REWARD_HOVERING_DISTANCE_THRESHOLD = 15.0
    REWARD_HOVERING_MULTIPLIER = 3.0
    REWARD_MOVEMENT_MULTIPLIER_NEAR_FIRE = 0.2
    REWARD_INACTIVITY_PENALTY = 0.2
    REWARD_RETURN_PROGRESS_MULTIPLIER = 2.0
    REWARD_FIRE_DISCOVERY_BONUS = 50.0
    REWARD_TRACKING_MAX = 15.0
    REWARD_TRACKING_THRESHOLD = 42.0
    REWARD_STOP_BONUS_MULTIPLIER = 5.0
    REWARD_MOVEMENT_PENALTY_MULTIPLIER = 20.0
    REWARD_INTENSITY_MAX = 10.0
    REWARD_INTENSITY_MULTIPLIER = 2.0
    REWARD_GUIDANCE_DISTANCE_THRESHOLD = 42.0
    REWARD_GUIDANCE_RANGE = 27.0
    REWARD_TOO_FAR_PENALTY = 1.0
    REWARD_FIRE_AREA_BONUS = 2.0
    REWARD_FIRE_AREA_PENALTY = 0.5
    REWARD_BOUNDARY_BUFFER = 8.0
    REWARD_BOUNDARY_BASE_PENALTY = 25.0
    REWARD_BOUNDARY_PROGRESSIVE_MULTIPLIER = 2.0
    REWARD_CRASH_PENALTY = 20.0
    REWARD_TIME_PENALTY = 0.1
    REWARD_EXPLORATION_MULTIPLIER = 1.0
    REWARD_COORDINATION_BONUS = 2.0
    REWARD_DEFAULT_FAIL = -10.0
    
    # Exploration tracking
    EXPLORATION_GRID_CELLS = 625  # 25x25 cells for 50x50m map
    
    # Global observation parameters
    GLOBAL_OBSERVATION_RESOLUTION = 16
    GLOBAL_OBSERVATION_KEY_FEATURES = 8
    GLOBAL_OBSERVATION_MAX_AGENTS = 8
    GLOBAL_OBSERVATION_SIZE = 512

# ==============================================================================
# WILDFIRE MODELS CONFIGURATION
# ==============================================================================

class WildfireModelsConfig:
    """Configuration for neural network models"""
    # QuadActor CNN parameters
    CNN_LAYER_1_FILTERS = 16
    CNN_LAYER_1_KERNEL = 3
    CNN_LAYER_1_STRIDE = 2
    CNN_LAYER_2_FILTERS = 32
    CNN_LAYER_2_KERNEL = 3
    CNN_LAYER_2_STRIDE = 2
    CNN_FLATTEN_SIZE = 32 * 7 * 7  # 1568
    CNN_OUTPUT_SIZE = 64
    
    # MLP parameters
    MLP_FUSION_SIZE = 128
    ACTION_HEAD_SIZE = 4  # Direct actions: [roll, pitch, yaw, throttle]
    
    # Initialization parameters
    ACTION_BIAS_THROTTLE = 0.1
    GENTLE_TRAINING_SCALE_CLAMP_MIN = 0.01
    GENTLE_TRAINING_SCALE_CLAMP_MAX = 0.3
    STANDARD_SCALE_CLAMP_MAX = 1.0
    
    # FixedWingActor parameters
    FIXED_WING_ATTENTION_HEADS = 2
    FIXED_WING_MLP_HIDDEN = 128
    FIXED_WING_OUTPUT_SIZE = 8

# ==============================================================================
# SIMULATION CONFIGURATION
# ==============================================================================

class SimulationConfig:
    """Configuration for simulation parameters"""
    # Water drop parameters
    WATER_DROP_BASE_AMOUNT = 200.0
    WATER_DROP_HEIGHT_EFFECTIVENESS_THRESHOLD = 200.0
    WATER_DROP_RADIUS_BASE = 10.0
    WATER_DROP_RADIUS_ALTITUDE_MULTIPLIER = 0.3
    WATER_DROP_SIGMA_DIVISOR = 2.5
    WATER_DROP_INFLUENCE_RADIUS_MULTIPLIER = 3
    WATER_DROP_GAUSSIAN_MULTIPLIER = 1000.0
    WATER_DROP_MIN_SIGMA = 1.0
    
    # Grid optimization parameters
    GRID_OPTIMIZATION_PADDING = 10

# ==============================================================================
# ENVIRONMENT CONFIGURATION  
# ==============================================================================

class EnvironmentConfig:
    """Configuration for environment parameters"""
    # Weather initial settings
    INITIAL_VISIBILITY = 1000.0
    INITIAL_PRECIPITATION = 0.0
    INITIAL_WIND_TIMER = 0.0
    INITIAL_WIND_INTERVAL_MIN = 5.0
    INITIAL_WIND_INTERVAL_MAX = 15.0
    
    # Grid dimensions (default)
    DEFAULT_GRID_WIDTH = 200.0
    DEFAULT_GRID_HEIGHT = 200.0
    DEFAULT_CELL_SIZE = 2.0
    
    # Fire simulation parameters
    DEFAULT_FIRE_DT = 0.1
    FIRE_TIME_ACCUMULATOR_INITIAL = 0.0
    FIRE_ENABLED_INITIAL = False
    
    # Burn rates per m²
    BURN_RATE_GRASS_PER_M2 = 0.01  # 30s per 1x1m cell
    BURN_RATE_FOREST_PER_M2 = 0.0067  # 2 min per 1x1m cell  
    BURN_RATE_BUILDING_PER_M2 = 0.0015  # 10 min per 1x1m cell
    
    # Fuel characteristics (fuel_level, burn_rate)
    FUEL_WATER = (0.0, 0.0)
    FUEL_GRASS_BASE = 0.3
    FUEL_FOREST_BASE = 0.8  
    FUEL_BUILDING_BASE = 0.9
    
    # Physical spread speed
    PHYSICAL_SPREAD_SPEED = 0.01
    
    # Fire simulation constants
    FIRE_K_SLOPE = 1.0
    
    # City block parameters
    CITY_BLOCK_DEFAULT_SIZE = [5, 5, 10]
    CITY_BLOCK_DEFAULT_COLOR = [0.7, 0.7, 0.7, 1]
    
    # Lake parameters  
    LAKE_VISUAL_HEIGHT = 0.1
    LAKE_DEFAULT_COLOR = [0.1, 0.5, 0.9, 0.8]
    
    # Fire visualization parameters
    FIRE_VIS_HEIGHT_BASE = 0.5
    FIRE_VIS_HEIGHT_MULTIPLIER = 2.0
    FIRE_VIS_RADIUS_MULTIPLIER = 0.4
    FIRE_VIS_COLOR = [1, 0, 0, 0.8]
    
    # Environment map colors
    ENV_MAP_GRASS_COLOR = [0.8, 0.9, 0.6]
    ENV_MAP_FOREST_COLOR = [0.1, 0.4, 0.1]  
    ENV_MAP_BUILDING_COLOR = [0.5, 0.5, 0.5]
    ENV_MAP_WATER_COLOR = [0.2, 0.5, 0.9]
    ENV_MAP_FIGURE_SIZE = (10, 10)
    ENV_MAP_DPI = 150
    
    # Random position ranges
    RANDOM_POSITION_X_RANGE = (-200, 200)
    RANDOM_POSITION_Y_RANGE = (-200, 200)  
    RANDOM_POSITION_Z_RANGE = (30, 80)
    
    # Obstacle parameters
    OBSTACLE_DEFAULT_SIZE = 10.0
    OBSTACLE_COLOR = [0, 1, 1, 0.4]  # Cyan with transparency
    OBSTACLE_RADIUS_TOLERANCE = 2.0
    
    # Wind parameters
    WIND_SPEED_MIN = 3.0
    WIND_SPEED_MAX = 25.0
    WIND_SPEED_CHANGE_RANGE = (-1, 1)
    WIND_SPEED_CLAMP_MIN = 2.0
    WIND_SPEED_CLAMP_MAX = 15.0
    WIND_ANGLE_CHANGE_RANGE = (-0.2, 0.2)
    WIND_CHANGE_INTERVAL_MIN = 5.0
    WIND_CHANGE_INTERVAL_MAX = 15.0

# ==============================================================================
# OBSERVATION PROCESSOR CONFIGURATION
# ==============================================================================

class ObservationProcessorConfig:
    """Configuration for observation processing"""
    # Window and resolution settings
    DEFAULT_WINDOW_SIZE = 30.0  # Reduced from 40.0 for better boundary awareness
    DEFAULT_RESOLUTION = 32
    DEFAULT_LIDAR_RAYS = 8
    DEFAULT_LIDAR_DISTANCE = 50.0
    
    # Map bounds (should match environment)
    MAP_BOUNDS = 100.0  # Unified with WildfireGymConfig!
    
    # Self-state vector size
    SELF_STATE_SIZE = 16

# ==============================================================================
# MAPPO CONFIGURATION
# ==============================================================================

class MAPPOConfig:
    """Configuration for MAPPO training"""
    # Data collection
    FRAMES_PER_BATCH = 6000
    N_ITERATIONS = 10
    TOTAL_FRAMES_MULTIPLIER = 1  # total_frames = frames_per_batch * n_iters * this
    
    # PPO training settings
    NUM_EPOCHS = 30
    MINIBATCH_SIZE = 400
    LEARNING_RATE = 3e-4
    MAX_GRAD_NORM = 1.0
    
    # PPO loss settings
    CLIP_EPSILON = 0.2
    GAMMA = 0.99
    LAMBDA = 0.9
    ENTROPY_EPS = 1e-4
    NORMALIZE_ADVANTAGE = False
    
    # Environment settings
    MAX_STEPS = 100
    N_AGENTS = 3
    CONTINUOUS_ACTIONS = True
    
    # Network architecture
    SHARE_PARAMETERS_POLICY = True
    SHARE_PARAMETERS_CRITIC = True
    ACTOR_CENTRALIZED = False
    CRITIC_CENTRALIZED = False  # Set to True for MAPPO
    NETWORK_DEPTH = 2
    NETWORK_NUM_CELLS = 256
    N_AGENT_OUTPUTS = 1
    
    # Demo/testing settings
    DEMO_MAX_STEPS = 400
    DEMO_NUM_ENVS = 1
    DEMO_RETURN_LOG_PROB = False
    DEMO_AUTO_CAST_TO_DEVICE = True
    DEMO_BREAK_WHEN_ANY_DONE = False

# ==============================================================================
# MAP IMPORTER CONFIGURATION
# ==============================================================================

class MapImporterConfig:
    """Configuration for map importing from OSM"""
    DEFAULT_RADIUS = 1500  # meters
    DEFAULT_HEIGHT = 10.0  # meters
    
    # OSM tags for different features
    OSM_TAGS = {
        'landuse': ['residential', 'commercial', 'industrial', 'forest', 'grass', 'meadow', 'reservoir'],
        'natural': ['wood', 'water', 'wetland', 'scrub'],
        'waterway': ['river', 'stream', 'canal', 'drain'],
        'building': True
    }

# ==============================================================================
# DEMO CONFIGURATION
# ==============================================================================

class DemoConfig:
    """Configuration for demo scripts"""
    # Quadcopter demos
    QUAD_WIND_TEST_FIGURE_SIZE = (10, 8)
    QUAD_WIND_INITIAL = [10.0, 0.0, 0.0]
    QUAD_WIND_INCREASED = [20.0, 0.0, 0.0]
    QUAD_WIND_CHANGE_TIME = 5.0
    QUAD_SPAWN_HEIGHT = 5
    QUAD_HOVER_COMMAND = [0, 0, 0, 0]
    
    # Staircase demo
    STAIRCASE_DURATION = 25.0
    STAIRCASE_SPAWN_HEIGHT = 2
    STAIRCASE_PITCH_HALF = 0.5
    STAIRCASE_YAW_RATE = 0.4
    STAIRCASE_YAW_RATE_NEG = -0.4
    
    # Forward demo
    FORWARD_DURATION = 15.0
    FORWARD_PITCH_POSITIVE = 0.5
    FORWARD_PITCH_NEGATIVE = -0.5
    FORWARD_VELOCITY_MULTIPLIER = 15.0
    FORWARD_SPAWN_HEIGHT = 2
    
    # Quadcopter complex demo
    COMPLEX_DURATION = 36.0
    COMPLEX_SPAWN_HEIGHT = 2
    COMPLEX_PITCH_FORWARD = 0.2
    COMPLEX_YAW_RATE = -1
    COMPLEX_ROLL_RIGHT = 0.2
    COMPLEX_ROLL_LEFT = -0.6
    COMPLEX_VERT_UP = 1
    COMPLEX_VERT_DOWN = -0.5
    
    # Separated steps demo
    SEPARATED_STEP_DURATION = 5.0
    SEPARATED_SPAWN_HEIGHT = 2
    SEPARATED_PITCH_CMD = 0.5
    SEPARATED_ROLL_CMD = 0.5  
    SEPARATED_VERT_CMD = 1.0
    SEPARATED_YAW_CMD = 0.1
    
    # Fire influence demo
    FIRE_DEMO_FIGURE_SIZE = (10, 10)
    FIRE_DEMO_FIGURE_SIZE_ANALYSIS = (10, 8)
    
    # Visualization parameters
    VIZ_GRID_ALPHA = 0.3
    VIZ_LINE_STYLE = ':'
    VIZ_LINEWIDTH = 2
    VIZ_MARKERSIZE = 3
    VIZ_ALPHA = 0.7
    VIZ_START_MARKERSIZE = 8
    VIZ_END_MARKERSIZE = 8

# ==============================================================================
# GRID MAPPER CONFIGURATION
# ==============================================================================

class GridMapperConfig:
    """Configuration for grid mapping"""
    # Clamping bounds (used in world_to_cell methods)
    CLAMP_MIN_INDEX = 0
    # Max indices are determined dynamically based on grid size