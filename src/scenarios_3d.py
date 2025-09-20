from simulation import Simulation
from environment import Environment, TerrainZone, TerrainType


def create_3d_altitude_layers_scenario():
    """Create a 3D scenario with drones flying at different altitude layers."""
    
    # Create 3D environment with layered restrictions
    env = Environment(bounds=(-20, 100, -20, 80), name="3D Altitude Layers")
    env.generate_3d_environment()
    
    sim = Simulation(
        name="3D Altitude Layers Navigation",
        duration=120,
        plot_bounds=(-20, 100, -20, 80),
        environment=env
    )
    
    # Low altitude layer (20-40m) - Small quadcopters
    sim.add_drone("quadcopter", [0, 10, 25], 45, [80, 60, 35])     # Low altitude corridor
    sim.add_drone("quadcopter", [10, 0, 30], 90, [70, 70, 25])     # Low altitude crossing
    
    # Medium altitude layer (60-100m) - Mixed aircraft
    sim.add_drone("fixedwing", [0, 50, 80], 0, [90, 50, 90])       # Straight flight
    sim.add_drone("quadcopter", [80, 10, 70], 180, [10, 60, 80])   # High speed crossing
    
    # High altitude layer (120-150m) - Long range fixed-wing
    sim.add_drone("fixedwing", [20, 0, 140], 45, [60, 70, 130])    # High altitude transit
    
    return sim


def create_3d_obstacle_avoidance_scenario():
    """Create a 3D scenario with altitude-based obstacle avoidance."""
    
    # Create custom environment with vertical obstacles
    env = Environment(bounds=(-10, 80, -10, 80), name="3D Obstacle Course")
    
    # Tower obstacles at different heights
    env.add_terrain_zone(TerrainZone(
        TerrainType.NO_FLY_ZONE,
        (20, 25, 20, 25),
        altitude_restriction=(0, 60),  # Tower up to 60m
        avoidance_priority=1000
    ))
    
    env.add_terrain_zone(TerrainZone(
        TerrainType.NO_FLY_ZONE,
        (50, 55, 40, 45),
        altitude_restriction=(0, 80),  # Taller tower up to 80m
        avoidance_priority=1000
    ))
    
    # High altitude restricted zone
    env.add_terrain_zone(TerrainZone(
        TerrainType.NO_FLY_ZONE,
        (30, 50, 10, 30),
        altitude_restriction=(100, 200),  # High altitude restriction
        avoidance_priority=800
    ))
    
    sim = Simulation(
        name="3D Obstacle Avoidance Challenge",
        duration=100,
        plot_bounds=(-10, 80, -10, 80),
        environment=env
    )
    
    # Drones must navigate around obstacles at different altitudes
    sim.add_drone("quadcopter", [0, 22], 0, [70, 22, 50], altitude=40)    # Must go over first tower
    sim.add_drone("quadcopter", [0, 42], 0, [70, 42, 50], altitude=50)    # Must go over second tower  
    sim.add_drone("fixedwing", [10, 20, 120], 0, [60, 20, 90])            # High alt, must descend
    
    return sim


def create_3d_vertical_separation_scenario():
    """Create a scenario demonstrating vertical separation for collision avoidance."""
    
    env = Environment(bounds=(-10, 60, -10, 60), name="Vertical Separation")
    env.generate_natural_environment()  # Standard terrain
    
    sim = Simulation(
        name="3D Vertical Separation",
        duration=80,
        plot_bounds=(-10, 60, -10, 60),
        environment=env
    )
    
    # Multiple drones on converging paths - should separate vertically
    sim.add_drone("quadcopter", [0, 0, 30], 45, [50, 50, 30])      # Diagonal low
    sim.add_drone("quadcopter", [50, 0, 35], 135, [0, 50, 35])     # Diagonal crossing
    sim.add_drone("quadcopter", [25, 0, 40], 90, [25, 50, 40])     # Straight through middle
    sim.add_drone("fixedwing", [0, 25, 80], 0, [50, 25, 80])       # High altitude crossing
    
    return sim


def create_3d_climb_descent_scenario():
    """Create a scenario with significant altitude changes."""
    
    env = Environment(bounds=(-20, 80, -20, 80), name="Climb and Descent")
    
    # Mountain terrain requiring altitude changes
    env.add_terrain_zone(TerrainZone(
        TerrainType.MOUNTAIN,
        (20, 40, 30, 50),
        altitude_restriction=(60, 1000),  # High mountain
        speed_modifier=0.7
    ))
    
    env.add_terrain_zone(TerrainZone(
        TerrainType.MOUNTAIN,
        (50, 70, 10, 30),
        altitude_restriction=(80, 1000),  # Higher peak
        speed_modifier=0.6
    ))
    
    sim = Simulation(
        name="3D Climb and Descent Challenge",
        duration=150,
        plot_bounds=(-20, 80, -20, 80),
        environment=env
    )
    
    # Drones must climb over mountains and descend
    sim.add_drone("quadcopter", [0, 40, 20], 0, [60, 40, 25])      # Must climb over mountains
    sim.add_drone("fixedwing", [0, 20, 50], 0, [70, 20, 120])      # Gradual climb
    sim.add_drone("fixedwing", [70, 60, 150], 225, [10, 10, 40])   # Steep descent
    
    return sim


def create_3d_formation_flight_scenario():
    """Create a scenario with drones maintaining 3D formation."""
    
    env = Environment(bounds=(-10, 100, -10, 60), name="Formation Flight")
    env.generate_natural_environment()
    
    sim = Simulation(
        name="3D Formation Flight",
        duration=100,
        plot_bounds=(-10, 100, -10, 60),
        environment=env
    )
    
    # Formation of quadcopters at different altitudes
    formation_center = [80, 30, 60]
    
    # Diamond formation with vertical spacing
    sim.add_drone("quadcopter", [0, 30, 60], 0, formation_center)                    # Lead
    sim.add_drone("quadcopter", [0, 25, 65], 0, [formation_center[0], formation_center[1]-5, formation_center[2]+5])   # Left wing high
    sim.add_drone("quadcopter", [0, 35, 55], 0, [formation_center[0], formation_center[1]+5, formation_center[2]-5])   # Right wing low
    sim.add_drone("quadcopter", [0, 30, 70], 0, [formation_center[0], formation_center[1], formation_center[2]+10])    # High trail
    
    return sim


def run_3d_scenario(scenario_func, output_file=None):
    """Run a specific 3D scenario and generate appropriate visualization."""
    sim = scenario_func()
    
    print(f"\n{'='*60}")
    print(f"Running 3D scenario: {sim.name}")
    print(f"Environment: {sim.environment}")
    print(f"{'='*60}")
    
    # Run simulation
    results = sim.run()
    
    # Create smart animation (will choose 3D if appropriate)
    if output_file is None:
        output_file = f"{sim.name.lower().replace(' ', '_')}_3d.gif"
    
    sim.create_smart_animation(output_file)
    
    return results


def main():
    """Run 3D drone simulation scenarios."""
    
    print("🌍 Running 3D drone simulation scenarios...")
    
    # Scenario 1: Altitude Layers
    results1 = run_3d_scenario(create_3d_altitude_layers_scenario, "altitude_layers_3d.gif")
    
    # Scenario 2: Obstacle Avoidance  
    results2 = run_3d_scenario(create_3d_obstacle_avoidance_scenario, "obstacle_avoidance_3d.gif")
    
    # Scenario 3: Vertical Separation
    results3 = run_3d_scenario(create_3d_vertical_separation_scenario, "vertical_separation_3d.gif")
    
    # Scenario 4: Climb and Descent
    results4 = run_3d_scenario(create_3d_climb_descent_scenario, "climb_descent_3d.gif")
    
    # Scenario 5: Formation Flight
    results5 = run_3d_scenario(create_3d_formation_flight_scenario, "formation_flight_3d.gif")
    
    # Print summary
    print(f"\n{'='*60}")
    print("3D SIMULATION SUMMARY")
    print(f"{'='*60}")
    
    scenarios = [results1, results2, results3, results4, results5]
    for i, result in enumerate(scenarios, 1):
        print(f"Scenario {i}: {result['name']}")
        print(f"  - Drones: {result['drones']}")
        print(f"  - Steps: {result['steps']}")
        print(f"  - Collisions: {result['collisions']}")
        print(f"  - Collision rate: {result['collisions']/result['steps']*100:.1f}%")
        print()
    
    print("✅ All 3D scenarios completed!")
    print("📁 Generated 3D animation files:")
    print("   - altitude_layers_3d.gif")
    print("   - obstacle_avoidance_3d.gif") 
    print("   - vertical_separation_3d.gif")
    print("   - climb_descent_3d.gif")
    print("   - formation_flight_3d.gif")


if __name__ == "__main__":
    main()