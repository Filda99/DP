from simulation import Simulation
from environment import Environment, TerrainZone, TerrainType


def create_natural_environment_scenario():
    """Create a drone scenario in a natural environment with forests, lakes, etc."""
    
    # Create environment with terrain features
    env = Environment(bounds=(-10, 100, -10, 80), name="Natural Environment")
    env.generate_natural_environment()
    
    # Create simulation with the environment
    sim = Simulation(
        name="Natural Environment Navigation",
        duration=120,
        plot_bounds=(-10, 100, -10, 80),
        environment=env
    )
    
    # Add drones with starting positions that avoid difficult terrain
    sim.add_drone("quadcopter", [5., 5.], 45., [85., 70.])      # Long diagonal flight
    sim.add_drone("quadcopter", [85., 5.], 135., [5., 70.])     # Crossing flight
    sim.add_drone("fixedwing", [5., 40.], 0., [90., 40.])       # Straight east flight
    
    return sim


def create_urban_environment_scenario():
    """Create a drone scenario in an urban environment."""
    
    # Create urban environment
    env = Environment(bounds=(-10, 100, -10, 80), name="Urban Environment")
    env.generate_urban_environment()
    
    # Add some weather for realism
    env.set_weather(wind_speed=8.0, wind_direction=45.0, visibility=800.0)
    
    sim = Simulation(
        name="Urban Drone Delivery",
        duration=100,
        plot_bounds=(-10, 100, -10, 80),
        environment=env
    )
    
    # Add delivery drones navigating between buildings
    sim.add_drone("quadcopter", [10., 10.], 0., [80., 60.])
    sim.add_drone("quadcopter", [80., 10.], 90., [20., 60.])
    
    return sim


def create_mixed_terrain_scenario():
    """Create a custom scenario with mixed terrain types."""
    
    # Create custom environment
    env = Environment(bounds=(-20, 120, -20, 100), name="Mixed Terrain")
    
    # Add custom terrain zones
    # Forest area
    env.add_terrain_zone(TerrainZone(
        TerrainType.FOREST,
        (10, 40, 20, 60),
        speed_modifier=0.7,
        avoidance_priority=3
    ))
    
    # Lake
    env.add_terrain_zone(TerrainZone(
        TerrainType.LAKE,
        (50, 80, 10, 40),
        speed_modifier=1.2
    ))
    
    # Mountain range
    env.add_terrain_zone(TerrainZone(
        TerrainType.MOUNTAIN,
        (70, 100, 50, 80),
        altitude_restriction=(60.0, 1000.0),
        speed_modifier=0.6,
        avoidance_priority=5
    ))
    
    # No-fly zone (military base)
    env.add_terrain_zone(TerrainZone(
        TerrainType.NO_FLY_ZONE,
        (30, 50, 70, 90),
        avoidance_priority=1000
    ))
    
    # Urban area
    env.add_terrain_zone(TerrainZone(
        TerrainType.URBAN,
        (80, 110, 10, 30),
        altitude_restriction=(30.0, 1000.0),
        speed_modifier=0.8,
        avoidance_priority=4
    ))
    
    # Set challenging weather
    env.set_weather(wind_speed=12.0, wind_direction=270.0, precipitation=0.3)
    
    sim = Simulation(
        name="Mixed Terrain Challenge",
        duration=150,
        plot_bounds=(-20, 120, -20, 100),
        environment=env
    )
    
    # Add multiple drones with challenging routes
    sim.add_drone("quadcopter", [0., 0.], 45., [100., 80.])     # Diagonal across all terrain
    sim.add_drone("fixedwing", [0., 50.], 0., [110., 50.])      # East across terrain
    sim.add_drone("quadcopter", [50., 0.], 90., [50., 90.])     # North through no-fly zone
    sim.add_drone("quadcopter", [100., 0.], 180., [0., 90.])    # West then north
    
    return sim


def create_weather_challenge_scenario():
    """Create a scenario with challenging weather conditions."""
    
    env = Environment(bounds=(-10, 80, -10, 60), name="Weather Challenge")
    env.generate_natural_environment()
    
    # Set severe weather
    env.set_weather(
        wind_speed=15.0,      # Strong winds
        wind_direction=180.0, # South wind
        visibility=300.0,     # Low visibility
        precipitation=0.7     # Heavy rain
    )
    
    sim = Simulation(
        name="Storm Navigation",
        duration=80,
        plot_bounds=(-10, 80, -10, 60),
        environment=env
    )
    
    # Add drones that must navigate in bad weather
    sim.add_drone("quadcopter", [10., 10.], 0., [60., 40.])
    sim.add_drone("quadcopter", [60., 10.], 90., [10., 40.])
    
    return sim


def run_scenario(scenario_func, output_file=None):
    """Run a specific scenario and generate animation."""
    sim = scenario_func()
    
    print(f"\n{'='*60}")
    print(f"Running scenario: {sim.name}")
    print(f"Environment: {sim.environment}")
    print(f"Weather: Wind {sim.environment.weather_conditions['wind_speed']:.1f}m/s, "
          f"Visibility {sim.environment.weather_conditions['visibility']:.0f}m")
    print(f"{'='*60}")
    
    # Run simulation
    results = sim.run()
    
    # Create animation
    if output_file is None:
        output_file = f"{sim.name.lower().replace(' ', '_')}.gif"
    
    sim.create_animation(output_file)
    
    return results


def main():
    """Run enhanced simulation scenarios with terrain and weather."""
    
    print("🌍 Running enhanced drone simulation scenarios with terrain and weather...")
    
    # Scenario 1: Natural environment
    results1 = run_scenario(create_natural_environment_scenario, "natural_environment.gif")
    
    # Scenario 2: Urban environment
    results2 = run_scenario(create_urban_environment_scenario, "urban_environment.gif")
    
    # Scenario 3: Mixed terrain challenge
    results3 = run_scenario(create_mixed_terrain_scenario, "mixed_terrain.gif")
    
    # Scenario 4: Weather challenge
    results4 = run_scenario(create_weather_challenge_scenario, "weather_challenge.gif")
    
    # Print summary
    print(f"\n{'='*60}")
    print("ENHANCED SIMULATION SUMMARY")
    print(f"{'='*60}")
    
    scenarios = [results1, results2, results3, results4]
    for i, result in enumerate(scenarios, 1):
        print(f"Scenario {i}: {result['name']}")
        print(f"  - Drones: {result['drones']}")
        print(f"  - Steps: {result['steps']}")
        print(f"  - Collisions: {result['collisions']}")
        print(f"  - Collision rate: {result['collisions']/result['steps']*100:.1f}%")
        print()
    
    print("✅ All enhanced scenarios completed!")
    print("📁 Generated files:")
    print("   - natural_environment.gif")
    print("   - urban_environment.gif") 
    print("   - mixed_terrain.gif")
    print("   - weather_challenge.gif")


if __name__ == "__main__":
    main()
