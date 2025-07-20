from simulation import Simulation


def create_basic_scenario():
    """Create a basic two-drone crossing scenario."""
    sim = Simulation(
        name="Basic Quadcopter Crossing",
        duration=30,
        plot_bounds=(-10, 60, -10, 100)
    )
    
    # Add two quadcopters
    sim.add_drone("quadcopter", [0., 0.], 0., [40., 40.])
    sim.add_drone("quadcopter", [40., 0.], 0., [0., 40.])
    
    return sim


def create_linear_scenario():
    """Create a basic two-drone crossing scenario."""
    sim = Simulation(
        name="Basic Quadcopter Crossing",
        duration=60,
        plot_bounds=(-10, 60, -10, 100)
    )
    
    # Add two quadcopters
    sim.add_drone("quadcopter", [0., 0.], 0., [0., 40.])
    sim.add_drone("quadcopter", [5., 0.], 0., [5., 40.])
    
    return sim


def create_multi_drone_scenario():
    """Create a more complex scenario with multiple drones."""
    sim = Simulation(
        name="Multi-Drone Navigation",
        duration=80,
        plot_bounds=(-20, 80, -20, 80)
    )
    
    # Add multiple quadcopters in a challenging configuration
    sim.add_drone("quadcopter", [0., 0.], 0., [60., 60.])      # Bottom-left to top-right
    sim.add_drone("quadcopter", [60., 0.], 180., [0., 60.])    # Bottom-right to top-left
    sim.add_drone("quadcopter", [30., 0.], 90., [30., 60.])    # Bottom-center to top-center
    sim.add_drone("quadcopter", [0., 30.], 0., [60., 30.])     # Left-center to right-center
    
    return sim


def create_mixed_scenario():
    """Create a scenario with both quadcopters and fixed-wing aircraft."""
    sim = Simulation(
        name="Mixed Aircraft Types",
        duration=100,
        plot_bounds=(-10, 100, -10, 80)
    )
    
    # Add quadcopters
    sim.add_drone("quadcopter", [0., 10.], 0., [40., 40.])
    sim.add_drone("quadcopter", [10., 0.], 90., [30., 60.])
    
    # Add fixed-wing aircraft
    sim.add_drone("fixedwing", [50., 0.], 90., [50., 70.])
    
    return sim


def run_scenario(scenario_func, output_file=None):
    """Run a specific scenario and generate animation."""
    sim = scenario_func()
    
    print(f"\n{'='*50}")
    print(f"Running scenario: {sim.name}")
    print(f"{'='*50}")
    
    # Run simulation
    results = sim.run()
    
    # Create animation
    if output_file is None:
        output_file = f"{sim.name.lower().replace(' ', '_')}.gif"
    
    sim.create_animation(output_file)
    
    return results


def main():
    """Run different simulation scenarios."""
    
    # Run basic scenario
    print("🎬 Running simulation scenarios...")
    
    # Scenario 1: Basic crossing
    results1 = run_scenario(create_basic_scenario, "basic_crossing.gif")
    # results1 = run_scenario(create_linear_scenario, "basic_crossing.gif")
    
    # Scenario 2: Multi-drone
    # results2 = run_scenario(create_multi_drone_scenario, "multi_drone.gif")
    
    # Scenario 3: Mixed aircraft types
    # results3 = run_scenario(create_mixed_scenario, "mixed_aircraft.gif")
    
    # Print summary
    print(f"\n{'='*50}")
    print("SIMULATION SUMMARY")
    print(f"{'='*50}")
    
    # scenarios = [results1, results2, results3]
    scenarios = [results1]
    for i, result in enumerate(scenarios, 1):
        print(f"Scenario {i}: {result['name']}")
        print(f"  - Drones: {result['drones']}")
        print(f"  - Steps: {result['steps']}")
        print(f"  - Collisions: {result['collisions']}")
        print(f"  - Collision rate: {result['collisions']/result['steps']*100:.1f}%")
        print()
    
    print("✅ All scenarios completed!")
    print("📁 Generated files:")
    print("   - basic_crossing.gif")
    print("   - multi_drone.gif") 
    print("   - mixed_aircraft.gif")


if __name__ == "__main__":
    main()
