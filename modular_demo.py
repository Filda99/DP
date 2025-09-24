"""
Modular Drone Simulation Demo

Demonstrates the new modular architecture with multiple drones,
environmental features, and realistic physics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import Simulation
import numpy as np


def simple_quadcopter_scenario(step, time, drones):
    """Simple square flight pattern for quadcopter."""
    controls = {}
    
    if 'quad1' in drones:
        # Simple square pattern based on time
        if time < 2.0:
            controls['quad1'] = [0.0, 0.0, 0.0]  # Hover
        elif time < 5.0:
            controls['quad1'] = [-1.0, 0.0, 0.0]  # Left
        elif time < 6.0:
            controls['quad1'] = [0.0, 0.0, 0.0]  # Hover
        elif time < 9.0:
            controls['quad1'] = [0.0, 1.0, 0.0]  # Forward
        elif time < 10.0:
            controls['quad1'] = [0.0, 0.0, 0.0]  # Hover
        elif time < 13.0:
            controls['quad1'] = [1.0, 0.0, 0.0]  # Right
        elif time < 14.0:
            controls['quad1'] = [0.0, 0.0, 0.0]  # Hover
        elif time < 17.0:
            controls['quad1'] = [0.0, -1.0, 0.0]  # Back
        else:
            controls['quad1'] = [0.0, 0.0, 0.0]  # Final hover
    
    return controls


def multi_drone_scenario(step, time, drones):
    """Scenario with multiple drones of different types."""
    controls = {}
    
    # Quadcopter 1 - square pattern
    if 'quad1' in drones:
        if time < 2.0:
            controls['quad1'] = [0.0, 0.0, 0.0]
        elif time < 4.0:
            controls['quad1'] = [-0.8, 0.0, 0.0]
        elif time < 6.0:
            controls['quad1'] = [0.0, 0.8, 0.0]
        elif time < 8.0:
            controls['quad1'] = [0.8, 0.0, 0.0]
        elif time < 10.0:
            controls['quad1'] = [0.0, -0.8, 0.0]
        else:
            controls['quad1'] = [0.0, 0.0, 0.0]
    
    # Quadcopter 2 - vertical maneuvers
    if 'quad2' in drones:
        if time < 3.0:
            controls['quad2'] = [0.0, 0.0, 0.0]
        elif time < 6.0:
            controls['quad2'] = [0.0, 0.0, 0.8]  # Up
        elif time < 8.0:
            controls['quad2'] = [0.0, 0.0, 0.0]  # Hover high
        elif time < 11.0:
            controls['quad2'] = [0.0, 0.0, -0.4]  # Down
        else:
            controls['quad2'] = [0.0, 0.0, 0.0]
    
    # Fixed-wing - circular pattern
    if 'fixedwing1' in drones:
        # Fixed-wing needs constant forward motion
        turn = 0.3 * np.sin(time * 0.5)  # Gentle turns
        controls['fixedwing1'] = [turn, 0.6, 0.1 * np.sin(time * 0.3)]  # Turn, throttle, climb
    
    return controls


def main():
    """Main demonstration."""
    print("🚁 Modular PyBullet Drone Simulation Demo")
    print("=" * 60)
    
    # Initialize simulation
    sim = Simulation(gui=False)  # Set to True for visual debugging
    sim.start_simulation()
    
    print("\n🏗️ Setting up environment...")
    
    # Choose environment type
    environment_choice = input("Choose environment (1=City, 2=Natural, 3=Mixed, Enter=Mixed): ").strip()
    
    if environment_choice == "1":
        sim.setup_city_environment()
    elif environment_choice == "2":
        sim.setup_natural_environment()
    else:
        sim.setup_mixed_environment()
    
    # Set weather conditions
    print("\n🌤️ Setting weather conditions...")
    sim.set_wind([2.0, 1.0, 0.0], turbulence=0.5)  # Light wind with turbulence
    sim.set_weather(visibility=800, precipitation=0.2)  # Moderate visibility, light rain
    
    # Add drones
    print("\n🚁 Adding drones...")
    
    scenario_choice = input("Choose scenario (1=Single Quad, 2=Multi-drone, Enter=Multi): ").strip()
    
    if scenario_choice == "1":
        # Single quadcopter scenario
        sim.add_quadcopter("quad1", position=[0, 0, 5])
        scenario_func = simple_quadcopter_scenario
        steps = 4000  # ~17 seconds
        title = "Single Quadcopter City Flight"
        
    else:
        # Multi-drone scenario
        sim.add_quadcopter("quad1", position=[-5, -5, 5])
        sim.add_quadcopter("quad2", position=[5, 5, 5])
        sim.add_fixedwing("fixedwing1", position=[0, -10, 8])
        scenario_func = multi_drone_scenario
        steps = 3000  # ~12.5 seconds
        title = "Multi-Drone Environmental Flight"
    
    # Show initial status
    print("\n📊 Initial drone status:")
    for name, status in sim.get_all_drone_status().items():
        print(f"  {name}: {status['type']} at {status['position']}")
    
    # Run simulation
    print(f"\n🚀 Running scenario...")
    sim.run_scenario(scenario_func, steps)
    
    # Final status
    print("\n📊 Final simulation results:")
    summary = sim.get_simulation_summary()
    print(f"  Total time: {summary['total_time']:.2f}s")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Collisions: {summary['collisions']}")
    
    print("\n📍 Final drone positions:")
    for name, status in summary['drones'].items():
        pos = status['position']
        print(f"  {name}: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
    
    # Create visualizations for each drone
    print("\n📊 Creating visualizations...")
    for drone_name in sim.drones.keys():
        sim.create_visualization(drone_name, title)
    
    # Stop simulation
    sim.stop_simulation()
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"✅ Generated visualization files for {len(sim.drones)} drones")
    print("🚁 Modular architecture working perfectly!")


if __name__ == "__main__":
    main()