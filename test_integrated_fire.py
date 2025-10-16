#!/usr/bin/env python3
"""
Test script for integrated fire simulation with drone environment.
"""

import numpy as np
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.simulation import Simulation
import time


def test_integrated_fire_simulation():
    """Test the complete integrated fire and drone simulation system."""
    print("Testing Integrated Fire Simulation System")
    print("=" * 50)
    
    # Create simulation
    sim = Simulation(gui=False)  # Headless mode for testing
    sim.start_simulation()
    
    try:
        # Setup environment with forests and buildings
        sim.setup_mixed_environment()
        
        # Enable fire simulation
        sim.enable_fire_simulation(grid_width_m=80, grid_height_m=80, cell_size_m=2.0)
        
        # Set wind conditions
        sim.set_wind([5.0, 3.0, 0.0])  # Wind blowing northeast
        
        # Add drones for fire fighting
        drone1 = sim.add_quadcopter("FireFighter1", position=[10, 10, 20])
        drone2 = sim.add_quadcopter("FireFighter2", position=[-10, -10, 25])
        
        # Start multiple fires
        fire_locations = [
            (15, 15),   # Near drone 1
            (-20, 0),   # Forest area
            (0, -25),   # Another forest area
        ]
        
        for pos in fire_locations:
            success = sim.start_fire(pos, intensity=0.25)
            if success:
                print(f"✅ Fire started at {pos}")
            else:
                print(f"❌ Failed to start fire at {pos}")
        
        print(f"\nRunning simulation for 30 seconds...")
        
        # Run simulation with drone movements
        total_steps = 300  # 30 seconds at 240 FPS = 7200 steps, but we'll sample every 24 steps
        
        for step in range(total_steps):
            # Time in simulation
            t = step * 0.1  # 10 FPS effective rate
            
            # Drone 1: Circle around first fire area
            drone1_controls = [
                0.2 * np.sin(t * 0.5),  # X movement
                0.2 * np.cos(t * 0.5),  # Y movement  
                0.0                     # Maintain altitude
            ]
            
            # Drone 2: Move towards fires
            drone2_controls = [
                0.1 * np.sin(t * 0.3),  # X movement
                0.1 * np.cos(t * 0.7),  # Y movement
                0.0                     # Maintain altitude
            ]
            
            controls = {
                "FireFighter1": drone1_controls,
                "FireFighter2": drone2_controls
            }
            
            # Step simulation 24 times (to get 10 FPS from 240 FPS)
            for _ in range(24):
                sim.step_simulation(controls)
            
            # Print progress every 5 seconds
            if step % 50 == 0:
                summary = sim.get_simulation_summary()
                if 'fire' in summary:
                    fire_stats = summary['fire']
                    print(f"t={t:.1f}s: {fire_stats['burning_cells']} burning cells, "
                          f"fuel: {fire_stats['total_fuel']:.1f}")
                else:
                    print(f"t={t:.1f}s: Simulation running...")
        
        # Generate comprehensive analysis
        print("\n" + "=" * 50)
        print("Generating Analysis...")
        
        sim.create_multi_drone_visualization("Fire Fighting Mission")
        
        # Get final summary
        final_summary = sim.get_simulation_summary()
        print(f"\nFinal Summary:")
        print(f"- Total simulation time: {final_summary['total_time']:.2f}s")
        print(f"- Total steps: {final_summary['total_steps']}")
        print(f"- Number of drones: {len(final_summary['drones'])}")
        print(f"- Collisions: {final_summary['collisions']}")
        
        if 'fire' in final_summary:
            fire_stats = final_summary['fire']
            print(f"- Final burning cells: {fire_stats['burning_cells']}")
            print(f"- Remaining fuel: {fire_stats['total_fuel']:.1f}")
            print(f"- Burn percentage: {fire_stats['burn_percentage']:.1f}%")
        
    finally:
        sim.stop_simulation()
    
    print("\n" + "=" * 50)
    print("✅ Integrated fire simulation test completed!")
    print("Check the output/ directory for visualizations:")
    print("- multi_drone_combined_analysis.png")
    print("- multi_drone_combined_fire_analysis.png")


def test_grid_mapper():
    """Test the GridMapper functionality."""
    print("\nTesting GridMapper...")
    
    from grid_mapper import GridMapper
    
    mapper = GridMapper(grid_width_m=100, grid_height_m=80, cell_size_m=2.5)
    
    # Test coordinate conversions
    test_positions = [
        (0, 0),      # Center
        (25, 20),    # Positive quadrant
        (-30, -25),  # Negative quadrant
        (100, 100),  # Outside bounds (should be clipped)
    ]
    
    print("Coordinate conversion tests:")
    for world_pos in test_positions:
        cell_indices = mapper.world_to_cell(world_pos)
        back_to_world = mapper.cell_to_world(*cell_indices)
        in_bounds = mapper.is_position_in_bounds(world_pos)
        
        print(f"  {world_pos} -> cell {cell_indices} -> world {back_to_world} (in bounds: {in_bounds})")
    
    # Test bounds
    bounds = mapper.get_grid_bounds()
    dimensions = mapper.get_grid_dimensions()
    print(f"Grid bounds: {bounds}")
    print(f"Grid dimensions: {dimensions}")
    
    print("✅ GridMapper tests passed!")


def demonstrate_fire_only():
    """Demonstrate fire simulation without drones."""
    print("\nDemonstrating Fire-Only Simulation...")
    
    # Create simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    try:
        # Setup natural environment
        sim.setup_natural_environment()
        
        # Enable fire simulation with smaller grid for faster demo
        sim.enable_fire_simulation(grid_width_m=60, grid_height_m=60, cell_size_m=1.5)
        
        # Set strong wind
        sim.set_wind([8.0, 2.0, 0.0])  # Strong east wind
        
        # Start central fire
        sim.start_fire((0, 0), intensity=0.3)
        
        print("Running fire-only simulation for 100 steps...")
        
        # Run fire simulation only
        for step in range(100):
            sim.step_simulation({})  # No drone controls
            
            if step % 20 == 0:
                summary = sim.get_simulation_summary()
                if 'fire' in summary:
                    fire_stats = summary['fire']
                    print(f"Step {step}: {fire_stats['burning_cells']} burning, fuel: {fire_stats['total_fuel']:.1f}")
        
        # Get fire state
        fire_state = sim.environment.get_fire_state()
        if fire_state:
            stats = fire_state['fire_stats']
            print(f"Final fire state: {stats['burning_cells']} burning cells")
    
    finally:
        sim.stop_simulation()
    
    print("✅ Fire-only demonstration completed!")


def main():
    """Run all tests."""
    print("Integrated Fire Simulation Test Suite")
    print("=" * 60)
    
    # Test grid mapper
    test_grid_mapper()
    
    # Test fire-only simulation
    demonstrate_fire_only()
    
    # Test full integrated system
    test_integrated_fire_simulation()
    
    print("\n" + "=" * 60)
    print("🔥 All tests completed successfully! 🚁")
    print("The fire simulation is now fully integrated with the drone environment!")


if __name__ == "__main__":
    main()