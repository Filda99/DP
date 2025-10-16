#!/usr/bin/env python3
"""
Complete Fire Fighting Simulation Demo

This demo creates a realistic wildfire scenario with multiple drones
fighting the fire. It generates comprehensive visualizations showing:
- Drone flight paths
- Fire spread over time
- Fuel consumption
- Suppression effectiveness
- 3D trajectory views
- Fire analysis plots
"""

import numpy as np
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation


def run_fire_fighting_demo():
    """
    Run a complete fire fighting simulation with visualization.
    
    This creates:
    1. A mixed environment with forests, lakes, and buildings
    2. Multiple fires started in forest areas
    3. Two quadcopter drones patrolling and fighting fires
    4. Comprehensive visualizations of the entire mission
    """
    
    print("=" * 70)
    print("🔥 WILDFIRE FIGHTING SIMULATION DEMO 🚁")
    print("=" * 70)
    print()
    print("This demo will:")
    print("  1. Create a mixed environment (forests, lakes, buildings)")
    print("  2. Enable fire simulation with wind effects")
    print("  3. Start multiple wildfire sources")
    print("  4. Deploy 2 firefighting drones")
    print("  5. Run 60-second simulation with fire fighting")
    print("  6. Generate comprehensive visualizations")
    print()
    print("=" * 70)
    
    # Create simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    try:
        # 1. Setup environment
        print("\n📍 Setting up environment...")
        sim.setup_mixed_environment()
        
        # 2. Enable fire simulation with aggressive spread parameters
        print("🔥 Enabling fire simulation...")
        # Use smaller cell size and higher base spread rate for more dramatic fire
        sim.enable_fire_simulation(
            grid_width_m=100, 
            grid_height_m=100, 
            cell_size_m=3.0  # Larger cells = faster apparent spread
        )
        
        # Manually adjust fire spread parameters for more aggressive fire
        if sim.environment.fire_enabled:
            # Increase base lambda (fire spread rate)
            sim.environment.fire_grid.l_base *= 3.0  # 3x faster spread
            # Reduce alpha (less distance decay = fire spreads further)
            sim.environment.fire_grid.alpha = 0.3  # Was 1.0, now spreads much further
            print("   🔥 Fire spread parameters adjusted for aggressive wildfire behavior")
        
        # 3. Set weather conditions (wind affects fire spread!)
        print("💨 Setting wind conditions...")
        sim.set_wind([6.0, 4.0, 0.0], turbulence=0.5)
        
        # 4. Add firefighting drones
        print("🚁 Deploying firefighting drones...")
        drone1 = sim.add_quadcopter("FireFighter_Alpha", position=[15, 15, 35])
        drone2 = sim.add_quadcopter("FireFighter_Bravo", position=[-15, -15, 40])
        
        # 5. Start wildfires in strategic locations
        print("🔥 Starting wildfires...")
        fire_locations = [
            (22, 22),    # Inside first forest area [20, 20] radius 12
            (-23, 2),    # Inside second forest area [-25, 0] radius 10
            (18, 18),    # Another spot in first forest
        ]
        
        fires_started = 0
        for pos in fire_locations:
            if sim.start_fire(pos, intensity=0.3):
                fires_started += 1
        
        print(f"   ✓ {fires_started} fires ignited")
        
        # 6. Run simulation
        print("\n⏱️  Running 60-second fire fighting mission...")
        print("   (This may take a minute...)")
        print()
        
        total_steps = 600  # 60 seconds at 10 Hz effective rate
        progress_markers = [0, 15, 30, 45, 60]
        
        for step in range(total_steps):
            # Calculate simulation time
            t = step * 0.1
            
            # Drone 1: Patrol pattern - circular around fire zone
            angle1 = t * 0.3
            radius1 = 20
            drone1_controls = [
                0.3 * np.cos(angle1),
                0.3 * np.sin(angle1),
                0.05 * np.sin(t * 0.2)  # Slight altitude variation
            ]
            
            # Drone 2: Different patrol pattern - figure-8
            angle2 = t * 0.4
            drone2_controls = [
                0.25 * np.sin(angle2),
                0.2 * np.sin(2 * angle2),
                0.03 * np.cos(t * 0.3)
            ]
            
            controls = {
                "FireFighter_Alpha": drone1_controls,
                "FireFighter_Bravo": drone2_controls
            }
            
            # Step simulation 24 times (240 FPS -> 10 Hz)
            for _ in range(24):
                sim.step_simulation(controls)
            
            # Progress updates
            current_time = int(t)
            if current_time in progress_markers:
                summary = sim.get_simulation_summary()
                if 'fire' in summary:
                    fire_stats = summary['fire']
                    print(f"   [{current_time:2d}s] Burning: {fire_stats['burning_cells']:4d} cells | "
                          f"Fuel: {fire_stats['total_fuel']:6.1f} | "
                          f"Burn%: {fire_stats['burn_percentage']:5.1f}%")
                progress_markers.remove(current_time)
        
        # 7. Generate visualizations
        print("\n📊 Generating comprehensive visualizations...")
        print("   This includes:")
        print("   - 3D drone trajectories with environment")
        print("   - Top-view flight paths with terrain")
        print("   - Force and velocity analysis")
        print("   - Fire spread over time")
        print("   - Fuel consumption graphs")
        print("   - Final fire state map with drone paths")
        print()
        
        sim.create_multi_drone_visualization("Fire Fighting Mission")
        
        # 8. Final summary
        print("\n" + "=" * 70)
        print("📊 MISSION SUMMARY")
        print("=" * 70)
        
        final_summary = sim.get_simulation_summary()
        
        print(f"\n🚁 Drones:")
        for drone_name, drone_info in final_summary['drones'].items():
            print(f"   {drone_name}:")
            print(f"      Final position: {drone_info['position']}")
            print(f"      Distance traveled: {drone_info.get('distance_traveled', 'N/A')}")
        
        print(f"\n🌍 Environment:")
        env_info = final_summary['environment']
        print(f"   Buildings: {env_info['obstacles']}")
        print(f"   Terrain zones: {env_info['terrain_zones']}")
        print(f"   Wind: {env_info['weather']['wind_velocity']} m/s")
        
        if 'fire' in final_summary:
            fire_stats = final_summary['fire']
            print(f"\n🔥 Fire Statistics:")
            print(f"   Currently burning: {fire_stats['burning_cells']} cells")
            print(f"   Burn percentage: {fire_stats['burn_percentage']:.1f}%")
            print(f"   Remaining fuel: {fire_stats['total_fuel']:.1f} units")
            print(f"   Average intensity: {fire_stats['avg_intensity']:.3f}")
        
        print(f"\n⚠️  Collisions: {final_summary['collisions']}")
        print(f"⏱️  Total simulation time: {final_summary['total_time']:.2f} seconds")
        
        print("\n" + "=" * 70)
        print("✅ SIMULATION COMPLETE!")
        print("=" * 70)
        print("\n📁 Output files generated:")
        print("   📈 output/multi_drone_combined_analysis.png")
        print("      └─ Drone trajectories, forces, velocities, environment map")
        print("   🔥 output/multi_drone_combined_fire_analysis.png")
        print("      └─ Fire spread, fuel consumption, intensity, final state")
        print()
        print("💡 Tip: Open these PNG files to see the complete visualization!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.stop_simulation()


def quick_fire_only_demo():
    """
    Quick demo showing just fire spread without drones.
    Useful for testing fire mechanics.
    """
    print("\n" + "=" * 70)
    print("🔥 QUICK FIRE-ONLY DEMO")
    print("=" * 70)
    
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    try:
        sim.setup_natural_environment()
        sim.enable_fire_simulation(grid_width_m=60, grid_height_m=60, cell_size_m=1.5)
        sim.set_wind([10.0, 3.0, 0.0])  # Strong wind
        
        sim.start_fire((0, 0), intensity=0.5)
        
        print("Running fire simulation for 50 steps...")
        for step in range(50):
            sim.step_simulation({})
            
            if step % 10 == 0:
                summary = sim.get_simulation_summary()
                if 'fire' in summary:
                    stats = summary['fire']
                    print(f"  Step {step}: {stats['burning_cells']} cells burning")
        
        print("✅ Fire-only demo complete!")
        
    finally:
        sim.stop_simulation()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick fire-only demo
        quick_fire_only_demo()
    else:
        # Full fire fighting demo with visualizations
        run_fire_fighting_demo()
    
    print("\n🎉 Demo finished successfully!")
