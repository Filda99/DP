#!/usr/bin/env python3
"""
Strong Wind Hover Test

Test quadcopter hovering capability in extreme wind conditions.
Demonstrates when wind forces exceed drone's compensation ability.
"""

from src.simulation import Simulation
import numpy as np

def strong_wind_hover_scenario(step, time, drones):
    """Hover scenario with progressively stronger wind."""
    controls = {}
    
    # Try to hover in place despite strong wind
    for drone_name in drones:
        controls[drone_name] = [0.0, 0.0, 0.0]  # Pure hover attempt
    
    return controls

def main():
    """Test hover in extreme wind conditions."""
    print("🌪️ Strong Wind Hover Test")
    print("=" * 50)
    
    # Initialize simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    # Create simple environment (no obstacles to focus on wind effects)
    print("🏗️ Setting up open environment...")
    
    # Add single quadcopter
    sim.add_quadcopter("test_quad", position=[0, 0, 5], mass=0.5)
    
    # Test different wind strengths
    wind_tests = [
        {"wind": [5.0, 0.0, 0.0], "name": "Light Wind (5 m/s)", "steps": 1000},
        {"wind": [15.0, 0.0, 0.0], "name": "Strong Wind (15 m/s)", "steps": 1000}, 
        {"wind": [25.0, 0.0, 0.0], "name": "Hurricane Wind (25 m/s)", "steps": 1000},
        {"wind": [35.0, 0.0, 0.0], "name": "Extreme Wind (35 m/s)", "steps": 1000}
    ]
    
    for test in wind_tests:
        print(f"\n🌬️ Testing: {test['name']}")
        
        # Reset drone position
        sim.drones["test_quad"].position = np.array([0, 0, 5])
        
        # Set wind conditions
        sim.set_wind(test["wind"], turbulence=2.0)  # High turbulence
        
        # Get initial position
        initial_pos = sim.drones["test_quad"].get_position()
        print(f"  Initial position: [{initial_pos[0]:.1f}, {initial_pos[1]:.1f}, {initial_pos[2]:.1f}]")
        
        # Run hover test
        sim.run_scenario(strong_wind_hover_scenario, test["steps"])
        
        # Check final position 
        final_pos = sim.drones["test_quad"].get_position()
        print(f"  Final position: [{final_pos[0]:.1f}, {final_pos[1]:.1f}, {final_pos[2]:.1f}]")
        
        # Calculate drift
        drift = np.linalg.norm(final_pos - initial_pos)
        print(f"  Total drift: {drift:.1f} meters")
        
        # Analyze result
        if drift < 2.0:
            print(f"  ✅ HOVER SUCCESSFUL - Wind compensated")
        elif drift < 10.0:
            print(f"  ⚠️ PARTIAL HOVER - Some drift but controlled")
        else:
            print(f"  ❌ HOVER FAILED - Wind too strong, drone blown away")
        
        # Create visualization for this test
        title = f"Hover Test - {test['name']} - Drift: {drift:.1f}m"
        # Clean title for filename (remove problematic characters)
        clean_title = title.replace("/", "_per_").replace(":", "_").replace(" ", "_")
        sim.create_visualization("test_quad", clean_title)
        
        print(f"  📊 Saved: output/{clean_title.lower()}_test_quad_analysis.png")
    
    # Stop simulation
    sim.stop_simulation()
    
    print(f"\n🎉 Wind hover tests completed!")
    print("📊 Check generated PNG files to see drone behavior in different wind conditions")

if __name__ == "__main__":
    main()