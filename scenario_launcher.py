#!/usr/bin/env python3
"""
PyBullet Drone Scenario Launcher

Interactive launcher for running individual drone test scenarios.
Choose which movement test you want to run.
"""

import sys
from test_scenarios import run_scenario


def main():
    """Interactive scenario launcher."""
    
    print("🚁 PyBullet Drone Scenario Launcher")
    print("=" * 50)
    
    # Define available scenarios
    scenarios = {
        "1": ("Hover Test", [
            ([0.0, 0.0, 0.0], 100, "Pure hover - no movement"),
            ([0.0, 0.0, 0.0], 50, "Extended hover test"),
        ]),
        
        "2": ("Horizontal Movement Left", [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([-1.0, 0.0, 0.0], 80, "Full left movement"), 
            ([0.0, 0.0, 0.0], 50, "Final hover"),
        ]),
        
        "3": ("Horizontal Movement Right", [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([1.0, 0.0, 0.0], 80, "Full right movement"),
            ([0.0, 0.0, 0.0], 50, "Final hover"),
        ]),
        
        "4": ("Horizontal Movement Forward", [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([0.0, 1.0, 0.0], 80, "Full forward movement"),
            ([0.0, 0.0, 0.0], 50, "Final hover"),
        ]),
        
        "5": ("Horizontal Movement Backward", [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([0.0, -1.0, 0.0], 80, "Full backward movement"),
            ([0.0, 0.0, 0.0], 50, "Final hover"),
        ]),
        
        "6": ("Vertical Movement Up", [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([0.0, 0.0, 1.0], 60, "Full upward movement"),
            ([0.0, 0.0, 0.0], 50, "Final hover"),
        ]),
        
        "7": ("Vertical Movement Down", [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([0.0, 0.0, -0.5], 60, "Controlled downward movement"),
            ([0.0, 0.0, 0.0], 50, "Final hover"),
        ]),
        
        "8": ("Diagonal Movement", [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([0.7, 0.7, 0.0], 80, "Diagonal (NE) movement"),
            ([0.0, 0.0, 0.0], 50, "Final hover"),
        ]),
        
        "9": ("Complex 3D Movement", [
            ([0.0, 0.0, 0.0], 30, "Initial hover"),
            ([0.5, 0.5, 0.5], 60, "3D diagonal up movement"),
            ([0.0, 0.0, 0.0], 30, "Mid hover"),
            ([-0.5, -0.5, -0.3], 60, "3D diagonal down movement"),
            ([0.0, 0.0, 0.0], 40, "Final hover"),
        ]),
    }
    
    # Display menu
    print("Available test scenarios:")
    print("-" * 30)
    for key, (name, _) in scenarios.items():
        print(f"{key}. {name}")
    print("0. Run all scenarios")
    print("q. Quit")
    
    # Get user choice
    choice = input("\nSelect scenario (1-9, 0 for all, q to quit): ").strip()
    
    if choice.lower() == 'q':
        print("Goodbye! 👋")
        return
    
    if choice == '0':
        print("\n🚀 Running ALL scenarios...")
        from test_scenarios import main as run_all_scenarios
        run_all_scenarios()
        return
    
    if choice in scenarios:
        scenario_name, commands = scenarios[choice]
        print(f"\n🚀 Running: {scenario_name}")
        result = run_scenario(scenario_name, commands, visualize=True)
        
        print(f"\n✅ Scenario completed successfully!")
        print(f"📊 Steps: {result['total_steps']}")
        print(f"📍 Final position: [{result['final_position'][0]:.1f}, {result['final_position'][1]:.1f}, {result['final_position'][2]:.1f}]")
        print(f"📏 Distance from start: {result['distance_from_start']:.1f}m")
        
    else:
        print("❌ Invalid choice. Please select 1-9, 0, or q.")


if __name__ == "__main__":
    main()