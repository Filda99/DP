#!/usr/bin/env python3
"""
Demo script showing how to use the FireGrid wildfire simulation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from fire_grid import FireGrid


def basic_simulation_demo():
    """Demonstrate basic FireGrid usage."""
    print("Basic FireGrid Simulation Demo")
    print("-" * 40)
    
    # Create a 20x20 grid
    grid = FireGrid(
        H=20, W=20,          # Grid dimensions
        dt=0.1,              # Time step
        alpha=1.0,           # Distance decay factor
        k_wind=1.5,          # Wind influence
        k_slope=0.5,         # Slope influence  
        wind_dir=np.pi/4     # Wind direction (45 degrees)
    )
    
    print(f"Created {grid.H}x{grid.W} grid")
    print(f"Initial state: {grid.get_stats()}")
    
    # Run simulation for 50 steps
    print("\nRunning simulation...")
    
    step_data = []
    for step in range(50):
        stats = grid.get_stats()
        step_data.append({
            'step': step,
            'burning_cells': stats['burning_cells'],
            'total_fuel': stats['total_fuel'],
            'avg_intensity': stats['avg_intensity']
        })
        
        # Add some suppression efforts randomly
        suppression = {}
        if step > 10 and np.random.random() < 0.3:  # 30% chance of suppression
            # Target random burning cells for suppression
            burning_positions = np.where(grid.B)
            if len(burning_positions[0]) > 0:
                idx = np.random.randint(len(burning_positions[0]))
                i, j = burning_positions[0][idx], burning_positions[1][idx]
                suppression[(i, j)] = [0.4, 0.3]  # Multiple suppression efforts
        
        grid.step(suppression)
        
        if step % 10 == 0:
            print(f"Step {step}: {stats['burning_cells']} burning cells, "
                  f"fuel: {stats['total_fuel']:.1f}")
    
    return step_data


def create_plots(step_data):
    """Create plots showing simulation results."""
    steps = [d['step'] for d in step_data]
    burning = [d['burning_cells'] for d in step_data]
    fuel = [d['total_fuel'] for d in step_data]
    intensity = [d['avg_intensity'] for d in step_data]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Burning cells over time
    ax1.plot(steps, burning, 'r-', linewidth=2, label='Burning Cells')
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Number of Burning Cells')
    ax1.set_title('Fire Spread Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Total fuel over time
    ax2.plot(steps, fuel, 'g-', linewidth=2, label='Total Fuel')
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Total Fuel Remaining')
    ax2.set_title('Fuel Consumption Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Average intensity over time
    ax3.plot(steps, intensity, 'orange', linewidth=2, label='Avg Intensity')
    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('Average Fire Intensity')
    ax3.set_title('Fire Intensity Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('/home/filip/tmp/DP/output/fire_simulation_analysis.png', dpi=150)
    plt.close()
    
    print("✓ Analysis plots saved to output/fire_simulation_analysis.png")


def advanced_scenario_demo():
    """Demonstrate advanced scenarios with the FireGrid."""
    print("\nAdvanced Scenario Demo")
    print("-" * 40)
    
    # Create a larger grid with varying base lambda values
    H, W = 25, 25
    l_base = np.linspace(0.5, 2.0, H)  # Varying fire propensity by row
    
    grid = FireGrid(
        H=H, W=W,
        dt=0.05,           # Smaller time step for more detailed simulation
        alpha=0.8,         # Moderate distance decay
        k_wind=2.0,        # Strong wind effect
        k_slope=1.0,       # Moderate slope effect
        wind_dir=0.0,      # North wind
        l_base=l_base      # Custom base lambda
    )
    
    print(f"Created {H}x{W} grid with varying fire propensity")
    print(f"Base lambda range: {l_base.min():.2f} to {l_base.max():.2f}")
    
    # Set up a specific fire scenario
    grid.B.fill(False)
    grid.F = np.random.uniform(0.5, 1.0, (H, W))  # Random fuel distribution
    grid.I.fill(0.0)
    
    # Start multiple fires
    fire_starts = [(5, 5), (5, 20), (20, 12)]
    for i, j in fire_starts:
        grid.B[i, j] = True
        grid.I[i, j] = 0.2
    
    print(f"Started fires at: {fire_starts}")
    
    # Define suppression strategy - protect high-value areas
    protected_zones = [
        (slice(10, 15), slice(10, 15)),  # Center area
        (slice(18, 23), slice(18, 23))   # Bottom-right corner
    ]
    
    # Run advanced simulation
    for step in range(100):
        # Dynamic suppression based on protected zones
        suppression = {}
        for zone_i, zone_j in protected_zones:
            zone_burning = grid.B[zone_i, zone_j]
            positions = np.where(zone_burning)
            for idx in range(len(positions[0])):
                abs_i = zone_i.start + positions[0][idx]
                abs_j = zone_j.start + positions[1][idx]
                suppression[(abs_i, abs_j)] = [0.6, 0.4]  # Strong suppression in protected zones
        
        grid.step(suppression)
        
        if step % 20 == 0:
            stats = grid.get_stats()
            print(f"Step {step}: {stats['burning_cells']} burning, "
                  f"fuel: {stats['total_fuel']:.1f}")
    
    final_stats = grid.get_stats()
    print(f"\nFinal state: {final_stats}")


def demonstrate_mathematical_formulas():
    """Show the mathematical formulas implemented in FireGrid."""
    print("\nMathematical Formulas Used in FireGrid")
    print("=" * 50)
    
    print("1. Fire Spread Rate:")
    print("   λ_xy = l_base[y] × exp(-α×d) × wind_gain × slope_gain")
    print("   where:")
    print("   - l_base[y]: base fire rate for row y")
    print("   - α: distance decay factor")
    print("   - d: Euclidean distance between cells")
    print("   - wind_gain: 1 + k_wind × cos(wind_direction - spread_direction)")
    print("   - slope_gain: simplified slope factor")
    
    print("\n2. Ignition Probability:")
    print("   P_xy = 1 - exp(-λ_xy × dt)")
    print("   r1(x) = 1 - ∏(1 - P_xy × B[y]) for all burning neighbors y")
    
    print("\n3. Suppression Probability:")
    print("   r2(x) = 1 - ∏(1 - Q_i(x)) for all suppression efforts i")
    
    print("\n4. Update Order per Step:")
    print("   1. Ignition: new fires start based on r1(x)")
    print("   2. Suppression: fires extinguished based on r2(x)")
    print("   3. Fuel decrease: F[x] -= I[x] × dt")
    print("   4. Burn-out: stop burning if F[x] ≤ 0")


def main():
    """Run the complete demo."""
    print("FireGrid Wildfire Simulation - Usage Demo")
    print("=" * 50)
    
    # Show mathematical formulas
    demonstrate_mathematical_formulas()
    
    # Run basic simulation
    step_data = basic_simulation_demo()
    
    # Create analysis plots
    create_plots(step_data)
    
    # Run advanced scenario
    advanced_scenario_demo()
    
    print("\n" + "=" * 50)
    print("Demo completed! Check the output/ directory for visualizations.")
    print("\nKey features demonstrated:")
    print("- Fire spread based on probabilistic rules")
    print("- Wind and environmental effects")
    print("- Suppression strategies") 
    print("- Fuel consumption and burn-out")
    print("- State monitoring and analysis")


if __name__ == "__main__":
    main()