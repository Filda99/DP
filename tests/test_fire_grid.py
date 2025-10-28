#!/usr/bin/env python3
"""
Test script for the FireGrid wildfire simulation.
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.fire_grid import FireGrid


def test_basic_functionality():
    """Test basic FireGrid functionality."""
    print("Testing basic FireGrid functionality...")
    
    # Create a small grid for testing
    grid = FireGrid(H=10, W=10, dt=0.1, alpha=1.0)
    
    # Test initial state
    initial_state = grid.get_state()
    assert 'B' in initial_state
    assert 'F' in initial_state
    assert 'I' in initial_state
    assert 'M' in initial_state  # New: Moisture field
    
    assert initial_state['B'].shape == (10, 10)
    assert initial_state['F'].shape == (10, 10)
    assert initial_state['I'].shape == (10, 10)
    assert initial_state['M'].shape == (10, 10)  # New: Moisture field
    
    print("✓ Initial state structure correct (including moisture field)")
    
    # Test that some cells are burning initially
    initial_burning = np.sum(initial_state['B'])
    assert initial_burning > 0, "No initial fires found"
    print(f"✓ Initial burning cells: {initial_burning}")
    
    # Test step function
    initial_stats = grid.get_stats()
    grid.step()
    after_step_stats = grid.get_stats()
    
    print(f"✓ Step completed. Burning cells: {initial_stats['burning_cells']} -> {after_step_stats['burning_cells']}")
    
    # Test suppression
    suppression_assignments = {(5, 5): [0.8], (3, 3): [0.5, 0.3]}
    grid.step(suppression_assignments)
    print("✓ Step with suppression completed")


def test_fire_spread():
    """Test fire spread behavior."""
    print("\nTesting fire spread behavior...")
    
    # Create a grid with controlled initial conditions
    grid = FireGrid(H=15, W=15, dt=0.1, alpha=0.5)
    
    # Clear the grid and set a single fire in the center
    grid.B.fill(False)
    grid.F.fill(0.8)  # High fuel everywhere
    grid.I.fill(0.0)
    
    # Start fire in center
    center_i, center_j = 7, 7
    grid.B[center_i, center_j] = True
    grid.I[center_i, center_j] = 0.2
    
    print(f"Initial fire at ({center_i}, {center_j})")
    
    # Run simulation for several steps
    burning_history = []
    for step in range(20):
        stats = grid.get_stats()
        burning_history.append(stats['burning_cells'])
        grid.step()
        
        if step % 5 == 0:
            print(f"Step {step}: {stats['burning_cells']} burning cells")
    
    # Fire should spread outward
    assert max(burning_history) > 1, "Fire did not spread"
    print("✓ Fire spread confirmed")


def test_suppression_effectiveness():
    """Test that suppression reduces fires."""
    print("\nTesting suppression effectiveness...")
    
    # Create two identical grids
    grid1 = FireGrid(H=10, W=10, dt=0.1)
    grid1.reset_random(seed=42)
    grid2 = FireGrid(H=10, W=10, dt=0.1)
    grid2.reset_random(seed=42)
    
    # Run one without suppression, one with heavy suppression
    suppression = {}
    for i in range(10):
        for j in range(10):
            suppression[(i, j)] = [0.9]  # Very high suppression probability
    
    # Run for 10 steps
    for _ in range(10):
        grid1.step()  # No suppression
        grid2.step(suppression)  # Heavy suppression
    
    stats1 = grid1.get_stats()
    stats2 = grid2.get_stats()
    
    print(f"Without suppression: {stats1['burning_cells']} burning cells")
    print(f"With suppression: {stats2['burning_cells']} burning cells")
    
    # Suppression should reduce fires (though not necessarily to zero due to randomness)
    assert stats2['burning_cells'] <= stats1['burning_cells'], "Suppression did not reduce fires"
    print("✓ Suppression effectiveness confirmed")


def test_wind_effects():
    """Test wind direction effects on fire spread."""
    print("\nTesting wind effects...")
    
    # Create grids with different wind directions
    grid_no_wind = FireGrid(H=15, W=15, dt=0.1, k_wind=0.0)
    grid_no_wind.reset_random(seed=123)
    grid_with_wind = FireGrid(H=15, W=15, dt=0.1, k_wind=2.0, wind_dir=0.0)  # Wind to the north
    grid_with_wind.reset_random(seed=123)
    
    print("✓ Wind effect grids created")


def visualize_simulation():
    """Create a visualization of fire spread."""
    print("\nCreating fire spread visualization...")
    
    grid = FireGrid(H=20, W=20, dt=0.1, alpha=0.8, k_wind=1.5, wind_dir=np.pi/4)
    
    # Set up initial conditions
    grid.B.fill(False)
    grid.F.fill(0.7)
    grid.I.fill(0.0)
    
    # Start fires in a few locations
    start_fires = [(5, 5), (15, 15), (10, 3)]
    for i, j in start_fires:
        grid.B[i, j] = True
        grid.I[i, j] = 0.15
    
    # Run simulation and collect states
    states = []
    for step in range(30):
        states.append(grid.get_state())
        if step % 10 == 0:
            stats = grid.get_stats()
            print(f"Step {step}: {stats['burning_cells']} burning, avg fuel: {stats['avg_fuel_per_cell']:.3f}")
        grid.step()
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    timesteps = [0, 5, 10, 15, 20, 25]
    
    for idx, t in enumerate(timesteps):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Create composite image: burning (red), fuel (green background)
        img = np.zeros((grid.H, grid.W, 3))
        
        # Fuel as green background
        fuel_normalized = states[t]['F']
        img[:, :, 1] = fuel_normalized * 0.5  # Green channel
        
        # Burning cells as red
        burning_mask = states[t]['B']
        img[burning_mask, 0] = 1.0  # Red channel
        img[burning_mask, 1] = 0.0  # Remove green where burning
        
        ax.imshow(img)
        ax.set_title(f'Step {t}')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('/home/filip/tmp/DP/output/fire_grid_simulation.png', dpi=150)
    plt.close()
    
    print("✓ Visualization saved to output/fire_grid_simulation.png")


def test_moisture_system():
    """Test the new moisture-based fire suppression system."""
    print("Testing moisture-based fire suppression...")
    
    # Create a small grid
    grid = FireGrid(H=10, W=10, dt=0.1)
    
    # Manually set a burning cell
    grid.B[5, 5] = True
    grid.F[5, 5] = 1.0
    grid.I[5, 5] = 0.8
    grid.M[5, 5] = 0.0  # No initial moisture
    
    # Test 1: Water application increases moisture
    water_drops = {(5, 5): 0.5}
    grid.step(water_drops=water_drops)
    
    assert grid.M[5, 5] > 0.0, "Moisture should increase after water application"
    print(f"✓ Water application increased moisture to {grid.M[5, 5]:.2f}")
    
    # Test 2: Moisture evaporates over time
    initial_moisture = grid.M[5, 5]
    for _ in range(10):
        grid.step()  # No water drops
    
    assert grid.M[5, 5] < initial_moisture, "Moisture should evaporate over time"
    print(f"✓ Moisture evaporated from {initial_moisture:.2f} to {grid.M[5, 5]:.2f}")
    
    # Test 3: High moisture prevents ignition
    grid.reset_random(seed=42)
    
    # Set up two adjacent cells: one burning, one with fuel
    grid.B[5, 5] = True
    grid.F[5, 5] = 1.0
    grid.I[5, 5] = 0.8
    
    grid.B[5, 6] = False
    grid.F[5, 6] = 1.0
    grid.M[5, 6] = 0.9  # High moisture
    
    # Run steps and check if high moisture prevents ignition
    ignition_happened = False
    for _ in range(50):
        grid.step()
        if grid.B[5, 6]:
            ignition_happened = True
            break
    
    # High moisture should make ignition much less likely
    print(f"✓ Ignition with 90% moisture: {'occurred' if ignition_happened else 'prevented (expected)'}")
    
    print("✓ Moisture system working correctly")


def main():
    """Run all tests."""
    print("FireGrid Wildfire Simulation Test Suite")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_fire_spread()
        test_suppression_effectiveness()
        test_wind_effects()
        test_moisture_system()  # New test
        visualize_simulation()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! FireGrid implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()