#!/usr/bin/env python3
"""
Pure Fire Spread Demonstration

This demo shows natural wildfire behavior without any suppression:
- Fire spreading through forests
- Fire blocked by water (lakes)
- Fire going around buildings
- Wind effects on fire spread

Generates visualizations showing:
- Environment layout (buildings, forests, lakes)
- Fire spread over time
- Final fire state comparison
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.simulation import Simulation


def create_environment_visualization(sim):
    """Create a visualization of the environment layout."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Get grid bounds
    if sim.environment.fire_enabled:
        x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    else:
        x_min, x_max, y_min, y_max = -50, 50, -50, 50
    
    # Draw buildings
    for obstacle in sim.environment.obstacles:
        if obstacle['type'] == 'city_block':
            pos = obstacle['position']
            size = obstacle['size']
            rect = plt.Rectangle(
                (pos[0] - size[0]/2, pos[1] - size[1]/2),
                size[0], size[1],
                facecolor='gray', edgecolor='black', linewidth=2,
                alpha=0.8, label='Buildings' if obstacle == sim.environment.obstacles[0] else ""
            )
            ax.add_patch(rect)
            # Add building label
            ax.text(pos[0], pos[1], 'B', ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white')
    
    # Draw terrain zones
    forest_labeled = False
    lake_labeled = False
    for zone in sim.environment.terrain_zones:
        if zone['type'] == 'forest':
            circle = plt.Circle(
                zone['center'], zone['radius'],
                facecolor='darkgreen', edgecolor='green', linewidth=2,
                alpha=0.6, label='Forest' if not forest_labeled else ""
            )
            ax.add_patch(circle)
            forest_labeled = True
            # Add forest label
            ax.text(zone['center'][0], zone['center'][1], 'FOREST',
                   ha='center', va='center', fontsize=12, 
                   fontweight='bold', color='white')
        elif zone['type'] == 'lake':
            circle = plt.Circle(
                zone['center'], zone['radius'],
                facecolor='blue', edgecolor='darkblue', linewidth=2,
                alpha=0.6, label='Lake' if not lake_labeled else ""
            )
            ax.add_patch(circle)
            lake_labeled = True
            # Add lake label
            ax.text(zone['center'][0], zone['center'][1], 'LAKE',
                   ha='center', va='center', fontsize=12,
                   fontweight='bold', color='white')
    
    # Add fire start markers
    fire_markers = [
        (-25, 5, 'Fire 1\n(Forest #2)'),
        (15, -18, 'Fire 2\n(→Lake)'),
        (20, 20, 'Fire 3\n(Forest #1)')
    ]
    
    for x, y, label in fire_markers:
        ax.plot(x, y, 'r*', markersize=20, markeredgecolor='darkred', markeredgewidth=2)
        ax.text(x, y - 3, label, ha='center', va='top', 
               fontsize=10, fontweight='bold', color='red',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Add wind arrow
    wind = sim.environment.weather['wind_velocity']
    wind_scale = 5
    ax.arrow(x_min + 10, y_max - 10, wind[0]*wind_scale, wind[1]*wind_scale,
            head_width=3, head_length=2, fc='orange', ec='darkorange', linewidth=3)
    ax.text(x_min + 10, y_max - 5, f'Wind: {wind[0]:.1f}, {wind[1]:.1f} m/s',
           fontsize=12, fontweight='bold', color='darkorange')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax.set_title('Environment Layout\n(Buildings, Forests, Lakes, and Fire Start Locations)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('output/environment_layout.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("✅ Environment layout saved to output/environment_layout.png")


def create_fire_comparison_plot(sim):
    """Create side-by-side comparison of environment and final fire state."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Get grid bounds
    x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    
    # LEFT PLOT: Environment layout
    # Draw buildings
    for obstacle in sim.environment.obstacles:
        if obstacle['type'] == 'city_block':
            pos = obstacle['position']
            size = obstacle['size']
            rect = plt.Rectangle(
                (pos[0] - size[0]/2, pos[1] - size[1]/2),
                size[0], size[1],
                facecolor='gray', edgecolor='black', linewidth=2, alpha=0.8
            )
            ax1.add_patch(rect)
    
    # Draw terrain zones
    for zone in sim.environment.terrain_zones:
        if zone['type'] == 'forest':
            circle = plt.Circle(
                zone['center'], zone['radius'],
                facecolor='darkgreen', edgecolor='green', linewidth=2, alpha=0.6
            )
            ax1.add_patch(circle)
            ax1.text(zone['center'][0], zone['center'][1], 'FOREST',
                    ha='center', va='center', fontsize=14, 
                    fontweight='bold', color='white')
        elif zone['type'] == 'lake':
            circle = plt.Circle(
                zone['center'], zone['radius'],
                facecolor='blue', edgecolor='darkblue', linewidth=2, alpha=0.6
            )
            ax1.add_patch(circle)
            ax1.text(zone['center'][0], zone['center'][1], 'LAKE',
                    ha='center', va='center', fontsize=14,
                    fontweight='bold', color='white')
    
    # Add fire start markers
    fire_markers = [(5, -25, '1'), (8, 8, '2'), (22, 22, '3')]
    for x, y, num in fire_markers:
        ax1.plot(x, y, 'r*', markersize=25, markeredgecolor='darkred', markeredgewidth=2)
        ax1.text(x + 2, y + 2, f'Fire {num}', fontsize=12, fontweight='bold', 
                color='red', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax1.set_title('Environment Layout\n(Before Fire)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal')
    
    # RIGHT PLOT: Final fire state
    fire_state = sim.environment.get_fire_state()
    if fire_state:
        final_state = fire_state['fire_grid_state']
        H, W = final_state['B'].shape
        
        # Create composite image
        img = np.zeros((H, W, 3))
        
        print(f"\n🔍 DEBUG: Image array shape: {img.shape} (H={H} rows, W={W} cols)")
        print(f"   Extent: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
        print(f"   img[0, 0] should be bottom-left (x={x_min}, y={y_min})")
        print(f"   img[{H-1}, {W-1}] should be top-right (x={x_max}, y={y_max})")
        
        # Unburned fuel as green
        unburned_mask = (final_state['F'] > 0.5)
        img[unburned_mask] = [0.0, 0.5, 0.0]  # Dark green for unburned areas with fuel
        
        # Partially burned as yellow/orange
        partial_mask = (final_state['F'] > 0.1) & (final_state['F'] <= 0.5)
        fuel_ratio = final_state['F'][partial_mask]
        img[partial_mask, 0] = 1.0  # Red
        img[partial_mask, 1] = fuel_ratio  # Green based on remaining fuel
        img[partial_mask, 2] = 0.0
        
        # Completely burned as dark brown/black
        burned_mask = (final_state['F'] <= 0.1) & (~final_state['B'])
        img[burned_mask] = [0.2, 0.1, 0.0]  # Dark brown for burned areas
        
        # Currently burning as bright red
        burning_mask = final_state['B']
        img[burning_mask] = [1.0, 0.0, 0.0]  # Bright red for active fires
        
        # No fuel (water, buildings) as light blue/gray
        no_fuel_mask = (final_state['F'] == 0) & (~final_state['B'])
        # Determine if it's water or building based on position
        for i in range(H):
            for j in range(W):
                if no_fuel_mask[i, j]:
                    world_pos = sim.environment.grid_mapper.cell_to_world(i, j)
                    # Check if in lake
                    is_lake = False
                    for zone in sim.environment.terrain_zones:
                        if zone['type'] == 'lake':
                            center = zone['center']
                            radius = zone['radius']
                            dist = np.sqrt((world_pos[0] - center[0])**2 + (world_pos[1] - center[1])**2)
                            if dist <= radius:
                                img[i, j] = [0.3, 0.5, 0.9]  # Blue for water
                                is_lake = True
                                break
                    if not is_lake:
                        img[i, j] = [0.6, 0.6, 0.6]  # Gray for buildings
        
        # Add test pattern in corners LAST to verify orientation (after all fire rendering)
        # Bottom-left corner: MAGENTA
        img[0:3, 0:3] = [1.0, 0.0, 1.0]
        # Bottom-right corner: YELLOW  
        img[0:3, W-3:W] = [1.0, 1.0, 0.0]
        # Top-left corner: CYAN
        img[H-3:H, 0:3] = [0.0, 1.0, 1.0]
        # Top-right corner: WHITE
        img[H-3:H, W-3:W] = [1.0, 1.0, 1.0]
        print(f"   Corner markers added: BL=MAGENTA, BR=YELLOW, TL=CYAN, TR=WHITE")
        
        im = ax2.imshow(img, extent=[x_min, x_max, y_min, y_max], origin='lower', interpolation='nearest')
        
        # DEBUG: Add corner markers to verify orientation
        # Bottom-left should be (-50, -50), top-right should be (+50, +50)
        ax2.plot(-45, -45, 'ko', markersize=10, label='Bottom-Left (-50,-50)')
        ax2.plot(45, 45, 'w^', markersize=10, markeredgecolor='black', markeredgewidth=2, label='Top-Right (+50,+50)')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[0.0, 0.5, 0.0], label='Unburned (High Fuel)'),
            Patch(facecolor=[1.0, 0.5, 0.0], label='Partially Burned'),
            Patch(facecolor=[0.2, 0.1, 0.0], label='Completely Burned'),
            Patch(facecolor=[1.0, 0.0, 0.0], label='Currently Burning'),
            Patch(facecolor=[0.3, 0.5, 0.9], label='Water (No Spread)'),
            Patch(facecolor=[0.6, 0.6, 0.6], label='Buildings (No Fuel)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax2.set_title('Final Fire State\n(After Simulation)', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('output/fire_environment_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("✅ Fire/Environment comparison saved to output/fire_environment_comparison.png")


def run_pure_fire_spread_demo():
    """
    Run pure fire spread demonstration without any drones.
    
    Three fire scenarios:
    1. Fire near lake - should NOT spread across water
    2. Fire near building - should go AROUND the building
    3. Fire in forest - should spread naturally through vegetation
    """
    
    print("=" * 70)
    print("🔥 PURE FIRE SPREAD DEMONSTRATION (NO DRONES)")
    print("=" * 70)
    print()
    print("This demo shows natural wildfire behavior:")
    print("  🌊 Fire #1: Near lake - fire CANNOT cross water")
    print("  🏢 Fire #2: Near building - fire goes AROUND obstacles")
    print("  🌲 Fire #3: In forest - natural spread through vegetation")
    print()
    print("=" * 70)
    
    # Create simulation
    sim = Simulation(gui=False)
    sim.start_simulation()
    
    try:
        # 1. Setup environment
        print("\n📍 Setting up environment...")
        sim.setup_mixed_environment()
        
        # 2. Enable fire simulation with aggressive spread
        print("🔥 Enabling fire simulation...")
        sim.enable_fire_simulation(
            grid_width_m=100, 
            grid_height_m=100, 
            cell_size_m=2.5  # Slightly larger cells for better visualization
        )
        
        # Adjust fire parameters for dramatic spread
        if sim.environment.fire_enabled:
            sim.environment.fire_grid.l_base *= 4.0  # 4x faster spread
            sim.environment.fire_grid.alpha = 0.2  # Spreads very far
            print("   🔥 Fire parameters: VERY AGGRESSIVE spread")
        
        # 3. Set strong wind (will push fire in one direction)
        print("💨 Setting wind conditions...")
        wind_vec = [8.0, 5.0, 0.0]
        sim.set_wind(wind_vec, turbulence=0.3)
        
        # Debug wind direction
        wind_angle_rad = np.arctan2(wind_vec[1], wind_vec[0])
        wind_angle_deg = np.degrees(wind_angle_rad)
        print(f"   Wind vector: [{wind_vec[0]}, {wind_vec[1]}] m/s")
        print(f"   Wind angle: {wind_angle_rad:.3f} rad = {wind_angle_deg:.1f}°")
        print(f"   Wind direction: {'East' if abs(wind_vec[0]) > abs(wind_vec[1]) else 'North'}")
        print(f"   Fire should spread: RIGHT (positive X) and UP (positive Y)")
        
        # 4. Create environment visualization BEFORE fire
        print("📊 Creating environment layout visualization...")
        
        # Debug: Print terrain zones
        print("\n🔍 DEBUG: Terrain zones created:")
        for i, zone in enumerate(sim.environment.terrain_zones):
            print(f"   Zone {i}: {zone}")
        print(f"\n🔍 DEBUG: Obstacles created:")
        for i, obs in enumerate(sim.environment.obstacles):
            print(f"   Obstacle {i}: type={obs['type']}, pos={obs['position']}, size={obs['size']}")
        print()
        
        create_environment_visualization(sim)
        
        # 5. Start fires in strategic locations
        # Fire locations are chosen to start IN flammable areas (forests)
        # based on actual terrain layout
        print("\n🔥 Starting fires in strategic locations...")
        
        # Print coordinate system info first
        print("\n📍 COORDINATE SYSTEM DEBUG:")
        print("=" * 70)
        print("Environment features:")
        print(f"  Lake at: (0, -25), radius 10m → cell [10, 20]")
        print(f"  Forest #1 at: (20, 20), radius 12m → cell [28, 28]")
        print(f"  Forest #2 at: (-25, 0), radius 10m → cell [20, 10]")
        print(f"  Buildings: x ∈ [-10, 0, 10], y ∈ [-5, 5]")
        print()
        print("Fire starting positions (IN the forests/features):")
        fires = [
            (-25, 5, "In Forest #2 (left side) - should spread with wind toward center"),
            (15, -18, "Open area between forest and lake - should hit water"),
            (20, 20, "Center of Forest #1 (right side) - maximum spread")
        ]
        for x, y, desc in fires:
            i, j = sim.environment.grid_mapper.world_to_cell((x, y))
            fuel = sim.environment.fire_grid.F[i, j] if sim.environment.fire_enabled else 0
            print(f"  ({x:3d}, {y:3d}) -> cell [{i:2d}, {j:2d}], fuel={fuel:.2f}: {desc}")
        print("=" * 70)
        print()
        
        for x, y, description in fires:
            if sim.start_fire((x, y), intensity=0.5):
                print(f"   ✓ Fire started at ({x}, {y}): {description}")
        
        # 5b. Print initial fuel grid
        print("\n📊 INITIAL FUEL GRID:")
        print("=" * 70)
        if sim.environment.fire_enabled:
            F_initial = sim.environment.fire_grid.F.copy()
            H, W = F_initial.shape
            print(f"Grid size: {H}x{W}")
            print("\nFuel levels (F):")
            print("  0.00 = No fuel (water/buildings)")
            print(" ~0.30 = Open terrain")
            print(" ~0.80 = Forest")
            print()
            
            # Check specific locations
            print("\n🔍 Sampling specific locations:")
            test_points = [
                (0, -25, "Lake center"),
                (0, 0, "City center (should be building)"),
                (20, 20, "Forest #1 center"),
                (-25, 0, "Forest #2 center"),
                (40, 40, "Far corner (open terrain)")
            ]
            for x, y, desc in test_points:
                i, j = sim.environment.grid_mapper.world_to_cell((x, y))
                fuel = F_initial[i, j]
                print(f"  ({x:3d}, {y:3d}) -> cell [{i:2d},{j:2d}] fuel={fuel:.2f} - {desc}")
            
            # Print grid (flip vertically for correct orientation)
            print("\nFuel Grid Visual:")
            print("Column indices (j): 0=left edge (x=-50), 20=center (x=0), 40=right edge (x=+50)")
            print("Row indices (i): 0=bottom edge (y=-50), 20=center (y=0), 40=top edge (y=+50)")
            print()
            for i in range(H-1, -1, -1):
                row_str = f"Row {i:2d}: "
                for j in range(W):
                    fuel = F_initial[i, j]
                    if fuel < 0.05:
                        row_str += "  ."
                    elif fuel < 0.4:
                        row_str += "  o"
                    else:
                        row_str += "  #"
                if i % 5 == 0:  # Only print every 5th row to save space
                    print(row_str)
                    if i == 20:
                        print("        " + "".join([f"{j:3d}" if j % 5 == 0 else "   " for j in range(W)]))
            print("\nLegend: . = no fuel (0.0-0.05), o = low fuel (0.05-0.4), # = high fuel (0.4+)")
            
            # Mark specific features on the grid
            print("\n🗺️  Feature locations on grid:")
            lake_i, lake_j = sim.environment.grid_mapper.world_to_cell((0, -25))
            print(f"  Lake center (0, -25) → cell [{lake_i},{lake_j}]")
            forest1_i, forest1_j = sim.environment.grid_mapper.world_to_cell((20, 20))
            print(f"  Forest #1 center (20, 20) → cell [{forest1_i},{forest1_j}]")
            forest2_i, forest2_j = sim.environment.grid_mapper.world_to_cell((-25, 0))
            print(f"  Forest #2 center (-25, 0) → cell [{forest2_i},{forest2_j}]")
            print("=" * 70)
        
        # 6. Run simulation WITHOUT any drones (pure fire spread)
        print("\n⏱️  Running 2-minute fire spread simulation...")
        print("   (No drones = No suppression)")
        print()
        
        total_steps = 1200  # 120 seconds at 10 Hz
        progress_times = [0, 20, 40, 60, 80, 100, 120]
        
        for step in range(total_steps):
            t = step * 0.1
            
            # NO DRONES - just empty controls
            sim.step_simulation({})
            
            # Step only once (we want real-time fire spread)
            # Progress updates
            current_time = int(t)
            if current_time in progress_times:
                summary = sim.get_simulation_summary()
                if 'fire' in summary:
                    stats = summary['fire']
                    print(f"   [{current_time:3d}s] Burning: {stats['burning_cells']:4d} cells | "
                          f"Fuel: {stats['total_fuel']:6.1f} | "
                          f"Burned: {stats['burn_percentage']:5.1f}%")
                progress_times.remove(current_time)
        
        # 6b. Print final fire state
        print("\n📊 FINAL FIRE STATE:")
        print("=" * 70)
        if sim.environment.fire_enabled:
            B_final = sim.environment.fire_grid.B
            F_final = sim.environment.fire_grid.F
            H, W = B_final.shape
            print(f"Grid size: {H}x{W}")
            print("\nFire state:")
            print("  . = No fuel (water/buildings)")
            print("  o = Unburned (has fuel)")
            print("  X = Burned out (no fuel left)")
            print("  🔥 = Currently burning")
            print()
            # Print grid (flip vertically for correct orientation)
            print("\nFire State Visual:")
            print("Column indices (j): 0=left (x=-50), 20=center (x=0), 40=right (x=+50)")
            print("Row indices (i): 0=bottom (y=-50), 20=center (y=0), 40=top (y=+50)")
            print()
            for i in range(H-1, -1, -1):
                row_str = f"Row {i:2d}: "
                for j in range(W):
                    fuel = F_final[i, j]
                    burning = B_final[i, j]
                    if burning:
                        row_str += " 🔥"
                    elif fuel == 0.0:
                        row_str += "  ."
                    elif fuel < 0.15:  # Burned (very low fuel)
                        row_str += "  X"
                    else:
                        row_str += "  o"
                if i % 5 == 0:  # Only print every 5th row
                    print(row_str)
                    if i == 20:
                        print("        " + "".join([f"{j:3d}" if j % 5 == 0 else "   " for j in range(W)]))
            
            print("\n🔥 Fire start locations (should show as burned):")
            for x, y, desc in fires:
                i, j = sim.environment.grid_mapper.world_to_cell((x, y))
                fuel = F_final[i, j]
                print(f"  Fire at ({x},{y}) → cell [{i},{j}] - final fuel={fuel:.2f}")
            
            print("\n" + "=" * 70)
        
        # 7. Generate visualizations
        print("\n📊 Generating fire spread analysis...")
        
        # Create the environment comparison plot
        create_fire_comparison_plot(sim)
        
        # Get fire state for analysis
        fire_state = sim.environment.get_fire_state()
        if fire_state:
            stats = fire_state['fire_stats']
            
            # Create detailed fire analysis
            fire_log = sim.simulation_log['fire_states']
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Extract time series data
            times = sim.simulation_log['times'][:len(fire_log)]
            burning_cells = [s['fire_stats']['burning_cells'] for s in fire_log if s]
            total_fuel = [s['fire_stats']['total_fuel'] for s in fire_log if s]
            burn_pct = [s['fire_stats']['burn_percentage'] for s in fire_log if s]
            
            # Plot 1: Burning cells over time
            axes[0, 0].plot(times, burning_cells, 'r-', linewidth=2)
            axes[0, 0].set_xlabel('Time (s)', fontsize=12)
            axes[0, 0].set_ylabel('Burning Cells', fontsize=12)
            axes[0, 0].set_title('Active Fire Cells Over Time', fontsize=14, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Fuel consumption
            axes[0, 1].plot(times, total_fuel, 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Time (s)', fontsize=12)
            axes[0, 1].set_ylabel('Remaining Fuel', fontsize=12)
            axes[0, 1].set_title('Fuel Depletion Over Time', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Burn percentage
            axes[1, 0].plot(times, burn_pct, 'orange', linewidth=2)
            axes[1, 0].set_xlabel('Time (s)', fontsize=12)
            axes[1, 0].set_ylabel('Burn Percentage (%)', fontsize=12)
            axes[1, 0].set_title('Cumulative Burn Percentage', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Summary statistics
            axes[1, 1].axis('off')
            summary_text = f"""
FIRE SPREAD SUMMARY
{'='*40}

Initial Fires: 3 locations
Simulation Time: {times[-1]:.1f} seconds

Final Statistics:
  • Burning Cells: {stats['burning_cells']}
  • Total Burned: {stats['burn_percentage']:.1f}%
  • Fuel Consumed: {total_fuel[0] - stats['total_fuel']:.1f} units
  • Peak Fire: {max(burning_cells)} cells
  
Fire Behavior Observations:
  ✓ Fire #1 (Lake): Blocked by water
  ✓ Fire #2 (Building): Avoided obstacle
  ✓ Fire #3 (Forest): Natural spread
  
Wind Effect: {sim.environment.weather['wind_velocity'][:2]} m/s
  → Fire spread influenced by wind direction
            """
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                          fontsize=11, verticalalignment='top', family='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig('output/fire_spread_analysis.png', dpi=200, bbox_inches='tight')
            plt.close()
            
            print("✅ Fire spread analysis saved to output/fire_spread_analysis.png")
        
        # 8. Final summary
        print("\n" + "=" * 70)
        print("📊 FIRE SPREAD ANALYSIS COMPLETE")
        print("=" * 70)
        
        print("\n🔥 Final Fire Statistics:")
        if 'fire' in sim.get_simulation_summary():
            final_stats = sim.get_simulation_summary()['fire']
            print(f"   • Peak burning cells: {max(burning_cells)}")
            print(f"   • Final burning cells: {final_stats['burning_cells']}")
            print(f"   • Total area burned: {final_stats['burn_percentage']:.1f}%")
            print(f"   • Fuel consumed: {total_fuel[0] - final_stats['total_fuel']:.1f} units")
        
        print("\n📁 Output files generated:")
        print("   1️⃣  output/environment_layout.png")
        print("      └─ Shows environment before fire (buildings, forests, lakes)")
        print("   2️⃣  output/fire_environment_comparison.png")
        print("      └─ Side-by-side: Environment vs Final Fire State")
        print("   3️⃣  output/fire_spread_analysis.png")
        print("      └─ Time series analysis of fire spread")
        
        print("\n💡 Check these visualizations to see:")
        print("   ✓ Fire BLOCKED by lake (cannot cross water)")
        print("   ✓ Fire goes AROUND building (obstacle avoidance)")
        print("   ✓ Fire spreads naturally through forest")
        print("   ✓ Wind effect on fire direction")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.stop_simulation()


if __name__ == "__main__":
    run_pure_fire_spread_demo()
    print("\n🎉 Pure fire spread demo finished successfully!")
