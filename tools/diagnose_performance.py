#!/usr/bin/env python3
"""
Diagnostic Tool: Analyze Fire Simulation Performance

Shows where time is being spent during simulation setup and execution.
"""

import time
import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.simulation import Simulation


def time_operation(name, func):
    """Time a function and print results."""
    print(f"\n⏱️  {name}...", end='', flush=True)
    start = time.time()
    result = func()
    elapsed = time.time() - start
    print(f" {elapsed:.2f}s")
    return result, elapsed


def analyze_simulation_performance():
    """Analyze where time is spent in simulation."""
    print("=" * 70)
    print("🔍 FIRE SIMULATION PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    times = {}
    
    # 1. Create simulation
    def create_sim():
        sim = Simulation()
        sim.start_simulation()
        return sim
    
    sim, times['create'] = time_operation("Creating simulation", create_sim)
    
    # 2. Download OSM data
    location = "Tišnov, Czech Republic"
    distance_m = 1000  # Reduced from 3000m
    use_city_boundaries = True  # Use city boundaries instead of individual buildings
    
    def download_osm():
        sim.setup_osm_environment(location, 
                                  default_building_height=8.0,
                                  distance_m=distance_m,
                                  use_city_boundaries=use_city_boundaries)
    
    _, times['osm_download'] = time_operation(
        f"Downloading OSM data ({distance_m}m radius, {'city boundaries' if use_city_boundaries else 'individual buildings'})", 
        download_osm
    )
    
    # Print what was loaded
    print(f"\n   📊 Loaded:")
    print(f"      Buildings: {len(sim.environment.obstacles):,}")
    print(f"      Forests: {sum(1 for z in sim.environment.terrain_zones if z['type'] == 'forest')}")
    print(f"      Lakes: {sum(1 for z in sim.environment.terrain_zones if z['type'] == 'lake')}")
    
    # 3. Enable fire simulation
    grid_size = 2000  # Reduced from 6000m
    cell_size = 10.0  # Reduced from 20.0m
    
    def enable_fire():
        sim.enable_fire_simulation(
            grid_width_m=grid_size,
            grid_height_m=grid_size,
            cell_size_m=cell_size
        )
    
    _, times['fire_grid'] = time_operation(
        f"Creating fire grid ({grid_size}m, {cell_size}m cells)", 
        enable_fire
    )
    
    # Print fire grid stats
    if sim.environment.fire_grid:
        H, W = sim.environment.fire_grid.H, sim.environment.fire_grid.W
        print(f"\n   📊 Fire Grid:")
        print(f"      Size: {H}×{W} = {H*W:,} cells")
        
        fuel = sim.environment.fire_grid.fuel_burn_rate
        zero_fuel = np.sum(fuel == 0.0)
        low_fuel = np.sum((fuel > 0.0) & (fuel < 0.05))
        high_fuel = np.sum(fuel >= 0.05)
        
        print(f"      No fuel: {zero_fuel:,} ({100*zero_fuel/(H*W):.1f}%)")
        print(f"      Low fuel: {low_fuel:,} ({100*low_fuel/(H*W):.1f}%)")
        print(f"      High fuel: {high_fuel:,} ({100*high_fuel/(H*W):.1f}%)")
    
    # 4. Start a fire
    def start_fire():
        return sim.start_fire((0, 0), intensity=0.5)
    
    _, times['start_fire'] = time_operation("Starting fire", start_fire)
    
    # 5. Run simulation steps
    print(f"\n⏱️  Running 10 simulation steps...")
    step_times = []
    
    for i in range(10):
        start = time.time()
        sim.step_simulation({})
        elapsed = time.time() - start
        step_times.append(elapsed)
        print(f"      Step {i+1}: {elapsed:.3f}s")
    
    times['avg_step'] = np.mean(step_times)
    times['total_steps'] = sum(step_times)
    
    # 6. Save visualization
    def save_viz():
        sim.environment.save_environment_map('output/diagnostic_map.png', 
                                            show_fire_grid=True, 
                                            detailed=False)
    
    _, times['visualization'] = time_operation(
        "Saving visualization", 
        save_viz
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 70)
    
    total_init = times['create'] + times['osm_download'] + times['fire_grid']
    
    print(f"\n🚀 Initialization (one-time):")
    print(f"   Create simulation:  {times['create']:6.2f}s")
    print(f"   Download OSM:       {times['osm_download']:6.2f}s ⚠️ SLOW")
    print(f"   Create fire grid:   {times['fire_grid']:6.2f}s ⚠️ SLOW")
    print(f"   Start fire:         {times['start_fire']:6.2f}s")
    print(f"   TOTAL INIT:         {total_init:6.2f}s")
    
    print(f"\n🔥 Per-Step Performance:")
    print(f"   Average step time:  {times['avg_step']:6.3f}s")
    print(f"   10 steps total:     {times['total_steps']:6.2f}s")
    
    print(f"\n📸 Visualization:")
    print(f"   Save map:           {times['visualization']:6.2f}s")
    
    print(f"\n⏰ Estimated Time for 10-second Simulation:")
    steps_needed = int(10 / sim.timestep)  # 10 seconds / timestep
    estimated_sim_time = steps_needed * times['avg_step']
    estimated_total = total_init + estimated_sim_time + times['visualization']
    
    print(f"   Timestep: {sim.timestep}s → {steps_needed} steps needed")
    print(f"   Init: {total_init:.1f}s")
    print(f"   Simulation: {steps_needed} × {times['avg_step']:.3f}s = {estimated_sim_time:.1f}s")
    print(f"   Visualization: {times['visualization']:.1f}s")
    print(f"   TOTAL: {estimated_total:.1f}s")
    
    if estimated_total > 60:
        print(f"\n   ❌ WILL TIMEOUT (60s limit)")
        over = estimated_total - 60
        print(f"   ⚠️  Need to reduce by {over:.1f}s")
    else:
        print(f"\n   ✅ Should complete in time!")
    
    # Bottleneck analysis
    print("\n" + "=" * 70)
    print("🎯 BOTTLENECK ANALYSIS")
    print("=" * 70)
    
    bottlenecks = [
        ("OSM Download", times['osm_download']),
        ("Fire Grid Creation", times['fire_grid']),
        ("Simulation Steps", times['total_steps']),
        ("Visualization", times['visualization'])
    ]
    
    bottlenecks.sort(key=lambda x: x[1], reverse=True)
    
    print("\n⏱️  Time spent (sorted):")
    for name, t in bottlenecks:
        pct = 100 * t / estimated_total
        bar = "█" * int(pct / 2)
        print(f"   {name:20s} {t:6.2f}s  {pct:5.1f}%  {bar}")
    
    # Optimization suggestions
    print("\n" + "=" * 70)
    print("💡 OPTIMIZATION SUGGESTIONS")
    print("=" * 70)
    
    if times['osm_download'] > 10:
        print(f"\n1. OSM Download is VERY SLOW ({times['osm_download']:.1f}s)")
        print(f"   → Reduce distance_m from {distance_m}m to 500-1000m")
        print(f"   → Focus on specific forest area instead of entire city")
        print(f"   → Expected savings: ~{times['osm_download']*0.8:.1f}s")
    
    if times['fire_grid'] > 10:
        print(f"\n2. Fire Grid Creation is SLOW ({times['fire_grid']:.1f}s)")
        print(f"   → Reduce grid size from {grid_size}m to 1000-2000m")
        print(f"   → Increase cell size from {cell_size}m to 50m")
        print(f"   → Expected savings: ~{times['fire_grid']*0.7:.1f}s")
    
    if times['avg_step'] > 0.5:
        print(f"\n3. Simulation Steps are SLOW ({times['avg_step']:.3f}s each)")
        print(f"   → Implement active region processing (only check cells near fire)")
        print(f"   → Expected savings: ~{estimated_sim_time*0.9:.1f}s")
    
    sim.stop_simulation()
    
    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    analyze_simulation_performance()
