#!/usr/bin/env python3
"""Quick test of OSM loading with waterways"""
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.simulation import Simulation

print("Testing OSM load with waterways...")

sim = Simulation(gui=False)
sim.start_simulation()

sim.setup_osm_environment("Křivoklát, Czech Republic", 
                          default_building_height=8.0,
                          distance_m=1500,  # Increased to capture big river
                          use_city_boundaries=True)

sim.enable_fire_simulation(grid_width_m=3000, grid_height_m=3000, cell_size_m=15.0)

print("\n📸 Saving environment map...")
sim.environment.save_environment_map('output/test_with_rivers.png', 
                                    show_fire_grid=True, 
                                    detailed=False)

print("✅ Done! Check output/test_with_rivers.png")
