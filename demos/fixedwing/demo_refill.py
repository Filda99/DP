"""
Demo: Refill Zone Test
Verifies that a fixed-wing drone automatically refills its water tank
when flying through the designated zone.
"""

import numpy as np
import time
import os
import sys

# Adjust path to import source modules
sys.path.append(os.getcwd())

from src.simulation import Simulation

def run_refill_test():
    print("🧪 STARTING REFILL ZONE TEST")
    print("============================")

    # 1. Start Simulation
    sim = Simulation()
    sim.start_simulation()
    
    # --- FIX: Vypnutí větru, aby dron letěl rovně ---
    sim.set_wind([0, 0, 0])
    
    # 2. Add Drone (FixedWing)
    # Starting at [0, 0, 50]
    drone_name = "FW_Test"
    drone = sim.add_fixedwing(drone_name, position=[0, 0, 50], water_capacity=50.0)
    
    # 3. Drain the Tank (Simulation Setup)
    print(f"🚰 Initial Water: {drone.current_water} L")
    drone.current_water = 0.0
    print(f"🚰 Water after draining: {drone.current_water} L")

    # 4. Create Refill Zone manually
    # Place it 100 meters ahead on the X axis, same altitude
    zone_pos = [100, 0, 50]
    sim.environment.create_refill_zone(center_pos=zone_pos, size=10.0)
    
    # 5. Flight Loop
    print("\n🛫 Drone launching towards the zone...")
    
    success = False
    
    # Simulate approx 10 seconds (600 steps)
    for step in range(600): 
        # Fly Straight: Roll=0, Throttle=0.8, Pitch=0
        controls = {
            drone_name: [0, 0.8, 0, 1] 
        }
        
        sim.step_simulation(controls)
        
        # Telemetry
        pos = drone.get_position()
        water = drone.current_water
        dist_to_zone = np.linalg.norm(pos - np.array(zone_pos))
        
        # Log every 20 steps
        if step % 20 == 0:
            print(f"Step {step}: Dist={dist_to_zone:.1f}m | Water={water:.1f} L")

        # CHECK SUCCESS CONDITION
        if water >= 50.0:
            print(f"\n✅ SUCCESS! Water refilled at step {step}!")
            print(f"   Distance to center: {dist_to_zone:.2f}m")
            print(f"   Tank Level: {water} L")
            success = True
            
        time.sleep(0.005) # Speed up simulation slightly

    if not success:
        print("\n❌ TEST FAILED: Drone missed the zone or refill failed.")
        print("   (Tip: Check if strong wind is pushing the drone off course)")

    sim.stop_simulation()

if __name__ == "__main__":
    run_refill_test()