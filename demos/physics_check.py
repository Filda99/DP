#!/usr/bin/env python3
"""
PHYSICS SANITY CHECK
Cíl: Zjistit, zda letadlo dokáže letět samo (bez autopilota).
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pybullet as p

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation

def run_physics_check():
    print("🧪 SPOUŠTÍM FYZIKÁLNÍ DIAGNOSTIKU...")
    
    sim = Simulation()
    sim.start_simulation()
    
    # 1. Přidáme letadlo vysoko (100m)
    # Vypneme vodu (lehčí letadlo)
    fw = sim.add_fixedwing("Test_Plane", position=[0, 0, 100], water_capacity=0.0)
    
    # 2. BRUTÁLNÍ NATVRDO VYNUCENÁ RYCHLOST
    # 30 m/s je cca 108 km/h - to MUSÍ letět
    p.resetBaseVelocity(fw.drone_id, linearVelocity=[30.0, 0.0, 0.0], physicsClientId=sim.physics_client)
    
    # 3. KONTROLA DAMPINGU (Kritické!)
    # Získáme aktuální nastavení dynamiky
    dyn_info = p.getDynamicsInfo(fw.drone_id, -1)
    print(f"🔍 Damping Info (před změnou): Linear={dyn_info[8]}, Angular={dyn_info[9]}")
    
    # Vynutíme nulový odpor (aby fungovala naše aerodynamika)
    p.changeDynamics(fw.drone_id, -1, linearDamping=0.0, angularDamping=0.0)
    
    # Data pro graf
    log_h = []
    log_v = []
    log_t = []
    
    print("🚀 Házím letadlo... (Input: [0, 1.0, 0.1])")
    # Vstup: Žádné zatáčení, Plný plyn, Lehký Pitch Up (aby to nešlo čumákem dolů)
    fixed_input = [0.0, 1.0, 0.1] 
    
    for i in range(300): # 5 sekund letu
        # Aplikujeme ovládání napřímo
        fw.apply_control(fixed_input)
        p.stepSimulation()
        
        # Telemetrie
        pos = fw.get_position()
        vel = fw.get_speed()
        
        log_h.append(pos[2])
        log_v.append(vel)
        log_t.append(i / 60.0)
        
        if pos[2] < 0.5:
            print(f"💥 NÁRAZ V ČASE {i/60.0:.2f}s")
            break

    sim.stop_simulation()
    
    # Vykreslení výsledku
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(log_t, log_h, 'b-')
    plt.title("Výška (Altitude)")
    plt.xlabel("Čas (s)")
    plt.ylabel("Metry")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(log_t, log_v, 'r-')
    plt.title("Rychlost (Speed)")
    plt.xlabel("Čas (s)")
    plt.ylabel("m/s")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("output/physics_check.png")
    print("✅ Graf uložen do output/physics_check.png")

if __name__ == "__main__":
    run_physics_check()