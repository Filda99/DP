#!/usr/bin/env python3
"""
DEMO: Precise Maneuvers (Kinematic Model)
Scénář: Rovně -> Doleva -> Rovně -> Doprava -> Rovně -> Vypnout.
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pybullet as p

# Cesta k projektu
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation

# ------------------------------------------------------------------------------
# LOGIKA LETOVÉHO PLÁNU
# ------------------------------------------------------------------------------
def get_flight_commands(t):
    """
    Vrací [Roll, Throttle, Pitch, Water] podle času.
    Roll > 0 ... Zatáčka DOLEVA (podle Beard-McLain implementace: phi = -input * max)
    Roll < 0 ... Zatáčka DOPRAVA
    """
    
    # Defaultní hodnoty (Let rovně, cestovní rychlost)
    roll = 0.0
    throttle = 0.7  # cca 40 m/s
    pitch = 0.0
    water = 0.0
    
    # 1. Letíme rovně (0s - 2s)
    if t < 2.0:
        roll = 0.0
        
    # 2. Zatáčka DOLEVA (2s - 3s)
    elif t < 3.0:
        roll = 1.0  # Plný náklon doleva
        
    # 3. Letíme rovně (3s - 6s)
    elif t < 6.0:
        roll = 0.0
        
    # 4. Zatáčka DOPRAVA (6s - 7s)
    elif t < 7.0:
        roll = -1.0 # Plný náklon doprava
        
    # 5. Letíme rovně (7s - 10s)
    elif t < 10.0:
        roll = 0.0
        
    # 6. KONEC - Vypnout vše (10s+)
    else:
        roll = 0.0
        throttle = 0.0 # Motor vypnut
        pitch = 0.0
        
    return [roll, throttle, pitch, water]

# ------------------------------------------------------------------------------
# HLAVNÍ FUNKCE
# ------------------------------------------------------------------------------
def run_maneuver_demo():
    print("=" * 60)
    print("✈️  DEMO: S-TURN MANEUVER (Beard-McLain Kinematics)")
    print("=" * 60)
    
    sim = Simulation()
    sim.start_simulation()
    
    # Inicializace prostředí (pro jistotu, ať je grid ready)
    sim.enable_fire_simulation(grid_width_m=1000, grid_height_m=1000, cell_size_m=2.0)
    
    # Startovní pozice (X=0, Y=0, Výška=100m)
    start_pos = [0, 0, 100]
    fw = sim.add_fixedwing("AgroPlane", position=start_pos, mass=600.0, max_thrust=4000.0)
    
    # Nastavení počáteční rychlosti (nutné pro kinematiku)
    # Beard-McLain model potřebuje V > 0, aby nedělil nulou při výpočtu zatáčky
    initial_speed = 35.0
    p.resetBaseVelocity(fw.drone_id, linearVelocity=[initial_speed, 0, 0])
    
    # Data pro grafy
    times = []
    pos_x, pos_y = [], []
    headings = []
    inputs_log = []
    speeds = []
    
    total_time = 12.0 # 10s manévr + 2s dojezd
    steps = int(total_time / sim.timestep)
    
    print(f"▶️  Startuji scénář ({total_time}s)...")
    
    for step in range(steps):
        t = sim.simulation_time
        
        # Získání povelů
        controls = get_flight_commands(t)
        
        # Aplikace
        sim.step_simulation({"AgroPlane": controls})
        
        # Logování (každý 5. krok)
        if step % 5 == 0:
            pos = fw.get_position()
            times.append(t)
            pos_x.append(pos[0])
            pos_y.append(pos[1])
            headings.append(np.degrees(fw.get_heading()))
            inputs_log.append(controls[0]) # Roll input
            speeds.append(fw.get_speed())
            
        # Výpis
        if step % 60 == 0:
            phase = "IDLE"
            if t < 2: phase = "STRAIGHT"
            elif t < 3: phase = "LEFT TURN"
            elif t < 6: phase = "STRAIGHT"
            elif t < 7: phase = "RIGHT TURN"
            elif t < 10: phase = "STRAIGHT"
            else: phase = "ENGINE OFF"
            
            print(f"T={t:4.1f}s | {phase:12s} | Speed={fw.get_speed():.1f} | Head={headings[-1]:.1f}°")

    sim.stop_simulation()
    
    # --------------------------------------------------------------------------
    # VIZUALIZACE
    # --------------------------------------------------------------------------
    print("📊 Generuji grafy...")
    output_dir = 'output/demo_maneuver'
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # 1. Top-Down Trajektorie (Mapa)
    ax_map = fig.add_subplot(gs[0, :])
    ax_map.plot(pos_x, pos_y, 'b-', linewidth=2, label='Trajektorie')
    ax_map.plot(pos_x[0], pos_y[0], 'go', label='Start')
    ax_map.plot(pos_x[-1], pos_y[-1], 'ro', label='End')
    ax_map.set_title('Pohled shora (Top-Down)', fontweight='bold')
    ax_map.set_xlabel('X (m)')
    ax_map.set_ylabel('Y (m)')
    ax_map.axis('equal') # Aby kruh vypadal jako kruh
    ax_map.grid(True)
    ax_map.legend()
    
    # 2. Vstupy (Kdy zatáčíme)
    ax_inp = fig.add_subplot(gs[1, :])
    ax_inp.plot(times, inputs_log, 'orange', linewidth=2)
    ax_inp.set_title('Vstup Zatáčení (Roll Command)', fontweight='bold')
    ax_inp.set_ylabel('Input (-1..1)')
    ax_inp.set_ylim(-1.2, 1.2)
    ax_inp.grid(True)
    
    # Přidání textových anotací do grafu vstupů
    ax_inp.text(1.0, 0.2, "Rovně", ha='center')
    ax_inp.text(2.5, 0.8, "DOLEVA", ha='center', color='green', fontweight='bold')
    ax_inp.text(4.5, 0.2, "Rovně", ha='center')
    ax_inp.text(6.5, -0.8, "DOPRAVA", ha='center', color='red', fontweight='bold')
    ax_inp.text(8.5, 0.2, "Rovně", ha='center')
    ax_inp.text(11.0, 0.2, "Vypnuto", ha='center', color='gray')

    # 3. Rychlost (Reakce na vypnutí)
    ax_spd = fig.add_subplot(gs[2, :], sharex=ax_inp)
    ax_spd.plot(times, speeds, 'k-', linewidth=2)
    ax_spd.set_title('Rychlost (Speed)', fontweight='bold')
    ax_spd.set_ylabel('m/s')
    ax_spd.set_xlabel('Čas (s)')
    ax_spd.grid(True)
    ax_spd.axvline(10.0, color='r', linestyle='--', label='Throttle=0')
    ax_spd.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/maneuver_result.png')
    print(f"✅ Hotovo! Graf uložen: {output_dir}/maneuver_result.png")

if __name__ == "__main__":
    run_maneuver_demo()