#!/usr/bin/env python3
"""
Demo 1 DEBUG: Rychlá simulace šíření ohně
Optimalizováno pro rychlost: Větší buňky, méně časté ukládání.
"""

import numpy as np
import sys
import os
import glob
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects 
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Styl grafů
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except ImportError:
    pass

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (18, 6),
    'figure.dpi': 100  # Sníženo DPI pro rychlejší ukládání
})

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import Simulation
from src.map_importer import load_environment_from_osm_cache

def save_frame(sim, state, frame_num, time, output_dir):
    """Vykreslí a uloží jeden snímek."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    ax1, ax2, ax3 = axes
    
    x_min, x_max, y_min, y_max = sim.environment.grid_mapper.get_grid_bounds()
    H, W = state['B'].shape
    extent = [x_min, x_max, y_min, y_max]
    
    # === PANEL 1: Prostředí + Oheň ===
    burn_rate = sim.environment.fire_grid.fuel_burn_rate
    env_img = np.zeros((H, W, 3))
    
    # Barvy terénu
    mask_water = (burn_rate == 0.0) 
    mask_forest = (burn_rate >= 0.03) & (burn_rate < 0.06)
    mask_grass = (burn_rate >= 0.06)
    mask_building = (burn_rate > 0.0) & (burn_rate < 0.01)

    env_img[mask_grass] = [0.6, 0.7, 0.4]    # Louka
    env_img[mask_forest] = [0.1, 0.4, 0.1]   # Les
    env_img[mask_building] = [0.5, 0.5, 0.5] # Budovy
    env_img[mask_water] = [0.2, 0.5, 0.9]    # Voda
    
    # Vrstva ohně
    burning = state['B']
    fire_overlay = np.zeros((H, W, 4))
    fire_overlay[burning] = [1.0, 0.2, 0.0, 0.8] # Červená
    
    ax1.imshow(env_img, origin='lower', extent=extent)
    ax1.imshow(fire_overlay, origin='lower', extent=extent)
    
    # Šipka větru
    wind_vel = sim.environment.weather['wind_velocity']
    wind_speed = np.linalg.norm(wind_vel[:2])
    arrow_x = x_max - (x_max - x_min) * 0.1
    arrow_y = y_max - (y_max - y_min) * 0.1
    
    ax1.arrow(arrow_x, arrow_y, wind_vel[0] * 15, wind_vel[1] * 15, 
              head_width=40, head_length=40, fc='yellow', ec='black', width=8, zorder=10)
    txt = ax1.text(arrow_x, arrow_y + 80, f"{wind_speed:.1f} m/s", color='yellow', 
             fontweight='bold', ha='center', zorder=11)
    txt.set_path_effects([matplotlib.patheffects.withStroke(linewidth=3, foreground="black")])
    
    ax1.set_title(f'Mapa a Oheň', fontweight='bold')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')

    # === PANEL 2: Palivo ===
    fuel = state['F']
    im2 = ax2.imshow(fuel, origin='lower', extent=extent, cmap='YlGn_r', vmin=0, vmax=1)
    ax2.set_title(f'Zbývající Palivo', fontweight='bold')
    ax2.set_yticklabels([]) 
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2).set_label('Palivo (0-1)')

    # === PANEL 3: Intenzita ===
    intensity = state['I']
    im3 = ax3.imshow(intensity, origin='lower', extent=extent, cmap='inferno', vmin=0, vmax=1)
    ax3.set_title(f'Intenzita Ohně', fontweight='bold')
    ax3.set_yticklabels([])
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3).set_label('Intenzita (0-1)')

    # Nadpis
    burning_count = np.sum(burning)
    burn_area_ha = (burning_count * sim.environment.grid_mapper.cell_size_m**2) / 10000
    plt.suptitle(f'Čas: {time:.1f}s | Spálená plocha: {burn_area_ha:.2f} ha', fontsize=16, fontweight='bold')
    
    plt.savefig(f'{output_dir}/frame_{frame_num:03d}.png')
    plt.close()

def run_fast_demo():
    print("=" * 70)
    print("🚀 RYCHLÁ SIMULACE OHNĚ (Optimalizováno)")
    print("=" * 70)
    
    LOCATION = "Pec pod Sněžkou, Czech Republic"
    CACHE_PREFIX = "Pec_pod_Sněžkou_Czechia"
    CENTER_LAT = 50.6868
    CENTER_LON = 15.7361
    RADIUS_M = 1000
    
    # 1. Start simulace
    sim = Simulation()
    sim.start_simulation()
    
    # 2. Načtení mapy (Cache / Download)
    cache_dir = "data"
    cache_pattern = f"{cache_dir}/{CACHE_PREFIX}_building_*.gpkg"
    
    if len(glob.glob(cache_pattern)) > 0:
        print(f"📂 Načítám z cache: {CACHE_PREFIX}")
        load_environment_from_osm_cache(
            environment=sim.environment,
            cache_dir=cache_dir,
            region_prefix=CACHE_PREFIX,
            center_lat=CENTER_LAT,
            center_lon=CENTER_LON,
            radius_m=RADIUS_M,
            default_height_m=8.0
        )
    else:
        print(f"🌍 Stahuji z OSM...")
        sim.setup_osm_environment(LOCATION, default_height_m=8.0, radius_m=RADIUS_M)
        
    # 3. NASTAVENÍ MŘÍŽKY - ZDE JE HLAVNÍ ZRYCHLENÍ
    # Zvětšili jsme cell_size_m z 2.0 na 5.0 -> 6.25x méně buněk!
    sim.enable_fire_simulation(
        grid_width_m=2*RADIUS_M,
        grid_height_m=2*RADIUS_M,
        cell_size_m=5.0,  # <--- Větší buňky = mnohem rychlejší výpočet
        dt=0.5            # <--- Větší časový krok fyziky ohně
    )
    
    # Start ohně
    print("\n🔥 Zakládám oheň...")
    fire_started = False
    fire_state = sim.environment.get_fire_state()
    
    if fire_state:
        state = fire_state['fire_grid_state']
        fuel_grid = state['F']
        H, W = fuel_grid.shape
        
        # Hledáme les ve spodní části mapy
        candidates = []
        for i in range(H // 4, H // 2):
            for j in range(W // 3, 2 * W // 3):
                if fuel_grid[i, j] > 0.7: # Les
                    candidates.append((i, j))
        
        if candidates:
            # Vyber náhodné místo v lese
            ci, cj = candidates[np.random.randint(len(candidates))]
            wx, wy = sim.environment.grid_mapper.cell_to_world(ci, cj)
            sim.start_fire((wx, wy), intensity=0.5)
            fire_started = True
            print(f"✅ Oheň zapálen na pozici ({wx:.0f}, {wy:.0f})")
    
    if not fire_started:
        sim.start_fire((-100, -100), intensity=0.5)

    # 4. NASTAVENÍ BĚHU
    output_dir = 'output/fast_frames'
    os.makedirs(output_dir, exist_ok=True)
    
    total_time = 300.0      # Simulujeme 5 minut (300 sekund)
    save_interval = 20.0    # Ukládáme obrázek každých 20 sekund
    
    # Přepočet na kroky
    sim_dt = sim.timestep   # 1/60 s
    steps_per_save = int(save_interval / sim_dt)
    total_steps = int(total_time / sim_dt)
    
    print(f"\n🏃 Běžím simulaci na {total_time}s...")
    print(f"   - Ukládám snímek každých {save_interval}s")
    print(f"   - Celkem snímků: {int(total_time/save_interval)}")
    
    frame = 0
    start_real_time = time.time()
    
    for step in range(total_steps):
        sim.step_simulation({})
        
        # Uložení snímku
        if step % steps_per_save == 0:
            current_sim_time = sim.simulation_time
            fire_state = sim.environment.get_fire_state()
            
            if fire_state:
                state = fire_state['fire_grid_state']
                burning = np.sum(state['B'])
                
                # Výpis progressu
                elapsed = time.time() - start_real_time
                print(f"  [{elapsed:.1f}s] Snímek {frame:03d} | SimTime: {current_sim_time:.0f}s | Hoří: {burning} buněk")
                
                save_frame(sim, state, frame, current_sim_time, output_dir)
                frame += 1
    
    print(f"\n✅ Hotovo! Snímky uloženy v {output_dir}/")

if __name__ == '__main__':
    run_fast_demo()