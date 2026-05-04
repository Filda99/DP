import numpy as np
import time
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.simulation import Simulation

def run_maneuver_test():
    print("✈️ Zahajuji test manévrovatelnosti Fixed-Wing...")
    sim = Simulation()
    sim.start_simulation()
    
    # Přidáme letadlo do výšky 100m, letící na východ (yaw=0)
    fw_name = "tester_0"
    sim.add_fixedwing(fw_name, position=[0, 0, 100], yaw=0, water_capacity=200)
    
    positions = []
    
    # 1. FÁZE: Rozlet (přímý let pro stabilizaci rychlosti)
    print(" - Nabírám rychlost...")
    for _ in range(100):
        # Akce: [Roll=0, Pitch=0, Throttle=0.8, Water=0]
        sim.step_simulation({fw_name: np.array([0.0, 0.0, 0.8, 0.0])})

    # 2. FÁZE: Maximální zatáčka
    print(" - Zahajuji maximální zatáčku (Roll 1.0)...")
    for _ in range(1000):
        # Akce: [Roll=1.0, Pitch=0.05, Throttle=0.8, Water=0] 
        # (Pitch 0.05 pomáhá udržet výšku v náklonu)
        sim.step_simulation({fw_name: np.array([1.0, 0.05, 0.8, 0.0])})
        
        pos = sim.drones[fw_name].get_position()
        positions.append(pos[:2].copy())

    sim.stop_simulation()
    
    # 3. VÝPOČET POLOMĚRU
    pos_array = np.array(positions)
    # Najdeme nejvzdálenější body v zatáčce (průměr)
    min_x, max_x = np.min(pos_array[:, 0]), np.max(pos_array[:, 0])
    min_y, max_y = np.min(pos_array[:, 1]), np.max(pos_array[:, 1])
    
    diameter_x = max_x - min_x
    diameter_y = max_y - min_y
    
    # Poloměr je polovina největšího rozměru opsané dráhy
    radius = max(diameter_x, diameter_y) / 2.0
    
    print("-" * 30)
    print(f"📊 VÝSLEDKY TESTU:")
    print(f"Šířka zatáčky: {diameter_x:.2f} m")
    print(f"Délka zatáčky: {diameter_y:.2f} m")
    print(f"=> MINIMÁLNÍ POLOMĚR (Radius): {radius:.2f} m")
    print("-" * 30)
    
    if radius < 50:
        print("💡 Letadlo je velmi obratné. Orbita 100m bude fungovat.")
    elif radius < 150:
        print("💡 Letadlo má střední poloměr. Orbita musí být alespoň 150-200m.")
    else:
        print("⚠️ Letadlo zatáčí velmi pomalu. Musíme zvětšit mapu i orbitu.")

if __name__ == "__main__":
    run_maneuver_test()