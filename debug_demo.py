#!/usr/bin/env python3
"""
DEBUG SCRIPT pro validaci demo prostředí
"""
import sys
import numpy as np
import torch

from src.wildfire_gym_wrapper import WildfireMARLEnv
from src.wildfire_models import QuadActor

def debug_environment():
    """Ověří základní chování prostředí"""
    print("🔍 DEBUG: Ověřuji prostředí...")
    
    env = WildfireMARLEnv(agents_config=["quad_1"])
    obs, _ = env.reset()
    
    print(f"✅ Prostředí inicializováno")
    print(f"   Start pozice drona: {env.sim.drones['quad_1'].get_position()}")
    
    # Zkontroluj fire state
    fire_state = env.sim.environment.get_fire_state()
    if fire_state:
        burning_cells = np.sum(fire_state['fire_grid_state']['B'])
        fire_intensity = np.sum(fire_state['fire_grid_state']['I'])
        print(f"🔥 Fire state: {burning_cells} hořících buněk, intenzita: {fire_intensity:.2f}")
    else:
        print("❌ Žádný fire state!")
    
    # Test 10 kroků s fixed akcemi
    print("\n🚁 Test pohybu drona...")
    positions = []
    
    for step in range(10):
        # Jednoduchá akce - let vpřed
        action_dict = {
            "quads": {
                "action": np.array([[0.1, 0.0, 0.0, 0.2]])  # Vpřed + trochu nahoru
            }
        }
        
        obs, reward, done, truncated, info = env.step(action_dict)
        pos = env.sim.drones['quad_1'].get_position()
        positions.append(pos.copy())
        
        print(f"   Krok {step+1}: Pozice {pos}, Reward: {reward:.1f}")
        
        if done or truncated:
            break
    
    # Analýza pohybu
    if len(positions) > 1:
        start_pos = np.array(positions[0])
        end_pos = np.array(positions[-1])
        distance = np.linalg.norm(end_pos - start_pos)
        print(f"\n📊 Celkový pohyb: {distance:.2f}m od {start_pos} do {end_pos}")
    else:
        print("❌ Dron se nepohnul!")
    
    env.close()
    return positions

def debug_model_actions():
    """Ověří, že model generuje rozumné akce"""
    print("\n🧠 DEBUG: Ověřuji model akce...")
    
    # Load model
    model_path = "models/training_20260210_132859/checkpoint_ep015.pt"
    model = QuadActor(message_dim=8)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['actor_state_dict'])
    model.eval()
    
    # Dummy input
    local_map = torch.randn(1, 1, 32, 32)  # Náhodná mapa
    self_state = torch.randn(1, 12)        # Náhodný state
    hidden = torch.zeros(1, 128)
    
    with torch.no_grad():
        actions, _, _ = model(local_map, self_state, hidden)
    
    mean = actions[:, :4]
    scale = torch.clamp(actions[:, 4:], 0.01, 1.0)
    
    print(f"   Action mean: {mean.numpy()}")
    print(f"   Action scale: {scale.numpy()}")
    
    # Ukázka sampling
    for i in range(5):
        action_sample = torch.normal(mean, scale)
        actions_np = torch.tanh(action_sample).squeeze().numpy()
        print(f"   Sample {i+1}: {actions_np}")

def debug_fire_setup():
    """Ověří, že se fire správně nastaví"""
    print("\n🔥 DEBUG: Ověřuji fire setup...")
    
    from src.simulation import Simulation
    
    sim = Simulation()
    sim.start_simulation()
    
    print("✅ Základní simulace nastartovala")
    
    # ===== DŮLEŽITÉ: ENABLE FIRE SIMULATION FIRST! =====
    print("🔧 Zapínám fire simulation...")
    sim.enable_fire_simulation(
        grid_width_m=200,   # Stejné jako v wrapper
        grid_height_m=200,
        cell_size_m=2.0,
        dt=0.5
    )
    
    # Přidej fire stejně jak v demo
    print("🔥 Zapahluji fire...")
    sim.environment.ignite_fire(x=0, y=8, intensity=3.0)
    sim.environment.ignite_fire(x=2, y=5, intensity=2.0)
    
    # Zjisti fire state
    fire_state = sim.environment.get_fire_state()
    if fire_state:
        state = fire_state['fire_grid_state']
        burning = state['B']
        intensities = state['I']
        
        print(f"   Hořící buňky: {np.sum(burning)}")
        print(f"   Max intenzita: {np.max(intensities):.2f}")
        print(f"   Celková intenzita: {np.sum(intensities):.2f}")
        
        # Pozice fire buněk
        burning_coords = np.where(burning)
        if len(burning_coords[0]) > 0:
            print(f"   Fire pozice: {list(zip(burning_coords[0][:5], burning_coords[1][:5]))}")
    else:
        print("❌ Fire state se nevytvořil!")
        return
    
    # Přidej dron a otestuj local observation
    quad = sim.add_quadcopter('test_quad', position=[0, 8, 10])  # Nad fire
    
    # Prostě krokem
    sim.step_simulation({})
    
    # Test local observation - SPRÁVNÝ způsob
    from src.wildfire_obs_processor import WildfireObsProcessor
    obs_proc = WildfireObsProcessor(window_size_m=20.0, resolution_px=32)
    
    try:
        obs_data = obs_proc.fetch(sim, 'test_quad')
        local_map = obs_data["local_map"]
        fire_in_view = np.sum(local_map[0])  # Prvních kanál
        print(f"🔍 Fire v local observation: {fire_in_view:.2f}")
        print(f"   Local map shape: {local_map.shape}")
    except Exception as e:
        print(f"❌ Chyba v local observation: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚁 DEMO DEBUG - Validace prostředí a modelu")
    print("=" * 60)
    
    debug_fire_setup()
    debug_environment()
    debug_model_actions()
    
    print("\n✅ Debug dokončen!")