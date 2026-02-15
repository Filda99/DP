import torch
import numpy as np
from env_core import DroneFireEnv
from models import MAPPOActor, MAPPOCritic

def count_parameters(model):
    """Spočítá celkový počet trénovatelných parametrů v modelu."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_models():
    print("📊 ANALÝZA VELIKOSTI NEURONOVÝCH SÍTÍ")
    print("="*40)

    # 1. Inicializace prostředí (jen abychom získali správné rozměry vstupů)
    # Parametry musí sedět s těmi v train.py!
    env = DroneFireEnv(num_quads=2, grid_size_m=200.0)
    
    # Získání rozměrů
    obs_space = env.observation_space("quad_0")
    self_state_size = obs_space["self_state"].shape[0]
    global_state_size = env.state_space.shape[0]
    action_dim = env.action_space("quad_0").shape[0]

    print(f"🔹 Vstupní vektor pro Actora (Self State): {self_state_size}")
    print(f"🔹 Vstupní vektor pro Kritika (Global State): {global_state_size}")
    print(f"🔹 Výstupní akce (Action Dim): {action_dim}")
    print("-" * 40)

    # 2. Inicializace modelů
    actor = MAPPOActor(self_state_size, action_dim)
    critic = MAPPOCritic(global_state_size)

    # 3. Spočítání parametrů
    actor_params = count_parameters(actor)
    critic_params = count_parameters(critic)

    # 4. Detailní rozpad Actora (protože je složený z CNN a MLP)
    actor_cnn_params = count_parameters(actor.cnn)
    actor_mlp_params = count_parameters(actor.mlp)
    actor_head_params = count_parameters(actor.action_mean)

    print(f"🧠 ACTOR (Mozek drona)")
    print(f"   - Celkem parametrů: {actor_params:,}")
    print(f"   - Z toho CNN (Oči): {actor_cnn_params:,}")
    print(f"   - Z toho MLP (Stav): {actor_mlp_params:,}")
    print(f"   - Z toho Action Head (Výstup): {actor_head_params:,}")
    print("")
    
    print(f"⚖️ CRITIC (Hodnotitel)")
    print(f"   - Celkem parametrů: {critic_params:,}")
    print("="*40)
    
    print("ZÁVĚR PRO DIPLOMKU:")
    if actor_params < 100000:
        print("✅ Síť je velmi lehká (<100k parametrů).")
        print("   To znamená, že je vhodná pro real-time inferenci přímo na palubě drona (Raspberry Pi/Jetson).")
    else:
        print("⚠️ Síť je větší, může vyžadovat silnější hardware.")

if __name__ == "__main__":
    analyze_models()