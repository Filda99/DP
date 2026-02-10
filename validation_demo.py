#!/usr/bin/env python3
"""
🎯 VALIDATION DEMO - Ukazuje natrénovaný model v akci
Načte uložený model a spustí 5 testovacích běhů bez učení
"""

import torch
import numpy as np
from src.wildfire_gym_wrapper import WildfireMARLEnv
from src.wildfire_models import QuadActor

def load_model(model_path):
    """Načte natrénovaný model"""
    model = QuadActor(message_dim=8)  # Správný constructor
    
    # Načti checkpoint a extrahuj model state_dict
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'actor_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['actor_state_dict'])
        print(f"✅ Model načten z epizody {checkpoint.get('episode', '?')}")
        print(f"   Nejlepší reward v checkpointu: {checkpoint.get('best_reward', '?'):.1f}")
    else:
        model.load_state_dict(checkpoint)  # Fallback pro starý formát
    
    model.eval()  # Evaluation mode - vypne dropout/batch norm
    return model

def demo_run(env, model, episode_num, verbose=True):
    """Spustí jeden testovací běh"""
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    max_steps = 200
    
    if verbose:
        print(f"\n🚁 === EPIZODA {episode_num} ===")
    
    # Inicializace hidden state pro GRU
    hidden_state = torch.zeros(1, 128)  # (num_agents, hidden_size)
    
    while steps < max_steps:
        # Extrakce quad observací (první agent)
        if "quads" in obs and obs["quads"]["local_map"].shape[0] > 0:
            local_map = torch.FloatTensor(obs["quads"]["local_map"][:1])  # První quad
            self_state = torch.FloatTensor(obs["quads"]["self_state"][:1])  # První quad
        else:
            # Fallback pokud nejsou quads
            local_map = torch.zeros(1, 1, 32, 32)
            self_state = torch.zeros(1, 7)
        
        # Inference - bez gradient computation
        with torch.no_grad():
            actions, message, hidden_state = model(local_map, self_state, hidden_state)
            
        # Action parameters jsou [mean, scale] pro 4 akce
        means = actions[:, :4]  # První 4 hodnoty
        scales = torch.clamp(actions[:, 4:], 0.01, 1.0)  # Další 4, clamp pro stabilitu
        
        # Sampling akcí z normal distribuce
        action_sample = torch.normal(means, scales)
        actions_np = torch.tanh(action_sample).squeeze().numpy()  # Normalize to [-1,1]
        
        # Krok v prostředí - WildfireMARLEnv očekává slovník akcí
        action_dict = {"quad_1": actions_np}  # Název prvního quad agenta
        obs, reward, done, truncated, info = env.step(action_dict)
        total_reward += reward
        steps += 1
        
        if verbose and steps % 50 == 0:
            print(f"   Krok {steps}: Reward {reward:.1f}, Total: {total_reward:.1f}")
        
        if done or truncated:
            break
    
    if verbose:
        print(f"   ✅ Dokončeno za {steps} kroků, Celkový reward: {total_reward:.1f}")
    
    return total_reward, steps

def main():
    print("🎯 VALIDATION DEMO - Testování natrénovaného modelu")
    print("=" * 60)
    
    # Cesta k nejlepšímu modelu (epizoda 15 před kolapsem)
    model_path = "models/training_20260210_124415/checkpoint_ep015.pt"
    
    try:
        # Načtení modelu
        print(f"📥 Načítám model: {model_path}")
        model = load_model(model_path)
        print("✅ Model úspěšně načten!")
        
        # Vytvoření prostředí - jen s jedním quad agentem
        print("🌍 Inicializuji prostředí...")
        env = WildfireMARLEnv(agents_config=["quad_1"])  # Pouze jeden quad agent
        print("✅ Prostředí připraveno!")
        
        # Spuštění 5 demo běhů
        print("\n🚀 Spouštím 5 testovacích běhů...")
        
        results = []
        for i in range(1, 6):
            reward, steps = demo_run(env, model, i, verbose=True)
            results.append((reward, steps))
        
        # Statistiky
        print("\n📊 VÝSLEDKY:")
        print("=" * 40)
        total_rewards = [r[0] for r in results]
        total_steps = [r[1] for r in results]
        
        for i, (reward, steps) in enumerate(results, 1):
            print(f"Run {i}: {reward:8.1f} reward za {steps:3d} kroků")
        
        print("-" * 40)
        print(f"Průměr: {np.mean(total_rewards):8.1f} reward")
        print(f"Median: {np.median(total_rewards):8.1f} reward") 
        print(f"Min:    {np.min(total_rewards):8.1f} reward")
        print(f"Max:    {np.max(total_rewards):8.1f} reward")
        print(f"Avg steps: {np.mean(total_steps):5.1f}")
        
        if np.mean(total_rewards) > 15000:
            print("\n🎉 VÝBORNÉ! Model funguje skvěle!")
        elif np.mean(total_rewards) > 5000:
            print("\n👍 DOBRÉ! Model funguje rozumně.")
        elif np.mean(total_rewards) > 1000:
            print("\n🤔 SLABÉ. Model potřebuje více tréninku.")
        else:
            print("\n😞 ŠPATNÉ. Model nefunguje.")
            
    except FileNotFoundError:
        print(f"❌ Model nenalezen: {model_path}")
        print("   Možné řešení:")
        print("   1. Zkontroluj cestu k modelu")
        print("   2. Spusť nejdřív trénink: python test_quick_train.py")
    except Exception as e:
        print(f"❌ Chyba: {e}")

if __name__ == "__main__":
    main()