#!/usr/bin/env python3
"""
🚁 QUADCOPTER FIRE DETECTION TRAINING - MAIN SCRIPT

JEDNODUCHÉ POUŽITÍ:
    python main.py train     # Spustí trénink
    python main.py demo      # Testuje natrénovaný model
    python main.py validate  # Validace prostředí
    
ČISTÝ PRODUKČNÍ SCRIPT PRO TRÉNOVÁNÍ KVADROKOPTÉRY
NA DETEKCI OHNĚ V PYBULLET SIMULACI
"""

import sys
import argparse
import torch
import numpy as np
from datetime import datetime
import time
from collections import deque
import os
from tqdm import tqdm

# Core imports
from src.wildfire_gym_wrapper import WildfireMARLEnv
from src.wildfire_models import QuadActor
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def create_demo_visualization(demo_log, env):
    """Vytvoří vizualizaci demo běhů s trajektoriemi a rewards"""
    import os
    
    # Vytvoř výstupní složku
    os.makedirs("output", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # === TRAJEKTORIE DRONŮ ===
    ax1.set_title("Trajektorie dronů - Demo běhy", fontsize=14, fontweight='bold')
    
    colors = ['blue', 'red', 'green']
    
    for i, (run_name, data) in enumerate(demo_log['drones'].items()):
        positions = data['positions']
        if len(positions) > 0:
            # Rozděl pozice na x, y, z
            xs = [pos[0] for pos in positions]
            ys = [pos[1] for pos in positions]
            zs = [pos[2] for pos in positions]
            
            # Vykresli trajektorii (pohled shora)
            color = colors[i % len(colors)]
            ax1.plot(xs, ys, '-o', color=color, label=f'Run {i+1}', 
                    linewidth=2, markersize=3, alpha=0.7)
            
            # Označ start a konec
            if len(xs) > 0:
                ax1.plot(xs[0], ys[0], 'o', color=color, markersize=8, 
                        markeredgecolor='black', markeredgewidth=1)
                ax1.plot(xs[-1], ys[-1], 's', color=color, markersize=8,
                        markeredgecolor='black', markeredgewidth=1)
    
    # Přidej pozice ohně - 1 oheň uprostřed!
    fire_positions = [[0.0, 0.0]]  # 1 oheň uprostřed 50x50m mapy
    for fire_pos in fire_positions:
        ax1.plot(fire_pos[0], fire_pos[1], '*', color='orange', markersize=15, 
                markeredgecolor='red', markeredgewidth=2, label='Oheň' if fire_pos == fire_positions[0] else "")
    
    ax1.set_xlabel('X pozice (m)')
    ax1.set_ylabel('Y pozice (m)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # === REWARDS V ČASE ===
    ax2.set_title("Vývoj rewards v čase", fontsize=14, fontweight='bold')
    
    for i, (run_name, data) in enumerate(demo_log['drones'].items()):
        rewards = data['rewards']
        if len(rewards) > 0:
            color = colors[i % len(colors)]
            steps = range(len(rewards))
            ax2.plot(steps, rewards, '-', color=color, label=f'Run {i+1}', 
                    linewidth=2, alpha=0.8)
            
            # Cumulative reward
            cumulative = np.cumsum(rewards)
            ax2_twin = ax2.twinx()
            ax2_twin.plot(steps, cumulative, '--', color=color, alpha=0.5, 
                         linewidth=1)
    
    ax2.set_xlabel('Krok')
    ax2.set_ylabel('Reward za krok')
    ax2_twin.set_ylabel('Kumulativní reward')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Uložení
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"output/demo_visualization_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

class SimplePPOTrainer:
    """Zjednodušený PPO trainer - bez komplexností"""
    def __init__(self, actor, device='cpu', lr=1e-4):  # Konzervativní LR
        self.actor = actor
        self.optimizer = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=1e-6)
        self.device = device
        self.memory = []
        
        # Hyperparameters
        self.gamma = 0.995  # Mírně vyšší discount
        self.eps_clip = 0.15  # Menší clipping range
        self.entropy_coef = 0.05  # Bonus za explorace
        self.max_grad_norm = 0.5  # Gradient clipping
    
    def store_transition(self, obs, action, reward, log_prob, value, done, hidden):
        self.memory.append((obs, action, reward, log_prob, value, done, hidden))
    
    def update_policy(self):
        if len(self.memory) < 10:  # Minimální velikost batch
            return
            
        # Compute returns using gamma
        returns = []
        R = 0
        for i in reversed(range(len(self.memory))):
            obs, action, reward, log_prob, value, done, hidden = self.memory[i]
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalizace
        
        # Policy update s entropií
        total_loss = 0
        entropy_loss = 0
        for i, (obs, action, reward, old_log_prob, value, done, hidden) in enumerate(self.memory):
            # Forward pass pro nový log_prob
            local_map, self_state = obs
            action_out, _, _ = self.actor(local_map, self_state, hidden)
            
            mean = action_out[:, :4]
            scale = torch.clamp(action_out[:, 4:], 0.01, 1.0)
            
            dist = torch.distributions.Normal(mean, scale)
            new_log_prob = dist.log_prob(action).sum()
            
            # Advantage 
            advantage = returns[i]
            
            # PPO loss s clipping
            ratio = torch.exp(new_log_prob - old_log_prob.detach())
            clipped_ratio = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
            
            # Entropy bonus pro explorace
            entropy = dist.entropy().sum()
            entropy_loss += entropy
            
            total_loss += policy_loss - self.entropy_coef * entropy
        
        # Backprop s gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Clear memory
        self.memory.clear()
        return total_loss.item()

def train_main():
    """Spustí trénink kvadrokoptéry"""
    print("🚁 SPOUŠTÍM TRÉNINK KVADROKOPTÉRY...")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")
    
    # Environment - 4 drony v rozích 50x50m mapy
    env = WildfireMARLEnv(agents_config=["quad_1", "quad_2", "quad_3", "quad_4"])
    
    # Actor
    actor = QuadActor(message_dim=8).to(device)
    
    # Trainer
    trainer = SimplePPOTrainer(actor, device, lr=1e-4)  # Použije nový konstruktor
    
    # Training settings
    max_episodes = 30  # KRATŠÍ pro rychlejší iteraci
    max_steps = 200
    save_every = 5  # Častější checkpointy
    
    save_dir = f"models/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 Modely se ukládají do: {save_dir}")
    
    # Training stats
    episode_rewards = deque(maxlen=10)
    best_reward = -float('inf')
    
    # Progress bar pro epizody
    episode_pbar = tqdm(range(max_episodes), desc="🚁 Trénink", leave=True)
    
    for episode in episode_pbar:
        # Reset environment
        obs_dict, _ = env.reset()
        hidden_state = torch.zeros(4, 128).to(device)  # 4 drony, každý má 128 hidden dim
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_steps):
            try:
                # Extract observations
                if "quads" in obs_dict:
                    # *** VŠECH 4 DRONŮ místo pouze prvního ***
                    local_map = torch.FloatTensor(obs_dict["quads"]["local_map"]).to(device)  # Všechny drony: (4, 1, 32, 32)
                    self_state = torch.FloatTensor(obs_dict["quads"]["self_state"]).to(device)  # Všechny drony: (4, 6)
                else:
                    break  # No quad agent
                
                # Forward pass
                actions, message, new_hidden = actor(local_map, self_state, hidden_state)
                
                # Sample action pro všechny drony
                mean = actions[:, :4]  # (4, 4) - 4 drony, každý má 4 akce
                scale = torch.clamp(actions[:, 4:], 0.01, 1.0)  # (4, 4)
                
                dist = torch.distributions.Normal(mean, scale)
                action = dist.sample()  # (4, 4) - 4 drony, každý má 4 akce
                log_prob = dist.log_prob(action).sum()
                
                # Simple value estimation
                value = torch.tensor(0.0, device=device)
                
                # Step environment - *** AKCE PRO VŠECH 4 DRONŮ ***
                action_np = torch.tanh(action).cpu().numpy()  # (4, 4)
                result = env.step({"quads": {"action": action_np}})  # Posílá všech 4 akcí!
                obs_dict, reward, done, info = result
                
                # Store transition
                trainer.store_transition(
                    (local_map, self_state), action, reward, log_prob, value, done, hidden_state
                )
                
                # Update
                hidden_state = new_hidden
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
                    
            except Exception as e:
                print(f"❌ Chyba v kroku {step}: {e}")
                break
        
        # Update policy every episode
        if episode % 2 == 0:  # Update každé 2 epizody
            loss = trainer.update_policy()
            if loss is not None:
                episode_pbar.write(f"📈 Policy updated, Loss: {loss:.3f}")
            else:
                episode_pbar.write("📈 Policy updated (no loss returned)")
        
        # Stats
        episode_rewards.append(episode_reward)
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        
        # Update tqdm popis
        episode_pbar.set_postfix({
            'Reward': f"{episode_reward:.1f}",
            'Avg': f"{avg_reward:.1f}", 
            'Best': f"{best_reward:.1f}",
            'Steps': episode_steps
        })
        
        # Early stopping pokud vidíme kolaps
        if episode > 10 and avg_reward < 500 and episode_steps < 50:
            episode_pbar.write(f"⚠️ POLICY COLLAPSE detekován v epizodě {episode+1}! Ukončuji trénink.")
            break
        
        # Save checkpoint
        if (episode + 1) % save_every == 0:
            checkpoint = {
                'episode': episode,
                'actor_state_dict': actor.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'episode_rewards': list(episode_rewards),
                'best_reward': best_reward
            }
            torch.save(checkpoint, f"{save_dir}/checkpoint_ep{episode+1:03d}.pt")
            episode_pbar.write(f"💾 Model uložen (epizoda {episode+1})")
    
    # Final save
    final_checkpoint = {
        'episode': max_episodes,
        'actor_state_dict': actor.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'episode_rewards': list(episode_rewards),
        'best_reward': best_reward
    }
    torch.save(final_checkpoint, f"{save_dir}/final_model.pt")
    
    # Uzavři progress bar
    episode_pbar.close()
    
    print(f"\n🎉 TRÉNINK DOKONČEN!")
    print(f"   Celkem epizod: {max_episodes}")
    print(f"   Nejlepší reward: {best_reward:.1f}")
    print(f"   Modely uloženy v: {save_dir}")
    
    env.close()
    return save_dir

def demo_main():
    """Spustí demo natrénovaného modelu"""
    print("🎬 DEMO NATRÉNOVANÉHO MODELU")
    
    # Najdi nejnovější model
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ Složka models/ neexistuje. Nejdřív spusť trénink!")
        return
    
    # Najdi nejnovější složku
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not subdirs:
        print("❌ Žádné natrénované modely nenalezeny!")
        return
    
    latest_dir = sorted(subdirs)[-1]
    
    # Hledáme nejnovější checkpoint před kolapsem
    checkpoint_files = [f for f in os.listdir(f"{models_dir}/{latest_dir}") if f.startswith("checkpoint_")]
    if checkpoint_files:
        # Seřadíme podle čísla epizody
        checkpoint_files.sort()
        # Pokusíme se najít před kolapsem (kolem epizody 15)
        good_checkpoints = [f for f in checkpoint_files if "015" in f or "010" in f or "005" in f]
        if good_checkpoints:
            model_path = f"{models_dir}/{latest_dir}/{good_checkpoints[-1]}"  # Nejnovější dobrý
        else:
            model_path = f"{models_dir}/{latest_dir}/{checkpoint_files[0]}"  # Jakýkoliv checkpoint
    else:
        model_path = f"{models_dir}/{latest_dir}/final_model.pt"  # Fallback
    
    if not os.path.exists(model_path):
        print(f"❌ Model {model_path} neexistuje!")
        return
    
    print(f"📥 Načítám model: {model_path}")
    
    # Load model
    model = QuadActor(message_dim=8)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['actor_state_dict'])
    model.eval()
    
    # Environment - 4 drony v rozích 50x50m mapy
    env = WildfireMARLEnv(agents_config=["quad_1", "quad_2", "quad_3", "quad_4"])
    
    # Definuj start pozice pro demo
    start_positions = [
        ("Levý dolní", [-20, -20, 8]),
        ("Pravý dolní", [20, -20, 8]),  
        ("Pravý horní", [20, 20, 8]),
        ("Levý horní", [-20, 20, 8])
    ]
    
    # Run demo pro každou ze 4 pozic
    demo_log = {'drones': {}, 'rewards': []}
    
    for i, (position_name, start_pos) in enumerate(start_positions):
        print(f"\n🚁 === DEMO {i+1}: {position_name} {start_pos} ===\n")
        
        # Environment pro 1 drona
        env = WildfireMARLEnv(agents_config=["quad_1"])
        obs, _ = env.reset()
        
        # Přestav dron na počáteční pozici
        import pybullet as p
        drone_id = env.sim.drones["quad_1"].drone_id
        p.resetBasePositionAndOrientation(drone_id, start_pos, [0, 0, 0, 1])
        obs = env._get_obs()  # Aktualizuj observace
        
        hidden_state = torch.zeros(1, 128)  # 1 dron
        total_reward = 0
        steps = 0
        
        # Log pro vizualizaci
        positions = []
        rewards = []
        
        while steps < 150:  # Delší pro jeden dron - 150 kroků
            if "quads" in obs:
                # *** JEDEN DRON ***
                local_map = torch.FloatTensor(obs["quads"]["local_map"])  # (1, 1, 32, 32)
                self_state = torch.FloatTensor(obs["quads"]["self_state"])  # (1, 6)
            else:
                break
            
            with torch.no_grad():
                actions, _, hidden_state = model(local_map, self_state, hidden_state)
                
            # Sample action pro jeden dron
            mean = actions[:, :4]  # (1, 4)
            scale = torch.clamp(actions[:, 4:], 0.1, 1.0) 
            action_sample = torch.normal(mean, scale)
            actions_np = torch.tanh(action_sample).numpy()  # (1, 4)
            
            # ===== AKCE PRO JEDEN DRON =====
            action_dict = {
                "quads": {
                    "action": actions_np  # (1, 4)
                }
            }
            
            obs, reward, done, info = env.step(action_dict)
            total_reward += reward
            steps += 1
            
            # Log pozice drona
            if "quad_1" in env.sim.drones:
                pos = env.sim.drones["quad_1"].get_position()
                positions.append(pos.copy())
                rewards.append(reward)
            
            # Progress každých 15 kroků
            if steps % 15 == 0:
                if "quad_1" in env.sim.drones:
                    pos = env.sim.drones["quad_1"].get_position()
                    fire_distance = ((pos[0])**2 + (pos[1])**2)**0.5
                    print(f"   Krok {steps}: Vzdálenost k ohni {fire_distance:.1f}m, Reward={reward:.2f}")
            
            if done:
                break
        
        # Ulož data pro vizualizaci
        demo_log['drones'][f'quad_{position_name.replace(" ", "_")}'] = {
            'positions': positions,
            'rewards': rewards
        }
        demo_log['rewards'].append(total_reward)
        
        # Výsledek
        start_distance = ((start_pos[0])**2 + (start_pos[1])**2)**0.5
        if positions:
            final_pos = positions[-1]
            final_distance = ((final_pos[0])**2 + (final_pos[1])**2)**0.5
            improvement = start_distance - final_distance
            print(f"   ✅ Hotovo! Start: {start_distance:.1f}m -> Konec: {final_distance:.1f}m")
            print(f"   🏆 Zlepšení: {improvement:+.1f}m, Reward: {total_reward:.1f}")
        
        env.close()
    
    # 🎨 VYTVOŘ VIZUALIZACI 
    try:
        print("\n🎨 Vytvářím vizualizaci demo běhu...")
        if demo_log['drones']:
            # Simulujeme environment pro vizualizaci
            temp_env = WildfireMARLEnv(agents_config=["quad_1"])
            viz_path = create_demo_visualization(demo_log, temp_env)
            print(f"📈 Vizualizace uložena: {viz_path}")
            temp_env.close()
        else:
            print("⚠️ Žádná data pro vizualizaci")
    except Exception as e:
        print(f"⚠️ Chyba při vizualizaci: {e}")

def validate_main():
    """Ověří že prostředí funguje"""
    print("✅ VALIDACE PROSTŘEDÍ")
    
    env = WildfireMARLEnv(agents_config=["quad_1", "quad_2", "quad_3", "quad_4"])
    obs, _ = env.reset()
    
    print(f"🌍 Prostředí úspěšně inicializováno")
    print(f"   Observace klíče: {list(obs.keys())}")
    if "quads" in obs:
        print(f"   Quad local_map shape: {obs['quads']['local_map'].shape}")
        print(f"   Quad self_state shape: {obs['quads']['self_state'].shape}")
    
    # Test jednoho kroku - hovering pro jeden dron
    action = {
        "quads": {
            "action": np.array([[0.0, 0.0, 0.0, 0.1]])  # quad_1: hovering
        }
    }
    obs, reward, done, info = env.step(action)
    print(f"✅ Test krok: reward={reward:.1f}")
    
    env.close()
    print("✅ Validace dokončena!")

def main():
    parser = argparse.ArgumentParser(description="Quadcopter Fire Detection Training")
    parser.add_argument("command", choices=["train", "demo", "validate"], 
                      help="Co spustit")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_main()
    elif args.command == "demo":
        demo_main()
    elif args.command == "validate":
        validate_main()

if __name__ == "__main__":
    main()