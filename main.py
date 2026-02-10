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
import matplotlib.patheffects
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    """Gentle Action PPO Trainer - STABILNÍ VERZE S POSTUPNÝM ZVYŠOVÁNÍM AKCÍ"""
    def __init__(self, actor, device='cpu', lr=3e-5, gentle_training=True):
        self.actor = actor
        self.optimizer = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=1e-5)
        self.device = device
        self.memory = []
        self.update_count = 0
        
        # Gentle training parameters
        self.gentle_training = gentle_training
        if gentle_training:
            self.action_scale = 0.1  # Začínáme malé!
            self.max_action_scale = 0.8
            self.action_scale_increment = 0.05
            print(f"🎯 GENTLE TRAINING ENABLED - starting action scale: {self.action_scale:.3f}")
        else:
            self.action_scale = 1.0
            
        # PPO hyperparameters - stabilní
        self.gamma = 0.99   
        self.eps_clip = 0.1  # Menší clipping pro gentle training
        self.entropy_coef = 0.05  # Umírněný entropy bonus
        self.max_grad_norm = 1.0
        self.min_loss_threshold = 0.01
        
        # Training tracking
        self.gradient_norms = []
        self.successful_episodes = 0
        self.episode_rewards = []
    
    def get_gentle_actions(self, action_params):
        """Bezpečné získání akcí s postupným zvyšováním magnitude"""
        if not self.gentle_training:
            # Standard approach
            mean = action_params[:, :4]  
            scale = torch.clamp(action_params[:, 4:], 0.01, 1.0)
            dist = torch.distributions.Normal(mean, scale)
            actions = dist.sample()
            return actions, dist, scale
        
        # GENTLE APPROACH
        mean = action_params[:, :4]  
        scale = torch.clamp(action_params[:, 4:], 0.01, 0.3)  # Menší variance
        
        dist = torch.distributions.Normal(mean, scale)
        actions = dist.sample()
        
        # GENTLE CLIPPING - postupně zvyšujeme rozsah
        actions = torch.clamp(actions, -self.action_scale, self.action_scale)
        
        return actions, dist, scale
    
    def maybe_increase_action_scale(self, episode_reward):
        """Postupně zvyšuj action scale při úspěchu"""
        if not self.gentle_training:
            return
            
        self.episode_rewards.append(episode_reward)
        
        # Zvyš action scale při stabilně dobrém výkonu
        if episode_reward > 5.0:  # Stabilní let bez crashe
            self.successful_episodes += 1
            
            # Po 3 úspěšných episodes v řadě
            if self.successful_episodes >= 3 and self.action_scale < self.max_action_scale:
                old_scale = self.action_scale
                self.action_scale += self.action_scale_increment
                self.action_scale = min(self.action_scale, self.max_action_scale)
                self.successful_episodes = 0
                print(f"🚀 Action scale increased: {old_scale:.3f} → {self.action_scale:.3f}")
        else:
            self.successful_episodes = 0  # Reset při špatném výkonu
    
    def store_transition(self, obs, action, reward, log_prob, value, done, hidden):
        self.memory.append((obs, action, reward, log_prob, value, done, hidden))
    
    def update_policy(self):
        if len(self.memory) < 10:  # Minimální velikost batch
            return None
            
        # Compute returns using gamma
        returns = []
        R = 0
        for i in reversed(range(len(self.memory))):
            obs, action, reward, log_prob, value, done, hidden = self.memory[i]
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalizace
        
        # Policy update s GENTLE ACTIONS a monitoring
        total_loss = 0
        entropy_loss = 0
        policy_losses = []
        
        for i, (obs, action, reward, old_log_prob, value, done, hidden) in enumerate(self.memory):
            # Forward pass pro nový log_prob
            local_map, self_state = obs
            action_out, _, _ = self.actor(local_map, self_state, hidden)
            
            # Použij gentle action distribution SAME as training
            if self.gentle_training:
                mean = action_out[:, :4]
                scale = torch.clamp(action_out[:, 4:], 0.01, 0.3)  # Stejná variance jako v training
            else:
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
            policy_losses.append(policy_loss.item())
            
            # Entropy bonus pro explorace
            entropy = dist.entropy().sum()
            entropy_loss += entropy
            
            total_loss += policy_loss - self.entropy_coef * entropy
            ratio = torch.exp(new_log_prob - old_log_prob.detach())
            clipped_ratio = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
            policy_losses.append(policy_loss.item())
            
            # Entropy bonus pro explorace
            entropy = dist.entropy().sum()
            entropy_loss += entropy
            
            total_loss += policy_loss - self.entropy_coef * entropy
        
        # *** OPRAVENÁ COLLAPSE DETECTION ***
        # NEDETEKUJ collapse když se model ZLEPŠUJE!
        if len(policy_losses) > 15:  # Čekej víc dat
            recent_avg = np.mean([abs(x) for x in policy_losses[-5:]])  # Posledních 5
            older_avg = np.mean([abs(x) for x in policy_losses[-10:-5]])  # Předchozích 5
            
            # Collapse = loss je malý A nedělá pokrok
            if recent_avg < 0.01 and abs(recent_avg - older_avg) < 0.001:
                print(f"⚠️ REAL POLICY COLLAPSE! Recent: {recent_avg:.4f}, Progress: {abs(recent_avg - older_avg):.4f}")
                return total_loss.item()  # Ukonči training
            else:
                print(f"✅ Model learning OK. Recent loss: {recent_avg:.3f}, Progress: {abs(recent_avg - older_avg):.4f}")
        
        # Backprop s pokročilým gradient monitoringem
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Monitor gradients PŘED clippingem
        total_norm = 0.0
        for p in self.actor.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # Varování při exploding gradients
        if total_norm > 100.0:
            print(f"⚠️ EXPLODING GRADIENTS! Norm: {total_norm:.1f} - možný problém s rewards")
            # Zmírni gradienty místo ořezání
            for p in self.actor.parameters():
                if p.grad is not None:
                    p.grad.data /= (total_norm / 10.0)  # Normalize místo clip
        
        # Normální clipping jen při mírném overflow
        clipped_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        if clipped_norm > self.max_grad_norm and total_norm < 100.0:
            print(f"🔧 Normal gradients clipped from {clipped_norm:.3f} to {self.max_grad_norm}")
        
        self.optimizer.step()
        
        # Learning rate decay každých 20 updates (místo 10)
        self.update_count += 1
        if self.update_count % 20 == 0:
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] > 1e-6:  # Minimum LR
                    param_group['lr'] *= 0.99  # Ještě pomalejší decay
        
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
    
    # Trainer - s gentle training enabled
    trainer = SimplePPOTrainer(actor, device, lr=3e-5, gentle_training=True)
    
    # Training settings
    max_episodes = 100  # Více episodes pro gentle progression
    max_steps = 150
    save_every = 20
    
    save_dir = f"models/gentle_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 Modely se ukládají do: {save_dir}")
    
    # Training stats
    episode_rewards = deque(maxlen=10)
    best_reward = -float('inf')
    crash_count = 0
    stable_episodes = 0
    
    # Progress bar pro epizody
    episode_pbar = tqdm(range(max_episodes), desc="🚁 Gentle Training", leave=True)
    
    for episode in episode_pbar:
        # Reset environment
        obs_dict, _ = env.reset()
        hidden_state = torch.zeros(4, 128).to(device)  # 4 drony, každý má 128 hidden dim
        episode_reward = 0
        episode_steps = 0
        episode_crashed = False
        
        for step in range(max_steps):
            try:
                # Extract observations
                if "quads" in obs_dict:
                    # Všech 4 dronů
                    local_map = torch.FloatTensor(obs_dict["quads"]["local_map"]).to(device)  # (4, 1, 32, 32)
                    self_state = torch.FloatTensor(obs_dict["quads"]["self_state"]).to(device)  # (4, 6)
                else:
                    break  # No quad agent
                
                # Forward pass
                action_params, message, new_hidden = actor(local_map, self_state, hidden_state)
                
                # *** GENTLE ACTIONS - použij trainer method ***
                actions, dist, scale = trainer.get_gentle_actions(action_params)
                log_prob = dist.log_prob(actions).sum()
                
                # Simple value estimation
                value = torch.tensor(0.0, device=device)
                
                # Step environment - AKCE PRO VŠECH 4 DRONŮ
                action_np = actions.cpu().numpy()  # Already sampled and clipped by gentle method
                result = env.step({"quads": {"action": action_np}})
                obs_dict, reward, done, info = result
                
                # Store transition
                trainer.store_transition(
                    (local_map, self_state), actions, reward, log_prob, value, done, hidden_state
                )
                
                # Update
                hidden_state = new_hidden
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    if reward < -5:  # Crashed episode
                        episode_crashed = True
                        crash_count += 1
                    break
                    
            except Exception as e:
                print(f"❌ Chyba v kroku {step}: {e}")
                episode_crashed = True
                break
        
        # === GENTLE PROGRESSION ===
        trainer.maybe_increase_action_scale(episode_reward)
        
        # Update policy každých 3 epizod - stabilnější
        if episode % 3 == 0 and episode > 0:
            loss = trainer.update_policy()
            if loss:
                episode_pbar.write(f"📊 Ep {episode}: Policy updated, Loss: {loss:.3f}")
            trainer.memory.clear()
        
        # Stats s crash tracking
        episode_rewards.append(episode_reward)
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Track stability
        if episode_reward > 0 and not episode_crashed:
            stable_episodes += 1
        else:
            stable_episodes = 0
            
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        crash_rate = crash_count / (episode + 1) * 100
        
        # Update tqdm popis s gentle training info
        episode_pbar.set_postfix({
            'Reward': f"{episode_reward:.1f}",
            'Avg': f"{avg_reward:.1f}", 
            'Scale': f"{trainer.action_scale:.2f}",
            'Crash%': f"{crash_rate:.1f}",
            'Stable': stable_episodes
        })
        
        # Success condition - stable flying with reasonable action scale
        if stable_episodes >= 10 and trainer.action_scale >= 0.4:
            episode_pbar.write(f"🏆 SUCCESS! Stable flight achieved with action scale {trainer.action_scale:.3f}")
            episode_pbar.write(f"    Stable episodes: {stable_episodes}, Crash rate: {crash_rate:.1f}%")
            
        # Early stopping pokud moc crashuje
        if episode > 20 and crash_rate > 80:
            episode_pbar.write(f"🛑 Too many crashes ({crash_rate:.1f}%) - ending training")
            break
        
        # Save checkpoint s gentle training info
        if (episode + 1) % save_every == 0:
            checkpoint = {
                'episode': episode,
                'actor_state_dict': actor.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'action_scale': trainer.action_scale,
                'episode_rewards': list(episode_rewards),
                'crash_count': crash_count,
                'best_reward': best_reward
            }
            torch.save(checkpoint, f"{save_dir}/checkpoint_ep{episode+1:03d}.pt")
            episode_pbar.write(f"💾 Model uložen (Ep {episode+1}, Scale: {trainer.action_scale:.3f})")
    
    # Final save s complete stats
    final_checkpoint = {
        'episode': max_episodes,
        'actor_state_dict': actor.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'action_scale': trainer.action_scale,
        'episode_rewards': list(episode_rewards),
        'crash_count': crash_count,
        'stable_episodes': stable_episodes,
        'best_reward': best_reward,
        'final_crash_rate': crash_rate
    }
    torch.save(final_checkpoint, f"{save_dir}/final_model.pt")
    
    # Uzavři progress bar
    episode_pbar.close()
    
    print(f"\n🎉 GENTLE TRAINING DOKONČEN!")
    print(f"   Celkem epizod: {episode+1}")
    print(f"   Nejlepší reward: {best_reward:.1f}")
    print(f"   Final action scale: {trainer.action_scale:.3f}")
    print(f"   Crash rate: {crash_rate:.1f}%")
    print(f"   Stable episodes in row: {stable_episodes}")
    print(f"   Modely uloženy v: {save_dir}")
    
    env.close()
    return save_dir

def create_demo_visualization(env, frame_num):
    """Vytvoří vizualizaci aktuálního stavu prostředí s drony a ohněm"""
    try:
        # Vytvoř output složku
        output_dir = 'output/demo_frames'
        os.makedirs(output_dir, exist_ok=True)
        
        # Získej fire state
        fire_state = env.sim.environment.get_fire_state()
        if not fire_state:
            return None
            
        state = fire_state['fire_grid_state']
        x_min, x_max, y_min, y_max = env.sim.environment.grid_mapper.get_grid_bounds()
        H, W = state['B'].shape
        extent = [x_min, x_max, y_min, y_max]
        
        # Vytvoř figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax1, ax2 = axes
        
        # === PANEL 1: Mapa + Oheň + Drony ===
        burn_rate = env.sim.environment.fire_grid.fuel_burn_rate
        env_img = np.zeros((H, W, 3))
        
        # Terrain colors
        mask_water = (burn_rate == 0.0) 
        mask_building = (burn_rate > 0.0) & (burn_rate <= 0.0002)
        mask_forest = (burn_rate > 0.0002) & (burn_rate <= 0.0005)
        mask_grass = (burn_rate >= 0.0005)

        env_img[mask_grass] = [0.6, 0.7, 0.4]    # Grass
        env_img[mask_forest] = [0.1, 0.4, 0.1]   # Forest
        env_img[mask_water] = [0.2, 0.5, 0.9]    # Water
        env_img[mask_building] = [0.5, 0.5, 0.5] # Buildings
        
        # Fire layer
        burning = state['B']
        fire_overlay = np.zeros((H, W, 4))
        fire_overlay[burning] = [1.0, 0.2, 0.0, 0.8] # Red fire
        
        ax1.imshow(env_img, origin='lower', extent=extent)
        ax1.imshow(fire_overlay, origin='lower', extent=extent)
        
        # Drony - zobraz pozice
        drone_colors = ['blue', 'green', 'red', 'purple']
        for i, (drone_name, drone) in enumerate(env.sim.drones.items()):
            pos = drone.get_position()
            color = drone_colors[i % len(drone_colors)]
            ax1.scatter(pos[0], pos[1], c=color, s=100, marker='o', edgecolors='white', linewidth=2, 
                       label=f'{drone_name} (h={pos[2]:.1f}m)', zorder=10)
        
        # Wind arrow
        wind_vel = env.sim.environment.weather['wind_velocity']
        wind_speed = np.linalg.norm(wind_vel[:2])
        if wind_speed > 0.1:
            arrow_x = x_max - (x_max - x_min) * 0.1
            arrow_y = y_max - (y_max - y_min) * 0.1
            direction = wind_vel[:2] / wind_speed
            visual_length = 8  # Kratší šipka
            dx = direction[0] * visual_length
            dy = direction[1] * visual_length
            ax1.arrow(arrow_x - 5, arrow_y - 5, dx, dy, 
                      head_width=2, head_length=2, fc='yellow', ec='black', width=1, zorder=9)
            
            txt = ax1.text(arrow_x + 2, arrow_y + 2, f"{wind_speed:.1f} m/s", color='yellow', 
                         fontsize=8, ha='center', zorder=11)
            txt.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground="black")])
        
        ax1.set_title('Mapa + Oheň + Drony')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # === PANEL 2: Palivo ===
        fuel = state['F']
        im2 = ax2.imshow(fuel, origin='lower', extent=extent, cmap='YlOrRd', vmin=0, vmax=1)
        ax2.set_title('Zbývající palivo')
        ax2.set_xlabel('X [m]')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.set_label('Palivo (0-1)')
        ax2.grid(True, alpha=0.3)
        
        # Statistiky
        burning_count = np.sum(burning)
        total_fuel = np.sum(fuel)
        active_drones = len(env.sim.drones)
        sim_time = env.sim.simulation_time
        
        plt.suptitle(f'Demo Frame {frame_num} | Čas: {sim_time:.1f}s | Hořící buňky: {burning_count} | Drony: {active_drones}/4', 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Ulož
        filename = f'{output_dir}/demo_frame_{frame_num:03d}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename
        
    except Exception as e:
        print(f"❌ Chyba při vytváření vizualizace: {e}")
        return None

def demo_main():
    """Spustí demo natrénovaného modelu"""
    print("🎬 DEMO GENTLE TRAINED MODELU")
    
    # FORCE použij dobrý model místo nejnovějšího
    model_dir = "models/gentle_training_20260210_211518"  # Dobrý model s reward 65.88
    
    if not os.path.exists(model_dir):
        print(f"❌ Dobrý model {model_dir} neexistuje! Hledám nejnovější...")
        # Fallback na automatický nejnovější
        models_dir = "models"
        if not os.path.exists(models_dir):
            print(f"❌ Složka {models_dir}/ neexistuje!")
            return
        
        # Najdi všechny gentle training složky
        gentle_dirs = [d for d in os.listdir(models_dir) 
                       if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("gentle_training_")]
        
        if not gentle_dirs:
            print("❌ Žádné gentle training modely nenalezeny!")
            return
        
        # Seřaď podle času (nejnovější poslední)
        gentle_dirs.sort()
        latest_gentle_dir = gentle_dirs[-1]
        model_dir = f"{models_dir}/{latest_gentle_dir}"
        
        print(f"🔍 Použiji nejnovější: {latest_gentle_dir}")
    else:
        print(f"🎯 Použiji DOBRÝ model: gentle_training_20260210_211518")
    
    # Preferuj final_model.pt, pak nejnovější checkpoint
    model_path = f"{model_dir}/final_model.pt"
    
    if not os.path.exists(model_path):
        # Fallback na nejnovější checkpoint
        checkpoints = [f for f in os.listdir(model_dir) if f.startswith("checkpoint_")]
        if checkpoints:
            checkpoints.sort()
            model_path = f"{model_dir}/{checkpoints[-1]}"
            print(f"📥 Použiji nejnovější checkpoint: {checkpoints[-1]}")
        else:
            print("❌ Žádné modely nalezeny!")
            return
    else:
        print(f"📥 Načítám final model: {model_path}")
    
    # Load model
    model = QuadActor(message_dim=8)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['actor_state_dict'])
    model.eval()
    
    # Načti action scale z modelu pokud existuje
    action_scale = checkpoint.get('action_scale', 0.5)  # Default na 0.5
    print(f"🎯 Action scale: {action_scale:.3f}")
    
    # Zobraz statistiky modelu
    if 'crash_count' in checkpoint:
        crash_count = checkpoint['crash_count']
        episode_count = checkpoint.get('episode', 100)
        crash_rate = (crash_count / episode_count) * 100
        print(f"📊 Model stats: Crash rate {crash_rate:.1f}%, Best reward {checkpoint.get('best_reward', 'N/A')}")
    
    # Environment - 4 drony v rozích 50x50m mapy
    env = WildfireMARLEnv(agents_config=["quad_1", "quad_2", "quad_3", "quad_4"])
    
    # Definuj start pozice pro demo
    start_positions = [
        ("Levý dolní", [-20, -20, 8]),
        ("Pravý dolní", [20, -20, 8]),  
        ("Pravý horní", [20, 20, 8]),
        ("Levý horní", [-20, 20, 8])
    ]
    
    # Run demo se všemi 4 drony současně + vizualizace
    print(f"\n🚁 === DEMO: Všechny 4 drony současně ===\n")
    
    # Environment se všemi 4 drony
    env = WildfireMARLEnv(agents_config=["quad_1", "quad_2", "quad_3", "quad_4"])
    obs, _ = env.reset()
    
    # Přestav drony na počáteční pozice
    import pybullet as p
    for i, (position_name, start_pos) in enumerate(start_positions):
        drone_name = f"quad_{i+1}"
        if drone_name in env.sim.drones:
            drone_id = env.sim.drones[drone_name].drone_id
            p.resetBasePositionAndOrientation(drone_id, start_pos, [0, 0, 0, 1])
    
    obs = env._get_obs()  # Aktualizuj observace
    
    # Hidden states pro všechny drony
    hidden_states = torch.zeros(4, 128)
    total_reward = 0
    steps = 0
    frame_count = 0
    last_viz_time = time.time()
    
    print("🎬 Spouštím DLOUHÉ demo s vizualizací každých 0.8s...")
    
    # První frame
    try:
        viz_path = create_demo_visualization(env, frame_count)
        if viz_path:
            print(f"📷 Frame {frame_count}: {viz_path}")
            frame_count += 1
    except Exception as e:
        print(f"⚠️ Chyba při vizualizaci frame 0: {e}")
    
    while steps < 4500:  # Kratší demo - 5 sekund ale s detailem
        try:
            if "quads" in obs:
                # *** VŠECHNY 4 DRONY ***
                local_map = torch.FloatTensor(obs["quads"]["local_map"])  # (4, 1, 32, 32)
                self_state = torch.FloatTensor(obs["quads"]["self_state"])  # (4, 6)
            else:
                break
            
            with torch.no_grad():
                actions, _, hidden_states = model(local_map, self_state, hidden_states)
                
            # Sample action s gentle training action scale
            mean = actions[:, :4]  # (4, 4)
            scale = torch.clamp(actions[:, 4:], 0.01, 0.3)  # Stejná variance jako při training
            action_sample = torch.normal(mean, scale)
            
            # GENTLE CLIPPING - použij action scale z modelu
            actions_clipped = torch.clamp(action_sample, -action_scale, action_scale)
            actions_np = actions_clipped.numpy()  # (4, 4)
            
            # ===== AKCE PRO VŠECHNY DRONY =====
            action_dict = {
                "quads": {
                    "action": actions_np  # (4, 4)
                }
            }
            
            obs, reward, done, info = env.step(action_dict)
            total_reward += reward
            steps += 1
            
            # Vizualizace každých 450 kroků (= 15s * 30fps)
            if steps % 450 == 0 and steps > 0:  # Každých 450 kroků = 10 snímků za 4500 kroků  
                try:
                    viz_path = create_demo_visualization(env, frame_count)
                    if viz_path:
                        print(f"📷 Frame {frame_count} (krok {steps}): {viz_path}")
                        frame_count += 1
                except Exception as e:
                    print(f"⚠️ Chyba při vizualizaci frame {frame_count}: {e}")
            
            # Progress každých 20 kroků
            if steps % 20 == 0:
                active_drones = len(env.sim.drones)
                fire_distances = []
                out_of_bounds = 0
                for drone_name, drone in env.sim.drones.items():
                    pos = drone.get_position()
                    fire_distance = ((pos[0])**2 + (pos[1])**2)**0.5
                    fire_distances.append(fire_distance)
                    
                    # Kontrola hranic mapy (50x50m = ±25m)
                    if abs(pos[0]) > 25 or abs(pos[1]) > 25:
                        out_of_bounds += 1
                
                if fire_distances:
                    avg_distance = np.mean(fire_distances)
                    boundary_warning = f" ⚠️ {out_of_bounds} dronů mimo mapu!" if out_of_bounds > 0 else ""
                    
                    # Fire info
                    fire_state = env.sim.environment.get_fire_state()
                    burning_cells = 0
                    if fire_state:
                        burning_cells = np.sum(fire_state['fire_grid_state']['B'])
                    fire_info = f" | 🔥 {burning_cells} hořících buněk"
                    
                    print(f"   Krok {steps}: Aktivní drony={active_drones}, Prům.vzdálenost={avg_distance:.1f}m, Reward={reward:.2f}{boundary_warning}{fire_info}")
            
            if done:
                break
                
        except Exception as e:
            print(f"❌ Chyba v kroku {steps}: {e}")
            break
    
    # Finální frame
    try:
        viz_path = create_demo_visualization(env, frame_count)
        if viz_path:
            print(f"📷 Finální frame {frame_count}: {viz_path}")
    except Exception as e:
        print(f"⚠️ Chyba při finální vizualizaci: {e}")
    
    # Výsledky
    active_drones = len(env.sim.drones)
    destroyed_count = len(env.sim.destroyed_drones)
    
    print(f"\n✅ Demo dokončeno!")
    print(f"   🚁 Aktivní drony: {active_drones}/4")
    print(f"   💥 Zničené drony: {destroyed_count}")
    print(f"   🏆 Celkový reward: {total_reward:.1f}")
    print(f"   📷 Vytvořeno {frame_count + 1} snímků")
    
    if frame_count > 0:
        print(f"   📁 Snímky uloženy v: output/demo_frames/")
    
    env.close()

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