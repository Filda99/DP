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

# TorchRL imports for proper PPO (simplified)
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patheffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config import MainConfig, WildfireGymConfig

def create_demo_visualization(demo_log, env):
    """Vytvoří vizualizaci demo běhů s trajektoriemi a rewards"""
    import os
    
    # Vytvoř výstupní složku
    os.makedirs("output", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=MainConfig.VISUALIZATION_FIGSIZE)
    
    # === TRAJEKTORIE DRONŮ ===
    ax1.set_title("Trajektorie dronů - Demo běhy", fontsize=14, fontweight='bold')
    
    colors = MainConfig.DRONE_COLORS[:3]  # Use first 3 colors
    
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
    fire_positions = [MainConfig.FIRE_POSITION_CENTER]  # 1 oheň uprostřed 50x50m mapy
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
    plt.savefig(filepath, dpi=MainConfig.VISUALIZATION_DPI, bbox_inches='tight')
    plt.close()
    
    return filepath

class TorchRLTrainer:
    """TorchRL-based PPO Trainer - SIMPLIFIED FOR COMPATIBILITY"""
    def __init__(self, actor, critic, device='cpu', lr=MainConfig.LEARNING_RATE, gentle_training=False):
        self.device = device
        self.gentle_training = gentle_training
        self.action_scale = 0.3 if gentle_training else 1.0  # Start conservative if gentle
        
        # TorchRL PPO Loss Module
        self.loss_module = ClipPPOLoss(
            actor_network=actor_module,
            critic_network=critic_module,
            clip_epsilon=MainConfig.EPS_CLIP,
            entropy_coeff=MainConfig.ENTROPY_COEF,
            normalize_advantage=False,  # Better for MARL
        )
        
        # Set appropriate keys for our environment
        self.loss_module.set_keys(
            reward="reward",
            action="action", 
            value="state_value",
            done="done",
            terminated="terminated",
        )
        
        # GAE Value Estimator - much better than our primitive returns
        self.loss_module.make_value_estimator(
            ValueEstimators.GAE, 
            gamma=MainConfig.GAMMA, 
            lmbda=0.9  # GAE lambda parameter
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.loss_module.parameters(), 
            lr=lr, 
            weight_decay=1e-5
        )
        
        # Training tracking
        self.update_count = 0
        self.successful_episodes = 0
    
    def get_gentle_actions(self, raw_actions):
        """Bezpečné clipping akcí s postupným zvyšováním magnitude"""
        # Standard approach - use tanh to bound actions
        actions = torch.tanh(raw_actions)
        
        # Apply gentle scaling if enabled
        if self.gentle_training:
            actions = actions * self.action_scale
        
        return actions
    
    def increase_action_scale(self, reward_improvement=False):
        """Postupně zvyšuje action scale při úspěšných epizodách"""
        if self.gentle_training and reward_improvement and self.action_scale < 1.0:
            self.action_scale = min(1.0, self.action_scale + 0.05)
            return True
        return False
    
    def prepare_batch_data(self, memory_transitions):
        """Convert our memory format to TorchRL TensorDict format"""
        if len(memory_transitions) < 10:
            return None
            
        batch_size = len(memory_transitions)
        
        # Extract components from memory
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        
        for obs, action, reward, log_prob, value, done, hidden in memory_transitions:
            local_map, self_state = obs
            observations.append({
                'local_map': local_map.cpu(),
                'self_state': self_state.cpu()
            })
            actions.append(action.cpu())
            rewards.append(reward)
            dones.append(done)
            values.append(value if isinstance(value, (int, float)) else value.cpu())
        
        # Create TensorDict batch
        batch_data = TensorDict({
            'local_map': torch.stack([obs['local_map'] for obs in observations]),
            'self_state': torch.stack([obs['self_state'] for obs in observations]),
            'action': torch.stack(actions),
            'reward': torch.tensor(rewards, dtype=torch.float32),
            'done': torch.tensor(dones, dtype=torch.bool),
            'terminated': torch.tensor(dones, dtype=torch.bool),  # For simplicity
            'state_value': torch.tensor(values, dtype=torch.float32),
        }, batch_size=[batch_size]).to(self.device)
        
        return batch_data
    
    def update_policy(self, memory_transitions):
        """TorchRL-based PPO update"""
        # Convert memory to TensorDict format
        batch_data = self.prepare_batch_data(memory_transitions)
        if batch_data is None:
            return None
            
        try:
            # Compute GAE advantages - this is much better than our primitive approach
            with torch.no_grad():
                self.loss_module.value_estimator(
                    batch_data,
                    params=self.loss_module.critic_network_params,
                    target_params=self.loss_module.target_critic_network_params,
                )
            
            # PPO Loss computation - proper clipped objective
            loss_vals = self.loss_module(batch_data)
            total_loss = (
                loss_vals["loss_objective"] +  # Clipped policy loss
                loss_vals["loss_critic"] +     # Value function loss  
                loss_vals["loss_entropy"]      # Entropy bonus
            )
            
            # Backprop with gradient clipping
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.loss_module.parameters(), 
                MainConfig.MAX_GRAD_NORM
            )
            self.optimizer.step()
            
            # Learning rate decay
            self.update_count += 1
            if self.update_count % 20 == 0:
                for param_group in self.optimizer.param_groups:
                    if param_group['lr'] > 1e-6:
                        param_group['lr'] *= 0.99
            
            return total_loss.item()
            
        except Exception as e:
            print(f"❌ TorchRL update error: {e}")
            return None

class SimpleCritic(torch.nn.Module):
    """Simple critic network for value estimation"""
    def __init__(self, self_state_size, hidden_size=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self_state_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size//2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size//2, 1)  # Single value output
        )
    
    def forward(self, self_state):
        return self.net(self_state)

def train_main():
    """Spustí trénink kvadrokoptéry"""
    print("🚁 SPOUŠTÍM TRÉNINK KVADROKOPTÉRY...")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")
    
    # Environment - 1 dron na mapě s hranicemi definovanými v prostředí
    marlEnv = WildfireMARLEnv(agents_config=["quad_1"])
    
    # Get self_state size from environment's observation processor
    self_state_size = marlEnv.obs_proc.get_self_state_size()
    
    # Actor and Critic networks
    actor = QuadActor(message_dim=MainConfig.ACTOR_MESSAGE_DIM, self_state_size=self_state_size).to(device)
    critic = SimpleCritic(self_state_size=self_state_size).to(device)
    
    # TorchRL Trainer - much better than our fake PPO
    trainer = TorchRLTrainer(
        actor=actor,
        critic=critic, 
        device=device, 
        lr=MainConfig.LEARNING_RATE, 
        gentle_training=True
    )
    
    # Training settings
    max_episodes = MainConfig.MAX_EPISODES
    max_steps = MainConfig.MAX_STEPS
    save_every = MainConfig.SAVE_EVERY
    
    save_dir = f"models/gentle_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 Modely se ukládají do: {save_dir}")
    
    # Training stats
    episode_rewards = deque(maxlen=10)
    best_reward = -float('inf')
    crash_count = 0
    stable_episodes = 0
    memory_buffer = []  # Store transitions for TorchRL
    
    # Progress bar pro epizody
    episode_pbar = tqdm(range(max_episodes), desc="🚁 TorchRL Training", leave=True)
    
    for episode in episode_pbar:
        # Reset environment
        obs_dict, _ = marlEnv.reset()
        hidden_state = torch.zeros(1, MainConfig.ACTOR_HIDDEN_SIZE).to(device)  # 1 dron
        episode_reward = 0
        episode_steps = 0
        episode_crashed = False
        
        for step in range(max_steps):
            try:
                # Extract observations
                if "quads" in obs_dict:
                    # Jeden dron
                    local_map = torch.FloatTensor(obs_dict["quads"]["local_map"]).to(device)  # (1, 1, 32, 32)
                    self_state = torch.FloatTensor(obs_dict["quads"]["self_state"]).to(device)  # (1, 6)
                else:
                    break  # No quad agent
                
                # Forward pass
                raw_actions, message, new_hidden = actor(local_map, self_state, hidden_state)
                
                # *** GENTLE ACTIONS - použij trainer method ***
                actions = trainer.get_gentle_actions(raw_actions)
                
                # Value estimation from critic
                with torch.no_grad():
                    value = critic(self_state).item()
                
                # Step environment - AKCE PRO 1 DRON
                action_np = actions.cpu().detach().numpy()  # Detach gradients before converting to numpy
                result = marlEnv.step({"quads": {"action": action_np}})
                obs_dict, reward, done, info = result
                
                # Store transition for TorchRL
                memory_buffer.append((
                    (local_map, self_state), actions, reward, 0.0, value, done, hidden_state
                ))
                
                # Update
                hidden_state = new_hidden
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    # Check if episode ended due to crash (altitude < 0.5m) rather than just low reward
                    drone_crashed = False
                    if "quads" in obs_dict:
                        for i in range(obs_dict["quads"]["self_state"].shape[0]):
                            altitude = obs_dict["quads"]["self_state"][i][0]  # First feature is altitude
                            if altitude < 0.5:
                                drone_crashed = True
                                break
                    
                    if drone_crashed or reward < -10:  # Actual crash or severe penalty
                        episode_crashed = True
                        crash_count += 1
                    break
                    
            except Exception as e:
                print(f"❌ Chyba v kroku {step}: {e}")
                episode_crashed = True
                break
        
        # Update policy každé 3 epizody - častější updates pro lepší learning
        if episode % 3 == 0 and episode > 0 and len(memory_buffer) > 0:
            loss = trainer.update_policy(memory_buffer)
            if loss:
                episode_pbar.write(f"📊 Ep {episode}: TorchRL PPO updated, Loss: {loss:.3f}")
            memory_buffer.clear()  # Clear after update
        
        # Stats s crash tracking
        episode_rewards.append(episode_reward)
        reward_improved = episode_reward > best_reward
        if reward_improved:
            best_reward = episode_reward
        
        # Track stability and increase action scale if improving
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
            
            # Také uložit jako nejnovější model pro demo
            newest_dir = "models/newest"
            os.makedirs(newest_dir, exist_ok=True)
            torch.save(checkpoint, f"{newest_dir}/latest_model.pt")
            episode_pbar.write(f"💾 Model uložen (Ep {episode+1}, Scale: {trainer.action_scale:.3f}) a zkopírován do newest/")
    
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
    
    # Uložit finální model také jako nejnovější pro demo
    newest_dir = "models/newest"
    os.makedirs(newest_dir, exist_ok=True)
    torch.save(final_checkpoint, f"{newest_dir}/latest_model.pt")
    
    # Uzavři progress bar
    episode_pbar.close()
    
    print(f"\n🎉 TORCHRL TRAINING DOKONČEN!")
    print(f"   Celkem epizod: {episode+1}")
    print(f"   Nejlepší reward: {best_reward:.1f}")
    print(f"   Final action scale: {trainer.action_scale:.3f}")
    print(f"   Crash rate: {crash_rate:.1f}%")
    print(f"   Stable episodes in row: {stable_episodes}")
    print(f"   Modely uloženy v: {save_dir}")
    
    marlEnv.close()
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

        env_img[mask_grass] = MainConfig.TERRAIN_COLORS['grass']    # Grass
        env_img[mask_forest] = MainConfig.TERRAIN_COLORS['forest']   # Forest
        env_img[mask_water] = MainConfig.TERRAIN_COLORS['water']    # Water
        env_img[mask_building] = MainConfig.TERRAIN_COLORS['building'] # Buildings
        
        # Fire layer
        burning = state['B']
        fire_overlay = np.zeros((H, W, 4))
        fire_overlay[burning] = MainConfig.FIRE_OVERLAY_COLOR # Red fire
        
        ax1.imshow(env_img, origin='lower', extent=extent)
        ax1.imshow(fire_overlay, origin='lower', extent=extent)
        
        # Drony - zobraz pozice
        for i, (drone_name, drone) in enumerate(env.sim.drones.items()):
            pos = drone.get_position()
            color = MainConfig.DRONE_COLORS[i % len(MainConfig.DRONE_COLORS)]
            ax1.scatter(pos[0], pos[1], c=color, s=100, marker='o', edgecolors='white', linewidth=2, 
                       label=f'{drone_name} (h={pos[2]:.1f}m)', zorder=10)
        
        # Wind arrow
        wind_vel = env.sim.environment.weather['wind_velocity']
        wind_speed = np.linalg.norm(wind_vel[:2])
        if wind_speed > 0.1:
            arrow_x = x_max - (x_max - x_min) * 0.1
            arrow_y = y_max - (y_max - y_min) * 0.1
            direction = wind_vel[:2] / wind_speed
            visual_length = MainConfig.WIND_ARROW_LENGTH
            dx = direction[0] * visual_length
            dy = direction[1] * visual_length
            ax1.arrow(arrow_x - 5, arrow_y - 5, dx, dy, 
                      head_width=MainConfig.WIND_ARROW_HEAD_WIDTH, 
                      head_length=MainConfig.WIND_ARROW_HEAD_LENGTH, 
                      fc=MainConfig.WIND_ARROW_COLOR, 
                      ec=MainConfig.WIND_ARROW_EDGE_COLOR, 
                      width=MainConfig.WIND_ARROW_WIDTH, zorder=9)
            
            txt = ax1.text(arrow_x + 2, arrow_y + 2, f"{wind_speed:.1f} m/s", color='yellow', 
                         fontsize=8, ha='center', zorder=11)
            txt.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground="black")])
        
        ax1.set_title('Mapa + Oheň + Drony')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # === PANEL 2: Real-time Rewards ===
        # Sleduj reward z posledních kroků
        recent_rewards = getattr(env, '_recent_rewards', [0] * 20)
        recent_rewards = recent_rewards[-20:]  # Posledních 20 kroků
        
        ax2.clear()
        steps_x = range(len(recent_rewards))
        ax2.plot(steps_x, recent_rewards, 'b-', linewidth=2, label='Reward za krok')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Dobrý reward (5+)')
        ax2.axhline(y=-5, color='red', linestyle='--', alpha=0.5, label='Špatný reward (-5)')
        
        ax2.set_title(f'Real-time Rewards (posledních {len(recent_rewards)} kroků)')
        ax2.set_xlabel('Kroky zpět')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Statistiky o rewards
        if recent_rewards:
            avg_reward = np.mean(recent_rewards)
            ax2.text(0.02, 0.98, f'Avg: {avg_reward:.2f}', transform=ax2.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        # Statistiky
        burning_count = np.sum(burning)
        active_drones = len(env.sim.drones)
        sim_time = env.sim.simulation_time
        
        # Current reward z tracking
        current_reward = env._recent_rewards[-1] if env._recent_rewards else 0.0
        
        plt.suptitle(f'Demo Frame {frame_num} | Čas: {sim_time:.1f}s | Hořící buňky: {burning_count} | Drony: {active_drones}/1 | Reward: {current_reward:.2f}', 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Ulož
        filename = f'{output_dir}/demo_frame_{frame_num:03d}.png'
        plt.savefig(filename, dpi=MainConfig.DEMO_VISUALIZATION_DPI, bbox_inches='tight')
        plt.close()
        
        return filename
        
    except Exception as e:
        print(f"❌ Chyba při vytváření vizualizace: {e}")
        return None

def demo_main():
    """Spustí demo natrénovaného modelu"""
    print("🎬 DEMO GENTLE TRAINED MODELU")
    
    # Použij nejnovější model z models/newest/
    model_path = "models/newest/latest_model.pt"
    
    if not os.path.exists(model_path):
        print("❌ Nejnovější model nenalezen! Spusťte nejdříve train.")
        print(f"   Očekávaný soubor: {model_path}")
        return
    
    print(f"📥 Načítám nejnovější model: {model_path}")
    
    # Load model with proper configuration
    # First create env to get configuration
    temp_env = WildfireMARLEnv(agents_config=["quad_1"])
    self_state_size = temp_env.obs_proc.get_self_state_size()
    temp_env.close()
    
    model = QuadActor(message_dim=MainConfig.ACTOR_MESSAGE_DIM, self_state_size=self_state_size)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['actor_state_dict'])
    model.eval()
    
    # Načti action scale z modelu ale SYNCHRONIZOVÁNO s reward thresholds
    model_action_scale = checkpoint.get('action_scale', 0.5)
    # Use model's actual action scale WITHOUT limiting for demo - let's see what happens
    action_scale = model_action_scale  # Use full trained scale
    print(f"🎯 Action scale z modelu: {model_action_scale:.3f} → použito {action_scale:.3f} pro demo (bez omezení!)")
    
    # Zobraz statistiky modelu
    if 'crash_count' in checkpoint:
        crash_count = checkpoint['crash_count']
        episode_count = checkpoint.get('episode', 100)
        crash_rate = (crash_count / episode_count) * 100
        print(f"📊 Model stats: Crash rate {crash_rate:.1f}%, Best reward {checkpoint.get('best_reward', 'N/A')}")
    
    # Environment with unlimited runtime for demo
    env = WildfireMARLEnv(agents_config=["quad_1"], demo_mode=True)
    obs, _ = env.reset()
    
    # Přidej reward tracking pro vizualizaci
    env._recent_rewards = []
    
    print("🚁 === DEMO: 1 dron na mapě (konzistentní s tréninkem) ===")
    
    # Hidden states pro 1 dron
    hidden_states = torch.zeros(1, MainConfig.ACTOR_HIDDEN_SIZE)
    total_reward = 0
    steps = 0
    frame_count = 0
    
    # První frame
    try:
        viz_path = create_demo_visualization(env, frame_count)
        if viz_path:
            print(f"📷 Frame {frame_count}: {viz_path}")
            frame_count += 1
    except Exception as e:
        print(f"⚠️ Chyba při vizualizaci frame 0: {e}")
    
    max_steps = MainConfig.DEMO_MAX_STEPS 
    while steps < max_steps:
        try:
            if "quads" in obs:
                local_map = torch.FloatTensor(obs["quads"]["local_map"])  # (1, 1, 32, 32)
                self_state = torch.FloatTensor(obs["quads"]["self_state"])  # (1, 6)
            else:
                break
            
            with torch.no_grad():
                raw_actions, _, hidden_states = model(local_map, self_state, hidden_states)
                
                # Direct action output - no distributions needed
                actions = torch.tanh(raw_actions)  # Bound to [-1, 1]
                
                # Apply gentle policy if needed - conservative scaling for safety
                if MainConfig.DEMO_CONSERVATIVE_POLICY:
                    actions = actions * 0.7  # More conservative for demo
                
                actions_clipped = torch.clamp(actions, -1.0, 1.0)  # Standard range
            
            actions_np = actions_clipped.numpy()  # (1, 4)
            
            # DEBUG: Print actions every 100 steps to see what model outputs
            if steps % 100 == 0 and steps < 500:
                # Also debug observation state to see if model gets proper input
                alt = self_state[0][0].item() if len(self_state[0]) > 0 else 0
                
                # BOUNDARY DEBUG - print key boundary info
                if len(self_state[0]) >= 14:
                    boundary_x = self_state[0][12].item()
                    boundary_y = self_state[0][13].item()
                    print(f"🐛 Krok {steps}: actions=[{actions_np[0][0]:.4f}, {actions_np[0][1]:.4f}, {actions_np[0][2]:.4f}, {actions_np[0][3]:.4f}]")
                    print(f"   ↳ Boundary info: X_dist={boundary_x:.1f}, Y_dist={boundary_y:.1f} (negative = outside!)")
                else:
                    print(f"🐛 Krok {steps}: actions=[{actions_np[0][0]:.4f}, {actions_np[0][1]:.4f}, {actions_np[0][2]:.4f}, {actions_np[0][3]:.4f}]")
                print(f"   ↳ altitude={alt:.1f}m, roll/pitch/yaw/throttle")
            
            # ===== AKCE PRO 1 DRON =====
            action_dict = {
                "quads": {
                    "action": actions_np  # (1, 4)
                }
            }
            
            obs, reward, done, info = env.step(action_dict)
            total_reward += reward
            steps += 1
            
            # Track rewards pro vizualizaci
            env._recent_rewards.append(reward)
            if len(env._recent_rewards) > MainConfig.DEMO_RECENT_REWARDS_MAX:  # Drž pouze posledních N kroků
                env._recent_rewards = env._recent_rewards[-MainConfig.DEMO_RECENT_REWARDS_MAX:]
            
            if steps % (max_steps / 10) == 0 and steps > 0:
                try:
                    viz_path = create_demo_visualization(env, frame_count)
                    if viz_path:
                        print(f"📷 Frame {frame_count} (krok {steps}): {viz_path}")
                        frame_count += 1
                except Exception as e:
                    print(f"⚠️ Chyba při vizualizaci frame {frame_count}: {e}")
            
            if steps % 100 == 0:
                active_drones = len(env.sim.drones)
                fire_distances = []
                out_of_bounds = 0
                drone_heights = []
                for drone_name, drone in env.sim.drones.items():
                    pos = drone.get_position()
                    fire_distance = ((pos[0])**2 + (pos[1])**2)**0.5
                    fire_distances.append(fire_distance)
                    drone_heights.append(pos[2])
                    
                    # Check boundary violations using environment's map bounds
                    map_bounds = env.map_bounds  # Use actual environment configuration
                    if abs(pos[0]) > map_bounds or abs(pos[1]) > map_bounds:
                        out_of_bounds += 1
                
                if fire_distances:
                    avg_distance = np.mean(fire_distances)
                    avg_height = np.mean(drone_heights) 
                    boundary_warning = f" ⚠️ {out_of_bounds} dronů mimo mapu!" if out_of_bounds > 0 else ""
                    
                    # Fire info
                    fire_state = env.sim.environment.get_fire_state()
                    burning_cells = 0
                    if fire_state:
                        burning_cells = np.sum(fire_state['fire_grid_state']['B'])
                    fire_info = f" | 🔥 {burning_cells} hořících buněk"
                    
                    print(f"   Krok {steps}: Aktivní drony={active_drones}, Vzdálenost={avg_distance:.1f}m, Výška={avg_height:.1f}m, Reward={reward:.2f}{boundary_warning}{fire_info}")
            
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
    print(f"   🚁 Aktivní drony: {active_drones}/1")
    print(f"   💥 Zničené drony: {destroyed_count}")
    print(f"   🏆 Celkový reward: {total_reward:.1f}")
    print(f"   📷 Vytvořeno {frame_count + 1} snímků")
    
    if frame_count > 0:
        print(f"   📁 Snímky uloženy v: output/demo_frames/")
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description="Quadcopter Fire Detection Training")
    parser.add_argument("command", choices=["train", "demo"], 
                      help="Co spustit")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_main()
    elif args.command == "demo":
        demo_main()

if __name__ == "__main__":
    main()