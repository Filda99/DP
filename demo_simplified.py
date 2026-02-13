#!/usr/bin/env python3
"""
DEMO PRO ZJEDNODUŠENÝ TRAINING - TESTOVÁNÍ NATRÉNOVANÉHO MODELU
"""

import torch
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

from src.wildfire_gym_wrapper import WildfireMARLEnv
from src.wildfire_models import QuadActor
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal
from config import MainConfig

# Vizualizace imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patheffects
from mpl_toolkits.axes_grid1 import make_axes_locatable

class SimpleCritic(torch.nn.Module):
    """Simple critic network for value estimation"""
    def __init__(self, self_state_size, hidden_size=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self_state_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size//2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size//2, 1)
        )
    
    def forward(self, self_state):
        return self.net(self_state)

class PolicyNetwork(torch.nn.Module):
    """Wrapper pro QuadActor aby výstupoval distribution parameters"""
    def __init__(self, actor_net):
        super().__init__()
        self.actor_net = actor_net
        
    def forward(self, observation):
        # Split observation
        local_map_flat = observation[:, :1024]
        self_state = observation[:, 1024:]
        
        # Reshape
        batch_size = observation.shape[0]
        local_map = local_map_flat.reshape(batch_size, 1, 32, 32)
        hidden_state = torch.zeros(batch_size, 128, device=observation.device)
        
        # Get actions
        dist_params, _, _ = self.actor_net(local_map, self_state, hidden_state)
        return dist_params

def setup_networks(device, self_state_size):
    """Nastavení sítí a policy (stejné jako v train_simplified)"""
    print("🧠 Nastavuji sítě pro demo...")
    
    # Base networks
    actor_net_base = QuadActor(message_dim=MainConfig.ACTOR_MESSAGE_DIM, self_state_size=self_state_size).to(device)
    
    # Policy network
    policy_net = torch.nn.Sequential(
        PolicyNetwork(actor_net_base),
        NormalParamExtractor(),
    )
    
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    
    policy = ProbabilisticActor(
        module=policy_module,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=False,  # Pro demo nepotřebujeme log_prob
    )
    
    return policy, actor_net_base

def create_demo_visualization(env, frame_num, recent_rewards):
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
        # Zobraz poslední rewards
        recent_rewards_display = recent_rewards[-20:] if len(recent_rewards) > 20 else recent_rewards
        
        ax2.clear()
        if recent_rewards_display:
            steps_x = range(len(recent_rewards_display))
            ax2.plot(steps_x, recent_rewards_display, 'b-', linewidth=2, label='Reward za krok')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Dobrý reward (5+)')
            ax2.axhline(y=-5, color='red', linestyle='--', alpha=0.5, label='Špatný reward (-5)')
            
            # Avg reward
            avg_reward = np.mean(recent_rewards_display)
            ax2.text(0.02, 0.98, f'Avg: {avg_reward:.2f}', transform=ax2.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        ax2.set_title(f'Real-time Rewards (posledních {len(recent_rewards_display)} kroků)')
        ax2.set_xlabel('Kroky zpět')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Statistiky
        burning_count = np.sum(burning)
        active_drones = len(env.sim.drones)
        sim_time = env.sim.simulation_time
        
        # Current reward
        current_reward = recent_rewards[-1] if recent_rewards else 0.0
        
        plt.suptitle(f'Demo Frame {frame_num} | Čas: {sim_time:.1f}s | Hořící buňky: {burning_count} | Drony: {active_drones}/1 | Reward: {current_reward:.2f}', 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Ulož
        filename = f'{output_dir}/demo_frame_{frame_num:03d}.png'
        plt.savefig(filename, dpi=MainConfig.DEMO_VISUALIZATION_DPI, bbox_inches='tight')
        plt.close()
        
        return filename
        
    except Exception as e:
        print(f"❌ Chyba ve vizualizaci: {e}")
        return None

def create_demo_summary(episode_data):
    """Vytvoří souhrnnou vizualizaci všech epizod"""
    try:
        os.makedirs("output", exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # === PANEL 1: Rewards comparison ===
        for i, (episode_name, data) in enumerate(episode_data.items()):
            rewards = data['rewards']
            if rewards:
                steps = range(len(rewards))
                cumulative = np.cumsum(rewards)
                
                color = MainConfig.DRONE_COLORS[i % len(MainConfig.DRONE_COLORS)]
                ax1.plot(steps, cumulative, '-', color=color, linewidth=2, 
                        label=f'Epizoda {i+1} (total: {cumulative[-1]:.1f})')
        
        ax1.set_title('Kumulativní rewards - srovnání epizod')
        ax1.set_xlabel('Krok')
        ax1.set_ylabel('Kumulativní reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # === PANEL 2: Average performance ===
        all_episode_totals = []
        all_episode_steps = []
        
        for episode_name, data in episode_data.items():
            if data['rewards']:
                total_reward = sum(data['rewards'])
                steps = len(data['rewards'])
                all_episode_totals.append(total_reward)
                all_episode_steps.append(steps)
        
        if all_episode_totals:
            episodes = range(1, len(all_episode_totals) + 1)
            ax2.bar(episodes, all_episode_totals, alpha=0.7, color='skyblue')
            
            # Average line
            avg_reward = np.mean(all_episode_totals)
            ax2.axhline(y=avg_reward, color='red', linestyle='--', 
                       label=f'Průměr: {avg_reward:.1f}')
        
        ax2.set_title('Celkové rewards za epizodu')
        ax2.set_xlabel('Epizoda')
        ax2.set_ylabel('Celkový reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Uložení
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"output/demo_summary_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=MainConfig.VISUALIZATION_DPI, bbox_inches='tight')
        plt.close()
        
        return filepath
        
    except Exception as e:
        print(f"❌ Chyba v summary vizualizaci: {e}")
        return None

def load_model(policy, model_path):
    """Načte model z uložené cesty"""
    try:
        print(f"📁 Načítám model: {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')
        policy.load_state_dict(state_dict)
        print("✅ Model úspěšně načten!")
        return True
    except Exception as e:
        print(f"❌ Chyba při načítání modelu: {e}")
        return False

def process_observations(obs_dict, device):
    """Zpracuje observace z prostředí"""
    if "quads" not in obs_dict:
        return []
    
    local_maps = torch.FloatTensor(obs_dict["quads"]["local_map"]).to(device)
    self_states = torch.FloatTensor(obs_dict["quads"]["self_state"]).to(device)
    
    observations = []
    for i in range(local_maps.shape[0]):
        local_map = local_maps[i:i+1]
        self_state = self_states[i:i+1]
        
        # Combine observation
        local_map_flat = local_map.reshape(1, -1)
        self_state_flat = self_state.reshape(1, -1)
        combined_obs = torch.cat([local_map_flat, self_state_flat], dim=-1)
        
        # Create TensorDict
        obs_td = TensorDict({
            "agents": TensorDict({
                "observation": combined_obs.to(device),
            }, batch_size=[1], device=device)
        }, batch_size=[1], device=device)
        
        observations.append(obs_td)
    
    return observations

def get_actions(observations, policy):
    """Získá akce z policy pro všechny agenty"""
    actions = []
    
    for obs_td in observations:
        with torch.no_grad():
            action_td = policy(obs_td)
            action = action_td[("agents", "action")][0]  # Shape: [4]
            actions.append(action)
    
    return actions

def demo_simplified():
    """Spustí demo se zjednodušeným modelem"""
    print("🎬 SPOUŠTÍM DEMO ZJEDNODUŠENÉHO MODELU...")
    
    # Setup
    device = "cpu"
    marlEnv = WildfireMARLEnv(agents_config=["quad_1"])
    self_state_size = marlEnv.obs_proc.get_self_state_size()
    
    # Networks (stejné jako v train_simplified)
    policy, actor_net_base = setup_networks(device, self_state_size)
    
    # Najdi nejnovější model
    model_dir = None
    for folder in sorted(os.listdir("models")):
        if folder.startswith("simple_training_"):
            model_dir = f"models/{folder}"
    
    if not model_dir:
        print("❌ Žádný model ze simple_training nenalezen!")
        return
    
    # Najdi poslední checkpoint
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not model_files:
        print(f"❌ Žádné .pt soubory v {model_dir}!")
        return
    
    latest_model = sorted(model_files)[-1]
    model_path = f"{model_dir}/{latest_model}"
    
    # Načti model
    if not load_model(policy, model_path):
        return
    
    # Nastav eval mode
    policy.eval()
    
    print(f"🎮 Spouštím demo s modelem: {model_path}")
    print("🎯 Pozoruj chování dronu...")
    
    # Demo parametry
    num_episodes = 1  # Změněno z 3 na 1 epizodu
    max_steps = 100
    
    # Tracking pro vizualizaci
    episode_data = {}  # Pro souhrnnou vizualizaci
    
    for episode in range(num_episodes):
        print(f"\n🎬 === EPIZODA {episode + 1}/{num_episodes} ===")
        
        # Reset prostředí
        obs_dict, _ = marlEnv.reset()
        episode_reward = 0
        episode_steps = 0
        episode_rewards = []  # Tracking rewards pro vizualizaci
        frame_count = 0
        
        for step in range(max_steps):
            try:
                # Zpracuj observace
                observations = process_observations(obs_dict, device)
                if not observations:
                    print("❌ Žádné observace dostupné")
                    break
                
                # Získej akce
                actions = get_actions(observations, policy)
                if not actions:
                    print("❌ Žádné akce dostupné")
                    break
                
                # KLÍČOVÁ OPRAVA: Vezmi jen první 4 hodnoty pro quadcopter!
                actions_np = torch.stack(actions).cpu().detach().numpy()  # Shape: [num_agents, 4]
                print(f"🔍 Step {step}: Actions shape: {actions_np.shape}, values: {actions_np[0] if len(actions_np) > 0 else 'none'}")
                
                # Environment step
                action_dict = {"quads": {"action": actions_np}}
                result = marlEnv.step(action_dict)
                obs_dict, reward, done, info = result
                
                # Tracking
                step_reward = reward if isinstance(reward, (int, float)) else np.mean(reward)
                episode_reward += step_reward
                episode_steps += 1
                episode_rewards.append(step_reward)
                
                # Vytvoř vizualizaci každých 20 kroků nebo na konci
                if step % 20 == 0 or done:
                    viz_path = create_demo_visualization(marlEnv, frame_count, episode_rewards)
                    if viz_path:
                        print(f"📷 Frame {frame_count}: {viz_path}")
                    frame_count += 1
                
                if step % 10 == 0:
                    print(f"  📊 Step {step}: reward={step_reward:.3f}, total={episode_reward:.3f}")
                
                if done:
                    print(f"✅ Epizoda dokončena ve step {step}")
                    break
                    
            except Exception as e:
                print(f"❌ Chyba ve step {step}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"🏆 Epizoda {episode + 1}: {episode_steps} kroků, reward: {episode_reward:.3f}")
        
        # Uložit data epizody pro souhrnnou vizualizaci
        episode_data[f"episode_{episode + 1}"] = {
            'rewards': episode_rewards,
            'total_reward': episode_reward,
            'steps': episode_steps
        }
    
    # Vytvoř souhrnnou vizualizaci
    print("\n📊 Vytvářím souhrnnou vizualizaci...")
    summary_path = create_demo_summary(episode_data)
    if summary_path:
        print(f"📊 Souhrnná vizualizace: {summary_path}")
    
    print("\n🎉 Demo dokončeno!")
    print(f"📁 Snímky uloženy v: output/demo_frames/")
    if summary_path:
        print(f"📊 Souhrn uložen: {summary_path}")

if __name__ == "__main__":
    demo_simplified()