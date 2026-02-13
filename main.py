#!/usr/bin/env python3
"""
QUADCOPTER FIRE DETECTION TRAINING - MAIN SCRIPT

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
from tensordict.nn import TensorDictModule, NormalParamExtractor

from torchrl.objectives import ClipPPOLoss, ValueEstimators  
from torchrl.modules import ProbabilisticActor, TanhNormal

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
    print("SPOUŠTÍM TRÉNINK KVADROKOPTÉRY...")
    
    # Setup - force CPU for now to avoid TorchRL device issues
    device = "cpu"  # Force CPU for stability
    
    # Environment - Multi-agent setup (více kvadrokoptér)
    marlEnv = WildfireMARLEnv(agents_config=["quad_1"])
    num_agents = 1
    
    # Get self_state size from environment's observation processor
    self_state_size = marlEnv.obs_proc.get_self_state_size()
    
    # Actor and Critic networks - TorchRL style probabilistic setup
    actor_net_base = QuadActor(message_dim=MainConfig.ACTOR_MESSAGE_DIM, self_state_size=self_state_size).to(device)
    critic_net = SimpleCritic(self_state_size=self_state_size).to(device)
    
    # Create policy network that outputs loc and scale (following mappo_hello_world pattern)
    # Need wrapper that handles our observation format AND outputs distribution parameters
    class PolicyNetwork(torch.nn.Module):
        def __init__(self, actor_net):
            super().__init__()
            self.actor_net = actor_net
            
        def forward(self, observation):
            # Split combined observation back to local_map + self_state
            # observation shape: (batch, features)
            
            # observation shape: (batch, 1024 + self_state_size) 
            local_map_flat = observation[:, :1024]  # First 1024 features  
            self_state = observation[:, 1024:]      # Remaining features (variable size)
            
            # Reshape local_map back to (batch, 1, 32, 32)
            batch_size = observation.shape[0]
            local_map = local_map_flat.reshape(batch_size, 1, 32, 32)
            
            # Add hidden state (zeros for now)
            hidden_state = torch.zeros(batch_size, 128, device=observation.device)
            
            # Get distribution parameters from actor (now returns 8 values: 4 loc + 4 scale)
            dist_params, _, _ = self.actor_net(local_map, self_state, hidden_state)
            
            return dist_params  # Shape: (batch, 8)
    
    policy_net = torch.nn.Sequential(
        PolicyNetwork(actor_net_base),
        NormalParamExtractor(),  # Splits into loc and scale
    )
    
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],  # [agents, batch, obs_dim]
        out_keys=[("agents", "loc"), ("agents", "scale")],  # [agents, batch, 4], [agents, batch, 4]
    )
    
    # ProbabilisticActor handles sampling from the distribution (like mappo_hello_world)
    # This will automatically generate log_prob for PPO!
    policy = ProbabilisticActor(
        module=policy_module,
        in_keys=[("agents", "loc"), ("agents", "scale")],  # [agents, batch, 4], [agents, batch, 4]
        out_keys=[("agents", "action")],  # [agents, batch, 4]
        distribution_class=TanhNormal,  # TanhNormal keeps actions inside [-1, 1] 
        distribution_kwargs={
            "low": -1.0,   # Our actions are in [-1, 1]
            "high": 1.0,
        },
        return_log_prob=True,  # Required for PPO loss calculation!
    )
    
    # Critic wrapper to handle combined observation  
    class CriticWrapper(torch.nn.Module):
        def __init__(self, critic_net):
            super().__init__()
            self.critic_net = critic_net
            
        def forward(self, observation):
            # Extract self_state from combined observation (variable size)
            # observation shape: (batch, features)
            
            self_state = observation[:, 1024:]  # Everything after local_map features
            value = self.critic_net(self_state)
            
            return value
    
    critic_wrapper = CriticWrapper(critic_net)
    
    critic_module = TensorDictModule(
        critic_wrapper,
        in_keys=[("agents", "observation")],  # [agents, batch, obs_dim]
        out_keys=[("agents", "state_value")],  # [agents, batch, 1]
    )
    
    # Manual PPO Loss (avoid TorchRL GAE issues with multi-agent setup)
    # We'll compute PPO loss manually with our own advantage calculation
    print("Using manual PPO loss computation (TorchRL GAE has multi-agent issues)")
    
    # Optimizer for manual PPO (policy + critic parameters)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(critic_module.parameters()), 
        lr=MainConfig.LEARNING_RATE
    )
    print("Manual PPO optimizer setup complete")
    
    # Training settings
    max_episodes = MainConfig.MAX_EPISODES
    max_steps = MainConfig.MAX_STEPS
    save_every = MainConfig.SAVE_EVERY
    
    save_dir = f"models/gentle_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Modely se ukládají do: {save_dir}")
    
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
        hidden_states = torch.zeros(num_agents, MainConfig.ACTOR_HIDDEN_SIZE).to(device)  # Multiple agents
        episode_reward = 0
        episode_steps = 0
        episode_crashed = False
        
        for step in range(max_steps):
            try:
                # Extract observations for all agents
                agent_observations = []
                agent_actions = []
                agent_log_probs = []
                agent_values = []
                
                if "quads" in obs_dict:
                    # Handle multiple agents
                    local_maps = torch.FloatTensor(obs_dict["quads"]["local_map"]).to(device)  # Shape will be [num_agents, 1, 32, 32]
                    self_states = torch.FloatTensor(obs_dict["quads"]["self_state"]).to(device)  # Shape will be [num_agents, self_state_size]
                    
                    # Process each agent
                    for i in range(min(num_agents, local_maps.shape[0])):
                        local_map = local_maps[i:i+1]  # [1, 1, 32, 32]
                        self_state = self_states[i:i+1]  # [1, 6]
                        
                        # Combine observation - adjust for actual self_state size
                        local_map_flat = local_map.reshape(1, -1)  # [1, 1024]
                        self_state_flat = self_state.reshape(1, -1)  # [1, actual_size] 
                        combined_obs = torch.cat([local_map_flat, self_state_flat], dim=-1)  # [1, 1024+actual_size]
                        
                        # Create TensorDict for this agent with nested structure
                        obs_td = TensorDict({
                            "agents": TensorDict({
                                "observation": combined_obs.to(device),  # [1, total_obs_dim]
                            }, batch_size=[1], device=device)
                        }, batch_size=[1], device=device)
                        
                        # Get action and value for this agent
                        with torch.no_grad():
                            action_td = policy(obs_td)
                            action = action_td[("agents", "action")][0]  # [4] 
                            action_log_prob = action_td[("agents", "action_log_prob")][0]  # scalar
                            
                            # Ensure action_log_prob is proper tensor shape
                            if action_log_prob.dim() == 0:
                                action_log_prob = action_log_prob.unsqueeze(0)  # [1]
                            
                            value_td = critic_module(obs_td)
                            value = value_td[("agents", "state_value")][0].item()  # scalar from [1,1] -> scalar
                        
                        agent_observations.append((local_map, self_state))
                        agent_actions.append(action)
                        agent_log_probs.append(action_log_prob)
                        agent_values.append(value)
                else:
                    break  # No agents available
                
                # Step environment with multiple agents
                if agent_actions:
                    # Convert actions to numpy for environment
                    actions_np = torch.stack(agent_actions).cpu().detach().numpy()  # [num_agents, 4]
                    
                    action_dict = {
                        "quads": {
                            "action": actions_np
                        }
                    }
                    
                    result = marlEnv.step(action_dict)
                    obs_dict, reward, done, info = result
                    
                    # Store transition for each agent (simplified - use average reward for now)
                    avg_reward = reward if isinstance(reward, (int, float)) else np.mean(reward)
                    avg_value = np.mean(agent_values) if agent_values else 0.0
                    
                    # Store one representative transition (for now)
                    if agent_observations and agent_actions:
                        memory_buffer.append((
                            agent_observations[0], agent_actions[0], avg_reward, 0.0, 
                            agent_values[0], done, hidden_states[0:1], agent_log_probs[0]
                        ))
                    
                    # Update hidden states for all agents
                    with torch.no_grad():
                        for i, (local_map, self_state) in enumerate(agent_observations[:len(hidden_states)]):
                            if i < len(hidden_states):
                                _, _, new_hidden = actor_net_base(local_map, self_state, hidden_states[i:i+1])
                                hidden_states[i] = new_hidden[0]
                    
                    episode_reward += avg_reward
                    episode_steps += 1
                else:
                    break
                
                if done:
                    # Check if episode ended due to crash for any agent
                    drone_crashed = False
                    if "quads" in obs_dict and obs_dict["quads"]["self_state"] is not None:
                        self_states = obs_dict["quads"]["self_state"]
                        for i in range(self_states.shape[0]):
                            altitude = self_states[i][0]  # First feature is altitude
                            if altitude < 0.5:
                                drone_crashed = True
                                break
                    
                    if drone_crashed or avg_reward < -10:  # Actual crash or severe penalty
                        episode_crashed = True
                        crash_count += 1
                    break
                    
            except Exception as e:
                print(f"❌ Chyba v kroku {step}: {e}")
                episode_crashed = True
                break
        
        # Update policy každé 3 epizody - častější updates pro lepší learning
        if episode % 3 == 0 and episode > 0 and len(memory_buffer) > 0:
            try:
                # Convert memory_buffer to TensorDict format for TorchRL
                batch_size = len(memory_buffer)
                
                # Extract data from memory_buffer
                observations = []
                next_observations = []
                actions = []
                rewards = []
                values = []
                next_values = []
                dones = []
                action_log_probs = []
                
                for i, transition in enumerate(memory_buffer):
                    obs_tuple, action, reward, next_obs, value, done, hidden, action_log_prob = transition
                    local_map, self_state = obs_tuple
                    
                    # Flatten observation for TensorDict (combine local_map + self_state)
                    local_map_flat = local_map.reshape(1, -1)  # (1, 1*32*32)
                    self_state_flat = self_state.reshape(1, -1)  # (1, self_state_size)
                    combined_obs = torch.cat([local_map_flat, self_state_flat], dim=-1)  # (1, 1024+self_state_size)
                    
                    observations.append(combined_obs)
                    actions.append(action.unsqueeze(0))  # Ensure [1, 4] shape
                    rewards.append(torch.tensor([reward], dtype=torch.float32))  # [1]
                    values.append(torch.tensor([value], dtype=torch.float32))   # [1]
                    dones.append(torch.tensor([done], dtype=torch.bool))        # [1]
                    action_log_probs.append(action_log_prob.detach())  # Already [1]
                    
                    # For "next" - use next transition or current for last step
                    if i < len(memory_buffer) - 1:
                        # Use next observation
                        next_transition = memory_buffer[i + 1]
                        next_obs_tuple = next_transition[0]  # (local_map, self_state)
                        next_local_map, next_self_state = next_obs_tuple
                        
                        next_local_map_flat = next_local_map.reshape(1, -1)
                        next_self_state_flat = next_self_state.reshape(1, -1)
                        next_combined_obs = torch.cat([next_local_map_flat, next_self_state_flat], dim=-1)
                        
                        next_observations.append(next_combined_obs)
                        next_values.append(torch.tensor([next_transition[4]], dtype=torch.float32))  # [1]
                    else:
                        # Last step - use current obs as next (or zeros if terminal)
                        if done:
                            next_observations.append(torch.zeros_like(combined_obs))
                            next_values.append(torch.tensor([0.0], dtype=torch.float32))  # [1]
                        else:
                            next_observations.append(combined_obs)
                            next_values.append(torch.tensor([value], dtype=torch.float32))  # [1]
                
                # Stack into tensors
                batch_obs = torch.cat(observations, dim=0)  # (batch_size, obs_dim)
                batch_next_obs = torch.cat(next_observations, dim=0)  # (batch_size, obs_dim)
                batch_actions = torch.cat(actions, dim=0)  # (batch_size, action_dim)
                batch_rewards = torch.cat(rewards, dim=0)  # (batch_size,)
                batch_values = torch.cat(values, dim=0)  # (batch_size,)
                batch_next_values = torch.cat(next_values, dim=0)  # (batch_size,)
                batch_dones = torch.cat(dones, dim=0)  # (batch_size,)
                batch_action_log_probs = torch.cat(action_log_probs, dim=0)  # (batch_size,)
                
                # Create TensorDict with structure matching mappo_hello_world EXACTLY  
                # Need to add top-level done/terminated as in mappo debug output
                tensordict_data = TensorDict({
                    # All data under "agents" like in mappo_hello_world
                    "agents": TensorDict({
                        "observation": batch_obs.to(device),  # [batch, obs_dim] 
                        "action": batch_actions.to(device),   # [batch, 4]
                        "sample_log_prob": batch_action_log_probs.to(device),  # [batch]
                        "reward": batch_rewards.to(device),   # [batch] - reward under agents!
                        "state_value": batch_values.to(device), # [batch]
                        "done": batch_dones.to(device),       # [batch] 
                        "terminated": batch_dones.to(device), # [batch]
                    }, batch_size=[batch_size], device=device),
                    
                    # Top-level done/terminated as seen in mappo debug output
                    "done": batch_dones.unsqueeze(-1).to(device),  # [batch, 1] 
                    "terminated": batch_dones.unsqueeze(-1).to(device),  # [batch, 1]
                    
                    "next": TensorDict({
                        "agents": TensorDict({
                            "observation": batch_next_obs.to(device),
                            "state_value": batch_next_values.to(device),
                            "done": batch_dones.to(device),
                            "terminated": batch_dones.to(device),
                        }, batch_size=[batch_size], device=device),
                        # Top-level done/terminated in next too
                        "done": batch_dones.unsqueeze(-1).to(device),
                        "terminated": batch_dones.unsqueeze(-1).to(device),
                    }, batch_size=[batch_size], device=device),
                }, batch_size=[batch_size], device=device)
                
                # 🧠 MANUAL ADVANTAGE COMPUTATION 🧠
                # Advantage = "Jak moc lepší byla tato akce než průměr"
                # Formula: Advantage = R + γ*V(s') - V(s)
                # kde: R = reward, γ = discount, V(s') = next state value, V(s) = current state value
                
                with torch.no_grad():
                    gamma = 0.99  # Discount factor - jak moc si ceníme budoucích rewards
                    
                    # TD Target = R + γ*V(next) * (1 - done)  (Temporal Difference target)
                    td_targets = batch_rewards + gamma * batch_next_values * (1 - batch_dones.float())
                    
                    # Advantage = TD_Target - V(current) = "Překvapení" - jak moc lepší/horší než očekávané
                    advantages = td_targets - batch_values
                    
                    # Normalize advantages (zlepšuje stabilitu tréninku)
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    print(f"📊 Advantage stats: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
                    print(f"📊 TD targets: mean={td_targets.mean():.4f}, rewards: mean={batch_rewards.mean():.4f}")
                
                # 🔥 MANUAL PPO LOSS COMPUTATION 🔥
                # PPO = Proximal Policy Optimization - "Nechoď příliš daleko od staré policy"
                
                try:
                    # 1. Policy ratio = new_prob / old_prob (jak moc jsme změnili policy)
                    with torch.no_grad():
                        old_log_probs = batch_action_log_probs  # Zalogované pravděpodobnosti z inference
                    
                    # 2. Forward pass přes current policy pro nové log_probs (use batch_obs místo reconstructing)
                    current_obs = batch_obs
                    policy_output = policy(TensorDict({
                        "agents": TensorDict({
                            "observation": current_obs.to(device)
                        }, batch_size=[batch_size], device=device)
                    }, batch_size=[batch_size], device=device))
                    
                    new_log_probs = policy_output[("agents", "action_log_prob")]
                    
                    # 3. Policy ratio
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    
                    # 4. Clipped PPO objective
                    clip_epsilon = 0.2
                    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                    
                    # 5. PPO loss = -min(ratio * advantage, clipped_ratio * advantage)
                    policy_loss_1 = ratio * advantages
                    policy_loss_2 = clipped_ratio * advantages
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                    
                    # 6. Value loss (critic chce predikovat správné TD targets)
                    current_values = critic_module(TensorDict({
                        "agents": TensorDict({
                            "observation": current_obs.to(device)
                        }, batch_size=[batch_size], device=device)
                    }, batch_size=[batch_size], device=device))[("agents", "state_value")]
                    
                    value_loss = torch.nn.functional.mse_loss(current_values.squeeze(), td_targets)
                    
                    # 7. Entropy bonus (encouraguje exploration)
                    entropy_coeff = 1e-4
                    # Pro TanhNormal distribution, entropy je trochu komplexní, použijeme approximation
                    entropy_bonus = 0.0  # Simplified for now
                    
                    # 8. Total loss
                    total_loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy_bonus
                    
                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(critic_module.parameters()), max_norm=0.5)
                    optimizer.step()
                    
                    loss = total_loss.item()
                    episode_pbar.write(f"📊 Ep {episode}: Manual PPO Loss: {loss:.4f} (Policy: {policy_loss:.4f}, Value: {value_loss:.4f})")
                    
                except Exception as e:
                    print(f"❌ Manual PPO loss computation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    loss = 0.0
                
            except Exception as e:
                episode_pbar.write(f"❌ PPO training error: {e}")
                loss = 0.0
                
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
            'Loss': f"{loss:.3f}" if 'loss' in locals() else "N/A",
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
                'actor_state_dict': actor_net_base.state_dict(),
                'critic_state_dict': critic_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': list(episode_rewards),
                'crash_count': crash_count,
                'best_reward': best_reward
            }
            torch.save(checkpoint, f"{save_dir}/checkpoint_ep{episode+1:03d}.pt")
            
            # Také uložit jako nejnovější model pro demo
            newest_dir = "models/newest"
            os.makedirs(newest_dir, exist_ok=True)
            torch.save(checkpoint, f"{newest_dir}/latest_model.pt")
            episode_pbar.write(f"💾 Model uložen (Ep {episode+1}) a zkopírován do newest/")
    
    # Final save s complete stats
    final_checkpoint = {
        'episode': max_episodes,
        'actor_state_dict': actor_net_base.state_dict(),
        'critic_state_dict': critic_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
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
    temp_env = WildfireMARLEnv(agents_config=["quad_1", "quad_2", "quad_3"])
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
    env = WildfireMARLEnv(agents_config=["quad_1", "quad_2", "quad_3"], demo_mode=True)  # 3 agents
    obs, _ = env.reset()
    num_agents = 3
    
    # Přidej reward tracking pro vizualizaci
    env._recent_rewards = []
    
    print("🚁 === DEMO: 3 drony na mapě (multi-agent) ===")
    
    # Hidden states for multiple agents
    hidden_states = torch.zeros(num_agents, MainConfig.ACTOR_HIDDEN_SIZE)
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
                # Handle multiple agents
                local_maps = torch.FloatTensor(obs["quads"]["local_map"])  # [num_agents, 1, 32, 32]
                self_states = torch.FloatTensor(obs["quads"]["self_state"])  # [num_agents, 6]
                
                agent_actions = []
                
                # Process each agent
                for i in range(min(num_agents, local_maps.shape[0])):
                    local_map = local_maps[i:i+1]  # [1, 1, 32, 32]
                    self_state = self_states[i:i+1]  # [1, 6]
                    
                    with torch.no_grad():
                        raw_actions, _, new_hidden = model(local_map, self_state, hidden_states[i:i+1])
                        
                        # Direct action output - no distributions needed
                        actions = torch.tanh(raw_actions)  # Bound to [-1, 1]
                        
                        # Apply gentle policy if needed - conservative scaling for safety
                        if MainConfig.DEMO_CONSERVATIVE_POLICY:
                            actions = actions * 0.7  # More conservative for demo
                        
                        actions_clipped = torch.clamp(actions, -1.0, 1.0)  # Standard range
                        agent_actions.append(actions_clipped)
                        
                        # Update hidden state for this agent
                        hidden_states[i] = new_hidden[0]
                
                if not agent_actions:
                    break
                    
                # Stack actions for environment step
                actions_np = torch.cat(agent_actions, dim=0).numpy()  # [num_agents, 4]
            
            # DEBUG: Print actions every 100 steps to see what model outputs
            if steps % 100 == 0 and steps < 500:
                # Debug first agent only for brevity
                if len(self_states) > 0:
                    alt = self_states[0][0].item()
                    
                    # BOUNDARY DEBUG - print key boundary info for first agent
                    if len(self_states[0]) >= 14:
                        boundary_x = self_states[0][12].item()
                        boundary_y = self_states[0][13].item()
                        print(f"🐛 Krok {steps}: Agent 0 actions=[{actions_np[0][0]:.4f}, {actions_np[0][1]:.4f}, {actions_np[0][2]:.4f}, {actions_np[0][3]:.4f}]")
                        print(f"   ↳ Boundary info: X_dist={boundary_x:.1f}, Y_dist={boundary_y:.1f} (negative = outside!)")
                    else:
                        print(f"🐛 Krok {steps}: Agent 0 actions=[{actions_np[0][0]:.4f}, {actions_np[0][1]:.4f}, {actions_np[0][2]:.4f}, {actions_np[0][3]:.4f}]")
                    print(f"   ↳ altitude={alt:.1f}m, {len(actions_np)} agents active")
            
            # ===== AKCE PRO VÍCE DRONŮ =====
            action_dict = {
                "quads": {
                    "action": actions_np  # [num_agents, 4]
                }
            }
            
            # DEBUG: Check action shape
            print(f"🔍 actions_np shape: {actions_np.shape}")
            print(f"🔍 actions_np content: {actions_np}")
            
            try:
                result = env.step(action_dict)
                print(f"🔍 env.step returned: {len(result)} items")
                if len(result) == 4:
                    obs, reward, done, info = result
                else:
                    print(f"❌ Unexpected return count: {len(result)}")
                    break
            except Exception as step_error:
                print(f"❌ Error in env.step: {step_error}")
                import traceback
                traceback.print_exc()
                break
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
    print(f"   🚁 Aktivní drony: {active_drones}/{num_agents}")
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