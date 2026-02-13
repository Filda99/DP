#!/usr/bin/env python3
"""
REFACTORED TRAINING CODE - ROZDĚLENO DO LOGICKÝCH FUNKCÍ
"""

import torch
import numpy as np
from datetime import datetime
import time
from collections import deque
import os
from tqdm import tqdm

from src.wildfire_gym_wrapper import WildfireMARLEnv
from src.wildfire_models import QuadActor
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal
from config import MainConfig

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

class CriticWrapper(torch.nn.Module):
    """Wrapper pro SimpleCritic aby zpracoval kombinovanou observaci"""
    def __init__(self, critic_net):
        super().__init__()
        self.critic_net = critic_net
        
    def forward(self, observation):
        self_state = observation[:, 1024:]
        value = self.critic_net(self_state)
        return value

def setup_networks(device, self_state_size):
    """Nastavení sítí a policy"""
    print("🧠 Nastavuji sítě...")
    
    # Base networks
    actor_net_base = QuadActor(message_dim=MainConfig.ACTOR_MESSAGE_DIM, self_state_size=self_state_size).to(device)
    critic_net = SimpleCritic(self_state_size=self_state_size).to(device)
    
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
        return_log_prob=True,
    )
    
    # Critic module
    critic_wrapper = CriticWrapper(critic_net)
    critic_module = TensorDictModule(
        critic_wrapper,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )
    
    return policy, critic_module, actor_net_base

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
        
        observations.append((obs_td, local_map, self_state))
    
    return observations

def get_agent_actions(observations, policy, critic_module):
    """Získá akce a hodnoty pro všechny agenty"""
    agent_data = []
    
    for obs_td, local_map, self_state in observations:
        with torch.no_grad():
            # Get action
            action_td = policy(obs_td)
            action = action_td[("agents", "action")][0]
            action_log_prob = action_td[("agents", "action_log_prob")][0]
            
            if action_log_prob.dim() == 0:
                action_log_prob = action_log_prob.unsqueeze(0)
            
            # Get value
            value_td = critic_module(obs_td)
            value = value_td[("agents", "state_value")][0].item()
            
            agent_data.append({
                'observation': (local_map, self_state),
                'action': action,
                'log_prob': action_log_prob,
                'value': value
            })
    
    return agent_data

def update_hidden_states(agent_data, actor_net_base, hidden_states):
    """Aktualizuje hidden states pro všechny agenty"""
    with torch.no_grad():
        for i, data in enumerate(agent_data):
            if i < len(hidden_states):
                local_map, self_state = data['observation']
                _, _, new_hidden = actor_net_base(local_map, self_state, hidden_states[i:i+1])
                hidden_states[i] = new_hidden[0]

def compute_advantages_manual(memory_buffer, gamma=0.99, lmbda=0.95):
    """Vlastní výpočet advantage (GAE)"""
    if len(memory_buffer) < 2:
        return []
    
    advantages = []
    gae = 0
    
    # Reverse iterate through buffer
    for i in reversed(range(len(memory_buffer) - 1)):
        obs, action, reward, next_value, value, done, hidden, log_prob = memory_buffer[i]
        
        if i == len(memory_buffer) - 1:
            next_value = 0 if done else value
        else:
            next_value = memory_buffer[i + 1][4]  # Next value
        
        delta = reward + gamma * next_value - value
        gae = delta + gamma * lmbda * gae * (1 - done)
        advantages.insert(0, gae)
    
    return advantages

def train_simplified():
    """Zjednodušená training funkce"""
    print("🚁 SPOUŠTÍM ZJEDNODUŠENÝ TRÉNINK...")
    
    # Setup
    device = "cpu"
    marlEnv = WildfireMARLEnv(agents_config=["quad_1"])
    num_agents = 1
    self_state_size = marlEnv.obs_proc.get_self_state_size()
    
    # Networks
    policy, critic_module, actor_net_base = setup_networks(device, self_state_size)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(critic_module.parameters()), 
        lr=MainConfig.LEARNING_RATE
    )
    
    # Training setup - delší epizody pro lepší learning
    max_episodes = 300  # Zvýšeno z 100 na 300 pro důkladnější učení
    max_steps = 1000  # Zvýšeno z 500 na 1000 pro delší epizody
    exploration_noise = 0.3  # Zvýšeno z 0.05 na 0.3 pro větší exploraci
    memory_buffer = []
    episode_rewards = deque(maxlen=10)
    
    save_dir = f"models/simple_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    episode_pbar = tqdm(range(max_episodes), desc="🚁 Zjednodušený Training")
    
    for episode in episode_pbar:
        obs_dict, _ = marlEnv.reset()
        hidden_states = torch.zeros(num_agents, MainConfig.ACTOR_HIDDEN_SIZE).to(device)
        episode_reward = 0
        
        for step in range(max_steps):
            try:
                # Process observations
                observations = process_observations(obs_dict, device)
                if not observations:
                    break
                
                # Get actions
                agent_data = get_agent_actions(observations, policy, critic_module)
                
                # Environment step with exploration
                actions_np = torch.stack([data['action'] for data in agent_data]).cpu().detach().numpy()
                # Add exploration noise during training
                if episode < max_episodes:  # Explorace po celou dobu tréninku
                    noise = np.random.normal(0, exploration_noise, actions_np.shape)
                    actions_np = np.clip(actions_np + noise, -1.0, 1.0)
                action_dict = {"quads": {"action": actions_np}}
                
                result = marlEnv.step(action_dict)
                obs_dict, reward, done, info = result
                
                # Store transitions
                avg_reward = reward if isinstance(reward, (int, float)) else np.mean(reward)
                
                if agent_data:
                    memory_buffer.append((
                        agent_data[0]['observation'], 
                        agent_data[0]['action'], 
                        avg_reward, 
                        0.0, 
                        agent_data[0]['value'], 
                        done, 
                        hidden_states[0:1], 
                        agent_data[0]['log_prob']
                    ))
                
                # Update hidden states
                update_hidden_states(agent_data, actor_net_base, hidden_states)
                
                episode_reward += avg_reward
                
                if done:
                    break
                    
            except Exception as e:
                print(f"❌ Chyba ve step {step}: {e}")
                break
        
        episode_rewards.append(episode_reward)
        
        # PPO Update (improved)
        if len(memory_buffer) >= 50 and episode % 3 == 0:
            # Improved learning update
            advantages = compute_advantages_manual(memory_buffer)
            
            if advantages and len(advantages) > 5:
                # Fixed policy loss calculation
                policy_losses = []
                
                for i, (obs, action, reward, _, value, done, hidden, log_prob) in enumerate(memory_buffer[:len(advantages)]):
                    if i < len(advantages) and log_prob is not None and log_prob.requires_grad:
                        advantage = torch.tensor(advantages[i], dtype=torch.float32)
                        loss = -log_prob * advantage
                        policy_losses.append(loss)
                
                if policy_losses:
                    optimizer.zero_grad()
                    total_loss = torch.stack(policy_losses).mean()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()
            
            memory_buffer.clear()
        
        # Progress update
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        episode_pbar.set_postfix({
            'reward': f"{avg_reward:.2f}",
            'buffer': len(memory_buffer)
        })
        
        # Save model - každých 20 epizod
        if episode > 0 and episode % 20 == 0:
            torch.save(policy.state_dict(), f"{save_dir}/policy_ep{episode:03d}.pt")
            print(f"💾 Model uložen: episode {episode}, avg_reward: {avg_reward:.2f}")
    
    print(f"✅ Trénink dokončen! Modely v: {save_dir}")

if __name__ == "__main__":
    train_simplified()