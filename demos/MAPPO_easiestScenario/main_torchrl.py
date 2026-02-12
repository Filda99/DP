# =============================================================================
# SECTION 1: IMPORTS AND HYPERPARAMETERS
# =============================================================================
# Theory:
# - MARL requires massive parallelization to stabilize training. We use 'Vectorization'
#   to run hundreds of environments simultaneously on the GPU.
# - TensorDict is the core data carrier here: it groups observations, actions, and
#   rewards from all these environments into single batches.
#
# Implementation:
# - We check for CUDA (GPU) availability.
# - We set PPO hyperparameters (learning rate, clip epsilon, etc.) standard for
#   continuous control tasks.
# - set_composite_lp_aggregate(False) is crucial for Multi-Agent logic: it ensures
#   log-probabilities are calculated per agent, not summed up for the whole team immediately.
# =============================================================================

import os
import torch
from torch import multiprocessing

# Tensordict modules
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys 

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.wildfire_gym_wrapper import WildfireMARLEnv

# Add path for our models
from src.wildfire_models import QuadActor
from main import SimpleCritic  # Import SimpleCritic from main.py
import config

torch.manual_seed(0)

# --- Hyperparameters ---

# Device configuration
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
vmas_device = device  # VMAS runs physics directly on the GPU

# Sampling settings
frames_per_batch = 6_000   # Total frames collected per iteration across all parallel envs
n_iters = 10               # How many times we collect data and train
total_frames = frames_per_batch * n_iters

# PPO Training settings
num_epochs = 30            # Number of times we loop over the collected data
minibatch_size = 400       # Size of gradient descent updates
lr = 3e-4                  # Learning rate
max_grad_norm = 1.0        # Clips gradients to prevent exploding gradients

# PPO Loss settings
clip_epsilon = 0.2         # PPO clipping (prevents policy from changing too fast)
gamma = 0.99               # Discount factor for future rewards
lmbda = 0.9                # GAE lambda (balances bias vs variance)
entropy_eps = 1e-4         # Entropy coefficient (encourages exploration)

# Disable automatic aggregation of log-probs (crucial for keeping agents distinct)
set_composite_lp_aggregate(False).set()


# =============================================================================
# SECTION 2: ENVIRONMENT (VMAS)
# =============================================================================
# Theory:
# - VMAS (Vectorized Multi-Agent Simulator) simulates physics on the GPU.
# - Unlike standard Gym envs, the "batch size" is built-in. We don't need
#   wrappers to parallelize; the tensors coming out are already batched.
#
# Implementation:
# - We calculate `num_vmas_envs` to ensure our buffer fills up exactly.
# - We wrap the environment in `TransformedEnv` with `RewardSum`. This automatically
#   tracks the cumulative reward of an episode for logging purposes.
#
# !! REPLACE !!
# =============================================================================

max_steps = 100  # Max steps per episode
num_vmas_envs = frames_per_batch // max_steps  # Number of parallel simulations
scenario_name = "navigation"
n_agents = 3

# Initialize our WildfireMARLEnv
env = WildfireMARLEnv(agents_config=["quad_1"])

# Print specs to inspect shapes
print("action_spec:", env.action_spec)
print("reward_spec:", env.reward_spec) 
print("observation_spec:", env.observation_spec)

# Check actual shapes we need
print("\n📏 Key shapes:")
local_map_shape = env.observation_spec["quads"]["local_map"].shape
self_state_shape = env.observation_spec["quads"]["self_state"].shape
hidden_state_shape = env.observation_spec["quads"]["hidden_state"].shape
action_shape = env.action_spec["quads"]["action"].shape
print(f"  local_map: {local_map_shape}")
print(f"  self_state: {self_state_shape}")
print(f"  hidden_state: {hidden_state_shape}")
print(f"  action: {action_shape}")

# Add device property that TransformedEnv expects\nsetattr(env, 'device', device)\n\n# Transformations: Compute sum of rewards for logging
# Our environment has simple "reward" structure
env = TransformedEnv(
    env,
    RewardSum(in_keys=["reward"], out_keys=["episode_reward"]),
)


# To something like this:
    # # Assume you have a custom gym env registered as 'MyCustomEnv-v0'
    # base_env = GymEnv("MyCustomEnv-v0", device=device) 

    # # 2. Check the keys! 
    # # Standard Gym usually has:
    # # - reward key: "reward"
    # # - done key: "done"
    # print(base_env.reward_key) # Likely prints: "reward"

    # # 3. Adapt the TransformedEnv
    # env = TransformedEnv(
    #     base_env,
    #     RewardSum(
    #         in_keys=["reward"],       # CHANGED: VMAS used [("agents", "reward")]
    #         out_keys=["episode_reward"] # CHANGED: We store the sum as "episode_reward"
    #     )
    # )

check_env_specs(env)


# =============================================================================
# SECTION 3: NETWORKS (ACTOR & CRITIC) - THE "MAPPO" LOGIC
# =============================================================================
# Theory: Centralized Training, Decentralized Execution (CTDE)
# - ACTOR: Must be Decentralized. It takes LOCAL observations and outputs actions.
#   It uses 'Parameter Sharing' (one net for all agents) to learn faster.
# - CRITIC: Can be Centralized (MAPPO). It takes GLOBAL information (all agents' obs)
#   to estimate the Value V(s). This helps reducing variance during training.
#
# Implementation:
# - MultiAgentMLP: A specialized TorchRL module that handles (Batch, Agents, Dim) data.
# - share_params=True: Enables parameter sharing.
# - centralised=True (for Critic): Concatenates all agent inputs for the critic.
#
# For our environment:
# - One cetralized critic, but two dinsrinct actor networks
#  (one for each agent type) could be used for better specialization.
# Actor A: viz studies
#
# !! REPLACE !!
# Needs to be 2 distinct actor networks for different agent types.
# =============================================================================

# --- Policy (Actor) ---
# Use our QuadActor model - need to handle complex observation structure
# Our QuadActor expects flattened input combining local_map + self_state
local_map_dim = env.observation_spec["quads"]["local_map"].shape[-3:].numel()  # Flatten CNN input
self_state_dim = env.observation_spec["quads"]["self_state"].shape[-1]
hidden_state_dim = env.observation_spec["quads"]["hidden_state"].shape[-1]
action_dim = env.action_spec["quads"]["action"].shape[-1]

print(f"\n🧠 Network dimensions:")
print(f"  local_map_dim: {local_map_dim}")
print(f"  self_state_dim: {self_state_dim}")
print(f"  hidden_state_dim: {hidden_state_dim}")
print(f"  action_dim: {action_dim}")

# For now, create a simple wrapper that combines the inputs for QuadActor
# This is a simplified approach - in production, you'd want proper CNN handling

quad_actor = QuadActor(
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_dim=config.HIDDEN_DIM
).to(device)

# Create policy network that outputs mean and std
policy_net = torch.nn.Sequential(
    quad_actor,
    NormalParamExtractor(),  # Splits output into Loc (mean) and Scale (std dev)
)

policy_module = TensorDictModule(
    policy_net,
    in_keys=["observation"],  # Our key
    out_keys=["loc", "scale"],  # Standard TorchRL keys for mean/std
)

# ProbabilisticActor handles sampling from the distribution
policy = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    out_keys=["action"],  # Our action key
    distribution_class=TanhNormal,  # TanhNormal keeps actions inside [-1, 1]
    distribution_kwargs={
        "low": env.action_spec["action"].space.low,
        "high": env.action_spec["action"].space.high,
    },
    return_log_prob=True,  # Required for PPO loss calculation
)

# --- Value (Critic) ---
# Use SimpleCritic instead of MultiAgentMLP
critic_net = SimpleCritic(
    self_state_size=obs_dim,
    hidden_size=config.HIDDEN_DIM
).to(device)

critic = TensorDictModule(
    module=critic_net,
    in_keys=["observation"],  # Our key
    out_keys=["state_value"],
)

# Sanity check
# print("Running policy:", policy(env.reset()))
# print("Running value:", critic(env.reset()))


# A. Define the Quadcopter Network (CNN + GRU)
# # This class acts as both the "Actor" (Pilot) and the "Sensor" (Messenger)
# class QuadActor(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Define CNN for map ...
#         # Define GRU for memory ...
#         # Define 2 Heads: one for Action, one for Message ...
    
#     def forward(self, map, self_state, hidden_state):
#         # ... logic ...
#         return action, message, new_hidden_state

# # B. Define the Fixed-Wing Network (Attention)
# class FixedActor(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Define Attention Layer ...
#         # Define Flight Control MLP ...
    
#     def forward(self, self_state, quad_messages):
#         # ... logic ...
#         return action

# # C. Stitch them into a Single Policy
# # We use TensorDictSequential to make them run one after another
# # 1. Quads run first (generate messages)
# quad_module = TensorDictModule(
#     QuadActor(), 
#     in_keys=[("quads", "map"), ("quads", "self_state"), "hidden_state"],
#     out_keys=[("quads", "action"), "message_stream", "next_hidden_state"]
# )

# # 2. Fixed wings run second (read messages)
# fixed_module = TensorDictModule(
#     FixedActor(),
#     in_keys=[("fixed", "self_state"), "message_stream"],
#     out_keys=[("fixed", "action")]
# )

# policy = TensorDictSequential(quad_module, fixed_module)


# =============================================================================
# SECTION 4: DATA COLLECTOR AND REPLAY BUFFER
# =============================================================================
# Theory:
# - PPO is On-Policy. We collect data, train on it, and DISCARD it.
# - SyncDataCollector: Steps the environment and gathers batches of size `frames_per_batch`.
#
# Implementation:
# - The ReplayBuffer is simple (not a priority buffer) because we just shuffle and
#   iterate through the most recent data.
# =============================================================================

collector = SyncDataCollector(
    env,
    policy,
    device=device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch, device=device),
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,
)


# =============================================================================
# SECTION 5: LOSS FUNCTION AND OPTIMIZER
# =============================================================================
# Theory:
# - ClipPPOLoss: Implements the PPO objective (clipped surrogate objective).
# - GAE (Generalized Advantage Estimation): Computes the "Advantage" (how much better
#   an action was than expected). This reduces variance in the gradient estimate.
#
# Implementation:
# - normalize_advantage=False: In MARL, normalizing across the agent dimension can
#   sometimes destabilize training, so we disable it or handle it carefully.
#
# !! MODIFY !!
# 2 separate loss modules for different actor networks 
# =============================================================================

loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    clip_epsilon=clip_epsilon,
    entropy_coeff=entropy_eps,
    normalize_advantage=False,
)

loss_module.set_keys(
    reward=env.reward_key,
    action=env.action_key,
    value=("agents", "state_value"),
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)

loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)
GAE = loss_module.value_estimator

optim = torch.optim.Adam(loss_module.parameters(), lr)


# # Loss for Quads
# loss_quad = ClipPPOLoss(
#     actor_network=quad_module,
#     critic_network=critic, # Centralized critic
#     clip_epsilon=clip_epsilon,
#     ...
# )
# loss_quad.set_keys(reward="team_reward", action=("quads", "action"), ...)

# # Loss for Fixed Wings
# loss_fixed = ClipPPOLoss(
#     actor_network=fixed_module,
#     critic_network=critic,
#     ...
# )
# loss_fixed.set_keys(reward="team_reward", action=("fixed", "action"), ...)


# =============================================================================
# SECTION 6: MAIN TRAINING LOOP
# =============================================================================
# Theory:
# 1. Collect Data: Run the agents in the environment.
# 2. Compute Advantage (GAE): Calculate returns and advantages without gradients.
# 3. Update Policy (PPO):
#    - Sample minibatches from the buffer.
#    - Calculate Loss (Actor Loss + Critic Loss + Entropy).
#    - Backpropagate and Step Optimizer.
#
# Implementation:
# - Note on Broadcasting: VMAS returns a global 'done' flag (Batch, 1). However,
#   TorchRL expects 'done' to match the reward shape (Batch, n_agents).
#   We manually expand/broadcast the done keys before calculating GAE.
#
# !! MODIFY !!
# Will stay: Collect -> GAE -> Buffer -> Train
# Modify: Since we added a GRU, must initialize the hidden_state 
# at the start of the episode and pass it along in the collector.
# Modify: The "Broadcasting" part where we fixed the done keys needs to 
# target our new group names ("quads" and "fixed") instead of the generic "agents".
# =============================================================================

pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")
episode_reward_mean_list = []

for tensordict_data in collector:
    
    # --- 1. Fix shapes for GAE ---
    # Expand 'done' and 'terminated' to match the agent dimension
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )
    tensordict_data.set(
        ("next", "agents", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )

    # --- 2. Compute GAE ---
    with torch.no_grad():
        GAE(
            tensordict_data,
            params=loss_module.critic_network_params,
            target_params=loss_module.target_critic_network_params,
        )

    # --- 3. Add to Buffer ---
    # Reshape to flatten batch and agent dimensions if needed, or just flatten batch
    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view)

    # --- 4. PPO Update ---
    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            
            # Sample a minibatch
            subdata = replay_buffer.sample()
            
            # Forward pass (Compute loss)
            loss_vals = loss_module(subdata)
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Backward pass
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    # Update policy weights (needed if collector is on a different device)
    collector.update_policy_weights_()

    # --- 5. Logging ---
    # We filter for 'done' states to get the final cumulative reward of the episode
    done = tensordict_data.get(("next", "agents", "done"))
    episode_reward_mean = (
        tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list.append(episode_reward_mean)
    pbar.set_description(f"episode_reward_mean = {episode_reward_mean:.2f}", refresh=False)
    pbar.update()

# --- Plot Results ---
plt.plot(episode_reward_mean_list)
plt.xlabel("Training iterations")
plt.ylabel("Reward")
plt.title("Episode reward mean")
plt.show()

# --- Render Final Policy ---
print("Rendering...")
with torch.no_grad():
    env.rollout(
        max_steps=max_steps,
        policy=policy,
        callback=lambda env, _: env.render(),
        auto_cast_to_device=True,
        break_when_any_done=False,
    )

# Uložení natrénovaného modelu
model_name = "mappo_policy_trained.pt"
torch.save(policy.state_dict(), model_name)
print(f"💾 Model uložen do souboru: {model_name}")