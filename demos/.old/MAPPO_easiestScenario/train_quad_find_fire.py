import torch
from torch import nn
from tqdm import tqdm
from torchrl.envs import TransformedEnv, InitTracker, GymWrapper
from torchrl.modules import ProbabilisticActor, TanhNormal, MLP, NormalParamExtractor
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.wildfire_gym_wrapper import WildfireMARLEnv
from src.wildfire_models import QuadActor

print("🚁 OPRAVENÝ KVADROKOPTÉRA TRÉNINK")
print("=" * 50)

# 1. NASTAVENÍ - Pouze jeden dron!
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Používám zařízení: {device}")

# Konfigurace pouze s kvadrokoptérou
base_env = WildfireMARLEnv(agents_config=["quad_1"]) 
env = TransformedEnv(GymWrapper(base_env, device=device), InitTracker())

# 2. JEDNODUCHÁ POLITIKA (Jen QuadActor)
quad_net = TensorDictSequential(
    TensorDictModule(QuadActor(message_dim=8), 
                     in_keys=[("quads", "local_map"), ("quads", "self_state"), ("quads", "hidden_state")],
                     out_keys=[("quads", "params"), ("quads", "msg"), ("quads", "next_hidden_state")]),
    TensorDictModule(NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=0.01), 
                     in_keys=[("quads", "params")], 
                     out_keys=[("quads", "loc"), ("quads", "scale")])
).to(device)

policy = ProbabilisticActor(
    module=quad_net,
    spec=env.action_spec["quads", "action"],
    in_keys=[("quads", "loc"), ("quads", "scale")],
    out_keys=[("quads", "action")],
    distribution_class=TanhNormal,
    return_log_prob=True,
    log_prob_key=("quads", "sample_log_prob")  # Správný klíč pro multi-agent
).to(device)

# 3. KRITIK A LOSS S GAE
critic = TensorDictModule(
    MLP(in_features=512, out_features=1, num_cells=[128, 128]), 
    in_keys=["global_observation"], 
    out_keys=["state_value"]
).to(device)

loss_module = ClipPPOLoss(
    actor_network=policy, 
    critic_network=critic,
    clip_epsilon=0.2,
    entropy_coeff=0.02,  # Více explorace
    normalize_advantage=True  # Zapnout normalizaci
).to(device)

# Nastavení klíčů pro loss
loss_module.set_keys(
    reward="reward",
    action=("quads", "action"), 
    value="state_value",
    done="done",
    terminated="terminated",
    sample_log_prob=("quads", "sample_log_prob")
)

# Přidání GAE
loss_module.make_value_estimator(ValueEstimators.GAE, gamma=0.99, lmbda=0.95)
GAE = loss_module.value_estimator

optim = torch.optim.Adam(loss_module.parameters(), lr=3e-4)

# 4. REPLAY BUFFER A COLLECTOR S KONZERVATIVNÍMI PARAMETRY
frames_per_batch = 500  # Menší batch pro stabilnější učení
minibatch_size = 100   # Menší minibatch
num_epochs = 3         # Méně epoch
total_frames = 20_000  # Kratší trénink pro test

collector = SyncDataCollector(
    env, 
    policy, 
    frames_per_batch=frames_per_batch, 
    total_frames=total_frames,
    device=device
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch, device=device),
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,
)

# Progress bars
pbar_total = tqdm(total=total_frames // frames_per_batch, desc="Celkový trénink")

# Debug prostředí
print(f"Environment action space: {env.action_spec}")
print(f"Environment observation space: {env.observation_spec}")

# Test reset
reset_data = env.reset()
print(f"Reset data keys: {reset_data.keys()}")

print("\n🎯 Začínám trénink...")

try:
    # 5. HLAVNÍ TRÉNOVACÍ SMYČKA
    for i, data in enumerate(collector):
        # ROBUSTNÍ fix done keys pro GAE - bez použití get_item_shape
        next_done = data.get(("next", "done"))
        next_terminated = data.get(("next", "terminated"))
        
        # Safely expand to match expected shape [batch, agents]
        if next_done.dim() == 2:  # [batch, 1]
            expanded_done = next_done  # Already correct shape
        else:  # [batch]
            expanded_done = next_done.unsqueeze(-1)  # [batch, 1]
            
        if next_terminated.dim() == 2:
            expanded_terminated = next_terminated
        else:
            expanded_terminated = next_terminated.unsqueeze(-1)
        
        data.set(("next", "quads", "done"), expanded_done)
        data.set(("next", "quads", "terminated"), expanded_terminated)
        data.set(("next", "quads", "reward"), data.get(("next", "reward")).unsqueeze(-1))
        # Výpočet GAE
        with torch.no_grad():
            try:
                GAE(
                    data,
                    params=loss_module.critic_network_params,
                    target_params=loss_module.target_critic_network_params,
                )
            except Exception as e:
                print(f"⚠️  GAE chyba: {e}")
                continue
        
        # Přidání do bufferu
        data_view = data.reshape(-1)
        replay_buffer.extend(data_view)
        
        # PPO update s více epochami
        total_loss = torch.tensor(0.0)
        for epoch in range(num_epochs):
            for _ in range(frames_per_batch // minibatch_size):
                try:
                    subdata = replay_buffer.sample()
                    
                    loss_vals = loss_module(subdata)
                    total_loss = (
                        loss_vals["loss_objective"] 
                        + loss_vals["loss_critic"] 
                        + loss_vals["loss_entropy"]
                    )
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                    optim.step()
                    optim.zero_grad()
                except Exception as e:
                    print(f"⚠️  Loss computation chyba: {e}")
                    continue
        
        # Update policy weights
        collector.update_policy_weights_()
        
        # Robustnější statistiky
        try:
            reward_data = data.get(("next", "reward"))
            if reward_data is not None and reward_data.numel() > 0:
                avg_reward = reward_data.mean().item()
            else:
                avg_reward = 0.0
        except:
            avg_reward = 0.0
        
        pbar_total.update(1)
        pbar_total.set_postfix({
            "Reward": f"{avg_reward:.3f}",
            "Loss": f"{total_loss.item():.3f}",
        })
        
        if i % 5 == 0:
            print(f"\nBatch {i}: Avg Reward={avg_reward:.3f}, Loss={total_loss.item():.3f}")

    pbar_total.close()
    print("\n✅ Trénink dokončen úspěšně!")
    print("🎯 Dron by se měl naučit základní hovering a hledání ohně")
    
except KeyboardInterrupt:
    print("\n⏹️  Trénink přerušen uživatelem")
except Exception as e:
    print(f"\n❌ Chyba během tréninku: {e}")
    import traceback
    traceback.print_exc()
    
print("\n🔚 KONEC")