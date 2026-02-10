import torch
from torch import nn
from tqdm import tqdm
from torchrl.envs import TransformedEnv, InitTracker, GymWrapper
from torchrl.modules import ProbabilisticActor, TanhNormal, MLP, NormalParamExtractor
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.wildfire_gym_wrapper import WildfireMARLEnv
from src.wildfire_models import QuadActor

# 1. NASTAVENÍ - Pouze jeden dron!
device = "cuda" if torch.cuda.is_available() else "cpu"
# Konfigurace pouze s kvadrokoptérou
base_env = WildfireMARLEnv(agents_config=["quad_1"]) 
env = TransformedEnv(GymWrapper(base_env, device=device), InitTracker())

# 2. JEDNODUCHÁ POLITIKA (Jen QuadActor)
quad_net = TensorDictSequential(
    TensorDictModule(QuadActor(message_dim=8), 
                     in_keys=[("quads", "local_map"), ("quads", "self_state"), ("quads", "hidden_state")],
                     out_keys=[("quads", "params"), ("quads", "msg"), ("quads", "next_hidden_state")]),
    TensorDictModule(NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=0.01), in_keys=[("quads", "params")], out_keys=[("quads", "loc"), ("quads", "scale")])
)

policy = ProbabilisticActor(
    module=quad_net,
    spec=env.action_spec["quads", "action"],
    in_keys=[("quads", "loc"), ("quads", "scale")],
    out_keys=[("quads", "action")],
    distribution_class=TanhNormal,
    return_log_prob=True,
    log_prob_key="sample_log_prob" # Standardní klíč pro single-agent PPO
)

# 3. JEDNODUCHÝ LOSS A KRITIK
# Kritik se dívá jen na globální pozorování (které obsahuje stav drona a ohně)
critic = TensorDictModule(MLP(in_features=512, out_features=1, num_cells=[128, 128]), 
                          in_keys=["global_observation"], out_keys=["state_value"]).to(device)

loss_module = ClipPPOLoss(actor=policy, critic=critic)
optim = torch.optim.Adam(loss_module.parameters(), lr=1e-4) # Nižší LR pro stabilitu

# 4. TRÉNOVACÍ SMYČKA
frames_per_batch = 512
total_frames = 50_000
collector = SyncDataCollector(env, policy, frames_per_batch=frames_per_batch, total_frames=total_frames)

# Hlavní bar pro iterace (celkový trénink)
pbar_total = tqdm(total=total_frames // frames_per_batch, desc="Celkový trénink", position=0)

# Progress bar pro vnitřní kroky simulace (3600 kroků = 120s)
pbar_sim = tqdm(total=3600, desc="Aktuální let (Epizoda)", position=1, leave=False)

# Propojení baru s prostředím
env.base_env.sim_pbar = pbar_sim

for i, data in enumerate(collector):
    # 1. Update vah sítě
    loss_vals = loss_module(data)
    total_loss = loss_vals["loss_objective"] + loss_vals["loss_critic"]
    total_loss.backward()
    optim.step()
    optim.zero_grad()
    
    # 2. Statistiky a logování
    avg_reward = data["next", "reward"].mean().item()
    
    pbar_total.update(1)
    pbar_total.set_postfix({"Rwd": f"{avg_reward:.2f}", "Loss": f"{total_loss.item():.2f}"})
    
    # Update hlavního baru
    pbar_total.update(1)
    pbar_total.set_postfix({"Rwd": f"{data['next', 'reward'].mean():.2f}"})

pbar_total.close()
pbar_sim.close()

print("✅ Trénink základního vznášení dokončen.")