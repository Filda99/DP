import torch
from torch import nn
import numpy as np
import os
import sys

# TorchRL a TensorDict
from torchrl.envs import TransformedEnv, InitTracker, GymWrapper
from torchrl.modules import ProbabilisticActor, TanhNormal, MLP, NormalParamExtractor
from tensordict.nn import TensorDictModule, TensorDictSequential, TensorDictModuleBase
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss

from tqdm import tqdm

# Projektové importy
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.wildfire_gym_wrapper import WildfireMARLEnv
from src.wildfire_models import QuadActor, FixedWingActor

# =============================================================================
# 1. HETEROGENNÍ DISTRIBUCE (Matematické spojení týmu)
# =============================================================================

class JointDist:
    """Spojuje distribuce drona a letadla. PPO ji používá pro výpočet log_prob a entropie."""
    def __init__(self, dist_q, dist_f):
        self.dist_q = dist_q
        self.dist_f = dist_f
    
    def log_prob(self, action):
        # Joint Log-Prob = součet log_probů jednotlivců (nezávislí agenti)
        act_q = action.get(("quads", "action"))
        act_f = action.get(("fixed", "action"))
        return self.dist_q.log_prob(act_q).sum(-1) + self.dist_f.log_prob(act_f).sum(-1)
    
    def entropy(self):
        # Celková neurčitost týmu
        return self.dist_q.entropy().sum(-1) + self.dist_f.entropy().sum(-1)
    
    def sample(self, sample_shape=torch.Size()):
        # Generování náhodných akcí pro trénink
        return TensorDict({
            "quads": TensorDict({"action": self.dist_q.sample(sample_shape)}, batch_size=sample_shape),
            "fixed": TensorDict({"action": self.dist_f.sample(sample_shape)}, batch_size=sample_shape),
        }, batch_size=sample_shape)

# =============================================================================
# 2. HETEROGENNÍ POLITIKA (Explicitní definice)
# =============================================================================

class TeamPolicy(TensorDictModuleBase):
    def __init__(self, env_action_spec):
        super().__init__()
        self.in_keys = [("quads", "local_map"), ("quads", "self_state"), 
                        ("quads", "hidden_state"), ("fixed", "self_state")]
        self.out_keys = [("quads", "action"), ("fixed", "action"), "sample_log_prob"]

        # --- EXPLICITNÍ ULOŽENÍ LOGIKY (Tím zmizí NotImplementedError) ---
        
        # Logika Kvadrokoptéry: [Vstupy] -> [Mean, Std]
        self.q_logic = TensorDictSequential(
            TensorDictModule(QuadActor(message_dim=8), 
                             in_keys=[("quads", "local_map"), ("quads", "self_state"), ("quads", "hidden_state")],
                             out_keys=[("quads", "params"), ("quads", "msg"), ("quads", "next_hidden_state")]),
            TensorDictModule(NormalParamExtractor(), in_keys=[("quads", "params")], out_keys=[("quads", "loc"), ("quads", "scale")])
        )
        self.quad_actor = ProbabilisticActor(module=self.q_logic, spec=env_action_spec["quads", "action"],
                                             in_keys=[("quads", "loc"), ("quads", "scale")], out_keys=[("quads", "action")],
                                             distribution_class=TanhNormal, return_log_prob=True, log_prob_key=("quads", "sample_log_prob"))

        # Logika Letadla: [Vstupy] -> [Mean, Std]
        self.f_logic = TensorDictSequential(
            TensorDictModule(FixedWingActor(message_dim=8), in_keys=[("fixed", "self_state"), ("quads", "msg")], out_keys=[("fixed", "params")]),
            TensorDictModule(NormalParamExtractor(), in_keys=[("fixed", "params")], out_keys=[("fixed", "loc"), ("fixed", "scale")])
        )
        self.fixed_actor = ProbabilisticActor(module=self.f_logic, spec=env_action_spec["fixed", "action"],
                                              in_keys=[("fixed", "loc"), ("fixed", "scale")], out_keys=[("fixed", "action")],
                                              distribution_class=TanhNormal, return_log_prob=True, log_prob_key=("fixed", "sample_log_prob"))

    def forward(self, td):
        # Standardní průchod: Dron -> Letadlo -> Součet log_prob
        td = self.quad_actor(td)
        td = self.fixed_actor(td)
        td.set("sample_log_prob", td.get(("quads", "sample_log_prob")).sum(-1) + td.get(("fixed", "sample_log_prob")).sum(-1))
        return td

    def get_dist(self, td):
        # PPO volá toto pro trénink. Voláme přímo naše uložené sekvence (self.q_logic / self.f_logic).
        td = self.q_logic(td)
        dist_q = self.quad_actor.get_dist(td)
        
        td = self.f_logic(td)
        dist_f = self.fixed_actor.get_dist(td)
        
        return JointDist(dist_q, dist_f)

# =============================================================================
# 3. LOSS MODUL (S kompletními anotacemi)
# =============================================================================

class HeteroPPOLoss(ClipPPOLoss):
    # Anotace zajistí, že TorchRL najde všechny parametry pro gradienty a skryje varování
    actor_network: TeamPolicy
    critic_network: TensorDictModule
    actor_network_params: nn.ParameterList
    critic_network_params: nn.ParameterList
    target_actor_network_params: nn.ParameterList
    target_critic_network_params: nn.ParameterList

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor_keys.sample_log_prob = "sample_log_prob"

    def _get_cur_log_prob(self, tensordict):
        dist = self.actor_network.get_dist(tensordict)
        return dist.log_prob(tensordict), dist, False

# =============================================================================
# 4. INICIALIZACE A SMYČKA
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Zařízení: {device}")

base_env = WildfireMARLEnv(agents_config=["quad_1", "fw_1"])
env = TransformedEnv(GymWrapper(base_env, device=device), InitTracker())

policy = TeamPolicy(env.action_spec).to(device)
critic_net = MLP(in_features=512, out_features=1, num_cells=[256, 256]).to(device)
critic = TensorDictModule(critic_net, in_keys=["global_observation"], out_keys=["state_value"])

loss_module = HeteroPPOLoss(actor=policy, critic=critic)
optim = torch.optim.Adam(loss_module.parameters(), lr=3e-4)

frames_per_batch = 1000
total_frames = 100_000
collector = SyncDataCollector(env, policy, frames_per_batch=frames_per_batch, total_frames=total_frames)



print("🚀 Zahajuji trénink...")

# Vypočítáme celkový počet iterací pro tqdm
total_iterations = total_frames // frames_per_batch
pbar = tqdm(total=total_iterations, desc="Training Progress")

for i, data in enumerate(collector):
    # Výpočet ztráty (PPO + Value Loss)
    loss_vals = loss_module(data)
    total_loss = loss_vals["loss_objective"] + loss_vals["loss_critic"]
    
    # Backpropagace (Update vah neuronů)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
    optim.step()
    optim.zero_grad()
    
    # Průběžné výsledky
    avg_reward = data["next", "reward"].mean().item()
    print(f"Iter {i:03d} | Odměna: {avg_reward:.4f} | Ztráta: {total_loss.item():.4f}")