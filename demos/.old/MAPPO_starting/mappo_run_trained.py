import torch
from torch import multiprocessing
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.envs import TransformedEnv, RewardSum

# 1. NASTAVENÍ (Musí být stejné jako při tréninku)
scenario_name = "navigation"
n_agents = 3
device = "cpu" # Pro vizualizaci stačí CPU, je to jednodušší
max_steps = 400 # Můžeme nechat běžet déle než při tréninku

print("🚀 Načítám prostředí...")
# Vytvoříme prostředí (stejné jako předtím)
env = VmasEnv(
    scenario=scenario_name,
    num_envs=1, # Tady stačí jen 1 simulace (chceme se dívat, ne trénovat paralelně)
    continuous_actions=True,
    max_steps=max_steps,
    device=device,
    n_agents=n_agents,
)

# 2. DEFINICE MOZKU (Musí být kopie architektury z tréninku)
# Bez tohoto PyTorch neví, jak "policy.pt" přečíst.
share_parameters_policy = True

policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=2 * env.full_action_spec[env.action_key].shape[-1],
        n_agents=env.n_agents,
        centralised=False,
        share_params=share_parameters_policy,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    ),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    policy_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "loc"), ("agents", "scale")],
)

policy = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec_unbatched,
    in_keys=[("agents", "loc"), ("agents", "scale")],
    out_keys=[env.action_key],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.full_action_spec_unbatched[env.action_key].space.low,
        "high": env.full_action_spec_unbatched[env.action_key].space.high,
    },
    return_log_prob=False, # Při testování nepotřebujeme log_prob
)

# 3. NAČTENÍ VAH (To je ta "předaná funkce")
try:
    model_path = "mappo_policy_trained.pt"
    # map_location='cpu' zajistí, že to poběží i když byl model trénován na GPU
    loaded_state = torch.load(model_path, map_location=torch.device(device))
    policy.load_state_dict(loaded_state)
    print("✅ Model úspěšně načten!")
except FileNotFoundError:
    print("❌ Chyba: Soubor s modelem nenalezen. Spusťte nejdřív trénink!")
    exit()

# 4. SPUŠTĚNÍ DEMO (INFERENCE)
print("🎥 Spouštím vizualizaci...")

# Policy přepneme do eval módu (vypne náhodný Dropout atd.)
policy.eval()

with torch.no_grad(): # Vypneme počítání gradientů (šetří paměť)
    env.rollout(
        max_steps=max_steps,
        policy=policy,
        callback=lambda env, _: env.render(), # Vykreslí každé okno
        auto_cast_to_device=True,
        break_when_any_done=False,
    )

print("🏁 Hotovo.")