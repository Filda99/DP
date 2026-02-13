import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

class MAPPOActor(nn.Module):
    """
    ACTOR (Herec): Rozhoduje, co dron udělá.
    Vstup: Pouze LOKÁLNÍ data (local_map + self_state).
    Výstup: Akce (Roll, Pitch, Yaw, Throttle).
    """
    def __init__(self, obs_vector_size, action_dim=4):
        super().__init__()
        
        # 1. Zpracování obrázku (Local Map 32x32) pomocí CNN
        self.cnn = nn.Sequential(
            # Vstup: 1 kanál, 32x32 -> Výstup: 16 filtrů, 15x15
            nn.Conv2d(1, 16, kernel_size=4, stride=2), 
            nn.ReLU(),
            # Vstup: 16 filtrů, 15x15 -> Výstup: 32 filtrů, 6x6
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            # 32 * 6 * 6 = 1152 hodnot. Stlačíme to do 128.
            nn.Linear(1152, 128),
            nn.ReLU()
        )
        
        # 2. Zpracování čísel (Self State) pomocí MLP
        self.mlp = nn.Sequential(
            nn.Linear(obs_vector_size, 64),
            nn.ReLU()
        )
        
        # 3. Spojení obou větví (128 z CNN + 64 z MLP = 192) a výpočet akce
        self.action_mean = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh() # Tanh zaručí, že výstup bude vždy mezi -1.0 a 1.0!
        )
        
        # 4. Odchylka (Explorace)
        # Sítě v PPO nevypivnou jen jednu akci, ale Gaussovu křivku (průměr a odchylku).
        # Odchylku se učí nezávisle na stavu.
        self.action_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, local_map, self_state, action=None):
        # Proženeme data sítěmi
        cnn_features = self.cnn(local_map)
        mlp_features = self.mlp(self_state)
        
        # Spojíme je dohromady
        combined = torch.cat([cnn_features, mlp_features], dim=1)
        
        # Vypočítáme průměrnou akci (mean)
        action_mean = self.action_mean(combined)
        
        # Vytvoříme pravděpodobnostní rozdělení (pro exploraci)
        action_std = torch.exp(self.action_logstd)
        probs = Normal(action_mean, action_std)
        
        # Pokud neposíláme akci (hrajeme), tak ji nasamplujeme
        if action is None:
            action = probs.sample()
            
        # Vrátíme akci, její logaritmickou pravděpodobnost a entropii (důležité pro PPO vzorce)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


class MAPPOCritic(nn.Module):
    """
    CRITIC (Kritik): Hodnotí, jak dobrý je současný stav.
    Vstup: GLOBÁLNÍ data (Pohled boha na mapu ohně a všechny drony).
    Výstup: Jedno číslo (očekávaná budoucí odměna).
    """
    def __init__(self, global_state_size):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(global_state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Vrací přesně jednu hodnotu!
        )

    def forward(self, global_state):
        return self.network(global_state)