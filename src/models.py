import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

# ==========================================
# POMOCNÉ MODULY (ATTENTION)
# ==========================================

class MultiHeadAttention(nn.Module):
    """
    Univerzální Attention blok.
    Pokud query == keys, funguje jako Self-Attention (pro sousedy Scoutů).
    Pokud query != keys, funguje jako Cross-Attention (pro zprávy Commanderovi).
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value, key_padding_mask=None):
        """
        query: [Batch, 1, Embed_Dim] (pro Commandera) nebo [Batch, Seq, Embed_Dim] (pro Scouta)
        key, value: [Batch, Seq_Len, Embed_Dim] (Sousedé nebo Zprávy)
        key_padding_mask: [Batch, Seq_Len] - True tam, kde jsou 'mrtví' agenti (padding)
        """
        # MultiheadAttention v PyTorchu vyžaduje masku: True = IGNOROVAT
        attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
        
        # Pokud jsou všichni sousedé/zprávy zamaskovaní (True), Attention vrátí NaN.
        # Musíme tyto NaN nahradit nulami, jinak se celá síť rozsype.
        if torch.isnan(attn_output).any():
            attn_output = torch.nan_to_num(attn_output, nan=0.0)
            
        return attn_output


class ScoutActor(nn.Module):
    """
    ACTOR (Herec): Rozhoduje, co dron udělá.
    Vstup: Pouze LOKÁLNÍ data (local_map + self_state).
    Výstup: Akce (Roll, Pitch, Yaw, Throttle).
    """
    def __init__(self, self_state_dim, action_dim=4, msg_dim=5, hidden_dim=128):
        super().__init__()
        
        # 1. Zpracování obrázku (Local Map 32x32) pomocí CNN
        # Bere černobílou mřížku ohně 32x32 a pomocí konvolučních vrstev z ní vytáhne tzv. Visual Features (vektor 128 čísel). Hledá tvary, hrany a intenzitu.
        self.cnn = nn.Sequential(
            # Vstup: 1 kanál, 32x32 -> Výstup: 16 filtrů, 15x15
            nn.Conv2d(1, 16, kernel_size=4, stride=2), 
            nn.ReLU(),
            # Vstup: 16 filtrů, 15x15 -> Výstup: 32 filtrů, 6x6
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            # 32 * 6 * 6 = 1152 hodnot. Stlačíme to do 128.
            nn.Linear(1152, hidden_dim),
            nn.ReLU()
        )

        # 2. Sousedé Scoutové (Self-Attention)
        # Vstup: Relativní pozice souseda [x, y, z] -> Embedding
        # Používáme Self-Attention . Dron se zeptá (Query = jeho vlastní stav): "Jsem tady a letím tamhle, kdo z vás 
        # je pro mě teď důležitý?" Sousedé (Keys/Values) odpoví svými pozicemi.
        #
        # Síť dynamicky ignoruje drony, které jsou na druhé straně mapy, a zaměří pozornost (dá vysoké váhy) dronům, 
        # se kterými by se mohla srazit nebo u kterých je oheň. 
        # Dokáže tak zpracovat 1 i 50 kolegů najednou a vždy z toho "vypadne" jeden fixní vektor (64 čísel). 
        # Navíc pomocí neighbor_mask umí ignorovat "mrtvé" drony.
        self.neighbor_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )
        self.neighbor_attention = MultiHeadAttention(embed_dim=64, num_heads=2)
        
        # 3. Zpracování čísel (Self State) pomocí MLP
        # Klasické MLP (Lineární vrstva). Vezme 10 čísel o rychlosti a vzdálenosti od stěn a udělá z nich vektor 64 čísel.
        self.self_embed = nn.Sequential(
            nn.Linear(self_state_dim, 64),
            nn.ReLU()
        )

        # 4.1 LAYER NORMALIZATION
        # Stabilizuje vstup do paměti GRU. Zabraňuje explozím gradientů.
        # Vstupní velikost je součet všech větví: 128 (CNN) + 64 (Self) + 64 (Neighbors) = 256
        self.layer_norm = nn.LayerNorm(128 + 64 + 64)

        # 4.2 PAMĚŤ (GRU)
        # Vstup do GRU: CNN(128) + Neighbor_Attn(64) + Self(64) = 256
        # Všechny tři vjemy (Oči + Tělo + Sousedé) se spojí dohromady a pošlou se do GRU vrstvy. 
        # To je paměťová buňka. Na vstup jde kromě vjemů i hidden_state (Stav paměti z minulého kroku).
        #
        # Dron si nyní dokáže pamatovat: "Před 5 sekundami jsem viděl oheň na východě, poletím to tam 
        # zkontrolovat znovu," nebo si dokáže vytvořit strategii hlídkování (zametání mapy), aby nelétal v kruzích.
        self.gru = nn.GRU(input_size=128 + 64 + 64, hidden_size=hidden_dim, batch_first=True)
        
        # 5. VÝSTUPNÍ HLAVY
        # 1. Action Head (Pohyb)
        # Už nepotřebuje "přemýšlet" (zmenšovat vrstvy), jen převede myšlenku (hidden_dim) na pohyb (action_dim).
        # Linear(128 -> 4)
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # 2. Message Head (Komunikace)
        # Vypouští vektor 5 čísel, která posíláme letadlu
        self.msg_head = nn.Sequential(
            nn.Linear(hidden_dim, msg_dim),
            nn.Tanh() # Zprávy budou v rozsahu -1 až 1
        )

    def forward(self, local_map, self_state, neighbor_states, neighbor_mask, hidden_state):
        """
        local_map: [Batch, 1, 32, 32]
        self_state: [Batch, self_state_dim]
        neighbor_states: [Batch, Max_Neighbors, 3]
        neighbor_mask: [Batch, Max_Neighbors] (True = padding/dead)
        hidden_state: [1, Batch, Hidden_Dim] (Stav paměti z minula)
        """
        batch_size = local_map.size(0)
        
        # 1. Extrakce příznaků
        # Proženeme data sítěmi
        vis_feat = self.cnn(local_map)     # [Batch, 128]
        self_feat = self.self_embed(self_state) # [Batch, 64]
        
        # 2. Zpracování sousedů (Attention)
        # Nejdřív embeddujeme každého souseda zvlášť
        neigh_embed = self.neighbor_embed(neighbor_states) # [Batch, N, 64]
        # Pak aplikujeme Self-Attention (každý s každým, ale nás zajímá agregace)
        # Použijeme self_feat jako Query, abychom zjistili "kdo je důležitý PRO MĚ"
        # Query: [Batch, 1, 64]
        query = self_feat.unsqueeze(1)
        neigh_context = self.neighbor_attention(query, neigh_embed, neigh_embed, key_padding_mask=neighbor_mask)
        neigh_context = neigh_context.squeeze(1) # [Batch, 64]

        # 3. Fúze a Paměť
        combined = torch.cat([vis_feat, self_feat, neigh_context], dim=1) # [Batch, 256]

        # 3.1 Layer Normalization
        combined = self.layer_norm(combined)
        
        # 3.2 GRU vyžaduje sekvenční dimenzi, přidáme ji [Batch, 1, Features]
        gru_out, new_hidden = self.gru(combined.unsqueeze(1), hidden_state)
        gru_out = gru_out.squeeze(1) # [Batch, Hidden]

        # 4. Výstupy
        # A) Akce
        action_mean = self.action_mean(gru_out)
        action_mean = torch.tanh(action_mean) # PPO akce -1..1
        
        action_std = torch.exp(self.action_logstd)
        dist = Normal(action_mean, action_std)

        # B) Zpráva
        message = self.msg_head(gru_out)

        return dist, message, new_hidden


# ==========================================
# 2. COMMANDER ACTOR (Letadlo)
# ==========================================

class CommanderActor(nn.Module):
    def __init__(self, self_state_dim, msg_input_dim=5, action_dim=4, hidden_dim=128):
        super().__init__()
        
        # 1. SELF STATE ENCODER (Jeho vlastní fyzický stav (rychlost, výška, hladina vody v nádrži, pozice))
        # Převede tato čísla na vektor (např. 64 čísel), který reprezentuje "Kdo jsem a co mám k dispozici".
        self.self_embed = nn.Sequential(
            nn.Linear(self_state_dim, 64),
            nn.ReLU()
        )

        # 2. MESSAGE ENCODER
        # Seznam zpráv od všech Scoutů (např. od 5 dronů). Každá zpráva je ten vektor 5 čísel, který vyplivl dron.
        # Každou zprávu "nafoukne" (pomocí Lineární vrstvy) na vektor 64 čísel
        self.msg_embed = nn.Sequential(
            nn.Linear(msg_input_dim, 64),
            nn.ReLU()
        )

        # 3. CROSS-ATTENTION (Naslouchání)
        # Query = Letadlo, Key/Value = Zprávy
        # Self-Attention (u dronů) porovnává sousedy navzájem (aby do sebe nenarazili).
        # Cross-Attention porovnává MĚ (Letadlo) s NIMI (Drony).
        #
        # Jak to funguje:
        # Query (Dotaz): "Jsem na pozici [100, 100]."
        # Keys (Nabídky):
        # Dron 1: "Jsem na [10, 10] a vidím trávu." -> Nízká shoda (daleko + nic zajímavého).
        # Dron 2: "Jsem na [90, 100] a vidím PEKLO." -> Vysoká shoda! (blízko + oheň).
        # Váhy (Attention Weights): Síť přiřadí Dronu 1 váhu 0.05 a Dronu 2 váhu 0.95.
        # Weighted Sum: Síť vezme 5 % zprávy Drona 1 a 95 % zprávy Drona 2 a sečte je.
        # Výsledek (context_vector): Vznikne jeden vektor, který obsahuje esenci toho nejdůležitějšího 
        # z celého bojiště, přesně na míru pro toto konkrétní letadlo.
        self.attention = MultiHeadAttention(embed_dim=64, num_heads=4)

        # 4.1 LAYER NORMALIZATION
        # Stabilizace vstupu před rozhodovací částí
        # 64 (Self) + 64 (Attention Context) = 128
        self.layer_norm = nn.LayerNorm(64 + 64)

        # 4.2 DECISION MLP
        # Spojí stav letadla + kontext ze zpráv
        # Jednoduše spojí (concatenation) vektor letadla a kontextový vektor.
        # Prožene to přes MLP (neuronové vrstvy), které vymyslí strategii: 
        # "Mám vodu + oheň je pode mnou = OTEVŘÍT NÁDRŽ." nebo "Nemám vodu + oheň je pode mnou = LETĚT DOPLNIT VODU."
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 5. ACTION HEAD
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, self_state, incoming_messages, message_mask):
        """
        self_state: [Batch, State_Dim]
        incoming_messages: [Batch, Num_Scouts, Msg_Dim]
        message_mask: [Batch, Num_Scouts] (True = padding/dead drone)
        """
        
        # 1. Encodery
        self_feat = self.self_embed(self_state)   # [Batch, 64]
        msg_feat = self.msg_embed(incoming_messages) # [Batch, N, 64]

        # 2. Cross-Attention
        # Letadlo (Query) se ptá zpráv (Key/Value)
        query = self_feat.unsqueeze(1) # [Batch, 1, 64]
        
        context_vector = self.attention(query, msg_feat, msg_feat, key_padding_mask=message_mask)
        context_vector = context_vector.squeeze(1) # [Batch, 64]

        # 3. Rozhodování
        combined = torch.cat([self_feat, context_vector], dim=1) # [Batch, 128]
        combined = self.layer_norm(combined)    # Stabilizace před fúzí
        features = self.fusion(combined)        # [Batch, Hidden]

        # 4. Akce
        action_mean = self.action_mean(features)
        action_mean = torch.tanh(action_mean)
        
        action_std = torch.exp(self.action_logstd)
        dist = Normal(action_mean, action_std)

        return dist, None, None # (Vracíme None pro kompatibilitu API s message/hidden)


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