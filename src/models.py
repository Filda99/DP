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
        # # MultiheadAttention v PyTorchu vyžaduje masku: True = IGNOROVAT
        # attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
        
        # # Pokud jsou všichni sousedé/zprávy zamaskovaní (True), Attention vrátí NaN.
        # # Musíme tyto NaN nahradit nulami, jinak se celá síť rozsype.
        # if torch.isnan(attn_output).any():
        #     attn_output = torch.nan_to_num(attn_output, nan=0.0)
            
        # return attn_output

        if key_padding_mask is not None:
            # Zjistíme, které řádky v batchi mají všechny prvky zamaskované (True)
            all_masked = key_padding_mask.all(dim=1)
            
            if all_masked.any():
                # Vytvoříme 'bezpečnou' masku – tam, kde jsou všichni mrtví, 
                # dočasně odmaskujeme první prvek, aby Softmax nehodil NaN
                safe_mask = key_padding_mask.clone()
                safe_mask[all_masked, 0] = False
                
                attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=safe_mask)
                
                # Výsledek pro ty, co mají být 'všichni mrtví', ručně vynulujeme
                # query.shape je (Batch, Seq, Dim), attn_output taky
                attn_output[all_masked] = 0.0
                return attn_output

        # Standardní cesta, pokud není vše zamaskováno
        attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
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

    # def forward(self, local_map, self_state, neighbor_states, neighbor_mask, hidden_state):
    #     """
    #     local_map: [Batch, 1, 32, 32]
    #     self_state: [Batch, self_state_dim]
    #     neighbor_states: [Batch, Max_Neighbors, 3]
    #     neighbor_mask: [Batch, Max_Neighbors] (True = padding/dead)
    #     hidden_state: [1, Batch, Hidden_Dim] (Stav paměti z minula)
    #     """
    #     batch_size = local_map.size(0)
    #     seq_len = local_map.size(1) if local_map.dim() > 3 else 1
        
    #     # 1. Extrakce příznaků
    #     # Proženeme data sítěmi
    #     # vis_feat = self.cnn(local_map)     # [Batch, 128]
    #     # self_feat = self.self_embed(self_state) # [Batch, 64]

    #     # Pokud máme sekvenci [Batch, Seq, 1024], musíme ji změnit na [Batch*Seq, 1, 32, 32]
    #     if local_map.dim() == 3: # Trénink (sekvence)
    #         # local_map má [15, 200, 1024]
    #         local_map_for_cnn = local_map.reshape(-1, 1, 32, 32)
    #     else: # Demo (jeden krok)
    #         # local_map má [1, 1, 32, 32]
    #         local_map_for_cnn = local_map

    #     # Teď CNN dostane správný tvar
    #     vis_feat = self.cnn(local_map_for_cnn) # [Batch*Seq, 128]

    #     # Pokud jsme v tréninku, vrátíme vis_feat do tvaru sekvence [Batch, Seq, 128]
    #     if local_map.dim() == 3:
    #         vis_feat = vis_feat.reshape(batch_size, seq_len, -1)
    #     # ----------------------

    #     # To samé musíme udělat pro self_state, pokud je to sekvence
    #     if self_state.dim() == 3:
    #         self_feat = self.self_embed(self_state) # [Batch, Seq, 64]
    #     else:
    #         self_feat = self.self_embed(self_state) # [Batch, 64]
        
    #     # 2. Zpracování sousedů (Attention)
    #     # Nejdřív embeddujeme každého souseda zvlášť
    #     neigh_embed = self.neighbor_embed(neighbor_states) # [Batch, N, 64]
    #     # Pak aplikujeme Self-Attention (každý s každým, ale nás zajímá agregace)
    #     # Použijeme self_feat jako Query, abychom zjistili "kdo je důležitý PRO MĚ"
    #     # Query: [Batch, 1, 64]
    #     query = self_feat.unsqueeze(1)
    #     neigh_context = self.neighbor_attention(query, neigh_embed, neigh_embed, key_padding_mask=neighbor_mask)
    #     neigh_context = neigh_context.squeeze(1) # [Batch, 64]

    #     # # 3. Fúze a Paměť
    #     # combined = torch.cat([vis_feat, self_feat, neigh_context], dim=1) # [Batch, 256]

    #     # # 3.1 Layer Normalization
    #     # combined = self.layer_norm(combined)
        
    #     # # 3.2 GRU vyžaduje sekvenční dimenzi, přidáme ji [Batch, 1, Features]
    #     # gru_out, new_hidden = self.gru(combined.unsqueeze(1), hidden_state)
    #     # gru_out = gru_out.squeeze(1) # [Batch, Hidden]

    #     # # 4. Výstupy
    #     # # A) Akce
    #     # action_mean = self.action_mean(gru_out)
    #     # action_mean = torch.tanh(action_mean) # PPO akce -1..1
        
    #     # action_std = torch.exp(self.action_logstd)
    #     # dist = Normal(action_mean, action_std)

    #     # # B) Zpráva
    #     # message = self.msg_head(gru_out)

    #     # return dist, message, new_hidden

    #     # 3. Fúze a Paměť
    #     combined = torch.cat([vis_feat, self_feat, neigh_context], dim=1) # [Batch, 256]
    #     combined = self.layer_norm(combined)
        
    #     # --- ZMĚNA PRO SEKVENČNÍ TRÉNINK ---
    #     # Pokud combined má 2 dimenze [Batch, Feat], jsme v DEMU/SIMULACI -> přidáme časovou dimenzi 1.
    #     # Pokud už má 3 dimenze [Batch, Seq, Feat], jsme v TRÉNINKU -> necháme ho, jak je.
    #     if combined.dim() == 2:
    #         combined = combined.unsqueeze(1) # [Batch, 1, Feat]
            
    #     gru_out, new_hidden = self.gru(combined, hidden_state)
        
    #     # GRU vrátí [Batch, Seq, 128]. Pro lineární vrstvy to musíme "zploštit" zpět do 2D.
    #     # Výsledek bude mít tvar [Batch * Seq, 128].
    #     features = gru_out.reshape(-1, 128)
    #     # -----------------------------------

    #     # 4. Výstupy (používáme 'features' místo 'gru_out')
    #     # A) Akce
    #     action_mean = self.action_mean(features)
    #     action_mean = torch.tanh(action_mean) # PPO akce -1..1
        
    #     action_std = torch.exp(self.action_logstd)
    #     dist = Normal(action_mean, action_std)

    #     # B) Zpráva
    #     message = self.msg_head(features)

    #     return dist, message, new_hidden


    def forward(self, local_map, self_state, neighbor_states, neighbor_mask, hidden_state):
        # 1. Detekce režimu
        # V tréninku dostáváme [15, 200, ...], v demu [1, ...]
        is_sequential = (local_map.dim() >= 4) # Maps jsou [Batch, Seq, 1, 32, 32]
        batch_size = local_map.size(0)
        seq_len = local_map.size(1) if is_sequential else 1

        # 2. TOTÁLNÍ ZPLOŠTĚNÍ (Flattening)
        # Spojíme Epizody a Kroky (15 * 200 = 3000) pro paralelní výpočet
        if is_sequential:
            local_map = local_map.reshape(-1, 1, 32, 32)
            self_state = self_state.reshape(-1, self_state.size(-1))
            # Sousedé: [15, 200, N, 3] -> [3000, N, 3]
            neighbor_states = neighbor_states.reshape(batch_size * seq_len, -1, neighbor_states.size(-1))
            # Maska: [15, 200, N] -> [3000, N]
            neighbor_mask = neighbor_mask.reshape(batch_size * seq_len, -1)
        else:
            # Demo/Inference režim
            if local_map.dim() == 2: local_map = local_map.reshape(-1, 1, 32, 32)
            if self_state.dim() == 1: self_state = self_state.unsqueeze(0)

        # 3. SENZORY (CNN + Attention)
        vis_feat = self.cnn(local_map)           # [3000, 128]
        self_feat = self.self_embed(self_state) # [3000, 64]
        
        # Attention nad sousedy (Nyní Query [3000, 1, 64] a Key [3000, N, 64] sedí!)
        neigh_embed = self.neighbor_embed(neighbor_states) 
        query = self_feat.unsqueeze(1)
        neigh_context = self.neighbor_attention(query, neigh_embed, neigh_embed, key_padding_mask=neighbor_mask)
        neigh_context = neigh_context.squeeze(1) # [3000, 64]

        # Fúze
        combined = torch.cat([vis_feat, self_feat, neigh_context], dim=1) # [3000, 256]
        combined = self.layer_norm(combined)

        # 4. ROZPLÉTÁNÍ PRO PAMĚŤ (GRU)
        # Vrátíme časovou dimenzi, aby GRU viděla plynulý film
        if is_sequential:
            combined = combined.view(batch_size, seq_len, -1) # [15, 200, 256]
        else:
            combined = combined.unsqueeze(1) # [Batch, 1, 256]
            
        gru_out, new_hidden = self.gru(combined, hidden_state)
        
        # Zploštění pro výstupní vrstvy
        features = gru_out.reshape(-1, 128)

        # 5. VÝSTUPY
        action_mean = torch.tanh(self.action_mean(features))
        dist = Normal(action_mean, torch.exp(self.action_logstd))
        message = self.msg_head(features)

        return dist, message, new_hidden


# ==========================================
# 2. COMMANDER ACTOR (Letadlo)
# ==========================================

class CommanderActor(nn.Module):
    def __init__(self, self_state_dim=15, msg_input_dim=5, action_dim=4, hidden_dim=128):
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
        # self.fusion = nn.Sequential(
        #     nn.Linear(64 + 64, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU()
        # )

        # # 5. ACTION HEAD
        # self.action_mean = nn.Linear(hidden_dim, action_dim)
        # self.action_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.gru = nn.GRU(input_size=64 + 64, hidden_size=hidden_dim, batch_first=True)

        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, self_state, incoming_messages, message_mask, hidden_state):
        # --- A) Detekce sekvence ---
        is_sequential = (self_state.dim() == 3)
        batch_size = self_state.size(0)
        seq_len = self_state.size(1) if is_sequential else 1

        # --- B) Zploštění pro Attention (pro trénink i demo) ---
        if is_sequential:
            self_state = self_state.reshape(-1, self_state.size(-1))
            incoming_messages = incoming_messages.reshape(-1, incoming_messages.size(-2), incoming_messages.size(-1))
            message_mask = message_mask.reshape(-1, message_mask.size(-1))

        # --- C) Senzory a Attention ---
        self_feat = self.self_embed(self_state)   
        msg_feat = self.msg_embed(incoming_messages) 
        query = self_feat.unsqueeze(1)
        
        # Cross-Attention (Letadlo se dívá na zprávy od dronů)
        context_vector = self.attention(query, msg_feat, msg_feat, key_padding_mask=message_mask)
        context_vector = context_vector.squeeze(1) 

        # --- D) Fúze a Paměť (GRU) ---
        combined = torch.cat([self_feat, context_vector], dim=1) 
        combined = self.layer_norm(combined)

        # Vrátíme časovou dimenzi pro GRU
        if is_sequential:
            combined = combined.view(batch_size, seq_len, -1)
        else:
            combined = combined.unsqueeze(1)
            
        gru_out, new_hidden = self.gru(combined, hidden_state)
        
        # Pro lineární vrstvy opět zploštíme na [Batch*Seq, Hidden]
        features = gru_out.reshape(-1, 128)

        # --- E) Výstup (Akce) ---
        action_mean = torch.tanh(self.action_mean(features))
        dist = Normal(action_mean, torch.exp(self.action_logstd))

        return dist, None, new_hidden

    # def forward(self, self_state, incoming_messages, message_mask):
    #     """
    #     self_state: [Batch, State_Dim]
    #     incoming_messages: [Batch, Num_Scouts, Msg_Dim]
    #     message_mask: [Batch, Num_Scouts] (True = padding/dead drone)
    #     """
        
    #     # 1. Encodery
    #     self_feat = self.self_embed(self_state)   # [Batch, 64]
    #     msg_feat = self.msg_embed(incoming_messages) # [Batch, N, 64]

    #     # 2. Cross-Attention
    #     # Letadlo (Query) se ptá zpráv (Key/Value)
    #     query = self_feat.unsqueeze(1) # [Batch, 1, 64]
        
    #     context_vector = self.attention(query, msg_feat, msg_feat, key_padding_mask=message_mask)
    #     context_vector = context_vector.squeeze(1) # [Batch, 64]

    #     # 3. Rozhodování
    #     combined = torch.cat([self_feat, context_vector], dim=1) # [Batch, 128]
    #     combined = self.layer_norm(combined)    # Stabilizace před fúzí
    #     features = self.fusion(combined)        # [Batch, Hidden]

    #     # 4. Akce
    #     action_mean = self.action_mean(features)
    #     action_mean = torch.tanh(action_mean)
        
    #     action_std = torch.exp(self.action_logstd)
    #     dist = Normal(action_mean, action_std)

    #     return dist, None, None # (Vracíme None pro kompatibilitu API s message/hidden)


class MAPPOCritic(nn.Module):
    """
    CRITIC (Kritik): Hodnotí, jak dobrý je současný stav.
    Vstup: 
        GLOBÁLNÍ data (Pohled boha na mapu ohně a všechny drony).
        Hidden state z minulého kroku (paměť).
    Výstup: Jedno číslo (očekávaná budoucí odměna).
    """
    def __init__(self, global_state_size, hidden_dim=128):
        super().__init__()
        # 1. Encoder (zmenšíme obří globální stav na rozumný vektor)
        self.encoder = nn.Sequential(
            nn.Linear(global_state_size, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU()
        )
        # 2. GRU Paměť (umožní kritikovi vidět časový trend)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # 3. Value Head (finální odhad odměny)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, global_state, hidden_state):
        # global_state v tréninku: [Batch, Seq, GS_DIM] | v rolloutu: [Batch, GS_DIM]
        is_sequential = (global_state.dim() == 3)
        batch_size = global_state.size(0)
        seq_len = global_state.size(1) if is_sequential else 1

        # Zploštění pro Lineární vrstvy
        x = global_state.reshape(-1, global_state.size(-1))
        x = self.encoder(x) # [Batch*Seq, hidden_dim]

        # Rozplétání pro GRU
        x = x.view(batch_size, seq_len, -1)
        
        # Průchod pamětí
        gru_out, new_hidden = self.gru(x, hidden_state)

        # Výstupní hodnota (opět zploštíme pro lineární hlavu)
        value = self.value_head(gru_out.reshape(-1, gru_out.size(-1)))

        return value, new_hidden