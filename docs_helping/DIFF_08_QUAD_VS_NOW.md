# Diff: 08_QuadTrainedWithDemo (e571a46, 2026-03-13) vs. HEAD

Commit: `e571a46` — *"Added dynamic learning rate, scheduler/entropy. Added penalty reward for too high Jerk. Added trained quad + demo"*

Od té doby proběhlo **93 commitů**. Níže je souhrn VŠECH změn v `src/` mezi tímto modelem a současným kódem.

---

## Přehled změněných souborů

| Soubor | Typ změny | Rozsah |
|--------|-----------|--------|
| `src/env_core.py` | Masivní refaktor | ~780 řádků změn |
| `src/models.py` | Přepsány modely | ~350 řádků změn |
| `src/reward_config.py` | **NOVÝ** | 82 řádků |
| `src/train_scout.py` | **NOVÝ** | 677 řádků (nahrazuje starý `train.py`) |
| `src/train_multi.py` | **NOVÝ** | 1343 řádků (multi-agent training) |
| `src/train.py` → `src/old/train.py` | Přesunut do old/ | rozšířen o ~480 řádků |
| `src/worker.py` → `src/old/worker.py` | Přesunut do old/ | rozšířen o ~95 řádků |
| `src/simulation.py` | Menší změny | 16 řádků |
| `src/environment.py` | Menší změny | 4 řádky |
| `src/drones/fixedwing.py` | Přemapování vstupů | 16 řádků |
| `src/drones/base_drone.py` | Kosmetické | 2 řádky |
| `src/check_params.py` | **SMAZÁN** | 61 řádků |
| `src/visualizer.py` | **SMAZÁN** (přesunuto do tools/) | 161 řádků |

---

## 1. NORMALIZACE OBSERVACÍ — Zásadní zlom

### Tehdy (e571a46):
- Pozice normalizované pomocí `self.map_bounds` (závisí na velikosti mapy!)
- Vzdálenosti k hranicím děleny `2000.0` (hardcoded)
- Relativní pozice sousedů děleny `self.grid_size_m`
- Kompas k ohni: `(fire_pos - drone_pos) / map_bounds` — závislé na mapě

### Nyní (HEAD):
- Globální konstanta `NORM_DIST = 1000.0` — **nezávislá na velikosti mapy**
- Veškeré pozice a vzdálenosti normalizovány přes `NORM_DIST`
- Kompas k ohni: **unit vector** (směr bez vzdálenosti) — `vec / np.hypot(vec)`
- Hranice: `distances / NORM_DIST`
- Sousedé: `rel_pos / NORM_DIST`

**Důsledek:** Model trénovaný s `08_` má naučené váhy kalibrovány na starou normalizaci (`map_bounds`). Pokud se pokusíš načíst jeho checkoint s novým `env_core.py`, observace budou mít **úplně jiný rozsah hodnot** → síť bude zmátená.

### Pro finetunování: KRITICKÉ
Buď musíš:
- (A) vrátit starou normalizaci v `env_core.py` a trénovat dál tím starým způsobem, nebo
- (B) akceptovat, že první fáze finetunings bude "re-calibrace" vah na nový rozsah (chvíli bude reward propadat, než se přizpůsobí)

---

## 2. REWARD SYSTÉM — Kompletní přepis

### Tehdy (e571a46):
```
Quad reward:
  - flat bonus (+1) za let nad ohněm
  - intensity bonus (avg_fire * 10)
  - speed penalty (-speed * 0.05)
  - ŽÁDNÝ approach shaping (kompas byl v observaci, ale ne v rewardu)

Physics shaping:
  - survival: +0.01/step
  - boundary: quadratická, threshold = map_bounds/2 * 0.10 (quad)
  - altitude: flat -0.05 za mimo [35, 150] m

Death penalty: -50
Reward scaling: rewards *= 0.1, clamp [-10, 10]

Team reward (extinguish):
  - Fire bonus: eff * 0.2 (oběma agentům!)
  - Waste penalty: -0.05

Commander reward: 3-state machine (MISSION → REFILL → PATROL)
  s orbital rewardem
```

### Nyní (HEAD):
```
Quad reward (z reward_config.py):
  - fire_flat_bonus: +0.5
  - fire_intensity_k: 10.0
  - fire_speed_pen: 0.005 (10x menší!)
  - approach_shaping_k: 0.0 (VYPNUT)
  - first_discovery_bonus: +5.0 (jednorázový!)
  - alt_sweet_bonus: +0.02 za let v [50, 100] m

Physics shaping:
  - survival: +0.03/step (3x víc)
  - boundary: threshold = 150m (fixed, ne %)
  - altitude: lineární penalty za mimo [30, 80] m + sweet spot bonus
  - ceiling death: 300m (quad), 450m (fixed)

Death penalty: -50 (nezměněn)
Reward scaling: BEZ globálního *0.1 ! Clamp [-10, 10] zůstává.

Team reward (extinguish):
  - Fire bonus: min(eff * 50.0, 3.0) — jen commanderovi!
  - Scout NEDOSTÁVÁ team bonus (zničil PPO signál)
  - Dropping near fire: +2.0/step
  - Dropping far from fire: -3.0
  - Fire spread penalty: delta_burned * 0.05 (max 2.0) — jen commanderovi

Commander reward: Zjednodušen na jen refill gradient
  - Žádná ground-truth fire pozice
  - Žádný orbital reward
  - Navigace čistě přes scout zprávy (cross-attention)
```

### Klíčové změny pro quad (scout):
1. **Fire bonus snížen z 1.0 na 0.5** — ale reward se už NESCALUJE *0.1, takže efektivně 5x vyšší signál!
2. **Speed penalty 10x menší** (0.005 vs 0.05)
3. **Survival bonus 3x vyšší** (0.03 vs 0.01)
4. **First discovery bonus +5.0** — obrovský jednorázový kick
5. **Altitude sweet spot [50-100m]** — dříve neexistoval
6. **Boundary threshold fixní 150m** — dříve záviselo na mapě

---

## 3. SPAWN / CURRICULUM — Kompletně přepsán

### Tehdy (e571a46):
```python
# Oheň:
if episode < 2000:  fire na [0, 0]
else:               fire random v 60% map_bounds

# Drony (oba typy):
if episode < 300:   start v [-10, 10] m (blízko centra)
else:               start random v safe_zone

# Grid: fixní 500m
```

### Nyní (HEAD):
```python
# Oheň: VŽDY random v 40% map_bounds, 3-6 zápalných bodů

# Quad spawn — probabilistický:
#   15% blízko (20-50m)
#   25% střed (50-200m)
#   60% daleko (200-600m)

# Grid: 1000m BASE, ale randomizace map_size_range (1000-2000m)
# → agent musí generalizovat přes různé velikosti arén
```

**Důsledek pro finetunování:** Model `08_` se naučil na jednoduchém curriculum (oheň na [0,0], blízký spawn). Nový systém je výrazně těžší — random spawn, random oheň, random mapa.

---

## 4. MODELY (models.py)

### ScoutActor:
| Vlastnost | Tehdy | Nyní |
|-----------|-------|------|
| `action_logstd` init | `zeros` (std≈1.0) | `full(-2.0)` (std≈0.135) |
| logstd clamp | [-3.0, 0.5] | [-4.0, 0.0] |
| Zprávy | Naučené přes `msg_head` (Linear+Tanh) | **Explicitní** z observace: `self_state[:, [0, 1, 14, 12, 13]]` |

**Zprávy:** Původně se scout UČIL co posílat commanderovi. Nyní posílá přímo `[pos_x, pos_y, intensity, rel_fire_x, rel_fire_y]` — žádné učení, čistě hardcoded slice z observací. → `msg_head` je ODSTRANĚN.

### CommanderActor:
| Vlastnost | Tehdy | Nyní |
|-----------|-------|------|
| hidden_dim | 128 | 64 |
| attn heads | 4 | 2 |
| encoder | 1-layer (64→64, ReLU) | 2-layer (17→64→64, Tanh) |
| comm_alpha gate | Ano (learnable) | Odstraněn |
| layer_norm | Ano (128-d) | Odstraněn |
| action_logstd init | `zeros` | `full(-0.5)` |
| aux_head | Ne | Ano (predikce polohy ohně) |
| self_state_dim | 15 | 17→19 (přibyl compass + danger flag) |
| Akce | [Roll, Pitch, Throttle, Water] přímo | Hierarchický: [heading_delta, target_alt, water] → flight controller |

### MAPPOCritic → PrivilegedCritic:
| Vlastnost | Tehdy | Nyní |
|-----------|-------|------|
| Název | `MAPPOCritic` | `PrivilegedCritic` |
| Vstup | `global_state` (flatten all agents + fire map 16×16) | privileged state = own obs + extras (fire pos, fire intensity, other agent pos) |
| Encoder | 2-layer (global→256→hidden) | 2-layer (input→hidden→hidden) |
| Sdílení | Jeden critic pro oba agenty | Oddělené instance (jiný input_dim) |

---

## 5. ACTION MAPPING (env_core.py step())

### Tehdy:
```python
# Quad: smoothing 0.8 * last + 0.2 * new
# Fixed: smoothing 0.8 * last + 0.2 * new
# Fixed: action mapping [Roll, Pitch, Throttle→0.4-1.0, Water→0-1]
```

### Nyní:
```python
# Quad: smoothing 0.5 * last + 0.5 * new (víc reaktivní!)
# Fixed: BEZ smoothing! (PPO credit assignment)
# Fixed: Hierarchický flight controller:
#   [0] heading_delta → roll_cmd (proportional controller)
#   [1] target_alt [-1,1] → [40, 250]m → PD altitude controller → pitch_cmd
#   [2] water_trigger
#   Throttle: automaticky z airspeed PID
```

**Důsledek:** Fixed-wing má teď 3 akce místo 4, a řízení je přes flight controller, ne přímo. Ale pro quad scout se liší jen smoothing faktor (0.8→0.5).

---

## 6. FIXED-WING (drones/fixedwing.py)

- **Přemapování vstupů:** `[Roll, Throttle, Pitch, Water]` → `[Roll, Pitch, Throttle, Water]`
- **Max pitch:** 45° → 15° (bezpečnější, méně letálních crash)

---

## 7. TRÉNINK (train.py → train_scout.py)

### Tehdy (e571a46 — `src/train.py`):
```
grid_size_m = 500
max_steps = 500
num_workers = 20
eps_per_worker = 3 (→ 60 eps/batch)
lr_scout = 3e-4
lr_critic = 3e-4
hidden_dim = 128 (scout)
N_QUADS = 1, N_FIXED = 0
Critic = MAPPOCritic(global_state_dim) — sdílený
Comm aux loss = MSE(msg[0:3], self_state[12:15]) * 0.5
Entropy coef = 0.01
Minibatches = 4
```

### Nyní (`src/train_scout.py`):
```
grid_size_m = 1000
map_size_range = (1000, 2000) — RANDOMIZACE
max_steps = 500 (nezměněn)
num_workers = 15
eps_per_worker = 2 (→ 30 eps/batch — poloviční!)
lr_scout = 3e-4 (nezměněn)
lr_critic = 3e-4 (nezměněn)
hidden_dim = 128 (nezměněn)
N_QUADS = 1, N_FIXED = 0
Critic = PrivilegedCritic(scout_self_dim + 6) — oddělený, s privilegovaným vstupem
ŽÁDNÝ comm aux loss (zprávy jsou explicitní)
Entropy coef = 0.1 (10x vyšší!)
Minibatches = ? (pravděpodobně 4, potřeba ověřit)
```

### Klíčový rozdíl: ENTROPY COEF
Starý model: `0.01` — nízká explorace, síť se rychle specializuje
Nový model: `0.1` — **10x vyšší** explorace → síť víc zkoumá, pomaleji konverguje, ale méně uvízne v lokálních minimech.

### Klíčový rozdíl: BATCH SIZE
Starý: 60 epizod/batch → více dat pro každý PPO update
Nový: 30 epizod/batch → méně dat, ale možná stabilnější gradienty

---

## 8. SIMULACE (simulation.py)

- Refill zone detekce: 3D vzdálenost → **2D (XY only)** — fixed-wing letí ve výšce 40-150m, 3D by nikdy nedetekoval
- Refill radius: `(size/2 + 2)²` → `150² = 22500` — mnohem větší detekční zóna
- Spotřeba vody: `200.0 * dt` → `5.0 * dt` — **40x pomalejší** vypouštění (240 kroků na prázdný tank)
- Logování: vypnuto (`_setup_logging`, `_save_log` zakomentovány)

---

## 9. ENVIRONMENT (environment.py)

- Refill zone default size: 10.0 → 30.0
- Refill zone radius pro detekci: `(size/2 + 2)²` → `150²`

---

## SHRNUTÍ: Proč 08_Quad fungoval a co je nyní jiné

### Co fungovalo v 08_ (a proč):
1. **Jednoduchý curriculum** — oheň na [0,0] prvních 2000 epizod, blízký spawn prvních 300 epizod → agent se rychle naučil "leť dopředu, zamiř na oheň"
2. **Normalizace závislá na mapě** (`map_bounds`) + malá mapa (500m) → observace byly v hezky škálovaném rozsahu
3. **Vysoký fire bonus** (efektivně 1.0 + 10*intensity, ALE scaleno *0.1)
4. **Nízký entropy coef (0.01)** → rychlá konvergence, méně tápání
5. **Větší batche (60 eps)** → stabilnější gradienty

### Co je nyní jiné (a proč je to těžší):
1. **Mapa 2-4x větší** (1000-2000m vs 500m) → agent musí létat mnohem dál
2. **Random spawn daleko od ohně** (60% ve vzdálenosti 200-600m!) → problém najít oheň
3. **Normalizace fixní NORM_DIST=1000** → jiný rozsah observací
4. **Entropy 10x vyšší** → explorace brzdí konvergenci
5. **Menší batche (30 eps)** → nestabilnější PPO updates
6. **Zprávy explicitní** (ne naučené) → méně parametrů, ale msg_head chybí

### Doporučení pro finetunování:
1. **Model 08_ je NEKOMPATIBILNÍ s novým env_core.py** kvůli jiné normalizaci. Buď vrátit starou normalizaci, nebo akceptovat re-kalibrační fázi.
2. **msg_head odstraněn** — `08_` má natrénovaný `msg_head`, nový kód ho ignoruje. Pro solo-quad to nevadí (msg se nepoužívají), ale při přechodu na multi-agent ano.
3. **Reward scale chybí** — starý kód násobil `*0.1`, nový ne. Efektivní magnitude rewardů se zásadně liší → critic value estimates budou mimo.
4. **Spawn curriculum zmizel** — model 08_ se naučil na jednoduchém startu, nový environment rovnou zahazuje agenta daleko.

### Nejlepší strategie:
- Vrátit starý `env_core.py` normalizaci a reward scaling pro quad
- Finetunovat s novou spawn distribucí (postupně zvyšovat difficulty)
- Nebo: trénovat znovu od nuly s novým systémem, ale s curriculum (easy→hard)
