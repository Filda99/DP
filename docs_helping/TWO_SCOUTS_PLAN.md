# Phase 8: Dva scouty (N_QUADS = 2)

## Motivace
Jeden scout pokryje omezený prostor. Se dvěma scouty:
- Větší pokrytí ohne → commander dostane lepší informace
- Redundance — když jeden umře, druhý stále vysílá
- Příprava na škálovatelnost (3+)

---

## Checklist změn

### 1. `train_multi.py` — konfigurace
- [ ] `N_QUADS = 2`
- [ ] Ověřit, že `n_msg_slots = 2 * N_QUADS` → bude automaticky 4 (latest + best_fire per scout)

### 2. `train_multi.py` — worker smyčka
- [ ] Worker iteruje přes **oba** scouty (`for q_agent in local_env.quad_agents`)
- [ ] Ověřit: Scout forward pass je **nezávislý** (každý dostane vlastní obs, vlastní hidden state)
- [ ] `msg_buffer` → rozšířit na per-scout slovník: `msg_buffers = {q: [] for q in quad_agents}`
- [ ] Message construction pro commandera:
  - Teď: 2 sloty [latest_from_scout0, best_fire_from_scout0]
  - Nově: 4 sloty [latest_scout0, best_fire_scout0, latest_scout1, best_fire_scout1]
  - Pokud scout mrtvý → zero msg + mask=True
- [ ] Scout buffer: ukládat trajectorie obou scoutů (ne jen jednoho)
- [ ] Hidden state `scout_h`, `critic_scout_h` → per-scout dict

### 3. `train_multi.py` — PPO update
- [ ] Scout PPO: batch ze **všech** scoutů (konkateno nebo iterace přes oba)
- [ ] Shared weights: oba scouty sdílejí jeden `ScoutActor` (homogenní agenti)
- [ ] Gradient z obou scoutů se akumuluje do stejného modelu

### 4. `env_core.py` — spawn
- [ ] Oba scouty spawnovat nezávisle (každý jiný úhel od ohne)
- [ ] Minimum separace při spawnu (~100m od sebe), aby se neshlukovaly

### 5. `env_core.py` — reward: separační bonus
- [ ] Nový reward komponent v `_get_quad_reward()`:
  ```python
  # Separační bonus: odměna za rozestup od ostatních scoutů
  for other in self.quad_agents:
      if other == agent or other not in self.sim.drones:
          continue
      other_pos = self.sim.drones[other].get_position()
      sep_dist = np.linalg.norm(pos[:2] - other_pos[:2])
      if sep_dist < 100.0:
          reward -= 0.1 * (1.0 - sep_dist / 100.0)  # penalizace za blízkost
      elif sep_dist < 300.0:
          reward += 0.05  # bonus za dobrý rozestup
  ```
- [ ] Alternativa: coverage reward (kolik unikátních fire-cells vidí oba dohromady)

### 6. `env_core.py` — neighbor attention
- [ ] `max_neighbors = num_quads - 1` → automaticky bude 1 (jeden soused)
- [ ] `neighbor_states` a `neighbor_mask` se naplní relativní pozicí druhého scouta
- [ ] Ověřit, že ScoutActor self-attention nad sousedy funguje (teď je to maskované)

### 7. `models.py` — ScoutActor
- [ ] Žádná změna potřeba — self-attention nad sousedy je generická (N sousedů)
- [ ] Message output je per-agent (každý scout posílá vlastní zprávu)
- [ ] Ověřit: `message = self_state[:, [0, 1, 14, 12, 13]]` — indexy stále platí

### 8. `models.py` — CommanderActor
- [ ] Cross-attention: `incoming_messages` shape změní z `[B, 2, 5]` na `[B, 4, 5]`
- [ ] `message_mask` shape: `[B, 2]` → `[B, 4]`
- [ ] Architektura to zvládne beze změny (MultiheadAttention je variabilně-délkový)

### 9. Demo (`demo_both_training.py`)
- [ ] Přidat vizualizaci druhého scouta (jiná barva)
- [ ] Zobrazit zprávy obou scoutů

---

## Pořadí implementace

1. **env_core.py**: spawn 2 scoutů + separační reward (nezávislé na train kódu)
2. **train_multi.py worker**: per-scout msg_buffer + 4-slotová message construction
3. **train_multi.py PPO**: batch obou scoutů do jednoho update
4. **Test**: spustit 1 batch, ověřit shapes a že oba scouty žijí
5. **Demo**: vizualizace

## Rizika
- Scout policy collapse (oba se naučí totéž → shlukují se) → separační bonus řeší
- Message overflow (4 slotů místo 2) → cross-attention to zvládne, ale tréning bude pomalejší
- OOM: 2× víc scout dat v bufferu → snížit `eps_per_worker` pokud potřeba
