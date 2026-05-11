# Optimalizace shazování vody (Water Drop Pipeline)

## Problém

Commander (fixed-wing) se učil pomocí NN rozhodovat **kdy otevřít ventil** (water_raw ∈ [-1, 1]).
Rozhodování probíhalo jednou za 30 kroků (waypoint_steps=30), ale fyzika ventilu běží každý krok.
Výsledek: NN nedokázala trefit správný timing a plýtvala vodou.

### Diagnostika — stav PŘED změnami

```
14:31:18 | Batch 0027 (Ep 00810) | R:    +15.4 (   +22.4)  Scout_R:   +96.1  Cmdr_R:   -80.7
   Cmdr reward breakdown: r_extinguish=+69.142  r_water_waste=-93.000  r_fire_out=+13.667  r_spread=-4.387
   Water drops: 9483 total, 417 hit, 9066 miss (4% accuracy)  alt=139m [88-182]  dist=104m [1-367]

14:33:01 | Batch 0028 (Ep 00840) | R:    +32.5 (   +24.0)  Scout_R:  +107.4  Cmdr_R:   -74.9
   Cmdr reward breakdown: r_extinguish=+66.903  r_water_waste=-101.450  r_fire_out=+12.000  r_spread=-5.653
   Water drops: 10143 total, 402 hit, 9741 miss (4% accuracy)  alt=139m [89-184]  dist=116m [1-388]

14:36:34 | Batch 0030 (Ep 00900) | R:    +46.5 (   +43.1)  Scout_R:  +141.6  Cmdr_R:   -95.1
   Cmdr reward breakdown: r_extinguish=+71.081  r_water_waste=-108.567  r_fire_out=+13.000  r_spread=-7.015
   Water drops: 10848 total, 428 hit, 10420 miss (4% accuracy)  alt=141m [100-186]  dist=107m [2-421]
```

**Klíčové problémy:**
1. **~10 000 dropů/batch**, jen ~400 hit → **4% accuracy**
2. **Průměrná výška 139m** — příliš vysoko, voda se rozfoukne (Gaussian radius = 10 + 0.3×139 = 52m)
3. **Průměrná vzdálenost od ohně 104–116m** — FW lítá daleko od ohně
4. **Cmdr_R hluboce záporný** (-75 až -95) kvůli water_waste penalty
5. **NN_dec = 370–420/1980 (18–21%)** — NN rozhoduje jen v minoritě kroků, zbytek scripted refill

---

## Příčiny

### 1. NN nemůže ovládat timing ventilu

NN rozhoduje jednou za 30 kroků → nastaví `water_raw = 1.0` → ventil zůstane otevřený celých 30 kroků.
Tank (200L) se při drain rate 100 L/s vyprázdní za 2 sekundy (= 2 kroky při dt=1s).
Zbylých 28 kroků se nic neděje, ale NN příště opět nastaví water_raw.

### 2. Žádný gradient pro snížení výšky

Effectiveness = `max(0, 1 - alt/200)` — lineární pokles, velmi mírný.
Při 139m je effectiveness stále 30%, takže NN necítí silný tlak letět níž.

### 3. Exploitovatelné reward zóny (odstraněno dříve)

Zone 1 (proximity bonus u ohně) a Zone 2 (bonus za otevřený ventil u scouta) dávaly
stabilní +80/ep reward bez reálného hašení → NN optimalizovala na kroužení, ne na přesnost.

---

## Provedené změny

### Změna 1: Rule-based ventil (commander_control.py)

NN výstup `water_raw` se **ignoruje**. Ventil se ovládá pravidlově:

```python
# heading_action() — voláno KAŽDÝ fyzikální krok
valve = -1.0  # default: closed
if pos[2] < 120.0 and drone.current_water > 0:
    for i, q in enumerate(env.quad_agents):
        # msg[2] = fire intensity hlášená scoutem
        fire_intensity = msgs[i, 2]
        if fire_intensity < 0.01:
            continue  # scout nevidí oheň
        d_sq = distance(FW, scout_q)
        if d_sq < 50.0:
            valve = 1.0  # otevři!
            break
```

**Podmínky pro otevření:**
1. FW pod 120m výšky
2. FW má vodu v tanku
3. Existuje scout do 50m od FW
4. Ten scout **hlásí oheň** (msg[2] > 0.01) — použita legitimní komunikace, ne oracle

### Změna 2: Kvadratický pokles effectiveness (simulation.py)

```python
# PŘED (lineární):
effectiveness = max(0.0, 1.0 - altitude / 200.0)

# PO (kvadratický):
effectiveness = max(0.0, 1.0 - (altitude / 150.0) ** 2)
```

| Výška | Lineární (staré) | Kvadratické (nové) |
|-------|------------------|--------------------|
| 40m   | 80%              | **93%**            |
| 60m   | 70%              | **84%**            |
| 80m   | 60%              | **72%**            |
| 100m  | 50%              | **56%**            |
| 120m  | 40%              | **36%**            |
| 140m  | 30%              | **13%**            |
| 150m+ | 25%              | **0%**             |

Nad 150m je effectiveness = 0 (dříve až od 200m). Silná motivace pro NN letět pod 80m.

### Změna 3: Altitude shaping v rewardu (env_core.py)

Když je FW do 200m od scouta, dostává bonus za nízký let:
```python
reward += 0.1 * max(0, 1 - pos[2] / 200)
```

### Změna 4: Zúžený rozsah výšky

```python
# PŘED: target_alt = 40 + (raw+1)/2 * 210  → [40, 250]m
# PO:   target_alt = 40 + (raw+1)/2 * 140  → [40, 180]m
```

### Změna 5: Scout zprávy jako zdroj informace pro ventil

Zpráva scouta (msg_dim=5):
- `[0]` norm_pos_x — pozice scouta
- `[1]` norm_pos_y — pozice scouta
- **`[2]` fire_intensity** — intenzita ohně viděná scoutem
- `[3]` rel_x — relativní vektor k ohni
- `[4]` rel_y — relativní vektor k ohni

Commander legitimně přijímá tyto zprávy → valve logika používá `msg[2]` místo
přímého přístupu k `env._prev_fire_seen` (oracle). Zprávy se cachují v
`CommanderController.last_scout_msgs` a aktualizují každý krok.

---

## Výsledky — stav PO změnách

### První trénink s rule-based valve (scout proximity 80m, bez fire condition)

```
15:49:22 | Batch 0001 (Ep 00030) | R:    +57.7  Scout_R:   +60.2  Cmdr_R:    -2.5
   Cmdr reward breakdown: r_extinguish=+62.833  r_water_waste=-14.167  r_fire_out=+5.333  r_spread=-5.732
   Water drops: 1270 total, 377 hit, 893 miss (30% accuracy)  alt=95m [39-119]  dist=75m [2-170]

15:52:46 | Batch 0003 (Ep 00090) | R:   +175.4  Scout_R:  +158.2  Cmdr_R:   +17.2
   Cmdr reward breakdown: r_extinguish=+68.500  r_water_waste=-15.233  r_fire_out=+6.000  r_spread=-6.387
   Water drops: 1387 total, 411 hit, 976 miss (30% accuracy)  alt=97m [44-121]  dist=78m [3-157]

15:56:05 | Batch 0005 (Ep 00150) | R:   +118.7  Scout_R:  +124.5  Cmdr_R:    -5.8
   Cmdr reward breakdown: r_extinguish=+59.000  r_water_waste=-12.533  r_fire_out=+8.000  r_spread=-4.692
   Water drops: 1150 total, 354 hit, 796 miss (31% accuracy)  alt=96m [60-119]  dist=76m [8-184]
```

---

## Srovnání PŘED vs. PO

| Metrika                  | PŘED (NN valve)     | PO (rule-based)     | Změna       |
|--------------------------|---------------------|---------------------|-------------|
| Drops/batch              | ~10 000             | ~1 300              | **-87%**    |
| Hit/batch                | ~430                | ~400                | ~stejné     |
| **Accuracy**             | **4%**              | **30%**             | **+26pp**   |
| Průměrná výška           | 139m                | 95m                 | **-44m**    |
| Vzdálenost od ohně       | 110m                | 76m                 | **-34m**    |
| Cmdr_R                   | -75 až -95          | -5 až +17           | **+90**     |
| NN_dec (% NN rozhodnutí) | 18–21%              | 59–72%              | **+40pp**   |
| Water waste penalty      | -93 až -109         | -13 až -17          | **-85%**    |

---

## Architektura valve logiky

```
Každý krok (heading_action):
  ├── FW výška < 120m?
  │    └── Ne → valve CLOSED
  ├── FW má vodu?
  │    └── Ne → valve CLOSED
  ├── Pro každý scout:
  │    ├── Scout hlásí oheň (msg[2] > 0.01)?
  │    │    └── Ne → přeskočit
  │    └── FW do 50m od scouta?
  │         ├── Ano → valve OPEN, break
  │         └── Ne → pokračovat
  └── Žádný scout nesplnil → valve CLOSED
```

## Proč ne 100% accuracy?

I s 30% accuracy existují legitimní důvody pro miss:
- Oheň se šíří — centroid se posouvá, scout ho sleduje ale FW letí fixním heading
- Gaussovo rozptýlení při 95m: radius = 10 + 0.3×95 = 38.5m — okraj shozu nedosáhne na okraj požáru
- Scout hlásí fire intensity z local_map (32×32, ~160×160m) — vidí oheň, ale nemusí být přímo nad ním
- FW přeletí 50m zónu rychlostí ~15 m/s → ventil otevřen jen ~3 kroky

Další optimalizace (kvadratická effectiveness + fire condition ze zpráv) by měly
zlepšit accuracy směrem k 40–50%.
