# Scaling Evaluation — Výsledky a analýza

## 1. Přehled experimentu

Vyhodnocení natrénovaného multi-agentního systému (scout UAV + fixed-wing commander) přes kompletní škálu konfigurací.

| Parametr | Hodnoty |
|---|---|
| Počet scoutů | 1, 3, 5, 7, 10 |
| Počet FW | 1, 3, 5, 7, 10 |
| Počet ohnišť | 1, 3, 5, 7 |
| Velikost mapy | 700, 1100, 2000, 4000, 6000 m |
| Seedů na konfiguraci | 10 |
| **Celkem epizod** | **5000** (4949 úspěšných) |
| Max kroků / epizodu | 1200 |

**Modely:** `v10_finetune` (scout hidden=128, commander hidden=64), trénovány MAPPO na konfiguraci 3 scoutů + 5 FW, mapa 800–1500 m, 1–3 ohňů.

---

## 2. Rovnice a metriky

### 2.1 Simulace

Fyzikální simulace PyBullet běží na frekvenci 30 Hz s `frame_skip=5`, takže jeden RL krok = 5 × (1/30) = 0.167 s reálného času.

- **Scout rozhoduje** každý RL krok (6 Hz)
- **Commander rozhoduje** každých 30 RL kroků (0.2 Hz, tj. jednou za 5 s)

### 2.2 Šíření ohně

Oheň se šíří na mřížce s buňkami 5×5 m podle rovnice:

$$
B_{i,j}^{t+1} = B_{i,j}^{t} + v_{\text{spread}} \cdot \Delta t \cdot F_{i,j} \cdot \sum_{(k,l) \in \mathcal{N}(i,j)} B_{k,l}^{t}
$$

kde $v_{\text{spread}} = 0.1$ m/s, $\Delta t = 0.1$ s, $F_{i,j}$ je palivo v buňce a $\mathcal{N}$ je 4-okolí.

### 2.3 Efektivita shazování vody

Efektivita vodního shozu závisí na výšce FW:

$$
\eta_{\text{water}} = \max\left(0,\ 1 - \left(\frac{h}{150}\right)^2\right)
$$

kde $h$ je výška FW v metrech. Kapacita nádrže = 200 L, spotřeba = 100 L/s × dt.

### 2.4 Reward — Scout (quadcopter)

Celkový per-step reward scouta:

$$
r_{\text{scout}} = r_{\text{survival}} + r_{\text{fire}} + r_{\text{approach}} + r_{\text{altitude}} + r_{\text{separation}} + r_{\text{boundary}}
$$

Hlavní komponenty:

| Komponenta | Rovnice | Parametr |
|---|---|---|
| Přežití | $r_{\text{surv}} = +0.02$ | per-step |
| Viditelnost ohně | $r_{\text{fire}} = 0.5 + 2.0 \cdot \bar{I}_{\text{fire}}$ | kde $\bar{I}$ je průměrná intenzita v FOV |
| Přiblížení k ohni | $r_{\text{appr}} = 0.03 \cdot (d_{t-1} - d_t)$ | potential-based shaping |
| Kompas | $r_{\text{comp}} = 0.25 \cdot \hat{v} \cdot \hat{d}_{\text{fire}}$ | dot product rychlosti a směru k ohni |
| Altitude penalty | $r_{\text{alt}} = -0.05 \cdot \|h - [40, 80]\|$ | lineární mimo [40, 80] m |
| Sweet-spot bonus | $r_{\text{sweet}} = +0.02$ pokud $h \in [70, 100]$ m | |
| Ground danger | $r_{\text{gnd}} = -5.0 \cdot (1 - h/20)^2$ pokud $h < 20$ m | |
| Separace | $r_{\text{sep}} = \pm 0.05$ per pair, práh 30 m | bonus za rozestup, penalizace za shlukování |
| Opuštění ohně | $r_{\text{aband}} = -1.0 \cdot I_{\text{prev}}$ | při přechodu z vidění ohně na nevidění |
| Crash | $r_{\text{crash}} = -10$ | jednorázově |
| Clip | $r \in [-3, +3]$ per step | |

### 2.5 Reward — Commander (fixed-wing)

Commander je řízen hybridně: NN generuje waypoint, PD kontrolér řídí heading, skriptovaný systém ovládá vodní ventil a refill.

$$
r_{\text{cmdr}} = r_{\text{survival}} + r_{\text{approach}} + r_{\text{water}} + r_{\text{altitude}} + r_{\text{boundary}}
$$

| Komponenta | Rovnice | Parametr |
|---|---|---|
| Přiblížení ke scoutu | $r_{\text{appr}} = 0.25 \cdot (d_{t-1} - d_t)$ | scout jako proxy pro oheň |
| Zbytečný shoz vody | $r_{\text{waste}} = -0.5$ | per-step při shozu mimo oheň |
| Altitude penalty | $r_{\text{alt}} = -0.02 \cdot \|h - [30, 80]\|$ | |

### 2.6 Vyhodnocovací metriky

| Metrika | Definice |
|---|---|
| **Success rate** | % epizod kde `final_burning_cells == 0` |
| **Time to suppression** | Krok ve kterém `burning_cells` poprvé klesne na 0 |
| **Water accuracy** | `hits / total_drops × 100` |
| **Total burned** | Počet buněk které kdy hořely |
| **Peak fire** | Maximum současně hořících buněk |

---

## 3. Výsledky

### 3.1 Celkové

| Metrika | Hodnota |
|---|---|
| Celková success rate | **68.5 %** |
| Medián času suprese (úspěšné) | **149 kroků** (24.8 s) |
| Průměrný čas suprese | **224 kroků** (37.3 s) |
| Průměrná water accuracy | **62.0 %** |
| Scout deaths / epizoda | 0.38–1.80 (roste s počtem) |
| FW deaths / epizoda | ~0 |

### 3.2 Success rate — Scouts × FW

|  | 1 FW | 3 FW | 5 FW | 7 FW | 10 FW |
|---|---|---|---|---|---|
| **1 scout** | 34 % | 70 % | 84 % | 90 % | 96 % |
| **3 scouts** | 12 % | 72 % | 90 % | 89 % | 96 % |
| **5 scouts** | 14 % | 78 % | 82 % | 88 % | 92 % |
| **7 scouts** | 12 % | 64 % | 74 % | 78 % | 82 % |
| **10 scouts** | 10 % | 58 % | 68 % | 88 % | 90 % |

### 3.3 Success rate — FW × Mapa

|  | 700 m | 1100 m | 2000 m | 4000 m | 6000 m |
|---|---|---|---|---|---|
| **1 FW** | 32 % | 22 % | 10 % | 12 % | 6 % |
| **3 FW** | 88 % | 90 % | 82 % | 44 % | 38 % |
| **5 FW** | 96 % | 96 % | 90 % | 62 % | 54 % |
| **7 FW** | 98 % | 98 % | 94 % | 77 % | 65 % |
| **10 FW** | 98 % | 98 % | 98 % | 88 % | 74 % |

### 3.4 Vliv počtu ohnišť

Počet ohnišť nemá statisticky významný vliv na success rate:

| Ohnišť | Success rate | Peak fire cells | Total burned |
|---|---|---|---|
| 1 | 69.3 % | 90 | 112 |
| 3 | 68.4 % | 91 | 114 |
| 5 | 68.0 % | 91 | 114 |
| 7 | 68.4 % | 91 | 114 |

**Vysvětlení:** Rychlost šíření ohně ($v = 0.1$ m/s) je pomalá relativně k rychlosti FW (~40 m/s). Limitujícím faktorem je čas doletu FW k ohni, nikoli počet ohnišť.

### 3.5 Chování scoutů

| Scouts | Altitude | Čas nad ohněm | Separace | Deaths/ep |
|---|---|---|---|---|
| 1 | 153 m | 37.9 % | 0 m | 0.28 |
| 3 | 105 m | 42.2 % | 22 m | 0.52 |
| 5 | 108 m | 39.2 % | 41 m | 0.92 |
| 7 | 109 m | 44.4 % | 48 m | 1.25 |
| 10 | 107 m | 47.8 % | 47 m | 1.80 |

### 3.6 Chování FW

| FW | Drops | Hits | Accuracy | Drop dist. | Drop alt. | Refills |
|---|---|---|---|---|---|---|
| 1 | 25 | 19 | 79 % | 41 m | 51 m | 1.9 |
| 3 | 39 | 20 | 60 % | 42 m | 55 m | 3.0 |
| 5 | 42 | 18 | 55 % | 43 m | 58 m | 3.1 |
| 7 | 37 | 16 | 59 % | 41 m | 62 m | 2.6 |
| 10 | 38 | 16 | 57 % | 41 m | 64 m | 2.6 |

---

## 4. Klíčová pozorování

### 4.1 Počet FW je dominantní faktor

Success rate roste monotónně s počtem FW: 16 % (1 FW) → 91 % (10 FW). Každý přidaný FW zvyšuje pravděpodobnost, že alespoň jeden doletí včas. Toto je nejsilnější prediktor úspěchu.

### 4.2 Více scoutů = mírně horší výsledek

Paradoxně, zvýšení počtu scoutů z 1 na 10 snižuje success rate ze 75 % na 63 %. Příčiny:
- **Model trénován na 3 scoutech** — zobecnění na 7–10 je slabé
- **Kolize a deaths rostou** — 1.80 deaths/ep u 10 scoutů vs 0.28 u 1
- **Separace saturuje** — nad 5 scoutů se průměrná separace nezvyšuje (47–48 m)

### 4.3 Mapa > 2000 m dramaticky snižuje úspěšnost

Na mapách ≤ 2000 m je success rate 75–82 %. Na 4000 m klesá na 57 %, na 6000 m na 47 %. Důvod: FW potřebuje víc času na dolet a na refill. Oheň mezitím roste.

### 4.4 Scouti létají příliš vysoko

Průměrná výška 107–153 m je výrazně nad ideálním pásmem (40–80 m). Důsledek:
- Horší rozlišení ohně v lokální mapě
- Menší FOV coverage pro přesné navedení FW
- Altitude penalty existuje, ale je příliš slabá ($k = 0.05$/m) oproti bezpečnostní tendenci

### 4.5 Water accuracy klesá s počtem FW

1 FW má 79 % přesnost, 5+ FW jen 55–59 %. Více FW shazuje vodu paralelně, často na stejný oheň, po jehož uhašení zbylé shody minuly.

---

## 5. Doporučení pro zlepšení

### 5.1 Trénink s proměnlivým počtem scoutů

**Problém:** Model je natrénován na fixních 3 scoutech.

**Řešení:** Randomizovat `n_quads` v rozsahu [1, 7] během tréninku. Attention-based agregace v `ScoutActor` to umožňuje bez změny architektury. Přidání `n_alive_scouts` do `self_state` scouta.

### 5.2 Snížení letové výšky scoutů

**Problém:** Scouti létají 107–153 m místo ideálních 40–80 m.

**Řešení:**
- Zvýšit `alt_penalty` z 0.05 na 0.15–0.2
- Snížit `alt_ideal_max` z 80 na 60 m
- Přidat explicitní bonus za viditelnou plochu ohně (inverzně úměrný výšce)
- Alternativně: curriculum od nízké výšky (ceiling 100 m → 200 m → 300 m)

### 5.3 Škálování na velké mapy

**Problém:** Success 47 % na 6000 m i s 10 FW.

**Řešení:**
- Trénovat na mapách 2000–5000 m (aktuálně trénováno na 800–1500 m)
- Zvýšit `waypoint_range` z 200 m na 500+ m pro velké mapy (adaptivně dle map_bounds)
- Delší `max_steps` pro velké mapy (1200 kroků = 200 s, ale FW potřebuje na 6000 m mapu ~150 s jen na přelet)
- Refill navigace: aktuálně skriptovaná, na velkých mapách by měla být naučená

### 5.4 Koordinace FW (task allocation)

**Problém:** Water accuracy klesá s počtem FW (sdílení cíle).

**Řešení:**
- Přidat do pozorování FW `fw_neighbor_target` — kam ostatní FW míří
- Penalty za shozy vody na stejný oheň jako jiný FW v posledních N krocích
- Implementovat explicitní task allocation: každý FW dostane jiný cíl (fire centroid)

### 5.5 Lepší fire-approach shaping pro FW

**Problém:** FW se přibližuje ke scoutovi, ne k ohni.

**Řešení:** Dát FW přímý přístup k pozici ohně (ze scout zpráv) a rewardovat přibližování k fire centroidu:

$$
r_{\text{approach}} = k \cdot (d_{t-1}^{\text{fire}} - d_t^{\text{fire}})
$$

místo současného přibližování ke scouta.

### 5.6 Adaptivní max_steps

Na 6000 m mapě je 1200 kroků (200 s) nedostatečné. Doporučení:

$$
\text{max\_steps} = \left\lceil \frac{2 \cdot \text{map\_size}}{v_{\text{FW}} \cdot \Delta t_{\text{RL}}} \right\rceil
$$

Pro 6000 m, $v_{\text{FW}} \approx 40$ m/s, $\Delta t = 0.167$ s: $\text{max\_steps} \approx 1800$.

---

## 6. Závěr

Systém multi-agentního hašení požárů byl úspěšně natrénován a vyhodnocen přes **4949 epizod** v 500 různých konfiguracích. Hlavní výsledky:

1. **Systém funguje** — na mapách do 2000 m s ≥ 3 FW dosahuje **82–98 % úspěšnosti** hašení.

2. **Škálování FW je efektivní** — přidávání fixed-wing letadel monotónně zvyšuje success rate i zkracuje dobu hašení. Přechod z 1 na 5 FW zvedne úspěšnost z 16 % na 80 %.

3. **Škálování scoutů je problematické** — více scoutů nepřináší výrazné zlepšení a nad 5 scoutů výkon klesá kvůli kolizím. Řešitelné tréninkem s variabilním počtem.

4. **Bottleneck je čas doletu FW** — nikoli průzkum, odhalení ohně ani počet ohnišť. Scout spolehlivě nalezne oheň v prvních krocích (mean discovery @ step 0).

5. **Limit systému** jsou velké mapy (> 4000 m), kde i 10 FW dosahuje jen 74–88 %. Řešitelné delším horizontem, větším waypoint range a tréninkem na větších mapách.

Systém úspěšně demonstruje kooperativní multi-agentní přístup kde **scouti průzkumem navádějí commandery**, kteří koordinovaně hasí ohně. Architektura s attention-based komunikací a hierarchickým řízením (NN + skriptované subsystémy) se ukázala jako robustní a škálovatelná v počtu fixed-wing letadel.
