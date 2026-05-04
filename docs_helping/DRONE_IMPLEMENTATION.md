# Implementace dronů — jak to funguje

Osobní poznámky pro konferenci. Popis obou typů dronů lidsky.

---

## Třídy a dědičnost

Všechny drony dědí ze společné třídy BaseDrone (src/drones/base_drone.py).
Ta zajišťuje věci, které mají oba společné — čtení pozice a rychlosti z PyBulletu,
zásobník vody (pro FW), logování. Každý typ dronu pak implementuje svou vlastní
logiku pohybu v metodě apply_control.

Konkrétní třídy:
- Scout -> Quadcopter v src/drones/quadcopter.py
- Commander -> FixedWing v src/drones/fixedwing.py

---

## Co je PyBullet a proč ho používáme

PyBullet je fyzikální engine — normálně se používá na simulaci robotů.
My ho využíváme hlavně proto, že umí:
1. Řídit tuhá tělesa (rigid bodies) a integrovat jejich pohyb v čase
2. Vizualizovat scénu v 3D okně

Timestep simulace je 1/30 sekundy (30 fps). Každý krok RL prostředí = jeden fyzikální krok.

---

## Scout (Quadcopter)

### Co to je fyzicky

Model komerčního dronu jako DJI nebo Parrot — tedy ne detailní simulace
čtyř rotorů a jejich aerodynamiky, ale zjednodušený model na vyšší úrovni.
Funguje podobně jako firmware v komerčním dronu: pilot říká "leť dopředu rychlostí X"
a firmware se postará o to, aby to tak bylo. Stejný přístup je tady.

### Jak funguje fyzika v PyBulletu

Scout je v PyBulletu plnohodnotné tuhé těleso (0,5 kg, krabička 40x40x10 cm).
PyBullet ho simuluje standardně — gravitace ho táhne dolů, náš kód aplikuje síly nahoru
a do stran. Výsledkem je, že PyBullet sám vypočítá pohyb na základě Newtonových zákonů.

Je to dynamický model — pohyb vzniká ze sil.

### Jak neuronová síť ovládá drona

NN vydá 4 čísla v rozsahu -1 až 1:
- roll — pohyb doleva/doprava
- pitch — pohyb dopředu/dozadu
- yaw — otočení
- vert — výška (nahoru/dolů)

Kód je pak přemění na síly takhle:

1. Cílová rychlost — z -1..1 se spočítá, jak rychle chce dron letět (max 15 m/s horizontálně, 4 m/s vertikálně)

2. Otočení souřadnic — cílová rychlost se přepočítá z pohledu drona (jeho "dopředu") do souřadnic mapy.
   Takže pokud dron kouká na sever a dostane příkaz "leť dopředu", letí na sever.
   Pokud kouká na východ, letí na východ.

3. Síla = rozdíl rychlostí — kód spočítá, o kolik se liší aktuální rychlost od cílové,
   a aplikuje sílu úměrnou tomuto rozdílu. Vertikálně navíc přičte přesně tolik,
   aby kompenzoval gravitaci (hover = nulová síla z NN).
   Tyto síly se předají PyBulletu přes p.applyExternalForce().

4. Stabilizace — aby dron nelátal nakřivo, jsou tam dva PD regulátory (roll a pitch zpět na 0)
   a jeden pro yaw. Ty aplikují točivé momenty přes p.applyExternalTorque().

### Vítr

Dron ignoruje vítr do 14,7 m/s (specifikace ANAFI USA — má interní stabilizaci).
Nad touto hranicí se dopočítá aerodynamický odpor a aplikuje jako síla proti pohybu.

### Je to lossless model?

Ano. Žádná baterie, žádná spotřeba paliva, rotory se nemůžou přehřát.
Jediné fyzické limity jsou maximální rychlosti a odpor vzduchu při silném větru.
Jde o behaviorální aproximaci, ne termodynamickou simulaci.

---

## Commander (FixedWing)

### Co to je fyzicky

Malý letoun nesoucí zásobník vody (200 litrů) — hasičský bombardér.
Model je postavený na kinetickém guidance modelu (MathWorks UAV Guidance Model) —
standardní přístup v letecké simulaci pro autopiloty.

### Klíčový rozdíl oproti scoutovi

FixedWing je kinematický model — kód si sám počítá, kde dron bude,
a pak ho tam teleportuje pomocí p.resetBasePositionAndOrientation().
PyBullet tedy nefunguje jako fyzikální engine, ale jen jako vizualizační backend.
Síly se vůbec nepoužívají.

Proč? Protože správná aerodynamická simulace letounu by vyžadovala velmi krátké
timestepy, geometrii křídel, vztlak atd. To je pro účely RL zbytečně komplikované.

### Interní stav

Kód si udržuje 8 proměnných stavu:
- pozice (x, y, výška)
- rychlost vzduchu (airspeed) — co dělají motory
- kurz (chi) — kam dron letí (úhel v horizontální rovině)
- úhel stoupání/klesání (gamma) — jak strmě stoupá nebo klesá
- náklon (phi) a rychlost náklonu — pro zatáčení

### Jak NN ovládá letoun

NN vydá 4 čísla:
- roll -> požadovaný náklon (max 45°)
- pitch -> požadovaný úhel stoupání/klesání (max 15°)
- throttle -> požadovaná rychlost (0 až 30 m/s)
- water_trigger -> otevřít/zavřít ventil vody

Každý krok se pak numericky integrují diferenciální rovnice, které popisují,
jak se reálný letoun chová — jak se mění rychlost, kurz, náklon.
Výsledná pozice se zapíše do PyBulletu.

### Jak zatáčí

Letoun zatáčí tak, jak zatáčí v realitě — nakloní se (phi), a náklon způsobí
zakřivení trajektorie. Čím vyšší rychlost a menší náklon, tím větší poloměr zatáčení.
Dron nemůže stát na místě ani točit se na místě.

### Vítr a termika

Vítr se přičítá přímo k rychlosti přes zem (ground speed).
To znamená, že pokud je dron nad ohněm, termika (stoupavý vzduch z fire gridu)
ho automaticky nese nahoru — tenhle efekt vychází přirozeně z modelu.

### Stall logika

Pokud airspeed klesne pod 7 m/s:
- Nos se skloní dolů (kód přepíše požadovaný gamma na -15°)
- Náklon se tlumí
- Gravitace pomáhá znovu zrychlit

Bez tohohle by dron při pomalém pohybu prostě zamrzl ve vzduchu.

### Zásobník vody

200 litrů, spotřeba 5 L/s. Plní se automaticky při přeletu přes refill zónu
(detekce v 2D, výška se ignoruje — jinak by to bylo nemožné trefit).

### Je to lossless model?

Ano — žádné palivo, žádná únava konstrukce, žádný aerodynamický odpor (drag).
Jediné fyzické hranice jsou minimální rychlost (stall) a maximální náklon (45°)
a úhel stoupání (15°).

---

## OSM data — odkud se bere terén a jak se dostane do simulace

### Co je OSM a co se vlastně stahuje

OpenStreetMap (OSM) jsou vektorová data — ne satelitní snímky, ale databáze
geografických objektů. Každý objekt má geometrii (polygon, linie, bod) a sadu
tagů (klíč=hodnota), které říkají co to je. Například:

```
building=yes          → tohle je budova
landuse=forest        → tady je les
natural=water         → tady je voda
waterway=river        → řeka
```

Nestahují se tedy pixely z fotky, ale seznam polygonů s popiskami.
Stažení zajišťuje knihovna `osmnx`, která volá OSM Overpass API.

### Kdy se data stahují — cache

Systém má dvě varianty:

**A) Přímé stažení při spuštění** (`load_environment_from_osm`)
Zavolá `ox.features_from_point(gps_souřadnice, tags, dist=radius_m)` — stáhne
vše v okruhu `radius_m` metrů od středu. Výsledek okamžitě uloží do souborů
`data/{lokalita}_{kategorie}_0.gpkg`. Příště se to už nestahuje, jen načte z disku.

**B) Z předem stažené cache** (`load_environment_from_osm_cache`)
Používáme my. Data jsou v `data/` jako `.gpkg` soubory (GeoPackage — binární
formát pro vektorová geodata). Při každém resetu prostředí se z těchto souborů
čte, oříže se kruh kolem požadovaného středu a předá se dál.

**Ke tvé otázce — stahuje se to celé znova při jiné velikosti?**
Ne. Cache soubory obsahují celou oblast (řádově kilometry). Když chceš
500×500 m nebo 1000×1000 m, kód vždy čte ze stejného souboru a jen vybere
jinak velký výřez (`get_subregion_by_point` s jiným `radius_m`). Síť se netouche.

### Jak se vektorová data převedou na mřížku (rasterizace)

Tady je klíčový krok — přechod z vektorů na grid, který požití fire simulace.

**Krok 1: Projekce souřadnic**
GPS souřadnice (stupně) se přepočítají na metry pomocí UTM projekce
(konkrétní zóna se určí automaticky ze zeměpisné délky). Tím dostaneme
x, y v metrech. Střed scény se přeloží na (0, 0) — vše je pak relativní
vůči středu simulace.

**Krok 2: Filtrování do kategorií**
Všechny stažené objekty se roztřídí do tří vrstev podle tagů:
- Voda: `natural=water`, `waterway=river/canal`, `landuse=reservoir`
- Budovy: `building=*`, `landuse=residential/commercial/industrial`
- Les: `landuse=forest/wood`, `natural=wood/scrub`

**Krok 3: Rasterizace — Painter's Algorithm**

Fuel mapa (grid pro simulaci šíření ohně) se naplní takto:

```
Výchozí stav celé mřížky = tráva (fuel=0.3, burn_rate nízký)
↓
Překreslí se lesy        (fuel=0.8, hoří pomaleji)
↓
Překreslí se budovy      (fuel=0.9, hoří nejpomaleji)
↓
Překreslí se voda        (fuel=0.0, nehoří — firebreak)
```

Takže voda přepíše les, les přepíše trávu atd. — pozdější vrstvy mají vyšší
prioritu (proto "Painter's Algorithm" — jako malíř, který přemaluje předchozí vrstvu).

**Jak konkrétně se polygon převede na buňky?**
Ne cell-by-cell sekvenčně. Dělá se to vektorizovaně:

1. Vezme se polygon (např. polygon lesa)
2. Najdou se jeho bounds (min/max x, y) a přepočítají se na indexy v gridu
3. Pro všechny buňky v tomto obdélníku se vygeneruje mřížka souřadnic středů buněk
4. Zavolá se `matplotlib.path.contains_points()` — C-based funkce, která
   vektorizovaně řekne pro každý bod, jestli leží uvnitř polygonu nebo ne
5. Buňky, které padnou dovnitř polygonu, dostanou příslušné fuel/burn_rate hodnoty

Takže ano, každá buňka se zkontroluje jestli leží uvnitř polygonu — ale ne
v Pythonové smyčce. Jde to přes numpy mask, takže je to rychlé.

**Výsledné hodnoty v gridu:**

| Typ terénu | fuel (F) | burn_rate | Čas hoření (1 buňka 5×5m) |
|---|---|---|---|
| Tráva (default) | 0.3 | 0.01 / m² | ~30 s |
| Les | 0.8 | 0.0067 / m² | ~2 min |
| Budova | 0.9 | 0.0015 / m² | ~10 min |
| Voda | 0.0 | 0.0 | nehořit |

Burn rate se navíc škáluje podle plochy buňky — větší buňka hoří
proporcionálně déle (protože simulujeme stejnou hustotu paliva na m²).

### Jak to poznat z dat co se stáhnou z OSM?

Každý stažený řádek v GeoDataFrame má sloupce jako `building`, `landuse`,
`natural`, `waterway`. Kód se kouká přesně na tyto sloupce:

```python
# Voda
natural == 'water'  nebo  waterway == 'river'  nebo  landuse == 'reservoir'

# Les
landuse == 'forest'  nebo  natural == 'wood'  nebo  natural == 'scrub'

# Budova
building != None  nebo  landuse == 'residential'
```

Cokoliv co nepasuje do žádné kategorie, zůstane tráva (default).
Geometrie objektu (polygon) pak určí, které buňky gridu patří do dané kategorie.

---

## Trénink — jak se sítě učí

### Přehled: co se trénuje a čím

Trénujeme dvě sítě zároveň, každou zvlášť vlastním optimalizátorem:
- **ScoutActor** — učí se navigovat k ohni a posílat zprávy
- **CommanderActor** — učí se létat k ohni na základě zpráv a hasit

Navíc jsou tu dvě pomocné sítě **PrivilegedCritic** (jedna pro každý typ agenta),
které se samy trénují a slouží jen k lepšímu odhadu hodnoty stavu — do simulace
se nenasazují, existují jen při tréninku.

Algoritmus je **MAPPO** — Multi-Agent PPO (Proximal Policy Optimization)
s centralizovaným tréninkem a decentralizovaným vykonáváním (CTDE).

---

### Krok 1: Sběr dat — rollout

Na začátku každého batche se spustí 15 paralelních workerů (každý na vlastním
CPU procesu). Každý worker odehraje 2 epizody simulace — dron létá, dostává
odměny, a ukládá si záznamy každého kroku:

```
Pro každý krok scoutu:
  - co viděl (local_map, self_state, zprávy sousedů)
  - jakou akci zvolil (4 čísla)
  - jak pravděpodobná ta akce byla (log_prob)
  - co dostal za odměnu
  - odhad hodnoty stavu od kritika

Pro commandera: totéž, ale jen při každém 30. kroku (waypoint decision)
```

Celkem jeden batch = 15 workerů × 2 epizody × 1000 kroků × 2 scouti
= ~60 000 záznamů pro scouty.

---

### Krok 2: GAE — výpočet "jak dobré bylo rozhodnutí"

Samotná odměna jednoho kroku nestačí — potřebujeme vědět, jak moc ta akce
pomohla nebo uškodila v delším horizontu. K tomu slouží **GAE (Generalized
Advantage Estimation)**.

Počítá se zpětně od konce epizody:

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
GAE_t   = delta_t + gamma * lambda * GAE_{t+1}
return_t = GAE_t + V(s_t)
```

- `r_t` = odměna v kroku t
- `V(s_t)` = odhad kritika, jak dobrý je stav v kroku t
- `gamma = 0.99` pro scouta, `0.95` pro commandera — jak moc se dívám do budoucnosti
- `lambda = 0.95` — kompromis mezi bias a variancí

`GAE_t` pak říká: "tato akce byla o tolik lepší (nebo horší) než průměrné
chování v tomto stavu". Říkáme tomu **advantage** — výhoda.

---

### Krok 3: PPO update — vlastní backpropagation

Po sběru dat ze všech workerů se provede update sítě. Opakuje se 4× (update_epochs)
přes zamíchaná mini-batch data.

#### Policy loss (klíčová část)

Pro každý krok se spočítá, jak moc se změnila pravděpodobnost akce:

```python
ratio = exp(new_log_prob - old_log_prob)
```

`ratio > 1` znamená, že nová politika tu akci preferuje více.
`ratio < 1` znamená, že ji preferuje méně.

PPO pak použije clipping — zabraňuje příliš velkým skokům:

```python
pg1 = -advantage * ratio
pg2 = -advantage * clamp(ratio, 0.8, 1.2)   # clip_coef = 0.2
loss = mean( max(pg1, pg2) )                  # pesimistická varianta
```

Pokud byla akce dobrá (advantage > 0):
- chceme ji dělat víc (ratio > 1), ale ne moc — clipping zastřihne ratio na max 1.2
- výsledek: mírné zvýšení pravděpodobnosti

Pokud byla akce špatná (advantage < 0):
- chceme ji dělat méně (ratio < 1), ale ne prudce — clipping na min 0.8
- výsledek: mírné snížení pravděpodobnosti

Záporné znaménko před `advantage * ratio` je proto, že minimalizujeme loss
(PyTorch minimalizuje), ale chceme maximalizovat odměnu — proto negace.

#### Entropy loss

K policy loss se přidá bonus za entropii (míru "náhodnosti" politiky):

```python
loss = policy_loss - entropy_coef * entropy
```

Bez toho by síť příliš rychle zkolapsovala na deterministické chování
a přestala explorovat. `entropy_scout = 0.002`, `entropy_cmdr = 0.01`.

#### Auxiliary loss (jen commander)

Commander má navíc pomocnou hlavičku, která predikuje polohu a intenzitu
ohně (2 čísla). Chyba této predikce se přičítá do lossu:

```python
loss_c = policy_loss - entropy_cmdr * entropy + 0.05 * aux_loss
```

Váha 0.05 je záměrně malá — jde jen o slabý doplněk, fire compass v obs
(indexy 19–22) dává tuto informaci přímo.

#### Backpropagation a update vah

```python
optimizer.zero_grad()    # vynuluj staré gradienty
loss.backward()          # spočítej parciální derivace přes celý graf výpočtu
clip_grad_norm_(..., max_norm=0.5)  # ořízni příliš velké gradienty
optimizer.step()         # Adam: uprav váhy podle gradientů
```

`loss.backward()` projde celý výpočetní graf zpětně (backpropagation) a ke
každému parametru sítě přiřadí gradient — o kolik se má hodnota parametru
změnit, aby loss klesl. Adam optimizer pak provede update s adaptivním
learning rate pro každý parametr zvlášť.

Gradient clipping (`max_norm=0.5`) zajišťuje, že žádný krok není příliš
velký — bez toho by mohlo dojít ke kolapsu sítě.

#### Critic update (odděleně)

Critic se trénuje zvlášť minimalizací MSE mezi svým odhadem `V(s)` a skutečnými
`returns` z GAE:

```python
v_loss = mean( (V_pred - returns_normalized)^2 )
```

Normalizace returns (`mean=0, std=1`) zabraňuje, aby MSE exploze ovlivnila
škálování gradientů.

---

### Krok 4: Privilegovaný critic — co navíc vidí

Critic při tréninku dostává víc informací než actor při nasazení.
Konkrétně k `self_state` dostane navíc 6 čísel o globálním stavu ohně
(poloha, intenzita, počet hořících buněk...). To mu umožňuje lépe odhadnout
hodnotu stavu, čímž GAE dostane přesnější advantage — a actor se naučí lépe.

Při nasazení critic neexistuje, actor ho nepotřebuje.

---

### Jak GRU pamět ovlivňuje backpropagation

Scout má GRU (rekurentní vrstu) — jeho hidden state se přenáší mezi kroky.
To vytváří problém: backpropagation by musela procházet celou epizodu zpět
(1000 kroků), což je pomalé a numericky nestabilní.

Řešení: **Chunked BPTT** (Backpropagation Through Time po blocích):
- Epizoda se rozdělí na bloky po 128 krocích (`bptt_chunk = 128`)
- GRU hidden state se mezi bloky detachuje (`h_chunk = h_chunk.detach()`)
- Gradienty tečou jen 128 kroků zpět, ne celou epizodu

Tím se zachová krátkodobá paměť, ale zabrání se exploze gradientů přes celou epizodu.

---

### Souhrn tréninku v číslech (aktuální konfigurace)

| Parametr | Hodnota |
|---|---|
| Workers (paralelní epizody) | 15 |
| Epizody na batch | 30 (15×2) |
| Kroků na epizodu | 1000 |
| Update epochs per batch | 4 |
| Mini-batches | 4 |
| Learning rate scout | 3e-6 (fine-tuning) |
| Learning rate cmdr | 3e-4 |
| PPO clip | 0.2 |
| Gamma scout | 0.99 |
| Gamma cmdr | 0.95 |
| GAE lambda | 0.95 |

---

## Porovnání obou dronů

| Otázka | Scout | Commander |
|---|---|---|
| Fyzika v PyBulletu | Plná (síly, integrace) | Jen vizualizace |
| Může hovovat? | Ano | Ne |
| Minimální rychlost | 0 m/s | ~10 m/s |
| Jak zatáčí | Yaw torque (na místě) | Náklon + zakřivení dráhy |
| Voda | Ne | Ano (200 L) |
| NN frekvence | Každý krok (30 Hz) | Každých 30 kroků (~1 Hz) |
| Komunikace | Vysílá 5D zprávy | Přijímá zprávy od scoutů |
