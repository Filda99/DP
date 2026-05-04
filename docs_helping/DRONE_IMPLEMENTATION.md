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
