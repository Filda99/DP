# 🚁 Quadcopter Fire Detection - RL Training System

**Jednoduchý a čistý systém pro trénování kvadrokoptéry na detekci ohně pomocí reinforcement learningu v PyBullet simulaci.**

## 🚀 Rychlý start

```bash
# 1. Ověř že prostředí funguje
python main.py validate

# 2. Spusť trénink
python main.py train

# 3. Otestuj natrénovaný model  
python main.py demo
```

## 📁 Struktura projektu

```
├── main.py              # 🎯 Hlavní script (train/demo/validate)
├── demo.py              # 🎬 Demo různých funkcí
├── validation_demo.py   # ✅ Validace modelů
├── src/                 # 🏗️ Core kód
│   ├── simulation.py    # PyBullet simulace
│   ├── wildfire_gym_wrapper.py  # Gym prostředí
│   ├── wildfire_models.py       # Neural network modely
│   └── ...
├── models/              # 💾 Natrénované modely (nové)
└── trained_models/      # 📦 Staré experimenty
```

## 🎮 Použití

### Trénink
```bash
python main.py train    # Spustí 50 epizod tréninku
```

### Demo a testování  
```bash
python main.py demo     # Test nejnovějšího modelu
python demo.py physics  # Test fyziky drona
python demo.py fire     # Test detekce ohně
python demo.py env      # Kompletní test prostředí
```

### Validace
```bash  
python main.py validate        # Ověří prostředí
python validation_demo.py      # Detailní test modelu
```

## 🎯 Co systém dělá

1. **Kvadrokoptéra** se naučí létat v 3D prostoru
2. **Detekuje oheň** pomocí CNN na místní mapě 32x32px  
3. **Získává reward** za:
   - Přiblížení k ohni
   - Vizuální detekci ohně
   - Udržení stabilního letu
4. **Vyhýbá se crashům** (penalty -1000)

## 🧠 Architektura

- **CNN**: Zpracování fire mapy (32x32 → 64 features)  
- **GRU**: Paměť a sekvenční rozhodování (128 hidden)
- **PPO**: Policy gradient training
- **Action space**: 4 kontinuální akce [roll, pitch, yaw, throttle]

## 📊 Výsledky

- **Dobrý model**: 15,000+ reward za epizodu
- **Průměrný**: 5,000-15,000 reward  
- **Špatný**: < 1,000 reward

## ⚙️ Konfigurací

Hlavní parametry v `main.py`:
- `max_episodes = 50` - Počet trénovacích epizod
- `lr = 3e-4` - Learning rate
- `save_every = 10` - Jak často ukládat model

## 🐛 Známé problémy

1. **Policy collapse** - Model se může rozpadnout po ~20 epizodách
2. **Reward tuning** - Může potřebovat doladění pro různé scénáře

## 💡 Tips

- Model se ukládá každých 10 epizod - můžeš zastavit trénink a použít nejlepší checkpoint
- Pokud model crashuje moc často, zkus snížit learning rate
- Pro rychlejší experimenty uprav `max_episodes` na menší číslo

---
**Created for DP project - Quadcopter fire detection using RL** 🔥🚁