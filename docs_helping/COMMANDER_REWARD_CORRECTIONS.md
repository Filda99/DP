# Commander Rewards — LaTeX vs. Actual Code

Comparison of the LaTeX draft text vs. the current implementation in
`env_core.py`, `reward_config.py`, and `commander_control.py`.

---

## 1. Commander Objectives — What the text gets WRONG

### Boundary enforcement
- **Text says**: "triggers 300 m from the edge"
- **Code** (`reward_config.py`): `boundary_threshold_m = 200.0`
- **Fix**: Change to 200 m.

### Altitude corridor
- **Text says**: "40 m to 250 m"
- **Code**: `alt_ideal_min = 30`, `alt_ideal_max = 80`, `alt_ceiling = 200`
- **Fix**: Corridor is **30–80 m** (penalty outside this). Hard kill at **200 m**, not 250 m.

### alt_penalty
- **Text says**: "-0.1 / step"
- **Code**: `alt_penalty = 0.02` — **per metre** above/below the corridor, not flat per step.
- At 100 m altitude: penalty = -(100 - 80) × 0.02 = -0.4 per step.
- **Fix**: "-0.02 per metre outside 30–80 m corridor"

### Reward hacking / water-conditioned shaping
- **Text describes**: approach_fire (active only if water > 5%), zone_bonus_fire at 300m/150m,
  approach_refill (active only if empty), zone_bonus_refill at 100m.
- **NONE of this exists in the code.** The code has:
  - `fire_approach_k = 0.25` — potential shaping **towards nearest scout** (not fire directly),
    **always active regardless of water level**.
  - `r_alt_shape = 0.3 * alt_effectiveness` when within 200 m of a scout — rewards flying low near fire.
  - **No zone_bonus_fire, no zone_bonus_refill, no approach_refill.**
  - **No water-conditioned switching.** The NN does NOT learn the refill cycle at all.
    Refill is entirely scripted (see SCRIPTED_SYSTEMS.md).

### Config values that exist but are DEAD CODE (never used in env_core.py)
These are in `reward_config.py` under FIXED but never referenced in `env_core.py`:
- `donut_radius`, `donut_bonus` — NOT USED
- `rubber_band_k` — NOT USED
- `survival_base` — NOT USED (SHARED["survival_bonus"] is used instead)
- `water_trigger_bonus`, `water_trigger_dist`, `water_trigger_alt`, `water_trigger_thresh` — NOT USED
- `water_guidance_bonus` — NOT USED
- `refill_state_bonus` — NOT USED
- `refill_proximity_dist`, `refill_proximity_bonus` — NOT USED
- `survival_weight`, `mission_weight` — NOT USED
- `reward_scale` — NOT USED

---

## 2. What the commander ACTUALLY receives (per step)

### A. Physics shaping (`_apply_physics_shaping`) — for ALL agents
| Component | Value | Condition |
|-----------|-------|-----------|
| `survival_bonus` | +0.02 | Always (alive) |
| `boundary_penalty` | up to -1.5 | Quadratic, within 200m of edge |
| `alt_penalty` | -0.02 × excess_m | Outside 30–80m corridor |
| `crash_penalty` | -10.0 | Ground/boundary/ceiling kill |

### B. Mission reward (`_get_fixed_reward_nav`) — FW only
| Component | Value | Condition |
|-----------|-------|-----------|
| `fire_approach_k` | delta × 0.25 | Potential shaping towards nearest **scout** (not fire). Always active. |
| `alt_shape` | up to +0.3 | When within 200m of scout. `0.3 × max(0, 1 - (alt/150)²)` |

### C. Team reward (in `step()`) — event-based
| Component | Value | Condition |
|-----------|-------|-----------|
| `extinguish_bonus` | min(eff × 100, 10.0) | FW successfully extinguishes fire cells. Scouts get 15% of this. |
| `water_waste_penalty` | -0.5 | FW triggers valve but hits no fire (per step with valve open) |
| `fire_out_bonus` | +50.0 (FW), +10.0 (scout) | All fire extinguished |
| `spread_penalty` | up to -5.0 | New burning cells appear (phase ≥ 3). Shared. |
| `time_up_bonus` | +2.0 | Episode ends naturally (survived full episode) |

### D. Reward clip
All per-step rewards clipped to **[-3.0, +3.0]**.

---

## 3. Corrected LaTeX table

```latex
\begin{table}[H]
    \centering
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{|l|l|p{7.5cm}|}
        \hline
        \textbf{Reward Component} & \textbf{Value / Scale} & \textbf{Description \& Condition} \\
        \hline
        \texttt{survival\_bonus} & $+0.02$ / step & Shared base reward for staying alive. \\
        \texttt{crash\_penalty} & $-10.0$ & Ground collision, boundary violation, or ceiling breach. \\
        \texttt{boundary\_penalty} & up to $-1.5$ & Quadratic penalty within 200\,m of map edge. \\
        \texttt{alt\_penalty} & $-0.02 \times \Delta_\text{m}$ & Per-metre penalty outside the 30--80\,m altitude corridor. \\
        \hline
        \texttt{fire\_approach\_k} & $\Delta d \times 0.25$ & Potential-based shaping toward the nearest scout
                                                               (scouts hover over fire, so this is an indirect fire approach signal).
                                                               Always active regardless of water level. \\
        \texttt{alt\_shape} & up to $+0.3$ & Altitude effectiveness bonus when within 200\,m of a scout.
                                              Rewards lower flight: $0.3 \times \max(0,\; 1 - (z/150)^2)$. \\
        \hline
        \texttt{extinguish\_bonus} & $\min(\text{eff} \times 100,\; 10.0)$ & Actual fire suppression (event-based, from simulation). \\
        \texttt{water\_waste\_penalty} & $-0.5$ & Valve opened but no fire cells extinguished. \\
        \texttt{fire\_out\_bonus} & $+50.0$ & All fire in the episode extinguished. \\
        \texttt{spread\_penalty} & up to $-5.0$ & New cells ignite; shared across all agents. \\
        \hline
    \end{tabular}
    \caption{Reward parameters driving the fixed-wing commander's behaviour.}
    \label{tab:commander_rewards}
\end{table}
```

---

## 4. Team Credit Assignment — corrections

### Text says
- "massive extinguish_reward based on number of suppressed fire cells"
- "water_waste_penalty = -3.0"

### Code actually does
- `extinguish_bonus = min(eff * 100.0, 10.0)` where `eff` is the simulation's
  extinguish effectiveness (area-based, not cell count). Capped at 10.0.
- `water_waste_penalty = 0.5` (NOT 3.0). Defined in `reward_config.py`.
  Can be overridden per-curriculum-phase via `self.waste_penalty_override`.
- Scouts receive `fire_bonus * 0.15` when FW extinguishes. The text doesn't
  mention the 15% sharing ratio.
- Full fire-out bonus: FW gets +50, scouts get +10 each.
