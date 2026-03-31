#!/usr/bin/env python3
"""
demo_pid_altitude.py
====================
Diagnostic: proves the fixed-wing flight-path-angle physics and compares
hardcoded pitch constants vs. a PD altitude controller.

Physics of the fixed-wing model (src/drones/fixedwing.py):
  action[1] (pitch) → γ^c (flight-path angle setpoint)
    gamma_c = clip(pitch, -1, 1) × 45°
    γ̇ = kp_gamma × (γ^c - γ),  kp_gamma = 5.0
    ḣ  = Va × sin(γ)

  Consequence:
    pitch = 0.0  →  γ^c = 0°  →  level flight  (ḣ = 0)
    pitch = 0.15 →  γ^c = 6.75° → climb ≈ +1.8 m/s at Va=15 m/s
    pitch = -0.2 →  γ^c = -9°  → descent ≈ -2.4 m/s at Va=15 m/s

  Throttle mapping (done by env_core.py, AFTER the 0.8/0.2 smoothing):
    mapped_throttle = 0.55 + (nn_throttle + 1.0) × 0.225
    target_va = mapped_throttle × 30.0 m/s
    nn=0.0 → target_va = 23.25 m/s  (good cruise speed)

Usage (from repo root):
    python demos/fixedwing/demo_pid_altitude.py
Saves: demos/fixedwing/pid_altitude_result.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
# Fix cwd so env_core can find ../urdf/ relative paths
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env_core import DroneFireEnv

# ──────────────────────────────────────────────────────────────────────────────
STEPS     = 400   # steps per experiment (env max_steps must be >= this)
SPAWN_Z   = 60.0  # standard spawn altitude (metres)
TARGET_Z  = 60.0  # PD controller setpoint

# PD altitude controller gains (operate in NN pitch space [-1, 1])
# Design rationale:
#   We want 10 m altitude error → γ^c ≈ 3°
#   3° in NN space = 3/45 = 0.0667
#   → Kp = 0.0667 / 10 ≈ 0.007
#   Damping: 2 m/s vertical speed should produce ~0.03 pitch correction
#   → Kd = 0.03 / 2 = 0.015
Kp_alt = 0.007
Kd_alt = 0.015
# ──────────────────────────────────────────────────────────────────────────────


def run_experiment(pitch_fixed=None, use_pid=False):
    """
    Run one episode with a fixed-wing only (no quads).

    Parameters
    ----------
    pitch_fixed : float or None
        Constant pitch NN value to apply at every step.
        Ignored when use_pid=True.
    use_pid : bool
        If True, use the PD altitude controller instead.

    Returns
    -------
    altitudes : np.ndarray  [STEPS]
    gammas_deg: np.ndarray  [STEPS]  — actual flight-path angle in degrees
    """
    env = DroneFireEnv(num_quads=0, num_fixed=1,
                       grid_size_m=2000.0, max_steps=STEPS + 10)
    obs, _ = env.reset(epizode_number=42)

    altitudes  = []
    gammas_deg = []
    crashed_at = None

    for step in range(STEPS):
        drone = env.sim.drones.get("fixed_0")
        if drone is None or not hasattr(drone, 'state_pos'):
            # Crashed — pad with last known values
            altitudes.append(altitudes[-1] if altitudes else 0.0)
            gammas_deg.append(gammas_deg[-1] if gammas_deg else 0.0)
            continue

        alt   = drone.state_pos[2]
        gamma = np.degrees(drone.state_gamma)
        altitudes.append(alt)
        gammas_deg.append(gamma)

        if use_pid:
            vel     = drone.get_velocity()
            vz      = vel[2]              # positive = climbing
            alt_err = TARGET_Z - alt      # positive = too low
            pitch   = float(np.clip(Kp_alt * alt_err - Kd_alt * vz, -0.3, 0.3))
        else:
            pitch = pitch_fixed

        # Roll=0 (straight), Throttle=0 (→ 23.25 m/s), Water=-1 (off)
        action = np.array([0.0, pitch, 0.0, -1.0], dtype=np.float32)
        _, _, terms, _, _ = env.step({"fixed_0": action})

        if terms.get("fixed_0", False) and crashed_at is None:
            crashed_at = step
            # Continue loop — env pads with zeros after crash

    env.sim.stop_simulation()
    return np.array(altitudes), np.array(gammas_deg), crashed_at


def main():
    experiments = [
        # (pitch_fixed, use_pid, label, linestyle)
        (-0.20, False, "pitch = −0.20  (γ^c = −9°,  descent ≈ −2.4 m/s)", "-"),
        (-0.10, False, "pitch = −0.10  (γ^c = −4.5°, descent ≈ −1.2 m/s)", "-"),
        ( 0.00, False, "pitch =  0.00  (γ^c =  0°,   LEVEL FLIGHT)",        "-"),
        ( 0.10, False, "pitch = +0.10  (γ^c = +4.5°, climb  ≈ +1.2 m/s)",  "-"),
        ( 0.15, False, "pitch = +0.15  (γ^c = +6.75°, climb ≈ +1.8 m/s)",  "-"),
        ( None, True,  f"PD controller  (target = {TARGET_Z:.0f} m)",        "--"),
    ]

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(experiments)))

    fig, (ax_alt, ax_gam) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Fixed-wing altitude response: hardcoded pitch vs. PD controller", fontsize=13)

    for (pf, pid, label, ls), color in zip(experiments, colors):
        tag = label.split("(")[0].strip()
        print(f"⏳  {tag} ...", flush=True)
        alts, gams, crash = run_experiment(pitch_fixed=pf, use_pid=pid)
        lw = 2.5 if pid else 1.5
        ax_alt.plot(alts, label=label, color=color, ls=ls, lw=lw)
        ax_gam.plot(gams, label=label, color=color, ls=ls, lw=lw)
        if crash is not None:
            ax_alt.axvline(crash, color=color, lw=0.8, alpha=0.5)
            print(f"   ↳ crashed at step {crash}  (alt={alts[crash-1]:.1f} m)")

    # Reference lines
    ax_alt.axhline(TARGET_Z, color='gray', ls=':', lw=1.2, label=f'target = {TARGET_Z} m')
    ax_alt.axhline(0.0,      color='black', ls='-', lw=0.5)
    ax_alt.set_title("Altitude (m) over time")
    ax_alt.set_xlabel("Step")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.legend(fontsize=7.5, loc='upper right')
    ax_alt.grid(True, alpha=0.4)

    ax_gam.axhline(0.0, color='gray', ls=':', lw=1.2, label='level (γ = 0°)')
    ax_gam.set_title("Actual flight-path angle γ (degrees)")
    ax_gam.set_xlabel("Step")
    ax_gam.set_ylabel("γ (°)")
    ax_gam.legend(fontsize=7.5)
    ax_gam.grid(True, alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "pid_altitude_result.png")
    plt.savefig(out_path, dpi=130)
    plt.close()

    print(f"\n✅  Plot saved → {out_path}")
    print()
    print("Physics summary (from fixedwing.py):")
    print("  action[1] (pitch) → γ^c = pitch × 45°  (flight-path angle setpoint)")
    print("  Inner loop: γ̇ = 5.0 × (γ^c − γ)   (kp_gamma = 5.0)")
    print("  Altitude:   ḣ = Va × sin(γ)")
    print()
    print("  pitch = 0.0  →  γ^c = 0°  →  ḣ = 0  →  EXACT LEVEL FLIGHT")
    print()
    nn = 0.0
    mapped = 0.55 + (nn + 1.0) * 0.225
    print(f"  Throttle NN = {nn}  →  mapped = {mapped:.3f}  →  target_va = {mapped*30:.1f} m/s")
    nn = 0.3
    mapped = 0.55 + (nn + 1.0) * 0.225
    print(f"  Throttle NN = {nn}  →  mapped = {mapped:.3f}  →  target_va = {mapped*30:.1f} m/s")
    print()
    print("PD gains used in autopilot (worker.py):")
    print(f"  Kp_alt = {Kp_alt}  (10 m error → pitch ≈ {Kp_alt*10:.3f} ≈ {Kp_alt*10*45:.1f}° climb)")
    print(f"  Kd_alt = {Kd_alt}  (2 m/s vz  → damp  {Kd_alt*2:.3f})")


if __name__ == "__main__":
    main()
