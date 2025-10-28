"""
Visualization of Fire Convection Physics
Illustrates the correct airflow pattern around a fire plume
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection

def visualize_fire_convection():
    """Create a detailed visualization of fire convection airflow."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # ===== Panel 1: Side View (Vertical Cross-Section) =====
    ax1 = axes[0]
    
    # Fire source at ground
    fire_x = 0
    fire_width = 4
    ax1.add_patch(Rectangle((fire_x - fire_width/2, 0), fire_width, 1, 
                            color='red', alpha=0.7, label='Fire'))
    
    # Draw convection column
    height_max = 50
    plume_width_ground = 2
    plume_width_top = 8
    
    # Plume outline
    x_plume = [
        -plume_width_ground, -plume_width_top, plume_width_top, plume_width_ground
    ]
    y_plume = [0, height_max, height_max, 0]
    ax1.fill(x_plume, y_plume, color='orange', alpha=0.2, label='Convection Column')
    
    # Vertical velocity profile (w component)
    heights = np.linspace(0, height_max, 100)
    velocities = []
    for h in heights:
        norm_h = h / height_max
        if norm_h < 0.3:
            vel = norm_h / 0.3  # Rising to peak
        else:
            vel = (1.0 - norm_h) / 0.7  # Decaying from peak
        velocities.append(vel * 10)  # Scale to 10 m/s max
    
    # Plot vertical velocity profile
    ax1_twin = ax1.twiny()
    ax1_twin.plot(velocities, heights, 'b-', linewidth=2, label='Vertical Velocity')
    ax1_twin.set_xlabel('Upward Velocity w [m/s]', color='b')
    ax1_twin.tick_params(axis='x', labelcolor='b')
    ax1_twin.set_xlim([0, 12])
    ax1_twin.grid(True, alpha=0.3)
    
    # Draw radial flow arrows
    # Inward flow at low altitude
    for y in [5, 10, 15, 20]:
        ax1.annotate('', xy=(0, y), xytext=(-15, y),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax1.annotate('', xy=(0, y), xytext=(15, y),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Outward flow at high altitude
    for y in [30, 35, 40, 45]:
        ax1.annotate('', xy=(15, y), xytext=(0, y),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax1.annotate('', xy=(-15, y), xytext=(0, y),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    # Add labels
    ax1.text(-20, 12, 'Inward\nFlow', color='green', fontsize=12, ha='center', weight='bold')
    ax1.text(-20, 38, 'Outward\nFlow', color='blue', fontsize=12, ha='center', weight='bold')
    ax1.axhline(y=25, color='k', linestyle='--', alpha=0.5, label='Transition (~50% height)')
    
    ax1.set_xlim([-25, 25])
    ax1.set_ylim([0, 55])
    ax1.set_xlabel('Horizontal Distance [m]')
    ax1.set_ylabel('Height [m]')
    ax1.set_title('Side View: Fire Convection Column\n(Correct Physics)', fontsize=14, weight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # ===== Panel 2: Top View (Horizontal Cross-Section at Low Altitude) =====
    ax2 = axes[1]
    
    # Fire cell
    ax2.add_patch(Rectangle((-2, -2), 4, 4, color='red', alpha=0.7))
    
    # Draw inward radial flow arrows
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    for angle in angles:
        # Starting points (far from fire)
        start_r = 15
        start_x = start_r * np.cos(angle)
        start_y = start_r * np.sin(angle)
        
        # Ending points (near fire)
        end_r = 3
        end_x = end_r * np.cos(angle)
        end_y = end_r * np.sin(angle)
        
        ax2.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax2.set_xlim([-20, 20])
    ax2.set_ylim([-20, 20])
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_title('Top View: Low Altitude (<50% height)\nINWARD Radial Flow', fontsize=14, weight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.text(0, -18, 'Air rushes toward fire\nto replace rising column', 
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===== Panel 3: Top View (Horizontal Cross-Section at High Altitude) =====
    ax3 = axes[2]
    
    # Fire cell (fainter - it's below)
    ax3.add_patch(Rectangle((-2, -2), 4, 4, color='red', alpha=0.3))
    
    # Draw outward radial flow arrows
    for angle in angles:
        # Starting points (near fire)
        start_r = 3
        start_x = start_r * np.cos(angle)
        start_y = start_r * np.sin(angle)
        
        # Ending points (far from fire)
        end_r = 15
        end_x = end_r * np.cos(angle)
        end_y = end_r * np.sin(angle)
        
        ax3.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    ax3.set_xlim([-20, 20])
    ax3.set_ylim([-20, 20])
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.set_title('Top View: High Altitude (>50% height)\nOUTWARD Radial Flow', fontsize=14, weight='bold')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.text(0, -18, 'Rising air spreads out\nat top of plume', 
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('fire_convection_physics.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved: fire_convection_physics.png")
    plt.show()

def plot_current_vs_correct():
    """Compare current implementation vs. correct physics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Simulation parameters
    airflow_H = 50.0
    cell_size = 2.0
    fire_intensity = 1.0
    
    # Create grid of positions
    heights = np.linspace(0, airflow_H, 50)
    radii = np.linspace(0, 20, 50)
    
    # ===== CURRENT IMPLEMENTATION =====
    
    # Panel 1: Current vertical velocity
    ax1 = axes[0, 0]
    for r in [0, 5, 10, 15]:
        w_values = []
        for h in heights:
            if h < airflow_H:
                height_taper = max(0.0, 1.0 - h / airflow_H)
                w = fire_intensity * 0.5 * height_taper  # Current: gain=0.5, no radial attenuation
                w_values.append(w)
            else:
                w_values.append(0)
        ax1.plot(heights, w_values, label=f'r = {r}m', linewidth=2)
    
    ax1.set_xlabel('Height [m]')
    ax1.set_ylabel('Upward Velocity w [m/s]')
    ax1.set_title('CURRENT: Vertical Velocity Profile\n(No radial attenuation, weak magnitude)', 
                  fontsize=12, weight='bold', color='red')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Panel 2: Current radial velocity
    ax2 = axes[0, 1]
    for h in [5, 15, 25, 35, 45]:
        u_values = []
        for r in radii:
            if h < airflow_H and r > 1e-6:
                height_taper = max(0.0, 1.0 - h / airflow_H)
                w = fire_intensity * 0.5 * height_taper
                u = (w / 4.0)  # Current: always outward
                u_values.append(u)
            else:
                u_values.append(0)
        ax2.plot(radii, u_values, label=f'h = {h}m', linewidth=2)
    
    ax2.set_xlabel('Radial Distance [m]')
    ax2.set_ylabel('Radial Velocity u [m/s]')
    ax2.set_title('CURRENT: Radial Velocity (ALWAYS Outward)\n❌ WRONG - Should be inward at low altitude', 
                  fontsize=12, weight='bold', color='red')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # ===== CORRECT IMPLEMENTATION =====
    
    # Panel 3: Correct vertical velocity
    ax3 = axes[1, 0]
    for r in [0, 5, 10, 15]:
        w_values = []
        for h in heights:
            if h < airflow_H:
                # Peaked height profile
                norm_h = h / airflow_H
                if norm_h < 0.3:
                    height_taper = norm_h / 0.3
                else:
                    height_taper = (1.0 - norm_h) / 0.7
                
                # Radial attenuation
                plume_radius = cell_size * 2.0
                radial_taper = np.exp(-0.5 * (r / plume_radius)**2)
                
                w = fire_intensity * 8.0 * height_taper * radial_taper  # Realistic gain
                w_values.append(w)
            else:
                w_values.append(0)
        ax3.plot(heights, w_values, label=f'r = {r}m', linewidth=2)
    
    ax3.set_xlabel('Height [m]')
    ax3.set_ylabel('Upward Velocity w [m/s]')
    ax3.set_title('CORRECT: Vertical Velocity Profile\n(Peaked, radial attenuation, realistic magnitude)', 
                  fontsize=12, weight='bold', color='green')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Correct radial velocity
    ax4 = axes[1, 1]
    for h in [5, 15, 25, 35, 45]:
        u_values = []
        for r in radii:
            if h < airflow_H and r > 1e-6:
                # Calculate w for this position
                norm_h = h / airflow_H
                if norm_h < 0.3:
                    height_taper = norm_h / 0.3
                else:
                    height_taper = (1.0 - norm_h) / 0.7
                plume_radius = cell_size * 2.0
                radial_taper = np.exp(-0.5 * (r / plume_radius)**2)
                w = fire_intensity * 8.0 * height_taper * radial_taper
                
                # Height-dependent direction
                if norm_h < 0.5:
                    # Inward (negative)
                    inward_strength = (0.5 - norm_h) / 0.5
                    u = -w * 0.3 * inward_strength
                else:
                    # Outward (positive)
                    outward_strength = (norm_h - 0.5) / 0.5
                    u = w * 0.3 * outward_strength
                
                u_values.append(u)
            else:
                u_values.append(0)
        ax4.plot(radii, u_values, label=f'h = {h}m', linewidth=2)
    
    ax4.set_xlabel('Radial Distance [m]')
    ax4.set_ylabel('Radial Velocity u [m/s]')
    ax4.set_title('CORRECT: Radial Velocity\n✅ Inward (negative) at low altitude, outward (positive) at high altitude', 
                  fontsize=12, weight='bold', color='green')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=1.5)
    ax4.text(10, -0.5, 'INWARD (toward fire)', fontsize=10, color='green', weight='bold')
    ax4.text(10, 0.5, 'OUTWARD (away from fire)', fontsize=10, color='blue', weight='bold')
    
    plt.tight_layout()
    plt.savefig('current_vs_correct_airflow.png', dpi=150, bbox_inches='tight')
    print("✅ Comparison saved: current_vs_correct_airflow.png")
    plt.show()

if __name__ == "__main__":
    visualize_fire_convection()
    plot_current_vs_correct()
