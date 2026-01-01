import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend bez okna
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- OPRAVENÝ STYL ---
# Vynutíme vestavěný patkový font (DejaVu Serif), aby to nehledalo Times New Roman
params = {
    'text.usetex': False,          # Vypnout LaTeX compiler
    'font.family': 'serif',        # Chtít patkové písmo
    'font.serif': ['DejaVu Serif'] # Konkrétně toto (je vždy dostupné)
}

try:
    import scienceplots
    plt.style.use(['science', 'ieee'])
    plt.rcParams.update(params) # Přepsat fonty scienceplots našimi
except ImportError:
    # Fallback styl
    plt.rcParams.update(params)
    plt.rcParams.update({
        'axes.grid': False,
        'figure.figsize': (8, 8)
    })

def generate_memory_map_figure():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Grid size
    nx, ny = 30, 30
    x = np.linspace(0, nx, nx)
    y = np.linspace(0, ny, ny)
    X, Y = np.meshgrid(x, y)
    
    # --- DATA GENERATION ---
    # 1. Fuel (Smooth terrain-like)
    Z_fuel = np.sin(X/5) * np.cos(Y/5) * 0.5 + 0.5
    
    # 2. Moisture (Patchy)
    # Z_moist = np.exp(-((X-10)**2 + (Y-10)**2)/20)
    a = 4
    b = 3
    sigma = 2.5  # šířka gradientu

    distance = np.abs(a * X - Y + b) / np.sqrt(a**2 + 1)
    Z_moist = np.exp(-(distance**2) / (2 * sigma**2))

    # 3. Intensity (Fire center)
    Z_intens = np.exp(-((X-20)**2 + (Y-20)**2)/15)
    
    # 4. Burning (Binary Mask - Thresholded Intensity)
    Z_burn = (Z_intens > 0.3).astype(float)
    # Make non-burning transparent later by plotting only points or masking

    # --- PLOTTING LAYERS ---
    # Offsets for stacking
    offsets = [0, 15, 30, 45]
    labels = [
        r'$\mathbf{F}$ - Fuel Layer (Float)', 
        r'$\mathbf{M}$ - Moisture Layer (Float)', 
        r'$\mathbf{I}$ - Intensity Layer (Float)', 
        r'$\mathbf{B}$ - Burning Mask (Bool)'
    ]
    cmaps = ['Greens', 'Blues', 'Oranges', 'Reds']
    layer_alpha = [1, 0.85, 0.75, 0.7]
    
    # Draw logic from bottom to top
    for i, (data, offset, label, cmap) in enumerate(zip([Z_fuel, Z_moist, Z_intens, Z_burn], offsets, labels, cmaps)):
        ax.text(
            0,
            0,
            offset + 2,
            label,
            fontsize=11,
            fontweight='bold',
            ha='left',
            va='bottom',
            zdir='y',
            color='black',
            zorder=300
        )
        
        # Normalize transparency
        if i == 3: # Burning Layer (Sparse)
            # Plot only active cells as voxels/scatter or masked surface
            mask = data > 0.1
            ax.contourf(X, Y, data, zdir='z', offset=offset, cmap=cmap, alpha=layer_alpha[i], levels=[0.1, 1.0])
            # Add border to the plane
            ax.plot([0, nx, nx, 0, 0], [0, 0, ny, ny, 0], [offset]*5, color='black', lw=1, alpha=0.3)
        else:
            # Continuous layers
            # Use contourf for "map" look projected on the plane
            cset = ax.contourf(X, Y, data, zdir='z', offset=offset, cmap=cmap, alpha=layer_alpha[i])
            # Add wireframe box outline for the layer
            ax.plot([0, nx, nx, 0, 0], [0, 0, ny, ny, 0], [offset]*5, color='black', lw=1, alpha=0.3)

    # --- UPDATED DRILL-DOWN LINE ---
    tx, ty = 20, 20
    
    # 1. The dashed vertical line
    ax.plot([tx, tx], [ty, ty], [0, 45], color='black', linestyle='--', linewidth=1.5, zorder=100)
    
    # 2. The intersection markers (Crosses 'x')
    # Use s=80 for size, linewidth=2.5 for thickness
    # ax.scatter([tx]*4, [ty]*4, offsets, marker='o', color='black', s=80, linewidth=2.5, zorder=101)
    POINT_EPS = 0.02

    ax.scatter(
        [tx]*4,
        [ty]*4,
        [o + POINT_EPS for o in offsets],
        s=120,
        marker='X',
        facecolor='black',
        edgecolor='white',
        linewidth=1.5,
        depthshade=False,   # KLÍČOVÉ
        zorder=300
    )
    
    # Annotation
    ax.text(tx, ty+2, 55, "Single Cell State $(x, y)$", ha='center', fontsize=10, style='italic', fontweight='bold')

    # --- AXIS LABELS ---
    ax.set_xlabel("Grid Width $W$")
    ax.set_ylabel("Grid Height $H$")
    ax.set_zlabel("Memory Layers")
    
    # Hide Z ticks because they are arbitrary offsets
    ax.set_zticks([])
    
    # Adjust View
    ax.view_init(elev=25, azim=-60)
    
    # Make pane transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    plt.tight_layout()
    filename = "concept_memory_layers.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Image saved: {filename}")

if __name__ == "__main__":
    generate_memory_map_figure()