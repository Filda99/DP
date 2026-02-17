import matplotlib
# TOTO JE TA KLÍČOVÁ OPRAVA - musí být před importem pyplot
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_smart_arch():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # === 1. QUADCOPTER (SCOUT) - SMART VERSION ===
    ax1.set_title("1. QUADCOPTER (SCOUT) - The Sensor", fontsize=18, pad=20, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 14)
    ax1.axis('off')

    # Styly
    box_style = dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='#1565C0', lw=2)     # Modrá (Vstupy)
    enc_style = dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0', edgecolor='#E65100', lw=2)     # Oranžová (Encoders)
    attn_style = dict(boxstyle='round,pad=0.8', facecolor='#FFF9C4', edgecolor='#FBC02D', lw=2)    # Žlutá (Attention)
    mem_style = dict(boxstyle='round,pad=0.5', facecolor='#E1BEE7', edgecolor='#8E24AA', lw=2)     # Fialová (Paměť)
    out_style = dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', edgecolor='#C62828', lw=2)     # Červená (Výstupy)

    # --- VSTUPY ---
    ax1.text(2, 12.5, "INPUT: Local Map\n(10x10 Grid)", ha='center', bbox=box_style)
    ax1.text(5, 12.5, "INPUT: Self State\n(Vel, Pos, Angles)", ha='center', bbox=box_style)
    ax1.text(8, 12.5, "INPUT: Neighbors\nPositions [N x 3]", ha='center', bbox=box_style)

    # --- ENCODERY ---
    ax1.text(2, 10.5, "CNN Encoder\n(Visual Features)", ha='center', bbox=enc_style)
    ax1.text(5, 10.5, "MLP Encoder\n(Self Features)", ha='center', bbox=enc_style)
    
    # --- ATTENTION PRO Drony ---
    ax1.text(8, 9.5, "SELF-ATTENTION\n(Cooperation Layer)", ha='center', bbox=attn_style)
    ax1.text(8, 8.5, "Aggregates variable\nneighbor count", fontsize=9, ha='center')

    # --- FUSION & MEMORY ---
    ax1.text(5, 7.5, "Feature Concatenation\n(Vision + Self + Neighbors)", ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    ax1.text(5, 5.5, "GRU / LSTM\n(Temporal Memory)", ha='center', bbox=mem_style)

    # --- HEADS ---
    ax1.text(3, 3.5, "Actor Head\n(Flight Policy)", ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    ax1.text(7, 3.5, "Comm Head\n(Message Gen)", ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    # --- OUTPUTS ---
    ax1.text(3, 1.5, "OUTPUT: Action\n[Roll, Pitch, Yaw, Throttle]", ha='center', bbox=out_style)
    ax1.text(7, 1.5, "OUTPUT: Message (z_i)\n[Latent Vector]", ha='center', bbox=out_style)

    # Šipky (Quad)
    # Vstupy -> Encodery
    ax1.annotate("", xy=(2, 11.1), xytext=(2, 12.0), arrowprops=dict(arrowstyle="->"))
    ax1.annotate("", xy=(5, 11.1), xytext=(5, 12.0), arrowprops=dict(arrowstyle="->"))
    ax1.annotate("", xy=(8, 10.2), xytext=(8, 12.0), arrowprops=dict(arrowstyle="->")) # Neighbors to Attention

    # Encodery -> Concat
    ax1.annotate("", xy=(4.5, 7.8), xytext=(2, 9.9), arrowprops=dict(arrowstyle="->")) # CNN
    ax1.annotate("", xy=(5, 7.8), xytext=(5, 9.9), arrowprops=dict(arrowstyle="->"))   # Self MLP
    ax1.annotate("", xy=(5.5, 7.8), xytext=(8, 8.8), arrowprops=dict(arrowstyle="->")) # Attention Out

    # Concat -> GRU -> Heads
    ax1.annotate("", xy=(5, 6.1), xytext=(5, 7.2), arrowprops=dict(arrowstyle="->"))
    ax1.annotate("", xy=(3, 4.0), xytext=(4.8, 5.0), arrowprops=dict(arrowstyle="->"))
    ax1.annotate("", xy=(7, 4.0), xytext=(5.2, 5.0), arrowprops=dict(arrowstyle="->"))

    # Heads -> Outputs
    ax1.annotate("", xy=(3, 2.1), xytext=(3, 3.0), arrowprops=dict(arrowstyle="->"))
    ax1.annotate("", xy=(7, 2.1), xytext=(7, 3.0), arrowprops=dict(arrowstyle="->"))
    
    # Broadcast Line
    ax1.annotate("", xy=(13, 2.1), xytext=(8.5, 1.5), arrowprops=dict(arrowstyle="->", color='red', lw=2, linestyle="dashed"))
    ax1.text(10.5, 1.8, "BROADCAST MSG", color='red', fontweight='bold', ha='center')


    # === 2. FIXED WING (COMMANDER) ===
    ax2.set_title("2. FIXED WING (COMMANDER) - The Actor", fontsize=18, pad=20, fontweight='bold')
    ax2.set_xlim(10, 20) # Posunuto doprava
    ax2.set_ylim(0, 14)
    ax2.axis('off')

    # Vstupy
    ax2.text(12, 12.5, "INPUT: Messages [M1...MN]\n(From all Scouts)", ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFCCBC', edgecolor='#D84315', lw=2))
    ax2.text(18, 12.5, "INPUT: Self State\n(Water, Vel, Pos)", ha='center', bbox=box_style)

    # Attention Block
    ax2.text(15, 10.0, "CROSS-ATTENTION\n(Message Filter)", ha='center', bbox=attn_style)
    ax2.text(13, 10.5, "Keys/Values:\nScout Msgs", fontsize=8, ha='center', color='gray')
    ax2.text(17, 10.5, "Query:\nMy State", fontsize=8, ha='center', color='gray')

    # Fusion
    ax2.text(15, 7.5, "Context Fusion\n(Weighted Map + State)", ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    # Decision
    ax2.text(15, 5.5, "Commander MLP\n(Strategic Logic)", ha='center', bbox=mem_style)

    # Heads
    ax2.text(13, 3.5, "Flight Control", ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    ax2.text(17, 3.5, "Water Control", ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    # Outputs
    ax2.text(13, 1.5, "OUTPUT: Action\n[Roll, Pitch, Throttle]", ha='center', bbox=out_style)
    ax2.text(17, 1.5, "OUTPUT: Drop Water\n[Trigger 0/1]", ha='center', bbox=out_style)

    # Šipky (Fixed)
    ax2.annotate("", xy=(14.5, 10.7), xytext=(12, 11.9), arrowprops=dict(arrowstyle="->", color='red')) # Msg -> Attn
    ax2.annotate("", xy=(15.5, 10.7), xytext=(18, 11.9), arrowprops=dict(arrowstyle="->")) # State -> Attn Query

    ax2.annotate("", xy=(15, 8.0), xytext=(15, 9.3), arrowprops=dict(arrowstyle="->")) # Attn -> Fusion
    ax2.annotate("", xy=(15.5, 8.0), xytext=(18, 11.9), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", linestyle="dotted")) # State -> Fusion (Skip connection)

    ax2.annotate("", xy=(15, 6.1), xytext=(15, 7.2), arrowprops=dict(arrowstyle="->")) # Fusion -> MLP
    ax2.annotate("", xy=(13, 4.0), xytext=(14.8, 5.0), arrowprops=dict(arrowstyle="->"))
    ax2.annotate("", xy=(17, 4.0), xytext=(15.2, 5.0), arrowprops=dict(arrowstyle="->"))

    ax2.annotate("", xy=(13, 2.1), xytext=(13, 3.0), arrowprops=dict(arrowstyle="->"))
    ax2.annotate("", xy=(17, 2.1), xytext=(17, 3.0), arrowprops=dict(arrowstyle="->"))

    plt.tight_layout()
    plt.savefig("smart_architecture.png")
    print("✅ Diagram uložen jako 'smart_architecture.png'")

if __name__ == "__main__":
    try:
        draw_smart_arch()
    except Exception as e:
        print(f"❌ Chyba: {e}")