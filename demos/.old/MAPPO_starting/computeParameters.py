import torch
import torch.nn as nn

class QuadNetProof(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. CNN (Encoder)
        # Input: 1 kanál (mapa), Output: 16 kanálů, Kernel: 3x3
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        
        # 2. Fusion (Spojení)
        # Mapa 10x10 -> po conv stále 10x10 (díky paddingu) -> 16 kanálů * 10 * 10 = 1600 features
        # Plus 6 čísel GPS stavu + 8 čísel Lidaru = 1614 vstupů
        self.fusion = nn.Linear(16 * 10 * 10 + 6 + 8, 128)
        
        # 3. GRU (Paměť)
        # Input 128, Hidden 128
        self.gru = nn.GRU(input_size=128, hidden_size=128)
        
        # 4. Heads (Výstupy)
        self.action_head = nn.Linear(128, 3) # Pohyb
        self.msg_head = nn.Linear(128, 5)    # Komunikace

    def forward(self, x):
        pass # Jen pro demonstraci parametrů

model = QuadNetProof()
total_params = sum(p.numel() for p in model.parameters())
print(f"Celkový počet trénovatelných parametrů: {total_params}")