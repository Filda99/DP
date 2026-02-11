import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import MultiAgentMLP
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class QuadActor(nn.Module):
    def __init__(self, message_dim=8, self_state_size=16):
        super().__init__()
        # 1. CNN for Fire Map
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2), # 32x32 -> 15x15
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), # 15x15 -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64)
        )
        
        # 2. State & Memory - updated for expanded self_state size
        self.gru = nn.GRUCell(input_size=64 + self_state_size, hidden_size=128)
        self.self_state_size = self_state_size
        
        # 3. Heads - mnohem konzervativnější
        self.action_head = nn.Linear(128, 8) # 4 for Loc, 4 for Scale (Mean/Std)
        self.message_head = nn.Linear(128, message_dim)
        
        # KONZERVATIVNÍ inicializace pro bezpečné hovering
        with torch.no_grad():
            # Inicializuj všechny akce na malé hodnoty kolem nuly
            self.action_head.bias.fill_(0.0)  
            self.action_head.weight.fill_(0.0)
            
            # Nastavit jen malý pozitivní bias pro throttle mean (pro hover)
            self.action_head.bias[3] = 0.1   # throttle mean mírně pozitivní
            
            # Scale biasy nastavit na malé pozitivní hodnoty (menší variance)
            self.action_head.bias[4:] = -1.0  # všechny scale biasy malé

    def forward(self, local_map, self_state, hidden_state):
        # 1. Handle Scalar Batch (Data Collection) vs Batched (Training)
        # Expected Batched: (Batch, Agents, C, H, W) -> 5 dims
        # Observed Scalar:  (Agents, C, H, W)        -> 4 dims
        
        has_batch_dim = local_map.dim() == 5
        
        if not has_batch_dim:
            # Add fake batch dimension [1, ...]
            local_map = local_map.unsqueeze(0)
            self_state = self_state.unsqueeze(0)
            hidden_state = hidden_state.unsqueeze(0)

        # 2. Flatten Batch & Agents for Processing
        b, n, c, h, w = local_map.shape
        
        map_flat = local_map.reshape(b * n, c, h, w) # Use reshape instead of view for safety
        state_flat = self_state.reshape(b * n, -1)
        hidden_flat = hidden_state.reshape(b * n, -1)
        
        # 3. Run Network
        map_feats = self.cnn(map_flat)
        combined = torch.cat([map_feats, state_flat], dim=-1)
        
        # GRU expects (Batch, Input), output is (Batch, Hidden)
        new_hidden = self.gru(combined, hidden_flat)
        
        actions = self.action_head(new_hidden)
        message = self.message_head(new_hidden)
        
        # 4. Restore Dimensions: (Batch, Agents, Features)
        actions = actions.view(b, n, -1)
        message = message.view(b, n, -1)
        new_hidden = new_hidden.view(b, n, -1)
        
        # 5. Remove fake batch dimension if we added it
        if not has_batch_dim:
            actions = actions.squeeze(0)
            message = message.squeeze(0)
            new_hidden = new_hidden.squeeze(0)
        
        return actions, message, new_hidden

class FixedWingActor(nn.Module):
    def __init__(self, message_dim=8):
        super().__init__()
        # 1. Attention Layer (Fixed-Wing queries, Quads provide Keys/Values)
        self.attention = nn.MultiheadAttention(embed_dim=message_dim, num_heads=2, batch_first=True)
        
        # 2. Flight Control MLP
        self.mlp = nn.Sequential(
            nn.Linear(7 + message_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 8) # 4 Loc, 4 Scale
        )

    def forward(self, self_state, message_stream):
        # 1. Handle Scalar Batch
        # Expected Batched: (Batch, Agents, Feats) -> 3 dims
        # Observed Scalar:  (Agents, Feats)        -> 2 dims
        
        has_batch_dim = self_state.dim() == 3
        
        if not has_batch_dim:
            self_state = self_state.unsqueeze(0)
            message_stream = message_stream.unsqueeze(0)

        # 2. Prepare Query
        # self_state (7) -> needs to match message_dim (8) for Attention
        # We assume simplified padding for this demo
        padding = torch.zeros_like(self_state[..., :1])
        query = torch.cat([self_state, padding], dim=-1) # [B, N_fixed, 8]
        
        # 3. Attention
        # Query: Fixed Wings
        # Key/Val: Quads (message_stream)
        # Output: [B, N_fixed, 8]
        attn_output, _ = self.attention(query, message_stream, message_stream)
        
        # 4. Combine & Act
        combined = torch.cat([self_state, attn_output], dim=-1)
        params = self.mlp(combined)
        
        # 5. Remove fake batch dimension
        if not has_batch_dim:
            params = params.squeeze(0)
        
        return params