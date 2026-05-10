import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

# =============================================================================
# HELPER MODULES -- Reusable attention building blocks
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Generic Multi-Head Attention block.

    When query == key == value (same source), it performs Self-Attention
    (used in ScoutActor to reason over neighbouring drones).
    When query comes from a different source than key/value, it performs
    Cross-Attention (used in CommanderActor to read scout messages).

    PyTorch's nn.MultiheadAttention uses the convention:
        key_padding_mask[b, i] = True  ->  position i is IGNORED for batch b
    This is the opposite of the "attend-to" mask used in some other libraries.
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        # batch_first=True means tensors are (Batch, Seq, Embed) rather than (Seq, Batch, Embed).
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Parameters
        ----------
        query            : (Batch, Q_len,  Embed_Dim) -- what the agent is "asking"
        key, value       : (Batch, KV_len, Embed_Dim) -- what neighbours / messages offer
        key_padding_mask : (Batch, KV_len) bool       -- True where a position must be ignored
                           (e.g. dead/absent agents padded with zeros)

        Returns
        -------
        attn_output : (Batch, Q_len, Embed_Dim)
        """
        # Edge case: no keys at all (e.g. agent has zero neighbours).
        # Return a zero tensor with the same shape as query so downstream
        # layers still receive a valid-shaped input.
        if key.size(1) == 0:
            return torch.zeros_like(query)

        if key_padding_mask is not None:
            # Detect rows in the batch where EVERY key position is masked (True).
            # If softmax receives a row of all -inf (from masking), it produces NaN
            # because exp(-inf) / sum(exp(-inf)) = 0/0.
            all_masked = key_padding_mask.all(dim=1)  # (Batch,) bool

            if all_masked.any():
                # Safety fix: temporarily unmask position 0 for all-dead rows so
                # that softmax can compute a finite (but arbitrary) distribution.
                # We will zero out the output for those rows afterwards, making
                # the temporary unmasking completely harmless.
                safe_mask = key_padding_mask.clone()
                safe_mask[all_masked, 0] = False  # unblock one slot per bad row

                attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=safe_mask)

                # Overwrite the result for all-dead rows with zeros so the
                # GRU / downstream layers receive no spurious information.
                attn_output[all_masked] = 0.0
                return attn_output

        # Standard path -- at least one unmasked key per row.
        attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
        return attn_output


# =============================================================================
# 1. SCOUT ACTOR  (quadcopter drone)
# =============================================================================

class ScoutActor(nn.Module):
    """
    Actor network for a scout drone.  Follows the MAPPO actor convention:
    it maps purely LOCAL observations to a stochastic action distribution
    and an outbound communication message.

    Architecture overview (three parallel perception branches + GRU memory):

        local_map  --(CNN)-->  vis_feat  (128-d)  -\
        self_state --(MLP)-->  self_feat  (64-d)  --+--[cat]--[LayerNorm]--[GRU]--> dist, message
        neighbors  --(Attn)-> neigh_ctx  (64-d)  -/

    Inputs
    ------
    local_map       : (Batch [, Seq], 1, 32, 32)         -- single-channel fire-intensity grid
    self_state      : (Batch [, Seq], self_state_dim)     -- velocity, boundary distances, ...
    neighbor_states : (Batch [, Seq], N_neighbours, 3)   -- relative [dx, dy, dz] per neighbour
    neighbor_mask   : (Batch [, Seq], N_neighbours) bool -- True for absent/dead neighbours (padding)
    hidden_state    : (1, Batch, hidden_dim)             -- GRU memory carried across time steps

    Outputs
    -------
    dist        : Normal distribution over actions (Roll, Pitch, Yaw, Throttle)
    message     : (Batch*Seq, msg_dim) -- compact vector broadcast to the commander
    new_hidden  : updated GRU hidden state, same shape as hidden_state
    """
    def __init__(self, self_state_dim, action_dim=4, msg_dim=5, hidden_dim=128):
        super().__init__()

        # ------------------------------------------------------------------
        # Branch 1 -- Visual perception: CNN over the 32x32 fire map
        # ------------------------------------------------------------------
        # The local fire map is treated as a single-channel (greyscale) image.
        # Two strided Conv2d layers extract spatial features (edges, hotspots,
        # fire fronts), then a Linear layer compresses them to a 128-d vector.
        #
        # Output spatial size after each Conv2d  (formula: floor((W-K)/S + 1)):
        #   32 -> floor((32-4)/2 + 1) = 15   (after Conv1)
        #   15 -> floor((15-4)/2 + 1) = 6    (after Conv2)
        # Flattened: 32 filters x 6 x 6 = 1152 values -> compressed to 128.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2),   # (B,  1, 32, 32) -> (B, 16, 15, 15)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # (B, 16, 15, 15) -> (B, 32,  6,  6)
            nn.ReLU(),
            nn.Flatten(),                                # (B, 32, 6, 6) -> (B, 1152)
            nn.Linear(1152, hidden_dim),                 # compress to 128-d visual feature
            nn.ReLU()
        )

        # ------------------------------------------------------------------
        # Branch 2 -- Situational awareness: Self-Attention over neighbours
        # ------------------------------------------------------------------
        # Each neighbour is described by its relative offset [dx, dy, dz].
        # neighbor_embed lifts that 3-d vector into a 64-d embedding space
        # so the attention mechanism has richer representations to work with.
        #
        # MultiHeadAttention is used here as *Self-Attention*:
        #   Query  = own embedded state ("who matters to me right now?")
        #   Key    = neighbour embeddings ("here is what I can offer")
        #   Value  = same neighbour embeddings (the actual information payload)
        #
        # Advantages over a simple mean/max aggregation:
        #   * Dynamically weights nearby / relevant drones higher.
        #   * Handles a variable number of neighbours (1...N) and always
        #     produces a fixed-size 64-d context vector.
        #   * Gracefully ignores absent/dead drones via neighbor_mask.
        self.neighbor_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )
        self.neighbor_attention = MultiHeadAttention(embed_dim=64, num_heads=2)

        # ------------------------------------------------------------------
        # Branch 3 -- Proprioception: MLP over own state vector
        # ------------------------------------------------------------------
        # Encodes ego-centric information (velocity components, distances to
        # arena boundaries, altitude, etc.) into a 64-d feature vector.
        self.self_embed = nn.Sequential(
            nn.Linear(self_state_dim, 64),
            nn.ReLU()
        )

        # ------------------------------------------------------------------
        # Fusion layer -- Layer Normalisation before the GRU
        # ------------------------------------------------------------------
        # The three branches are concatenated: 128 (CNN) + 64 (self) + 64 (neighbours) = 256.
        # LayerNorm normalises across the feature dimension independently for each
        # sample, which stabilises the scale of the fused representation and
        # prevents gradient explosions at the GRU input gate.
        self.layer_norm = nn.LayerNorm(128 + 64 + 64)

        # ------------------------------------------------------------------
        # Temporal memory -- GRU
        # ------------------------------------------------------------------
        # The GRU receives the fused 256-d perception vector at every time step
        # and maintains a hidden state of size hidden_dim (128).
        # This gives the drone episodic memory: it can recall where fire was
        # spotted several steps ago and plan patrol routes without revisiting
        # areas it already covered.
        #
        # GRU over LSTM rationale:
        #   * Fewer parameters (no separate cell state) -> trains faster.
        #   * Empirically performs on par with LSTM for tasks of this scale.
        # batch_first=True keeps tensor shapes consistent with the rest of the network.
        self.gru = nn.GRU(input_size=128 + 64 + 64, hidden_size=hidden_dim, batch_first=True)

        # ------------------------------------------------------------------
        # Output head 1 -- Action distribution (movement)
        # ------------------------------------------------------------------
        # Maps the GRU output to action means.  A separate learned log-std
        # parameter (not state-dependent) controls exploration noise globally.
        # Using nn.Parameter means log_std is optimised directly by the PPO
        # objective alongside all other network weights.
        self.action_mean    = nn.Linear(hidden_dim, action_dim)
        self.action_logstd  = nn.Parameter(torch.full((1, action_dim), -0.5))  # std ≈ 0.135 at init

        # ------------------------------------------------------------------
        # Output head 2 -- Outbound message (communication)
        # ------------------------------------------------------------------
        # Zpráva má dvě části:
        #   [0] norm. pozice drona X   (self_state[0]) — přímá obs hodnota
        #   [1] norm. pozice drona Y   (self_state[1]) — přímá obs hodnota
        #   [2] intenzita ohně         (self_state[14]) — přímá obs hodnota
        #   [3] naučená zpráva 1       (z msg_head)     — volná latentní dim
        #   [4] naučená zpráva 2       (z msg_head)     — volná latentní dim
        #
        # Commander tak VŽDY ví kde scout je (dim 0-1) a co vidí (dim 2),
        # nemusí to od nuly odpozorovat z gradientů. Dimenze 3-4 zůstávají
        # volné pro cokoli dalšího co se scout naučí sdělovat.
        # self.msg_head = nn.Sequential(
        #     nn.Linear(hidden_dim, msg_dim - 3),  # pouze 2 naučené dimy
        #     nn.Tanh()
        # ) REMOVED: fixed explicit messages now

    def forward(self, local_map, self_state, neighbor_states, neighbor_mask, hidden_state):
        # ------------------------------------------------------------------
        # Step 1: Detect operating mode (training vs. demo / inference)
        # ------------------------------------------------------------------
        # During training the rollout buffer stores full episodes, so tensors
        # carry an extra sequence dimension:
        #   local_map : (Batch, Seq, 1, 32, 32)
        # During demo / single env.step() they arrive without that dimension:
        #   local_map : (Batch, 1, 32, 32)  or  (1, 32, 32)
        is_sequential = (local_map.dim() >= 4)  # True when Seq dimension is present
        batch_size = local_map.size(0)
        seq_len    = local_map.size(1) if is_sequential else 1

        # ------------------------------------------------------------------
        # Step 2: Flatten batch x sequence into a single batch axis
        # ------------------------------------------------------------------
        # CNN and Linear layers do not understand sequence dimensions, so we
        # merge (Batch, Seq) -> (Batch*Seq) to process all time steps in parallel.
        # Example for training: (15 envs, 200 steps) -> 3000 independent samples.
        if is_sequential:
            local_map       = local_map.reshape(-1, 1, 32, 32)
            self_state      = self_state.reshape(-1, self_state.size(-1))
            neighbor_states = neighbor_states.reshape(batch_size * seq_len, -1, neighbor_states.size(-1))
            neighbor_mask   = neighbor_mask.reshape(batch_size * seq_len, -1)
        else:
            # Demo / single-step inference -- ensure correct tensor rank.
            if local_map.dim() == 2:
                local_map  = local_map.reshape(-1, 1, 32, 32)
            if self_state.dim() == 1:
                self_state = self_state.unsqueeze(0)

        # ------------------------------------------------------------------
        # Step 3: Run all three perception branches
        # ------------------------------------------------------------------
        vis_feat  = self.cnn(local_map)          # (B*S, 128) -- visual features from fire map
        self_feat = self.self_embed(self_state)  # (B*S,  64) -- encoded own state

        # Self-Attention over neighbours:
        #   embed each neighbour -> (B*S, N, 64)
        #   query with own state -> (B*S, 1, 64) ("what is relevant to me?")
        #   output context       -> (B*S, 1, 64) -> squeeze to (B*S, 64)
        neigh_embed   = self.neighbor_embed(neighbor_states)             # (B*S, N, 64)
        query         = self_feat.unsqueeze(1)                           # (B*S, 1, 64)
        neigh_context = self.neighbor_attention(
            query, neigh_embed, neigh_embed, key_padding_mask=neighbor_mask
        )                                                                # (B*S, 1, 64)
        neigh_context = neigh_context.squeeze(1)                         # (B*S,   64)

        # ------------------------------------------------------------------
        # Step 4: Fuse all branches and normalise
        # ------------------------------------------------------------------
        combined = torch.cat([vis_feat, self_feat, neigh_context], dim=1)  # (B*S, 256)
        combined = self.layer_norm(combined)

        # ------------------------------------------------------------------
        # Step 5: Re-introduce the sequence dimension for the GRU
        # ------------------------------------------------------------------
        # GRU expects (Batch, Seq, Features) -- restore that shape here.
        if is_sequential:
            combined = combined.view(batch_size, seq_len, -1)  # (B, S, 256)
        else:
            combined = combined.unsqueeze(1)                   # (B, 1, 256)

        gru_out, new_hidden = self.gru(combined, hidden_state)
        # gru_out    : (B, S, hidden_dim=128)
        # new_hidden : (1, B, hidden_dim) -- passed back to the next env step

        # Flatten again for output Linear layers.
        features = gru_out.reshape(-1, 128)  # (B*S, 128)

        # ------------------------------------------------------------------
        # Step 6: Compute outputs
        # ------------------------------------------------------------------
        # Action distribution:
        #   mean  = tanh(linear(features)) -- bounded to (-1, 1), matching the
        #           normalised action space of the PyBullet drone controller.
        #   std   = exp(log_std)           -- always positive; shared across
        #           the batch (state-independent exploration noise).
        #   Normal(mean, std) supports .rsample() for the reparameterisation
        #   trick needed in PPO, and .log_prob() for the importance-sampling ratio.
        action_mean = torch.tanh(self.action_mean(features))
        log_std     = self.action_logstd.clamp(-4.0, 0.0)   # std in [e^-4≈0.018, e^0≈1.0]
        dist        = Normal(action_mean, torch.exp(log_std))

        # Outbound message for the commander:
        #   dims 0-2: strukturovaná část — přímo z observace (vždy interpretovatelná)
        #   dims 3-4: naučená část — latentní kontext
        # explicit_msg = self_state[:, [0, 1, 14]]           # (B*S, 3) — norm_pos_x, norm_pos_y, intenzita
        # learned_msg  = self.msg_head(features)              # (B*S, 2)
        # message = torch.cat([explicit_msg, learned_msg], dim=1)  # (B*S, 5)
        
        # Zpráva obsahuje: [norm_pos_x, norm_pos_y, intensity, rel_x, rel_y]
        # Indexy v self_state (16-dim): 0=x, 1=y, 15=intensity, 13=rel_x, 14=rel_y
        message = self_state[:, [0, 1, 15, 13, 14]]

        return dist, message, new_hidden


# =============================================================================
# 2. COMMANDER ACTOR  (fixed-wing water-bomber — waypoint strategy + messages)
# =============================================================================

class CommanderActor(nn.Module):
    """
    Actor for the commander fixed-wing aircraft.

    Receives scout messages via cross-attention and outputs strategic
    waypoints (dx, dy, target_alt, water_trigger).  The training worker
    converts waypoints to heading commands for the flight controller.

    Architecture:

        self_state (17) → encoder (17→64→64) → enc_out (64)
                                                    ↓
        messages (N×5) → msg_embed(5→64) → cross_attn(q=enc_out) → ctx (64)
                                                    ↓
                                   cat([enc_out, ctx]) (128) → GRU(128→64) → action (4)
    """
    def __init__(self, self_state_dim=17, msg_input_dim=5, action_dim=4,
                 hidden_dim=64, num_attn_heads=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- Flight core (encoder + action head same as SimpleFWActor) ---
        self.encoder = nn.Sequential(
            nn.Linear(self_state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim),
            nn.Tanh(),
        )
        # GRU input is wider: enc_out (64) + attention ctx (64) = 128
        self.gru = nn.GRU(input_size=hidden_dim * 2, hidden_size=hidden_dim,
                          batch_first=True)

        # --- Message processing (NEW in Phase 2) ---
        self.msg_embed = nn.Sequential(
            nn.Linear(msg_input_dim, hidden_dim),
            nn.ReLU()
        )
        self.cross_attention = MultiHeadAttention(embed_dim=hidden_dim,
                                                  num_heads=num_attn_heads)

        # --- FW neighbor awareness (other fixed-wing positions) ---
        self.fw_neighbor_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU()
        )
        self.fw_neighbor_attention = MultiHeadAttention(
            embed_dim=hidden_dim, num_heads=num_attn_heads)

        # Projection: cat(enc_out, scout_ctx, fw_ctx) = 3*hidden → 2*hidden
        # keeps GRU input_size unchanged → old weights load perfectly.
        self.pre_gru = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        # Identity-init: first 2*H dims pass through, fw_ctx (last H) ignored
        with torch.no_grad():
            self.pre_gru.weight.zero_()
            self.pre_gru.weight[:hidden_dim * 2, :hidden_dim * 2].copy_(
                torch.eye(hidden_dim * 2))
            self.pre_gru.bias.zero_()

        # --- Action head: 64→4 ---
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))

        # Pomocná hlava pro predikci polohy ohně
        self.aux_head = nn.Linear(hidden_dim, 2)

    def forward(self, self_state, incoming_messages=None, message_mask=None,
                hidden_state=None, fw_neighbor_states=None, fw_neighbor_mask=None):
        is_sequential = (self_state.dim() == 3)
        batch_size = self_state.size(0)
        seq_len = self_state.size(1) if is_sequential else 1

        # --- Encode own state ---
        if is_sequential:
            flat_state = self_state.reshape(-1, self_state.size(-1))
        else:
            flat_state = self_state if self_state.dim() == 2 else self_state.unsqueeze(0)

        enc_out = self.encoder(flat_state)  # (B*S, hidden_dim)

        # --- Cross-attention over scout messages ---
        if incoming_messages is not None:
            if is_sequential:
                bs = batch_size * seq_len
                msgs = incoming_messages.reshape(bs, incoming_messages.size(-2),
                                                  incoming_messages.size(-1))
                mask = message_mask.reshape(bs, message_mask.size(-1))
            else:
                msgs = incoming_messages
                mask = message_mask

            msg_feat = self.msg_embed(msgs)               # (B*S, N, hidden_dim)
            query = enc_out.unsqueeze(1)                   # (B*S, 1, hidden_dim)
            ctx = self.cross_attention(
                query, msg_feat, msg_feat, key_padding_mask=mask
            ).squeeze(1)                                   # (B*S, hidden_dim)
        else:
            ctx = torch.zeros_like(enc_out)

        # --- FW neighbor attention ---
        if fw_neighbor_states is not None:
            if is_sequential:
                bs = batch_size * seq_len
                fw_n = fw_neighbor_states.reshape(
                    bs, fw_neighbor_states.size(-2), fw_neighbor_states.size(-1))
                fw_m = fw_neighbor_mask.reshape(bs, fw_neighbor_mask.size(-1))
            else:
                fw_n = fw_neighbor_states
                fw_m = fw_neighbor_mask
            fw_feat = self.fw_neighbor_embed(fw_n)         # (B*S, N_fw, H)
            fw_query = enc_out.unsqueeze(1)                # (B*S, 1, H)
            fw_ctx = self.fw_neighbor_attention(
                fw_query, fw_feat, fw_feat, key_padding_mask=fw_m
            ).squeeze(1)                                   # (B*S, H)
        else:
            fw_ctx = torch.zeros_like(enc_out)

        # --- Fuse and feed into GRU ---
        fused = self.pre_gru(
            torch.cat([enc_out, ctx, fw_ctx], dim=1))      # (B*S, hidden_dim*2)

        if is_sequential:
            fused = fused.view(batch_size, seq_len, -1)
        else:
            fused = fused.unsqueeze(1)

        if hidden_state is None:
            hidden_state = torch.zeros(1, batch_size, self.hidden_dim,
                                       device=fused.device)

        gru_out, new_hidden = self.gru(fused, hidden_state)
        features = gru_out.reshape(-1, self.hidden_dim)    # (B*S, hidden_dim)

        # --- Action output ---
        action_mean = torch.tanh(self.action_mean(features))
        log_std = self.action_logstd.clamp(-3.0, 0.0)
        dist = Normal(action_mean, torch.exp(log_std))

        # Predikce polohy ohně pro pomocnou odměnu
        aux_pred = self.aux_head(features) # (B*S, 2)

        return dist, aux_pred, new_hidden


# =============================================================================
# 3. PRIVILEGED CRITIC  (CTDE — sees global state during training)
# =============================================================================

class PrivilegedCritic(nn.Module):
    """MLP+GRU value network that receives a privileged global-state vector
    during training.  Used for both scout and commander — one instance each,
    with different input dims and hidden dims.

    Global state = agent's own obs (flattened) + privileged extras:
        [fire_x_norm, fire_y_norm, fire_intensity,
         other_agent_x, other_agent_y, other_agent_z]

    This is the standard MAPPO CTDE approach — critics are centralised
    (see everything), actors remain decentralised (see only own obs).
    """

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim,
                          batch_first=True)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, global_state, hidden_state=None):
        """
        global_state : (B, S, input_dim) or (B, input_dim)
        hidden_state : (1, B, hidden_dim) or None
        Returns: value (B*S,), new_hidden
        """
        is_sequential = (global_state.dim() == 3)
        batch_size = global_state.size(0)
        seq_len = global_state.size(1) if is_sequential else 1

        x = global_state.reshape(-1, global_state.size(-1))
        x = self.encoder(x)

        x = x.view(batch_size, seq_len, -1)

        if hidden_state is None:
            hidden_state = torch.zeros(1, batch_size, self.hidden_dim,
                                       device=x.device)

        gru_out, new_hidden = self.gru(x, hidden_state)
        value = self.value_head(gru_out.reshape(-1, self.hidden_dim))
        return value.squeeze(-1), new_hidden
