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
        self.action_logstd  = nn.Parameter(torch.full((1, action_dim), -2.0))  # std ≈ 0.135 at init

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
        self.msg_head = nn.Sequential(
            nn.Linear(hidden_dim, msg_dim - 3),  # pouze 2 naučené dimy
            nn.Tanh()
        )

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
        explicit_msg = self_state[:, [0, 1, 14]]           # (B*S, 3) — norm_pos_x, norm_pos_y, intenzita
        learned_msg  = self.msg_head(features)              # (B*S, 2)
        message = torch.cat([explicit_msg, learned_msg], dim=1)  # (B*S, 5)

        return dist, message, new_hidden


# =============================================================================
# 2. COMMANDER ACTOR  (fixed-wing water-bomber aircraft)
# =============================================================================

class CommanderActor(nn.Module):
    """
    Actor network for the commander aircraft.

    The commander has no direct visual sensor (no local map); instead it
    receives summarised reports (messages) from all scout drones and uses
    Cross-Attention to selectively focus on the most relevant ones.

    Architecture overview:

        self_state  --(MLP)---> self_feat  (64-d) --\
        messages    --(MLP)---> msg_feat   (N, 64)   Cross-Attn -> ctx (64-d) --+-- [cat] --[gate]--[LayerNorm]--[GRU]--> dist
                                                                                /
                                                         comm_alpha gate ------/

    Inputs
    ------
    self_state        : (Batch [, Seq], self_state_dim)           -- velocity, altitude, water level
    incoming_messages : (Batch [, Seq], N_scouts, msg_input_dim) -- msg vectors from drones
    message_mask      : (Batch [, Seq], N_scouts) bool           -- True for silent/absent scouts
    hidden_state      : (1, Batch, hidden_dim)                   -- GRU memory from previous step

    Outputs
    -------
    dist        : Normal distribution over actions (Roll, Pitch, Yaw, Throttle)
    None        : placeholder (commander sends no messages in this architecture)
    new_hidden  : updated GRU hidden state
    """
    def __init__(self, self_state_dim=15, msg_input_dim=5, action_dim=4, hidden_dim=128):
        super().__init__()

        # ------------------------------------------------------------------
        # Branch 1 -- Own-state encoder
        # ------------------------------------------------------------------
        # Encodes the commander's proprioceptive state (position, velocity,
        # water remaining, ...) into a 64-d latent vector.
        self.self_embed = nn.Sequential(
            nn.Linear(self_state_dim, 64),
            nn.ReLU()
        )

        # ------------------------------------------------------------------
        # Branch 2 -- Scout message encoder
        # ------------------------------------------------------------------
        # Each incoming scout message is a msg_input_dim-dimensional vector.
        # This MLP lifts every message into the same 64-d embedding space
        # as the commander's own state, so attention dot-products are
        # dimensionally consistent.
        self.msg_embed = nn.Sequential(
            nn.Linear(msg_input_dim, 64),
            nn.ReLU()
        )

        # ------------------------------------------------------------------
        # Branch 3 -- Cross-Attention (commander listens to scout messages)
        # ------------------------------------------------------------------
        # Unlike ScoutActor (Self-Attention: agents compare each other),
        # the commander uses Cross-Attention:
        #   Query  = commander's own embedded state
        #            ("I am at position X with Y litres -- who is useful to me?")
        #   Key    = scout message embeddings  ("here is what I know")
        #   Value  = same scout message embeddings (the actual payload)
        #
        # The attention weights reflect relevance: a scout reporting a large
        # fire close to the commander gets high weight; a distant scout with
        # nothing to report gets near-zero weight.
        # The weighted sum (context vector) distils the entire swarm's
        # situational awareness into one 64-d vector tailored to the commander.
        self.attention = MultiHeadAttention(embed_dim=64, num_heads=4)

        # ------------------------------------------------------------------
        # Fusion -- Layer Normalisation
        # ------------------------------------------------------------------
        # 64 (self) + 64 (attention context) = 128
        self.layer_norm = nn.LayerNorm(64 + 64)

        # ------------------------------------------------------------------
        # Temporal memory -- GRU
        # ------------------------------------------------------------------
        # Same role as in ScoutActor: lets the commander remember past
        # observations (e.g. which sectors were already covered) and plan
        # multi-step attack runs across consecutive time steps.
        self.gru = nn.GRU(input_size=64 + 64, hidden_size=hidden_dim, batch_first=True)

        # ------------------------------------------------------------------
        # Output head -- Action distribution
        # ------------------------------------------------------------------
        self.action_mean   = nn.Linear(hidden_dim, action_dim)
        self.action_logstd = nn.Parameter(torch.full((1, action_dim), -2.0))  # std ≈ 0.135 at init

        # ------------------------------------------------------------------
        # Learnable communication gate (comm_alpha)
        # ------------------------------------------------------------------
        # comm_alpha is a scalar nn.Parameter initialised to 0.3.
        # It is passed through sigmoid() in forward() so the effective gate
        # value stays in (0, 1) regardless of the raw parameter value.
        # This acts as a soft switch controlling how much the attention context
        # influences the fused representation.  The network can learn to
        # suppress scout input (alpha -> 0) if messages are noisy or
        # uninformative, or to rely on it heavily (alpha -> 1) when useful.
        self.comm_alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, self_state, incoming_messages, message_mask, hidden_state):
        # ------------------------------------------------------------------
        # A) Detect operating mode
        # ------------------------------------------------------------------
        is_sequential = (self_state.dim() == 3)  # True when Seq dimension present
        batch_size = self_state.size(0)
        seq_len    = self_state.size(1) if is_sequential else 1

        # ------------------------------------------------------------------
        # B) Flatten batch x sequence for Linear / Attention layers
        # ------------------------------------------------------------------
        if is_sequential:
            target_batch      = batch_size * seq_len
            self_state        = self_state.reshape(target_batch, self_state.size(-1))
            incoming_messages = incoming_messages.reshape(target_batch, incoming_messages.size(-2), incoming_messages.size(-1))
            message_mask      = message_mask.reshape(target_batch, message_mask.size(-1))

        # ------------------------------------------------------------------
        # C) Run both perception branches
        # ------------------------------------------------------------------
        self_feat = self.self_embed(self_state)        # (B*S, 64)
        msg_feat  = self.msg_embed(incoming_messages)  # (B*S, N_scouts, 64)

        # Cross-Attention: commander queries the scout message embeddings.
        # unsqueeze(1) because the commander asks a single question per step.
        query = self_feat.unsqueeze(1)                 # (B*S, 1, 64)
        context_vector = self.attention(
            query, msg_feat, msg_feat, key_padding_mask=message_mask
        )                                              # (B*S, 1, 64)
        context_vector = context_vector.squeeze(1)     # (B*S, 64)

        # ------------------------------------------------------------------
        # D) Gated fusion and temporal memory
        # ------------------------------------------------------------------
        # sigmoid(comm_alpha) keeps the gate in (0, 1) and ensures smooth
        # gradient flow regardless of the raw parameter value.
        # Scaling context_vector by this gate lets the network learn whether
        # listening to scouts is beneficial for this particular task.
        # gate     = torch.sigmoid(self.comm_alpha)
        gate = 1.0
        combined = torch.cat([self_feat, gate * context_vector], dim=1)  # (B*S, 128)
        combined = self.layer_norm(combined)

        # Re-introduce sequence dimension for the GRU.
        if is_sequential:
            combined = combined.view(batch_size, seq_len, -1)  # (B, S, 128)
        else:
            combined = combined.unsqueeze(1)                   # (B, 1, 128)

        gru_out, new_hidden = self.gru(combined, hidden_state)

        # Flatten for output Linear layer.
        features = gru_out.reshape(-1, 128)  # (B*S, 128)

        # ------------------------------------------------------------------
        # E) Action output
        # ------------------------------------------------------------------
        action_mean = torch.tanh(self.action_mean(features))
        # It is a hard structural guardrail so that even if the loss 
        # pushes in the wrong direction, the network physically 
        # cannot go past those boundaries.
        log_std     = self.action_logstd.clamp(-4.0, 0.0)   # std in [e^-4≈0.018, e^0≈1.0]
        dist        = Normal(action_mean, torch.exp(log_std))

        # Second return value is None: the commander does not broadcast messages.
        return dist, None, new_hidden


# =============================================================================
# 3. MAPPO CRITIC  (centralised value function)
# =============================================================================

class MAPPOCritic(nn.Module):
    """
    Centralised critic for the MAPPO algorithm.

    MAPPO follows the CTDE paradigm (Centralised Training, Decentralised
    Execution): during training the critic has access to the *global* state
    -- the full fire map, all agent positions, and any other privileged
    information.  This richer input allows it to estimate V(s) (expected
    future cumulative reward) far more accurately than a per-agent critic
    that only sees local observations.

    During *execution* the critic is not used; only the actors run on-device.

    Inputs
    ------
    global_state : (Batch [, Seq], global_state_size) -- concatenated global observation
    hidden_state : (1, Batch, hidden_dim)             -- GRU memory from previous step

    Outputs
    -------
    value      : (Batch*Seq, 1) -- scalar state-value estimate V(s)
    new_hidden : updated GRU hidden state
    """
    def __init__(self, global_state_size, hidden_dim=128):
        super().__init__()

        # ------------------------------------------------------------------
        # State encoder -- two-layer MLP
        # ------------------------------------------------------------------
        # The global state vector can be very large (concatenation of all
        # agents' observations plus shared map features).  This MLP compresses
        # it from global_state_size -> 256 -> hidden_dim in two steps,
        # allowing the GRU to operate on a compact, dense representation.
        self.encoder = nn.Sequential(
            nn.Linear(global_state_size, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU()
        )

        # ------------------------------------------------------------------
        # Temporal memory -- GRU
        # ------------------------------------------------------------------
        # Gives the critic a sense of temporal dynamics: it can track whether
        # the fire is growing or shrinking and adjust value estimates
        # accordingly, rather than treating every time step as independent.
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # ------------------------------------------------------------------
        # Value head -- scalar output
        # ------------------------------------------------------------------
        # Projects the GRU hidden state to a single scalar V(s).
        # No activation function -- value estimates are unbounded real numbers.
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, global_state, hidden_state):
        # ------------------------------------------------------------------
        # Detect operating mode and flatten
        # ------------------------------------------------------------------
        # Training: global_state is (Batch, Seq, global_state_size)
        # Rollout:  global_state is (Batch,      global_state_size)
        is_sequential = (global_state.dim() == 3)
        batch_size = global_state.size(0)
        seq_len    = global_state.size(1) if is_sequential else 1

        # Flatten to (Batch*Seq, global_state_size) so the encoder MLP can
        # process all time steps simultaneously (no sequential loop needed).
        x = global_state.reshape(-1, global_state.size(-1))
        x = self.encoder(x)  # (B*S, hidden_dim)

        # Restore sequence dimension for the GRU: (B, S, hidden_dim)
        x = x.view(batch_size, seq_len, -1)

        gru_out, new_hidden = self.gru(x, hidden_state)

        # Flatten again for the value head.
        value = self.value_head(gru_out.reshape(-1, gru_out.size(-1)))  # (B*S, 1)

        return value, new_hidden


# =============================================================================
# 4. SIMPLE FIXED-WING ACTOR  (lightweight, no attention)
# =============================================================================

class SimpleFWActor(nn.Module):
    """
    Lightweight actor for the fixed-wing aircraft.

    Designed for Phase 1 training (learn to fly) where there are no scouts
    and the cross-attention branch is dead weight.  Architecture:

        self_state (17-d) → MLP (64 → 64) → GRU (64) → action dist (4-d)

    ~6K parameters vs ~118K in CommanderActor.
    Compatible interface: forward() returns (dist, None, new_hidden).
    """
    def __init__(self, self_state_dim=17, action_dim=4, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(self_state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        self.action_mean = nn.Linear(hidden_dim, action_dim)
        # Per-dimension std:
        #   [Roll, Pitch, Throttle, Water]
        # Pitch needs tighter init: std=0.6 → 27° average pitch → constant
        # diving/climbing → crashes.  std=0.3 → 13° is explorative but
        # survivable.  Roll/throttle/water can be wider.
        self.action_logstd = nn.Parameter(torch.tensor([[-0.5, -1.2, -0.5, -0.5]]))

    def forward(self, self_state, incoming_messages=None, message_mask=None, hidden_state=None):
        """
        Parameters match CommanderActor signature so train_fw_flight.py needs
        minimal changes.  incoming_messages and message_mask are ignored.
        """
        is_sequential = (self_state.dim() == 3)
        batch_size = self_state.size(0)
        seq_len = self_state.size(1) if is_sequential else 1

        if is_sequential:
            x = self_state.reshape(-1, self_state.size(-1))
        else:
            x = self_state if self_state.dim() == 2 else self_state.unsqueeze(0)

        x = self.encoder(x)  # (B*S, hidden_dim)

        if is_sequential:
            x = x.view(batch_size, seq_len, -1)
        else:
            x = x.unsqueeze(1)

        if hidden_state is None:
            hidden_state = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)

        gru_out, new_hidden = self.gru(x, hidden_state)
        features = gru_out.reshape(-1, self.hidden_dim)

        action_mean = torch.tanh(self.action_mean(features))
        log_std = self.action_logstd.clamp(-3.0, 0.0)  # std in [0.05, 1.0]
        dist = Normal(action_mean, torch.exp(log_std))

        return dist, None, new_hidden


# =============================================================================
# 5. COMMANDER ACTOR V2  (SimpleFWActor core + cross-attention into GRU)
# =============================================================================

class CommanderActorV2(nn.Module):
    """
    Two-stage actor for the commander aircraft.

    Phase 1 (flight training): use SimpleFWActor to learn basic flight.
    Phase 2 (mission training): upgrade to this class with scout messages.

    Key design: messages feed INTO the GRU alongside the encoded state.
    The GRU can then plan multi-step manoeuvres toward fire targets
    reported by scouts, integrating message history over time.

    Architecture:

        self_state (17) → encoder (17→64→64) → enc_out (64)
                                                    ↓
        messages (N×5) → msg_embed(5→64) → cross_attn(q=enc_out) → ctx (64)
                                                    ↓
                                   cat([enc_out, ctx]) (128) → GRU(128→64) → action (4)

    Weight transfer from SimpleFWActor
    -----------------------------------
    Transferred exactly (same shape, same semantics):
        encoder       (17→64→64)     — perception of own state
        action_mean   (64→4)         — action output
        action_logstd (1, 4)         — exploration noise
        gru.weight_hh (64×192)       — hidden→hidden dynamics (flight memory)

    Reinitialised (input size changed from 64 to 128):
        gru.weight_ih (128×192)      — how GRU reads new input
        gru.bias_ih                  — corresponding bias

    New (trained from scratch in Phase 2):
        msg_embed        (5→64)
        cross_attention  (64-d, 2 heads)
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

        # --- Action head (same size as SimpleFWActor: 64→4) ---
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))

    @classmethod
    def from_simple_fw(cls, simple_actor: 'SimpleFWActor',
                       msg_input_dim=5, num_attn_heads=2):
        """
        Create a CommanderActorV2 by transferring weights from a trained
        SimpleFWActor.

        Transferred: encoder, action_mean, action_logstd, gru.weight_hh/bias_hh.
        Reinitialised: gru.weight_ih/bias_ih (input size changed 64→128).
        New: msg_embed, cross_attention.

        Usage:
            simple = SimpleFWActor(...)
            simple.load_state_dict(torch.load("actor_best.pt"))
            commander = CommanderActorV2.from_simple_fw(simple)
        """
        hd = simple_actor.hidden_dim
        self_state_dim = simple_actor.encoder[0].in_features
        action_dim = simple_actor.action_mean.out_features

        model = cls(self_state_dim=self_state_dim,
                    msg_input_dim=msg_input_dim,
                    action_dim=action_dim,
                    hidden_dim=hd,
                    num_attn_heads=num_attn_heads)

        # Copy encoder, action head, logstd
        model.encoder.load_state_dict(simple_actor.encoder.state_dict())
        model.action_mean.load_state_dict(simple_actor.action_mean.state_dict())
        model.action_logstd.data.copy_(simple_actor.action_logstd.data)

        # Copy GRU hidden→hidden weights (flight dynamics memory)
        old_gru = simple_actor.gru.state_dict()
        with torch.no_grad():
            model.gru.weight_hh_l0.copy_(old_gru['weight_hh_l0'])
            model.gru.bias_hh_l0.copy_(old_gru['bias_hh_l0'])
            # weight_ih: old is (192, 64), new is (192, 128).
            # Copy old weights into the first 64 columns (enc_out half).
            # The second 64 columns (ctx half) stay at random init so the
            # GRU starts by mostly relying on its old enc_out pathway.
            model.gru.weight_ih_l0[:, :hd] = old_gru['weight_ih_l0']
            model.gru.weight_ih_l0[:, hd:] = 0.0
            model.gru.bias_ih_l0.copy_(old_gru['bias_ih_l0'])

        return model

    def forward(self, self_state, incoming_messages=None, message_mask=None,
                hidden_state=None):
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

        # --- Fuse and feed into GRU ---
        fused = torch.cat([enc_out, ctx], dim=1)           # (B*S, hidden_dim*2)

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

        return dist, None, new_hidden
