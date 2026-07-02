"""
EulerFormer
################################################

Reference:
    Zhen Tian et al. "EulerFormer: Sequential Recommendation with Euler Encoding." 

Reference:
    https://github.com/user/EulerFormer

Implementation note:
    All custom layers (EulerFormer rotation, EulerMultiHeadAttention,
    EulerTransformerLayer, EulerTransformerEncoder) are self-contained.
    The implementation faithfully follows the original EulerFormer source code.
"""

import copy
import math

import torch
from torch import nn
import torch.nn.functional as fn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


# =============================================================================
# Euler Transform Helper
# =============================================================================

def _euler_transform(tensor):
    """
    Decompose tensor into polar coordinates (radius, angle).
    Interleaved layout: even indices = r, odd indices = p.
    
    Returns: (radius, angle) pair
    """
    r = tensor[..., ::2]
    p = tensor[..., 1::2]
    return torch.sqrt(r ** 2 + p ** 2), torch.atan2(p, r)


# =============================================================================
# EulerFormer Rotation Module (Attention-level)
# =============================================================================

class _EulerFormerRotation(nn.Module):
    """
    EulerFormer rotation module applied to Q/K vectors before multi-head splitting.
    
    Based on Euler's formula, vectors are decomposed into polar coordinates (r, theta),
    rotated by learnable parameters (delta, alpha, bias), and reconstructed.
    
    - delta: learnable scaling factor for the rotation angle
    - alpha: sinusoidal positional encoding (fixed reference angles)
    - b: query-specific bias (optional)
    """

    def __init__(self, max_seq_len, hidden_size, euler_bias=True, init_factor=1.0):
        super().__init__()
        self.alpha = None
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        if euler_bias:
            self.b = nn.Parameter(torch.zeros(1))
        else:
            self.b = 0

        # delta scales the rotation angle; initialized from init_factor
        self.delta = nn.Parameter(torch.ones(1) * init_factor)

        # Pre-build alpha (sinusoidal position encoding)
        self._build_alpha(1, 1, max_seq_len, hidden_size)

    def _build_alpha(self, batch_size, num_heads, max_len, output_dim):
        """Build sinusoidal positional encoding as the base rotation angles."""
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
        ids = torch.arange(0, output_dim // 2, dtype=torch.float)
        theta = torch.pow(10000, -2 * ids / output_dim)
        embeddings = position * theta
        embeddings = torch.stack([embeddings, embeddings], dim=-1)
        self.alpha = nn.Parameter(embeddings)

    def _get_alpha(self, batch_size, num_heads, max_len, output_dim):
        """Expand alpha to target shape and extract the angle component."""
        embeddings = self.alpha.repeat((batch_size, num_heads, *([1] * len(self.alpha.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, num_heads, max_len, output_dim))
        return embeddings[..., ::2]  # Return only the angle part

    def forward(self, v, rot_type='ro'):
        """
        Args:
            v: [B, L, D] input tensor
            rot_type: 'ro' for rotation, 'queryro' for query rotation (+bias)
        
        Returns:
            rotated tensor of shape [B, L, D]
        """
        v = v.unsqueeze(1)  # [B, 1, L, D] - add head dim
        r = v[..., ::2]
        p = v[..., 1::2]
        batch_size = v.shape[0]
        nums_head = v.shape[1]
        max_len = v.shape[2]
        output_dim = v.shape[-1]

        # Euler Transform: Cartesian -> Polar
        lam = torch.sqrt(r ** 2 + p ** 2)
        theta = torch.atan2(p, r)

        if 'ro' in rot_type:
            # Apply rotation: theta' = theta * delta + alpha
            alpha = self._get_alpha(batch_size, nums_head, max_len, output_dim)
            theta = theta * self.delta + alpha.to(theta).data

            if 'query' in rot_type:
                # Query gets an additional global bias
                theta = theta + self.b

        # Reconstruct Cartesian coordinates
        r, p = lam * torch.cos(theta), lam * torch.sin(theta)
        embeddings = torch.stack([r, p], dim=-1)
        embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
        return embeddings.squeeze(1)

    @staticmethod
    def get_polar(v):
        """Extract polar angles from Cartesian (interleaved) representation."""
        r = v[..., ::2]
        p = v[..., 1::2]
        return torch.atan2(p, r)


# =============================================================================
# EulerFormer Multi-Head Attention (with contrastive loss)
# =============================================================================

class _EulerMultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention with EulerFormer rotation on Q and K,
    plus a contrastive (InfoNCE) regularization loss on polar angles.
    """

    def __init__(
        self, n_heads, hidden_size, max_seq_len,
        hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
        euler_bias=True, init_factor=1.0, tep=1.0, lamb=1e-5
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        # EulerFormer rotation module
        self.euler = _EulerFormerRotation(
            max_seq_len=max_seq_len,
            hidden_size=hidden_size,
            euler_bias=euler_bias,
            init_factor=init_factor
        )

        # Contrastive loss components
        self.dp = nn.Dropout(p=0.5)
        self.loss = 0
        self.w = nn.Parameter(torch.ones(hidden_size // 2))
        self.tep = tep
        self.lamb = lamb

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # EulerFormer rotation on Q and K
        mixed_query_layer, mixed_key_layer = \
            self.euler(mixed_query_layer, 'queryro'), self.euler(mixed_key_layer)

        # Compute contrastive loss on rotated Q/K
        self._compute_contrastive_loss(mixed_query_layer, mixed_key_layer)

        # Multi-head splitting
        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 1, 3)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)
        key_layer = key_layer.transpose(-2, -1)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    def _compute_contrastive_loss(self, q, k):
        """
        InfoNCE-style contrastive loss on polar angles.
        Encourages phase consistency between nearby positions.
        """
        pq = _EulerFormerRotation.get_polar(q)
        pk = _EulerFormerRotation.get_polar(k)

        def _info_nce(input_angles, target_angles):
            # Weighted cosine similarity: cos(pq) * w * cos(pk)^T + sin(pq) * w * sin(pk)^T
            cos_sim = (torch.cos(input_angles) * self.w) @ torch.cos(target_angles).transpose(-2, -1) + \
                      (torch.sin(input_angles) * self.w) @ torch.sin(target_angles).transpose(-2, -1)
            numerator = torch.diagonal(torch.exp(cos_sim / self.tep), dim1=-1, dim2=-2)
            denominator = torch.sum(torch.exp(cos_sim / self.tep), dim=(-1)) + 1e-5
            return torch.mean(-torch.log(numerator / denominator)) * self.lamb

        # Contrastive loss with dropout augmentation for both Q and K
        self.loss = _info_nce(pq, self.dp(pq)) + _info_nce(pk, self.dp(pk))


# =============================================================================
# EulerFormer Transformer Layers
# =============================================================================

class _EulerFeedForward(nn.Module):
    """Point-wise feed-forward layer."""

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self._get_hidden_act(hidden_act)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def _get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self._gelu,
            "relu": fn.relu,
            "swish": self._swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def _swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class _EulerTransformerLayer(nn.Module):
    """
    One transformer layer with EulerFormer attention + contrastive loss propagation.
    """

    def __init__(
        self, n_heads, hidden_size, max_seq_len, intermediate_size,
        hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,
        euler_bias=True, init_factor=1.0, tep=1.0, lamb=1e-5
    ):
        super().__init__()
        self.multi_head_attention = _EulerMultiHeadAttention(
            n_heads, hidden_size, max_seq_len,
            hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
            euler_bias=euler_bias, init_factor=init_factor,
            tep=tep, lamb=lamb
        )
        self.feed_forward = _EulerFeedForward(
            hidden_size, intermediate_size,
            hidden_dropout_prob, hidden_act, layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        self._propagate_loss()
        return feedforward_output

    def _propagate_loss(self):
        self.loss = self.multi_head_attention.loss


class _EulerTransformerEncoder(nn.Module):
    """
    Stack of EulerFormer TransformerLayers with contrastive loss aggregation.
    """

    def __init__(
        self, n_layers=2, n_heads=2, hidden_size=64, max_seq_len=50,
        inner_size=256, hidden_dropout_prob=0.5, attn_dropout_prob=0.5,
        hidden_act="gelu", layer_norm_eps=1e-12,
        euler_bias=True, init_factor=1.0, tep=1.0, lamb=1e-5
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            _EulerTransformerLayer(
                n_heads, hidden_size, max_seq_len, inner_size,
                hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,
                euler_bias=euler_bias, init_factor=init_factor,
                tep=tep, lamb=lamb
            ) for _ in range(n_layers)
        ])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        self._aggregate_loss()
        return all_encoder_layers

    def _aggregate_loss(self):
        self.loss = sum(model.loss for model in self.layer)


# =============================================================================
# EulerFormer Model
# =============================================================================

class EulerFormer(SequentialRecommender):
    r"""
    EulerFormer enhances SASRec with Euler encoding at two levels:
    
    1. **Embedding level**: Rotary position embedding rotates item embeddings
       in polar space before entering the transformer.
    2. **Attention level**: The EulerFormer rotation module further rotates
       Q/K vectors, with an InfoNCE contrastive loss as regularization.
    
    Formula: x' = lam * cos(theta + delta * alpha + bias) + lam * sin(...)
    """

    def __init__(self, config, dataset):
        super(EulerFormer, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # EulerFormer-specific config
        self.euler_bias = config["euler_bias"]
        self.init_factor = config["init_factor"]
        self.tep = config["tep"]
        self.lamb = config["lamb"]

        # define layers
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        # Rotary position embedding (Embedding-level rotation)
        self.rotary_embedding = nn.Embedding(self.max_seq_length, self.hidden_size // 2)

        # EulerFormer Transformer Encoder with contrastive loss
        self.trm_encoder = _EulerTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            max_seq_len=self.max_seq_length,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            euler_bias=self.euler_bias,
            init_factor=self.init_factor,
            tep=self.tep,
            lamb=self.lamb
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        # Rotary embedding initialized to zeros (identity rotation)
        nn.init.zeros_(self.rotary_embedding.weight)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        rotary_embedding = self.rotary_embedding(position_ids)

        # Item embedding + position embedding
        input_emb = self.item_embedding(item_seq) + position_embedding
        shape = input_emb.shape

        # Embedding-level Euler rotation: x' = lam * cos(theta + rotary_emb)
        lam, theta = _euler_transform(input_emb)
        input_emb = torch.stack(
            [lam * torch.cos(theta + rotary_embedding),
             lam * torch.sin(theta + rotary_embedding)], dim=-1
        )
        input_emb = input_emb.reshape(*shape)

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            # CE loss + contrastive regularization
            loss = self.loss_fct(logits, pos_items) + self.trm_encoder.loss
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
