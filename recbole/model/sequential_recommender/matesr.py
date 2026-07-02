# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import TransformerEncoder


class MultiscaleTemporalEncoding(nn.Module):
    """多尺度时间编码，捕捉不同时间粒度的行为模式"""

    def __init__(self, hidden_size, time_scales=['short', 'medium', 'long']):
        super(MultiscaleTemporalEncoding, self).__init__()
        self.time_scales = time_scales
        self.hidden_size = hidden_size

        # 不同尺度使用不同的嵌入维度
        self.scale_embeddings = nn.ModuleDict()
        for scale in time_scales:
            if scale == 'short':  # 短期（小时级）
                self.scale_embeddings[scale] = nn.Linear(1, hidden_size // 4)
            elif scale == 'medium':  # 中期（天级）
                self.scale_embeddings[scale] = nn.Linear(1, hidden_size // 4)
            else:  # 长期（周/月级）
                self.scale_embeddings[scale] = nn.Linear(1, hidden_size // 2)

        # 门控融合机制
        self.fusion_gate = nn.Linear(hidden_size, len(time_scales))
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, time_deltas):
        """
        time_deltas: [batch_size, seq_len] 时间间隔矩阵（秒为单位）
        基于时间间隔扰动增强时序信息
        """
        batch_size, seq_len = time_deltas.shape
        scale_representations = []

        for scale in self.time_scales:
            # 对不同尺度的时间间隔进行归一化处理
            if scale == 'short':
                # 短期关注小时级差异，log1p 平滑 delta=0 附近的过渡
                normalized_deltas = torch.log1p(time_deltas / 3600)  # 转换为小时
            elif scale == 'medium':
                # 中期关注天级差异
                normalized_deltas = time_deltas / (24 * 3600)  # 转换为天
            else:
                # 长期关注周级差异
                normalized_deltas = time_deltas / (7 * 24 * 3600)  # 转换为周

            # 应用尺度特定的嵌入
            emb = self.scale_embeddings[scale](normalized_deltas.unsqueeze(-1))
            scale_representations.append(emb)

        # 门控融合多尺度时间信息
        combined = torch.cat(scale_representations, dim=-1)
        gate_weights = F.softmax(self.fusion_gate(combined), dim=-1)

        # 加权融合各尺度表示
        temporal_emb = torch.zeros(batch_size, seq_len, self.hidden_size,
                                   device=time_deltas.device)
        start_dim = 0
        for i, scale_emb in enumerate(scale_representations):
            scale_dim = scale_emb.shape[-1]
            # 应用门控权重
            weighted_emb = gate_weights[:, :, i].unsqueeze(-1) * scale_emb
            temporal_emb[:, :, start_dim:start_dim + scale_dim] = weighted_emb
            start_dim += scale_dim

        return self.layer_norm(temporal_emb)

class ContinuousScaleTemporalEncoding(nn.Module):
    """连续尺度时间编码 v0.3.1 — 多头RBF(Radial Basis Function) + 可学习带宽 + 相位方向 + 动态门控

    增强特性:
    - 可学习带宽 (1a): 每个basis独立调节感受野, 短尺度窄/长尺度宽
    - 相位方向编码 (1b): sin/cos区分时间偏移方向, 打破RBF对称性
    - 多头RBF (3a): 多组独立basis并行编码不同时间子模式
    - 动态门控 (2a): 基于时间上下文逐维度自适应调节编码强度
    """
    # 60s=分钟, 30 * 24 * 3600=月
    def __init__(self, hidden_size, num_basis=8, num_heads=4,
                 min_scale=60, max_scale=30 * 24 * 3600):
        super(ContinuousScaleTemporalEncoding, self).__init__()
        self.hidden_size = hidden_size
        self.num_basis = num_basis
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 多头RBF: 每组独立basis centers (log-space)
        # log-uniform初始化 + 小随机扰动打破head间对称性
        log_min = np.log(min_scale)
        log_max = np.log(max_scale)
        init_centers = torch.linspace(log_min, log_max, num_basis).unsqueeze(0).repeat(num_heads, 1)
        init_centers = init_centers + torch.randn(num_heads, num_basis) * \
                       0.05 * (log_max - log_min) / max(num_basis, 1)
        self.basis_centers = nn.Parameter(init_centers)  # [num_heads, num_basis]

        # 可学习带宽 (per head, per basis), softplus保证正值
        self.basis_bandwidth = nn.Parameter(torch.ones(num_heads, num_basis))

        # 每个head独立映射: RBF特征(rbf+sin+cos) → head_dim
        self.head_weights = nn.ModuleList([
            nn.Linear(num_basis * 3, self.head_dim)
            for _ in range(num_heads)
        ])

        # 最终投影（当num_heads*head_dim ≠ hidden_size时纠正维度）
        concat_dim = num_heads * self.head_dim
        if concat_dim != hidden_size:
            self.output_proj = nn.Linear(concat_dim, hidden_size)
        else:
            self.output_proj = nn.Identity()

        # 动态门控: 基于完整RBF特征 + log(原始时间值) → 逐维度门
        gate_input_dim = num_heads * num_basis * 3 + 1  # +1 for log(delta)
        self.context_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def _radial_basis_function(self, x, centers, bandwidth):
        """带可学习带宽的RBF + 相位方向编码

        Args:
            x: [batch_size, seq_len] 原始时间间隔(秒)
            centers: [num_basis] RBF中心(log-space)
            bandwidth: [num_basis] 各basis带宽

        Returns:
            [batch_size, seq_len, num_basis*3] 拼接[rbf, sin相位, cos相位]
        """
        log_x = torch.log(x + 1e-8)  # [B, S]
        log_x = log_x.unsqueeze(-1)  # [B, S, 1]
        c = centers.unsqueeze(0).unsqueeze(0)  # [1, 1, num_basis]
        bw = F.softplus(bandwidth).unsqueeze(0).unsqueeze(0)  # [1, 1, num_basis], >0

        diff = (log_x - c) / bw  # 标准化距离 [B, S, num_basis]

        rbf = torch.exp(-0.5 * diff ** 2)        # 距离相似度 (对称核)
        phase_sin = torch.sin(diff) * rbf         # <0=比中心短, >0=比中心长
        phase_cos = torch.cos(diff) * rbf         # 归一化方向分量

        return torch.cat([rbf, phase_sin, phase_cos], dim=-1)

    def forward(self, time_deltas):
        batch_size, seq_len = time_deltas.shape

        all_rbf_features = []   # 存所有head的原始RBF特征 (给gate用)
        head_outputs = []       # 存每个head的投影输出

        for h in range(self.num_heads):
            rbf = self._radial_basis_function(
                time_deltas,
                self.basis_centers[h],    # [num_basis]
                self.basis_bandwidth[h]   # [num_basis]
            )  # [B, S, num_basis*3]
            all_rbf_features.append(rbf)
            head_out = self.head_weights[h](rbf)  # [B, S, head_dim]
            head_outputs.append(head_out)

        # 拼接多头输出 → hidden_size
        multi_head_out = self.output_proj(torch.cat(head_outputs, dim=-1))

        # 动态门控: 基于完整RBF特征 + 原始时间对数
        all_rbf = torch.cat(all_rbf_features, dim=-1)   # [B, S, num_heads*num_basis*3]
        log_delta = torch.log(time_deltas + 1e-8).unsqueeze(-1)  # [B, S, 1]
        gate = self.context_gate(torch.cat([all_rbf, log_delta], dim=-1))

        return self.layer_norm(multi_head_out * gate)

class AgentAttentionLayer(nn.Module):
    """代理注意力层，通过少量代理向量实现高效全局信息交互

    M 个可学习 Agent 各自独立学习一种注意力模式，实现与标准 Transformer
    多头注意力同构的多样化功能，但无需显式拆分 head。
    """

    def __init__(self, hidden_size, num_agents=8, dropout_prob=0.1):
        super(AgentAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_agents = num_agents

        # 可学习的代理向量 [num_agents, hidden_size]
        self.agents = nn.Parameter(torch.Tensor(num_agents, hidden_size))
        nn.init.xavier_uniform_(self.agents)

        # 查询、键、值的线性变换
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        # 输出变换
        self.out_linear = nn.Linear(hidden_size, hidden_size)

        # 正则化层
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, hidden_size = x.shape

        # 第一步：序列到代理的注意力 (Items-to-Agents)
        # 代理向量作为查询，序列作为键和值
        agent_q = self.q_linear(self.agents).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, num_agents, H]
        seq_k = self.k_linear(x)  # [B, seq_len, H]
        seq_v = self.v_linear(x)  # [B, seq_len, H]

        # 计算注意力分数 [batch_size, num_agents, seq_len]
        agent_attn_scores = torch.matmul(agent_q, seq_k.transpose(1, 2))  # [B, num_agents, seq_len]
        agent_attn_scores = agent_attn_scores / (self.hidden_size ** 0.5)

        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            # 处理RecBole的多维掩码 [B, 1, N, N]（因果+padding组合）
            if attention_mask.dim() == 4:
                # Phase 1: Agent（非位置实体）attend items — 仅需 padding mask，
                # 不应受因果约束。从组合掩码中提取纯 padding 信息：
                # 对于每个 key 位置 j：若 j 是 padding，所有 query 行都是 -inf；
                # 若 j 有效，至少 query 行 j 为 0。取 max 沿 query 维度分离二者。
                pad_mask = attention_mask.max(dim=2).values  # [B, 1, N]
            else:
                pad_mask = attention_mask
                if pad_mask.dim() == 2:
                    pad_mask = pad_mask.unsqueeze(1)  # [B, 1, N]

            # 扩展到 agent 维度
            expanded_mask = pad_mask.expand(-1, self.num_agents, -1)  # [B, num_agents, N]
            agent_attn_scores = agent_attn_scores + expanded_mask

        agent_attn_weights = F.softmax(agent_attn_scores, dim=-1)
        agent_context = torch.matmul(agent_attn_weights, seq_v)  # [B, num_agents, H]

        # 第二步：代理到序列的注意力 (Agents-to-Items)
        # 序列作为查询，代理上下文作为键和值
        seq_q = self.q_linear(x)  # [B, seq_len, H]
        agent_context_k = self.k_linear(agent_context)  # [B, num_agents, H]
        agent_context_v = self.v_linear(agent_context)  # [B, num_agents, H]

        seq_attn_scores = torch.matmul(seq_q, agent_context_k.transpose(1, 2))  # [B, seq_len, num_agents]
        seq_attn_scores = seq_attn_scores / (self.hidden_size ** 0.5)

        if attention_mask is not None:
            # Phase 2: items attend to agents（位置无关，仅需 padding mask）
            if attention_mask.dim() == 4:
                # 复用 Phase 1 中提取的纯 padding 掩码 [B, 1, N]
                pad_mask = attention_mask.max(dim=2).values  # [B, 1, N]
                pad_mask = pad_mask.squeeze(1).unsqueeze(-1)  # [B, N, 1]
            else:
                pad_mask = attention_mask
                if pad_mask.dim() == 2:
                    pad_mask = pad_mask.unsqueeze(-1)  # [B, N, 1]
            seq_attn_scores = seq_attn_scores + pad_mask

        seq_attn_weights = F.softmax(seq_attn_scores, dim=-1)
        output = torch.matmul(seq_attn_weights, agent_context_v)  # [B, seq_len, H]

        # 残差连接和层归一化
        output = self.out_linear(output)
        output = self.dropout(output)
        output = self.layer_norm(output + x)

        return output


class AgentPointWiseFeedForward(nn.Module):
    """点对点前馈网络 - 保持非线性能力"""

    def __init__(self, hidden_size, inner_size, dropout_rate=0.1, hidden_act='gelu'):
        super(AgentPointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_size, inner_size, kernel_size=1)
        self.conv2 = nn.Conv1d(inner_size, hidden_size, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)
        if hidden_act == 'gelu':
            self.activation = nn.GELU()
        elif hidden_act == 'relu':
            self.activation = nn.ReLU()
        elif hidden_act == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        # FFN结构
        outputs = self.conv2(self.activation(self.conv1(inputs.transpose(-1, -2))))
        outputs = outputs.transpose(-1, -2)
        outputs = self.dropout(outputs)
        outputs = self.layer_norm(outputs + inputs)
        return outputs


class AgentTransformerEncoder(nn.Module):
    """代理Transformer编码器"""

    def __init__(self, n_layers, hidden_size, inner_size,
                 hidden_dropout_prob, attn_dropout_prob,
                 num_agents=8, hidden_act='gelu'):
        super(AgentTransformerEncoder, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            # 每个层包含代理注意力层和前馈网络
            layer = nn.ModuleDict({
                'attention': AgentAttentionLayer(
                    hidden_size=hidden_size,
                    num_agents=num_agents,
                    dropout_prob=attn_dropout_prob
                ),
                'feed_forward': AgentPointWiseFeedForward(
                    hidden_size=hidden_size,
                    inner_size=inner_size,
                    dropout_rate=hidden_dropout_prob,
                    hidden_act=hidden_act
                )
            })
            self.layers.append(layer)

    def forward(self, hidden_states, attention_mask=None, output_all_encoded_layers=True):
        all_encoder_layers = []

        for layer_module in self.layers:
            # 代理注意力层
            hidden_states = layer_module['attention'](hidden_states, attention_mask)
            # 前馈网络层
            hidden_states = layer_module['feed_forward'](hidden_states)

            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers


class TemporalFusion(nn.Module):
    """内容-时间自适应门控融合

    逐维度 gate 决定每个特征维度上内容与时间的权重配比，
    实现内容语义与时间语义的对等、自适应耦合。
    """

    def __init__(self, base_hidden_size, temporal_hidden_size):
        super(TemporalFusion, self).__init__()
        self.base_hidden_size = base_hidden_size
        self.temporal_hidden_size = temporal_hidden_size

        # 拼接内容+时间后逐维度门控
        gate_input_size = base_hidden_size + temporal_hidden_size
        self.fusion_gate = nn.Linear(gate_input_size, base_hidden_size)
        self.gate_activ = nn.Sigmoid()

        # 投影层，将时间表示投影到基础表示维度
        if temporal_hidden_size != base_hidden_size:
            self.output_proj = nn.Linear(temporal_hidden_size, base_hidden_size)
        else:
            self.output_proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(base_hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, base_emb, temporal_emb):
        """
        base_emb: [batch_size, seq_len, base_hidden_size] 基础物品表示
        temporal_emb: [batch_size, seq_len, temporal_hidden_size] 时间表示
        """
        # 确保维度匹配（不应在此处静默截断，调用方应保证维度一致）
        assert temporal_emb.size(-1) == self.temporal_hidden_size, \
            f"TemporalFusion: expected dim {self.temporal_hidden_size}, got {temporal_emb.size(-1)}"

        # 拼接基础表示和时间表示 → 逐维度 gate
        combined = torch.cat([base_emb, temporal_emb], dim=-1)
        gate_weights = self.gate_activ(self.fusion_gate(combined))

        # 将时间表示投影到基础表示维度后加权融合
        temporal_emb_proj = self.output_proj(temporal_emb)
        fused_emb = gate_weights * base_emb + (1 - gate_weights) * temporal_emb_proj

        fused_emb = self.layer_norm(fused_emb)
        fused_emb = self.dropout(fused_emb)
        return fused_emb


class CrossScaleInteraction(nn.Module):
    """跨尺度交互，修复特征维度不匹配问题"""

    def __init__(self, hidden_size, num_scales=3):
        super(CrossScaleInteraction, self).__init__()
        self.hidden_size = hidden_size
        self.num_scales = num_scales

        # 计算各尺度的原始特征维度
        self.scale_dims = {
            0: hidden_size // 4,  # 短期尺度
            1: hidden_size // 4,  # 中期尺度
            2: hidden_size // 2  # 长期尺度
        }

        # 统一投影维度，取各尺度维度的最小值或平均值
        self.projected_size = hidden_size // 4  # 统一投影到最小维度

        # 为每个尺度创建投影层，统一特征维度
        self.scale_projections = nn.ModuleList([
            nn.Linear(self.scale_dims[i], self.projected_size)
            for i in range(num_scales)
        ])

        # 跨尺度注意力机制
        self.cross_scale_attn = nn.MultiheadAttention(
            self.projected_size, num_heads=4, batch_first=True
        )

        # 尺度间门控融合
        self.scale_gates = nn.Linear(self.projected_size * num_scales, num_scales)
        self.layer_norm = nn.LayerNorm(self.projected_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, scale_representations):
        """
        scale_representations: 包含3个尺度张量的列表
        每个张量形状: [batch_size, seq_len, scale_dim]
        """
        batch_size, seq_len, _ = scale_representations[0].shape

        # 1. 投影所有尺度到统一特征维度
        projected_scales = []
        for i, scale_emb in enumerate(scale_representations):
            # 应用尺度特定的投影层
            projected_emb = self.scale_projections[i](scale_emb)  # [B, seq_len, projected_size]
            projected_scales.append(projected_emb)

        # 2. 跨尺度注意力交互（修复后的逻辑）
        enhanced_scales = []
        for i in range(self.num_scales):
            # 当前尺度作为Query
            query = projected_scales[i]  # [B, seq_len, projected_size]

            # 拼接其他尺度作为Key和Value（现在特征维度一致）
            other_scales = [projected_scales[j] for j in range(self.num_scales) if j != i]

            # 沿序列维度拼接其他尺度
            key_value = torch.cat(other_scales, dim=1)  # [B, seq_len*(num_scales-1), projected_size]

            # 计算跨尺度注意力
            attn_output, _ = self.cross_scale_attn(query, key_value, key_value)
            enhanced_scale = attn_output + query  # 残差连接
            enhanced_scales.append(enhanced_scale)

        # 3. 门控重加权融合
        combined = torch.cat(enhanced_scales, dim=-1)  # [B, seq_len, projected_size*3]
        gate_weights = F.softmax(self.scale_gates(combined), dim=-1)

        # 加权融合各尺度表示
        final_representation = torch.zeros_like(enhanced_scales[0])
        for i, scale_emb in enumerate(enhanced_scales):
            final_representation += gate_weights[:, :, i].unsqueeze(-1) * scale_emb

        return self.layer_norm(final_representation)


class MATESR(SequentialRecommender):
    """基于代理注意力和多尺度时间建模"""

    def __init__(self, config, dataset):
        super(MATESR, self).__init__(config, dataset)

        # 加载基础参数
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.loss_type = config["loss_type"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]

        # 新增代理注意力参数
        self.num_agents = config["num_agents"]  # 代理向量数量
        self.use_temporal = config["use_temporal_encoding"]  # 是否使用时序编码
        # 计算时间表示的投影维度
        self.temporal_hidden_size = self.hidden_size // 2  # v0.3.1: 提升时间表示容量 (原 //4)
        self.bpr_weight = config.get("bpr_weight", 0.0)  # BPR 正则化权重（0=禁用）

        # 定义嵌入层
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # 多尺度时间编码
        if self.use_temporal:
            if config['temporal_encoder'] == 'cs':
                self.temporal_encoding = ContinuousScaleTemporalEncoding(self.hidden_size)
            else: # config['temporal_encoder'] == 'ms':
                time_scales = config.get('time_scales', ['short', 'medium', 'long'])
                self.temporal_encoding = MultiscaleTemporalEncoding(self.hidden_size, time_scales)

            # 时间表示投影层：将多尺度编码直接映射到TemporalFusion输入维度
            self.temporal_proj = nn.Linear(self.hidden_size, self.temporal_hidden_size)
            # 时序融合模块（始终使用逐维度门控融合）
            self.temporal_fusion = TemporalFusion(
                base_hidden_size=self.hidden_size,
                temporal_hidden_size=self.temporal_hidden_size,
            )


        # 用代理Transformer编码器替换原始编码器
        if config["agent_type"] == 'atf':
            self.agent_encoder = AgentTransformerEncoder(
                n_layers=self.n_layers,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                num_agents=self.num_agents,
                hidden_act=self.hidden_act
            )
        else:
            self.agent_encoder = TransformerEncoder(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
            )


        # 多位置预测头：聚合末位和倒数第2位增强排序质量
        self.pred_head = nn.Linear(2 * self.hidden_size, self.hidden_size)

        # 正则化层
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # 损失函数
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, time_deltas=None):
        """
        item_seq: 物品序列 [batch_size, seq_len]
        item_seq_len: 序列长度 [batch_size]
        time_deltas: 时间间隔矩阵 [batch_size, seq_len] (可选)
        """
        # 位置编码
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        # 物品嵌入
        item_emb = self.item_embedding(item_seq)
        # 组合基础嵌入
        base_emb = item_emb + position_embedding

        # 多尺度时间编码与高级融合
        if self.use_temporal and time_deltas is not None:
            # 1 生成多尺度时间表示
            temporal_emb = self.temporal_encoding(time_deltas)  # [B, seq_len, hidden_size]
            # 1.5 清零 padding 位置的时间嵌入，防止伪信号污染 LayerNorm/融合
            valid_mask = (item_seq != 0).float().unsqueeze(-1)  # [B, seq_len, 1]
            temporal_emb = temporal_emb * valid_mask
            # 2 投影到时间融合维度
            enhanced_temporal_emb = self.temporal_proj(temporal_emb)  # [B, seq_len, temporal_hidden_size]
            # 3 自适应时序融合
            input_emb = self.temporal_fusion(base_emb, enhanced_temporal_emb)
        else:
            input_emb = base_emb

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # 注意力掩码
        extended_attention_mask = self.get_attention_mask(item_seq)
        # 代理Transformer编码
        agent_output = self.agent_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = agent_output[-1]

        # 多位置聚合：末位 + 倒数第2位，残差连接保留末位主导信号
        output_last = self.gather_indexes(output, item_seq_len - 1)
        # 短序列（len<=1）回退到末位自身，避免 index out of bounds
        last2_idx = torch.clamp(item_seq_len - 2, min=0)
        output_last2 = self.gather_indexes(output, last2_idx)
        combined = torch.cat([output_last, output_last2], dim=-1)
        output = self.pred_head(combined) + output_last
        return output



    def calculate_loss(self, interaction):
        """计算损失函数 - 支持时间间隔输入和混合 CE+BPR 损失"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # 提取时间间隔信息（如果存在）
        time_deltas = None
        if self.use_temporal and 'time_deltas' in interaction:
            time_deltas = interaction['time_deltas']

        seq_output = self.forward(item_seq, item_seq_len, time_deltas)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

            # 可选 BPR 正则化项：增强 pairwise ranking 信号
            # 序列推荐 CE 模式没有内置负采样，直接随机均匀采
            if self.bpr_weight > 0:
                neg_items = torch.randint(0, self.n_items, pos_items.shape, device=pos_items.device)
                pos_emb = self.item_embedding(pos_items)
                neg_emb = self.item_embedding(neg_items)
                pos_score = torch.sum(seq_output * pos_emb, dim=-1)
                neg_score = torch.sum(seq_output * neg_emb, dim=-1)
                bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
                loss = loss + self.bpr_weight * bpr_loss
        return loss

    def predict(self, interaction):
        """预测函数 - 支持时间间隔输入"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        # 提取时间间隔信息（如果存在）
        time_deltas = None
        if self.use_temporal and 'time_deltas' in interaction:
            time_deltas = interaction['time_deltas']

        seq_output = self.forward(item_seq, item_seq_len, time_deltas)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """全排序预测 - 支持时间间隔输入"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # 提取时间间隔信息（如果存在）
        time_deltas = None
        if self.use_temporal and 'time_deltas' in interaction:
            time_deltas = interaction['time_deltas']

        seq_output = self.forward(item_seq, item_seq_len, time_deltas)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores

