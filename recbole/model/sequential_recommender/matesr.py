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
                # 短期关注小时级差异，使用对数变换增强小间隔的区分度
                normalized_deltas = torch.log(time_deltas / 3600 + 1e-8)  # 转换为小时
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
    """连续尺度时间编码，将时间尺度视为连续空间"""
    # 60s=分钟，30 * 24 * 3600=月
    def __init__(self, hidden_size, num_basis=8, min_scale=60, max_scale=30 * 24 * 3600):
        super(ContinuousScaleTemporalEncoding, self).__init__()
        self.hidden_size = hidden_size
        self.num_basis = num_basis
        self.min_scale = min_scale
        self.max_scale = max_scale

        # 基础尺度（在对数空间均匀分布）
        log_min = np.log(min_scale)
        log_max = np.log(max_scale)
        self.basis_scales = nn.Parameter(
            torch.linspace(log_min, log_max, num_basis),
            requires_grad=True  # 允许基础尺度学习调整
        )

        # 基础函数权重（类似核方法）
        self.basis_weights = nn.Linear(num_basis, hidden_size)

        # 尺度注意力机制
        self.scale_attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)

        self.layer_norm = nn.LayerNorm(hidden_size)

    def _radial_basis_function(self, x, centers):
        """径向基函数，计算时间间隔与基础尺度的相似度"""
        # x: [batch_size, seq_len], centers: [num_basis]
        batch_size, seq_len = x.shape

        # 将x转换为对数空间
        log_x = torch.log(x + 1e-8)  # [batch_size, seq_len]

        # 重塑张量形状以便广播
        log_x = log_x.unsqueeze(-1)  # [batch_size, seq_len, 1]
        centers_expanded = centers.unsqueeze(0).unsqueeze(0)  # [1, 1, num_basis]

        # 使用正确的广播：log_x会扩展为[batch_size, seq_len, num_basis]
        # centers_expanded会扩展为[batch_size, seq_len, num_basis]
        # 计算每个时间点与每个基础尺度的相似度
        diff = log_x - centers_expanded  # 广播后形状：[batch_size, seq_len, num_basis]
        similarities = torch.exp(-0.5 * (diff ** 2))

        return similarities  # [batch_size, seq_len, num_basis]

    def forward(self, time_deltas):
        batch_size, seq_len = time_deltas.shape

        # 计算与每个基础尺度的相似度
        basis_similarities = self._radial_basis_function(time_deltas, self.basis_scales)

        # 生成基础特征
        basis_features = self.basis_weights(basis_similarities)  # [batch_size, seq_len, hidden_size]

        # 尺度内自注意力增强
        attended_features, _ = self.scale_attention(
            basis_features, basis_features, basis_features
        )

        # 残差连接
        final_output = basis_features + attended_features

        return self.layer_norm(final_output)

class AgentAttentionLayer(nn.Module):
    """代理注意力层，通过少量代理向量实现高效全局信息交互"""

    def __init__(self, hidden_size, num_heads, num_agents=8, dropout_prob=0.1):
        super(AgentAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_agents = num_agents
        self.head_dim = hidden_size // num_heads

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
        agent_attn_scores = agent_attn_scores / (self.head_dim ** 0.5)

        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            # 处理RecBole的多维掩码
            if attention_mask.dim() == 4:
                seq_mask = attention_mask[..., 0, :].squeeze(1)  # [B, seq_len]
            else:
                seq_mask = attention_mask

            # 扩展掩码形状并应用
            expanded_mask = seq_mask.unsqueeze(1).expand(-1, self.num_agents, -1)  # [B, num_agents, seq_len]
            agent_attn_scores = agent_attn_scores + expanded_mask

        agent_attn_weights = F.softmax(agent_attn_scores, dim=-1)
        agent_context = torch.matmul(agent_attn_weights, seq_v)  # [B, num_agents, H]

        # 第二步：代理到序列的注意力 (Agents-to-Items)
        # 序列作为查询，代理上下文作为键和值
        seq_q = self.q_linear(x)  # [B, seq_len, H]
        agent_context_k = self.k_linear(agent_context)  # [B, num_agents, H]
        agent_context_v = self.v_linear(agent_context)  # [B, num_agents, H]

        seq_attn_scores = torch.matmul(seq_q, agent_context_k.transpose(1, 2))  # [B, seq_len, num_agents]
        seq_attn_scores = seq_attn_scores / (self.head_dim ** 0.5)

        if attention_mask is not None:
            # 扩展掩码形状并应用
            expanded_mask = seq_mask.unsqueeze(2).expand(-1, -1, self.num_agents)  # [B, seq_len, num_agents]
            seq_attn_scores = seq_attn_scores + expanded_mask

        seq_attn_weights = F.softmax(seq_attn_scores, dim=-1)
        output = torch.matmul(seq_attn_weights, agent_context_v)  # [B, seq_len, H]

        # 残差连接和层归一化
        output = self.out_linear(output)
        output = self.dropout(output)
        output = self.layer_norm(output + x)

        return output


class AgentPointWiseFeedForward(nn.Module):
    """点对点前馈网络 - 保持非线性能力"""

    def __init__(self, hidden_size, inner_size, dropout_rate=0.1):
        super(AgentPointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_size, inner_size, kernel_size=1)
        self.conv2 = nn.Conv1d(inner_size, hidden_size, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        # FFN结构
        outputs = self.conv2(self.relu(self.conv1(inputs.transpose(-1, -2))))
        outputs = outputs.transpose(-1, -2)
        outputs = self.dropout(outputs)
        outputs = self.layer_norm(outputs + inputs)
        return outputs


class AgentTransformerEncoder(nn.Module):
    """代理Transformer编码器"""

    def __init__(self, n_layers, n_heads, hidden_size, inner_size,
                 hidden_dropout_prob, attn_dropout_prob, num_agents=8):
        super(AgentTransformerEncoder, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            # 每个层包含代理注意力层和前馈网络
            layer = nn.ModuleDict({
                'attention': AgentAttentionLayer(
                    hidden_size=hidden_size,
                    num_heads=n_heads,
                    num_agents=num_agents,
                    dropout_prob=attn_dropout_prob
                ),
                'feed_forward': AgentPointWiseFeedForward(
                    hidden_size=hidden_size,
                    inner_size=inner_size,
                    dropout_rate=hidden_dropout_prob
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
    """增强时序融合，修复维度不匹配问题"""

    def __init__(self, base_hidden_size, temporal_hidden_size, fusion_type="gate"):
        super(TemporalFusion, self).__init__()
        self.base_hidden_size = base_hidden_size
        self.temporal_hidden_size = temporal_hidden_size
        self.fusion_type = fusion_type

        if fusion_type == "gate":
            # 动态计算输入维度：base_hidden_size + temporal_hidden_size
            gate_input_size = base_hidden_size + temporal_hidden_size
            self.fusion_gate = nn.Linear(gate_input_size, base_hidden_size)
            self.gate_activ = nn.Sigmoid()
        elif fusion_type == "cross_attention":
            # 如果维度不匹配，需要投影层
            if base_hidden_size != temporal_hidden_size:
                self.temporal_proj = nn.Linear(temporal_hidden_size, base_hidden_size)
            else:
                self.temporal_proj = nn.Identity()
            self.cross_attn = nn.MultiheadAttention(base_hidden_size, num_heads=4, batch_first=True)

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
        if self.fusion_type == "gate":
            # 确保维度匹配
            if temporal_emb.size(-1) != self.temporal_hidden_size:
                temporal_emb = temporal_emb[:, :, :self.temporal_hidden_size]

            # 拼接基础表示和时间表示
            combined = torch.cat([base_emb, temporal_emb], dim=-1)
            gate_weights = self.gate_activ(self.fusion_gate(combined))

            # 将时间表示投影到基础表示维度（如果需要）
            temporal_emb_proj = self.output_proj(temporal_emb)
            fused_emb = gate_weights * base_emb + (1 - gate_weights) * temporal_emb_proj

        elif self.fusion_type == "cross_attention":
            # 投影时间表示到基础表示维度
            temporal_emb_proj = self.temporal_proj(temporal_emb)
            # 跨尺度注意力融合
            attn_output, _ = self.cross_attn(base_emb, temporal_emb_proj, temporal_emb_proj)
            fused_emb = attn_output + base_emb
        else:
            # 默认残差连接（需要维度匹配）
            if temporal_emb.size(-1) != base_emb.size(-1):
                temporal_emb_proj = self.output_proj(temporal_emb)
                fused_emb = base_emb + temporal_emb_proj
            else:
                fused_emb = base_emb + temporal_emb

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
        self.temporal_hidden_size = self.hidden_size // 4  # CrossScaleInteraction的输出维度

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
                self.temporal_encoding = MultiscaleTemporalEncoding(self.hidden_size)

            # 跨尺度交互模块
            self.cross_scale_interaction = CrossScaleInteraction(
                hidden_size=self.hidden_size,
                num_scales=3  # 短期、中期、长期
            )
            # 增强时序融合模块
            self.temporal_fusion = TemporalFusion(
                base_hidden_size=self.hidden_size,  # 基础表示维度
                temporal_hidden_size=self.temporal_hidden_size,  # 时间表示维度
                fusion_type=config['fusion_type']
            )


        # 用代理Transformer编码器替换原始编码器
        if config["agent_type"] == 'atf':
            self.agent_encoder = AgentTransformerEncoder(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                num_agents=self.num_agents
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
            temporal_emb = self.temporal_encoding(time_deltas)  # [B, seq_len, hidden_size=64]
            # 2 跨尺度交互增强
            # 将多尺度表示分解为不同尺度
            scale_embeddings = self._decompose_scales(temporal_emb)
            enhanced_temporal_emb = self.cross_scale_interaction(scale_embeddings) # [B, seq_len, temporal_hidden_size=16]
            # 3 增强时序融合
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

        # 取序列最后一个位置作为输出
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def _decompose_scales(self, temporal_emb):
        """将多尺度时间表示分解为不同尺度分量"""
        # 假设temporal_emb的维度为 [batch_size, seq_len, hidden_size]
        # 根据MultiscaleTemporalEncoding的设计，不同尺度占用不同维度段
        # 添加维度校验
        if self.hidden_size % 4 != 0:
            raise ValueError(f"hidden_size({self.hidden_size}) must be divisible by 4")

        scale_dim = self.hidden_size // 4
        long_scale_dim = self.hidden_size // 2

        # 验证维度总和
        total_dim = scale_dim * 2 + long_scale_dim
        if total_dim != self.hidden_size:
            long_scale_dim = self.hidden_size - scale_dim * 2

        short_scale = temporal_emb[:, :, :scale_dim]
        medium_scale = temporal_emb[:, :, scale_dim:scale_dim * 2]
        long_scale = temporal_emb[:, :, scale_dim * 2:scale_dim * 2 + long_scale_dim]

        return [short_scale, medium_scale, long_scale]


    def calculate_loss(self, interaction):
        """计算损失函数 - 支持时间间隔输入"""
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

