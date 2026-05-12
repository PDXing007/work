#!/usr/bin/env python3
"""
时序自注意力预测模型 — 纯 NumPy 实现

架构:
  Input: (T, F) — T期特征序列
  → Linear Projection (F→D)
  → Sinusoidal Positional Encoding
  → Multi-Head Self-Attention (跨时间维)
  → Residual + LayerNorm
  → FFN (GELU)
  → Residual + LayerNorm
  → Mean Pooling → (D,)
  → Output: red(33,) + blue(16,) logits

设计:
- 自注意力在时间维上计算，不跨样本 → 纯前馈，无需 BPTT
- 注意力权重可解释：显示"模型参考了哪些历史期"
- 约300行纯NumPy，可在单个特征向量上做前向传播
"""

import math
import random

from nn_base import (
    matmul_1d, matmul_2d, softmax, softmax_2d,
    gelu, layer_norm, add, mul, he_init, zeros,
    outer, cross_entropy_loss, binary_cross_entropy,
)


class TemporalAttentionPredictor:
    """
    时序自注意力模型

    Args:
        n_features: 输入特征维度 F
        d_model: 隐藏维度 D (默认48)
        n_heads: 注意力头数 (默认2)
        T: 输入序列长度 (默认20期)
        dropout: dropout 比率
    """

    def __init__(self, n_features, d_model=48, n_heads=2, T=20, dropout=0.1):
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每头维度
        self.T = T
        self.dropout = dropout

        # === 输入投影 ===
        self.W_in = he_init((n_features, d_model))
        self.b_in = zeros(d_model)

        # === 多头注意力 ===
        # Q, K, V 投影: d_model → d_model
        self.W_q = he_init((d_model, d_model))
        self.W_k = he_init((d_model, d_model))
        self.W_v = he_init((d_model, d_model))
        # 输出投影
        self.W_o = he_init((d_model, d_model))

        # === LayerNorm (2个) ===
        self.ln1_gamma = [1.0] * d_model
        self.ln1_beta = [0.0] * d_model
        self.ln2_gamma = [1.0] * d_model
        self.ln2_beta = [0.0] * d_model

        # === FFN ===
        ffn_dim = d_model * 4
        self.W_ff1 = he_init((d_model, ffn_dim))
        self.b_ff1 = zeros(ffn_dim)
        self.W_ff2 = he_init((ffn_dim, d_model))
        self.b_ff2 = zeros(d_model)

        # === 输出头 ===
        self.W_red = he_init((d_model, 33))
        self.b_red = zeros(33)
        self.W_blue = he_init((d_model, 16))
        self.b_blue = zeros(16)

        # === 位置编码 (固定) ===
        self.pos_enc = self._build_positional_encoding(T, d_model)

    def _build_positional_encoding(self, T, d):
        """Sinusoidal positional encoding"""
        pe = [[0.0] * d for _ in range(T)]
        for pos in range(T):
            for i in range(0, d, 2):
                angle = pos / (10000 ** (i / d))
                pe[pos][i] = math.sin(angle)
                if i + 1 < d:
                    pe[pos][i + 1] = math.cos(angle)
        return pe

    # ==================== 前向传播 ====================

    def forward(self, X, return_attention=False):
        """
        前向传播

        Args:
            X: (T, F) 输入特征序列
            return_attention: 是否返回注意力权重

        Returns:
            red_logits (33,), blue_logits (16,)
            及可选的 attention_weights (T, T)
        """
        T_actual = len(X)
        if T_actual == 0:
            raise ValueError("Empty input sequence")

        # 1. 输入投影 + 位置编码
        # (T, F) → (T, D)
        H = []
        for t in range(T_actual):
            h_t = matmul_1d(X[t], self.W_in)
            h_t = add(h_t, self.b_in)
            # 加位置编码
            if t < self.T:
                h_t = add(h_t, self.pos_enc[t])
            H.append(h_t)

        # 2. Multi-Head Self-Attention
        attn_out, attn_weights = self._multihead_attention(H)

        # 3. Residual + LayerNorm
        H_attn = []
        for t in range(T_actual):
            h = add(H[t], attn_out[t])
            h = layer_norm(h, self.ln1_gamma, self.ln1_beta)
            H_attn.append(h)

        # 4. FFN (Position-wise)
        H_ffn = []
        for t in range(T_actual):
            # D → ffn_dim → D
            h = matmul_1d(H_attn[t], self.W_ff1)
            h = add(h, self.b_ff1)
            h = gelu(h)
            h = matmul_1d(h, self.W_ff2)
            h = add(h, self.b_ff2)
            H_ffn.append(h)

        # 5. Residual + LayerNorm
        H_out = []
        for t in range(T_actual):
            h = add(H_attn[t], H_ffn[t])
            h = layer_norm(h, self.ln2_gamma, self.ln2_beta)
            H_out.append(h)

        # 6. Mean Pooling over time → (D,)
        pooled = [0.0] * self.d_model
        for t in range(T_actual):
            for i in range(self.d_model):
                pooled[i] += H_out[t][i]
        for i in range(self.d_model):
            pooled[i] /= T_actual

        # 7. Output heads
        red_logits = add(matmul_1d(pooled, self.W_red), self.b_red)
        blue_logits = add(matmul_1d(pooled, self.W_blue), self.b_blue)

        if return_attention:
            return red_logits, blue_logits, attn_weights
        return red_logits, blue_logits

    def _multihead_attention(self, H):
        """
        多头自注意力

        Args:
            H: (T, D) 输入序列

        Returns:
            attn_out: (T, D) 注意力输出
            avg_weights: (T, T) 平均注意力权重 (用于可解释性)
        """
        T_actual = len(H)
        D = self.d_model

        all_head_outputs = []
        all_head_weights = []

        for head in range(self.n_heads):
            # 计算 Q, K, V
            # 每头使用 d_k = D // n_heads 维
            start = head * self.d_k
            end = start + self.d_k

            Q = [[0.0] * self.d_k for _ in range(T_actual)]
            K = [[0.0] * self.d_k for _ in range(T_actual)]
            V = [[0.0] * self.d_k for _ in range(T_actual)]

            for t in range(T_actual):
                h_t = H[t]
                # 投影并切片
                q_full = matmul_1d(h_t, self.W_q)
                k_full = matmul_1d(h_t, self.W_k)
                v_full = matmul_1d(h_t, self.W_v)
                for i in range(self.d_k):
                    Q[t][i] = q_full[start + i]
                    K[t][i] = k_full[start + i]
                    V[t][i] = v_full[start + i]

            # Scaled Dot-Product Attention
            # scores = Q @ K^T / sqrt(d_k) → (T, T)
            scale = math.sqrt(self.d_k)
            scores = [[0.0] * T_actual for _ in range(T_actual)]
            for i in range(T_actual):
                for j in range(T_actual):
                    s = 0.0
                    for d in range(self.d_k):
                        s += Q[i][d] * K[j][d]
                    scores[i][j] = s / scale

            # Softmax over keys (dim=1)
            weights = softmax_2d(scores)
            all_head_weights.append(weights)

            # Output = weights @ V → (T, d_k)
            head_out = [[0.0] * self.d_k for _ in range(T_actual)]
            for i in range(T_actual):
                for j in range(T_actual):
                    w = weights[i][j]
                    for d in range(self.d_k):
                        head_out[i][d] += w * V[j][d]
            all_head_outputs.append(head_out)

        # Concatenate heads → (T, D)
        concat = [[0.0] * D for _ in range(T_actual)]
        for t in range(T_actual):
            for head in range(self.n_heads):
                start = head * self.d_k
                for d in range(self.d_k):
                    concat[t][start + d] = all_head_outputs[head][t][d]

        # Output projection
        attn_out = []
        for t in range(T_actual):
            out_t = matmul_1d(concat[t], self.W_o)
            attn_out.append(out_t)

        # 平均注意力权重
        avg_weights = [[0.0] * T_actual for _ in range(T_actual)]
        for i in range(T_actual):
            for j in range(T_actual):
                s = sum(all_head_weights[h][i][j] for h in range(self.n_heads))
                avg_weights[i][j] = s / self.n_heads

        return attn_out, avg_weights

    # ==================== 预测接口 ====================

    def predict_proba(self, X):
        """输出红球和蓝球的概率分布"""
        red_logits, blue_logits = self.forward(X)
        return softmax(red_logits), softmax(blue_logits)

    def predict(self, X):
        """输出最可能的号码"""
        red_probs, blue_probs = self.predict_proba(X)
        # Top 6 红球 (按概率降序)
        top6_idx = sorted(range(33), key=lambda i: red_probs[i], reverse=True)[:6]
        red_pred = sorted([i + 1 for i in top6_idx])
        blue_pred = max(range(16), key=lambda i: blue_probs[i]) + 1
        return red_pred, blue_pred

    # ==================== 损失与梯度 ====================

    def compute_loss(self, X, y_red_target, y_blue_target):
        """
        计算损失

        Args:
            X: (T, F) 特征序列
            y_red_target: (33,) multi-hot (6个1)
            y_blue_target: (16,) one-hot

        Returns:
            loss: float
        """
        red_logits, blue_logits = self.forward(X)
        red_probs = softmax(red_logits)
        blue_probs = softmax(blue_logits)

        eps = 1e-10
        red_loss = 0.0
        for i in range(33):
            if y_red_target[i] > 0:
                red_loss -= math.log(max(red_probs[i], eps))
        # 也惩罚过高预测的非目标球
        for i in range(33):
            if y_red_target[i] == 0:
                red_loss -= math.log(max(1.0 - red_probs[i], eps)) * 0.01

        blue_loss = -math.log(max(blue_probs[max(range(16),
                                                  key=lambda i: y_blue_target[i])], eps))

        return red_loss + blue_loss

    # ==================== 参数存取 ====================

    def get_params(self):
        """返回所有可训练参数的扁平化字典"""
        return {
            "W_in": self.W_in, "b_in": self.b_in,
            "W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v, "W_o": self.W_o,
            "W_ff1": self.W_ff1, "b_ff1": self.b_ff1,
            "W_ff2": self.W_ff2, "b_ff2": self.b_ff2,
            "W_red": self.W_red, "b_red": self.b_red,
            "W_blue": self.W_blue, "b_blue": self.b_blue,
            "ln1_gamma": self.ln1_gamma, "ln1_beta": self.ln1_beta,
            "ln2_gamma": self.ln2_gamma, "ln2_beta": self.ln2_beta,
        }

    def set_params(self, params):
        """设置参数"""
        for name, value in params.items():
            if hasattr(self, name):
                setattr(self, name, value)
