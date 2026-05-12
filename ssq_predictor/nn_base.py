#!/usr/bin/env python3
"""
纯 NumPy 后端 — 所有矩阵运算、激活函数、归一化、优化器

设计原则:
- 零外部依赖
- 列表嵌套列表表示矩阵 (row-major)
- 所有函数支持标量和嵌套列表
- 与 NumPy API 兼容的命名以便未来迁移
"""

import math
import random


# ==================== 基础线性代数 ====================

def matmul_1d(A, B):
    """向量 @ 矩阵: x(f,) @ W(f,h) → (h,)"""
    f = len(A)
    h = len(B[0])
    result = [0.0] * h
    for j in range(h):
        s = 0.0
        for i in range(f):
            s += A[i] * B[i][j]
        result[j] = s
    return result


def matmul_2d(A, B):
    """矩阵 @ 矩阵: A(m,k) @ B(k,n) → (m,n)"""
    m, k = len(A), len(A[0])
    n = len(B[0])
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            result[i][j] = s
    return result


def outer(a, b):
    """外积: a(m,) ⊗ b(n,) → (m,n)"""
    return [[ai * bj for bj in b] for ai in a]


def transpose(A):
    """矩阵转置"""
    if not A or not isinstance(A[0], list):
        return [[x] for x in A]
    rows, cols = len(A), len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def add(A, B):
    """逐元素加法: 支持 scalar, 1D, 2D"""
    if isinstance(A, list) and A and isinstance(A[0], list):
        if isinstance(B, list) and B and isinstance(B[0], list):
            return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
        return [[A[i][j] + B for j in range(len(A[0]))] for i in range(len(A))]
    if isinstance(B, list) and B and isinstance(B[0], list):
        return [[A + B[i][j] for j in range(len(B[0]))] for i in range(len(B))]
    if isinstance(A, list):
        if isinstance(B, list):
            return [a + b for a, b in zip(A, B)]
        return [a + B for a in A]
    if isinstance(B, list):
        return [A + b for b in B]
    return A + B


def sub(A, B):
    """逐元素减法"""
    if isinstance(A, list) and A and isinstance(A[0], list):
        if isinstance(B, list) and B and isinstance(B[0], list):
            return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
        return [[A[i][j] - B for j in range(len(A[0]))] for i in range(len(A))]
    if isinstance(B, list) and B and isinstance(B[0], list):
        return [[A - B[i][j] for j in range(len(B[0]))] for i in range(len(B))]
    if isinstance(A, list):
        if isinstance(B, list):
            return [a - b for a, b in zip(A, B)]
        return [a - B for a in A]
    if isinstance(B, list):
        return [A - b for b in B]
    return A - B


def mul(A, s):
    """标量乘法"""
    if isinstance(A, list) and A and isinstance(A[0], list):
        return [[x * s for x in row] for row in A]
    if isinstance(A, list):
        return [x * s for x in A]
    return A * s


def elem_mul(A, B):
    """逐元素乘法"""
    if isinstance(A, list) and A and isinstance(A[0], list):
        if isinstance(B, list) and B and isinstance(B[0], list):
            return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
        return [[A[i][j] * B for j in range(len(A[0]))] for i in range(len(A))]
    if isinstance(A, list):
        if isinstance(B, list):
            return [a * b for a, b in zip(A, B)]
        return [a * B for a in A]
    return A * B


def elem_div(A, B, eps=1e-8):
    """逐元素除法"""
    if isinstance(A, list) and A and isinstance(A[0], list):
        if isinstance(B, list) and B and isinstance(B[0], list):
            return [[A[i][j] / (B[i][j] + eps) for j in range(len(A[0]))] for i in range(len(A))]
        return [[A[i][j] / (B + eps) for j in range(len(A[0]))] for i in range(len(A))]
    if isinstance(A, list):
        if isinstance(B, list):
            return [a / (b + eps) for a, b in zip(A, B)]
        return [a / (B + eps) for a in A]
    return A / (B + eps)


def sqrt(A):
    """逐元素平方根"""
    if isinstance(A, list) and A and isinstance(A[0], list):
        return [[math.sqrt(x) for x in row] for row in A]
    if isinstance(A, list):
        return [math.sqrt(x) for x in A]
    return math.sqrt(A)


def sum_all(A, axis=None):
    """求和"""
    if isinstance(A, list) and A and isinstance(A[0], list):
        if axis == 0:
            return [sum(A[i][j] for i in range(len(A))) for j in range(len(A[0]))]
        if axis == 1:
            return [sum(row) for row in A]
        return sum(sum(row) for row in A)
    return sum(A)


def mean(A, axis=None):
    """均值"""
    if isinstance(A, list) and A and isinstance(A[0], list):
        n_rows, n_cols = len(A), len(A[0])
        if axis == 0:
            return [sum(A[i][j] for i in range(n_rows)) / n_rows for j in range(n_cols)]
        if axis == 1:
            return [sum(row) / n_cols for row in A]
        return sum_all(A) / (n_rows * n_cols)
    return sum(A) / len(A)


def var(A, axis=None):
    """方差 (总体)"""
    m = mean(A, axis)
    if isinstance(A, list) and A and isinstance(A[0], list):
        if axis == 0:
            n_rows = len(A)
            return [sum((A[i][j] - m[j]) ** 2 for i in range(n_rows)) / n_rows for j in range(len(A[0]))]
        if axis == 1:
            n_cols = len(A[0])
            return [sum((x - m[i]) ** 2 for x in A[i]) / n_cols for i in range(len(A))]
        n = len(A) * len(A[0])
        return sum((A[i][j] - m) ** 2 for i in range(len(A)) for j in range(len(A[0]))) / n
    n = len(A)
    return sum((x - m) ** 2 for x in A) / n


def std(A, axis=None):
    """标准差"""
    return sqrt(var(A, axis))


# ==================== 激活函数 ====================

def relu(x):
    """ReLU"""
    if isinstance(x, list) and x and isinstance(x[0], list):
        return [[max(0, v) for v in row] for row in x]
    return [max(0, v) for v in x]


def gelu(x):
    """GELU: x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x^3)))"""
    c = math.sqrt(2.0 / math.pi)
    if isinstance(x, list) and x and isinstance(x[0], list):
        return [[0.5 * v * (1.0 + math.tanh(c * (v + 0.044715 * v * v * v)))
                 for v in row] for row in x]
    if isinstance(x, list):
        return [0.5 * v * (1.0 + math.tanh(c * (v + 0.044715 * v * v * v))) for v in x]
    return 0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x * x * x)))


def sigmoid(x):
    """Sigmoid"""
    if isinstance(x, list) and x and isinstance(x[0], list):
        return [[1.0 / (1.0 + math.exp(-v)) for v in row] for row in x]
    if isinstance(x, list):
        return [1.0 / (1.0 + math.exp(-v)) for v in x]
    return 1.0 / (1.0 + math.exp(-x))


def tanh_list(x):
    """Tanh"""
    if isinstance(x, list) and x and isinstance(x[0], list):
        return [[math.tanh(v) for v in row] for row in x]
    if isinstance(x, list):
        return [math.tanh(v) for v in x]
    return math.tanh(x)


# ==================== 归一化 ====================

def softmax(x):
    """Softmax: (n,) → (n,)"""
    max_v = max(x)
    exps = [math.exp(v - max_v) for v in x]
    s = sum(exps)
    return [e / s for e in exps]


def softmax_2d(X):
    """Softmax over last axis: (m,n) → (m,n)"""
    result = []
    for row in X:
        max_v = max(row)
        exps = [math.exp(v - max_v) for v in row]
        s = sum(exps)
        result.append([e / s for e in exps])
    return result


def log_softmax(x):
    """Log-Softmax"""
    max_v = max(x)
    exps = [math.exp(v - max_v) for v in x]
    s = sum(exps)
    log_s = math.log(s)
    return [v - max_v - log_s for v in x]


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """
    Layer Normalization: (d,) → (d,)
    Normalize to mean=0, var=1, then scale and shift.
    """
    d = len(x)
    mu = sum(x) / d
    sigma_sq = sum((xi - mu) ** 2 for xi in x) / d
    sigma = math.sqrt(sigma_sq + eps)
    inv_sigma = 1.0 / sigma

    if gamma is None:
        gamma = [1.0] * d
    if beta is None:
        beta = [0.0] * d

    return [(x[i] - mu) * inv_sigma * gamma[i] + beta[i] for i in range(d)]


# ==================== 丢弃 ====================

def dropout_mask(shape, rate):
    """生成 dropout mask"""
    if isinstance(shape, tuple):
        if len(shape) == 2:
            return [[1.0 if random.random() > rate else 0.0
                     for _ in range(shape[1])] for _ in range(shape[0])]
        return [1.0 if random.random() > rate else 0.0 for _ in range(shape[0])]
    return [1.0 if random.random() > rate else 0.0 for _ in range(shape)]


def apply_dropout(x, mask, scale=None):
    """应用 dropout mask 并缩放"""
    if scale is None:
        scale = 1.0 / max(1e-8, 1.0 - sum(1.0 for m in mask if m == 0.0) / max(1, len(mask)))
    if isinstance(x, list) and x and isinstance(x[0], list):
        return [[x[i][j] * mask[i][j] * scale for j in range(len(x[0]))] for i in range(len(x))]
    return [x[i] * mask[i] * scale for i in range(len(x))]


# ==================== 损失函数 ====================

def cross_entropy_loss(logits, target):
    """交叉熵: logits(n,) + target(n,) one-hot → scalar"""
    probs = softmax(logits)
    eps = 1e-10
    return -sum(target[i] * math.log(probs[i] + eps) for i in range(len(target)))


def binary_cross_entropy(probs, targets):
    """二元交叉熵 (multi-label)"""
    eps = 1e-10
    n = len(probs)
    loss = 0.0
    for i in range(n):
        loss += -targets[i] * math.log(probs[i] + eps) - (1 - targets[i]) * math.log(1 - probs[i] + eps)
    return loss / n


# ==================== 随机初始化 ====================

def random_normal(shape, mean=0.0, std=1.0):
    """正态随机初始化"""
    rows, cols = shape
    return [[random.gauss(mean, std) for _ in range(cols)] for _ in range(rows)]


def xavier_init(shape):
    """Xavier/Glorot 初始化: U(-limit, limit) where limit = sqrt(6/(fan_in+fan_out))"""
    rows, cols = shape
    limit = math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def he_init(shape):
    """He 初始化: N(0, sqrt(2/fan_in))"""
    rows, cols = shape
    std = math.sqrt(2.0 / rows)
    return [[random.gauss(0.0, std) for _ in range(cols)] for _ in range(rows)]


def zeros(shape):
    """全零矩阵"""
    if isinstance(shape, int):
        return [0.0] * shape
    if len(shape) == 1:
        return [0.0] * shape[0]
    rows, cols = shape
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def ones(shape):
    """全一矩阵"""
    if isinstance(shape, int):
        return [1.0] * shape
    if len(shape) == 1:
        return [1.0] * shape[0]
    rows, cols = shape
    return [[1.0 for _ in range(cols)] for _ in range(rows)]


# ==================== 形状操作 ====================

def shape(x):
    """返回张量形状"""
    if isinstance(x, list) and x and isinstance(x[0], list):
        return (len(x), len(x[0]))
    if isinstance(x, list):
        return (len(x),)
    return ()


def reshape_1d_to_2d(x):
    """(n,) → (n,1)"""
    return [[xi] for xi in x]


def flatten(x):
    """2D → 1D"""
    if x and isinstance(x[0], list):
        return [v for row in x for v in row]
    return x


# ==================== 优化器 ====================

class AdamW:
    """AdamW: Adam with decoupled weight decay"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def init_param(self, name, param):
        """初始化参数的动量缓存"""
        self.m[name] = zeros(shape(param))
        self.v[name] = zeros(shape(param))

    def step(self, name, param, grad):
        """更新一个参数，返回更新后的值"""
        self.t += 1

        if name not in self.m:
            self.init_param(name, param)

        m_t = self.m[name]
        v_t = self.v[name]

        # 更新动量
        m_new = add(mul(m_t, self.beta1), mul(grad, 1.0 - self.beta1))
        v_new = add(mul(v_t, self.beta2), mul(elem_mul(grad, grad), 1.0 - self.beta2))
        self.m[name] = m_new
        self.v[name] = v_new

        # 偏差修正
        m_hat = mul(m_new, 1.0 / (1.0 - self.beta1 ** self.t))
        v_hat = mul(v_new, 1.0 / (1.0 - self.beta2 ** self.t))

        # AdamW: 先对参数做 weight decay，再应用 Adam 更新
        param_wd = sub(param, mul(param, self.lr * self.weight_decay))

        step = elem_div(mul(m_hat, self.lr), add(sqrt(v_hat), self.eps))
        return sub(param_wd, step)


# ==================== 实用工具 ====================

def argmax(x):
    """返回最大值的索引"""
    return max(range(len(x)), key=lambda i: x[i])


def argsort(x, reverse=False):
    """返回排序后的索引"""
    return sorted(range(len(x)), key=lambda i: x[i], reverse=reverse)


def clip_gradients(grads, max_norm):
    """梯度裁剪"""
    total_norm_sq = 0.0
    for g in grads.values():
        if isinstance(g, list) and g and isinstance(g[0], list):
            total_norm_sq += sum(x * x for row in g for x in row)
        elif isinstance(g, list):
            total_norm_sq += sum(x * x for x in g)
        else:
            total_norm_sq += g * g

    total_norm = math.sqrt(total_norm_sq)
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for name in grads:
            grads[name] = mul(grads[name], scale)
    return grads


def one_hot(idx, n_classes):
    """生成 one-hot 向量"""
    vec = [0.0] * n_classes
    vec[idx] = 1.0
    return vec


def multi_hot(indices, n_classes):
    """生成 multi-hot 向量"""
    vec = [0.0] * n_classes
    for idx in indices:
        vec[idx] = 1.0
    return vec
