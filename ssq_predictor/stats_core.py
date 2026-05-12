#!/usr/bin/env python3
"""
统计核心 — 时间衰减频率、条件概率矩阵、特征分布拟合

所有统计都支持时间衰减参数 half_life。
"""

import math
from collections import Counter, defaultdict


def compute_weighted_freq(data, half_life=200):
    """
    时间衰减加权频率

    Args:
        data: 时间倒序列表
        half_life: 半衰期(期数), 0=无衰减

    Returns:
        freq_red: {1..33: weighted_count}
        freq_blue: {1..16: weighted_count}
        total_weight: float
    """
    freq_red = Counter()
    freq_blue = Counter()
    total_w = 0.0

    for i, record in enumerate(data):
        w = 1.0 if half_life <= 0 else math.exp(-math.log(2) * i / half_life)
        total_w += w
        for r in record["红球"]:
            freq_red[r] += w
        freq_blue[record["蓝球"]] += w

    for i in range(1, 34):
        freq_red.setdefault(i, 0.0)
    for i in range(1, 17):
        freq_blue.setdefault(i, 0.0)

    return dict(freq_red), dict(freq_blue), total_w


def compute_conditional_prob(data, half_life=200):
    """
    时间衰减条件概率矩阵 P(B|A)

    对于非衰减模式 (half_life=0):
    P(B|A) = count(A,B同时出现) / count(A出现)，行和=5

    对于衰减模式:
    加权版本

    Returns:
        cond_red: {a: {b: P(b|a)}}
        cond_blue: {b: P(b)}  (蓝球每期只有一个, 退化为边际概率)
    """
    n = len(data)

    co_occur = defaultdict(lambda: Counter())
    count_red = Counter()
    total_weight = 0.0

    for i, record in enumerate(data):
        w = 1.0 if half_life <= 0 else math.exp(-math.log(2) * i / half_life)
        total_weight += w
        reds = record["红球"]
        for a in reds:
            count_red[a] += w
            for b in reds:
                if a != b:
                    co_occur[a][b] += w

    cond_red = {}
    for a in range(1, 34):
        cond_red[a] = {}
        total_a = max(count_red.get(a, 0), 0.001)
        for b in range(1, 34):
            if a == b:
                cond_red[a][b] = 0.0
            else:
                cond_red[a][b] = co_occur[a].get(b, 0.0) / total_a

    freq_blue = Counter()
    for i, record in enumerate(data):
        w = 1.0 if half_life <= 0 else math.exp(-math.log(2) * i / half_life)
        freq_blue[record["蓝球"]] += w

    cond_blue = {b: freq_blue.get(b, 0.0) / max(total_weight, 0.001) for b in range(1, 17)}

    return cond_red, cond_blue


def compute_co_occurrence_matrix(data, half_life=200):
    """
    共现矩阵: M[a][b] = P(a和b同在一期)
    用于关联规则挖掘的补充
    """
    n = len(data)
    co_occur = defaultdict(lambda: Counter())
    total_weight = 0.0

    for i, record in enumerate(data):
        w = 1.0 if half_life <= 0 else math.exp(-math.log(2) * i / half_life)
        total_weight += w
        reds = record["红球"]
        for a in reds:
            for b in reds:
                co_occur[a][b] += w

    matrix = {}
    for a in range(1, 34):
        matrix[a] = {}
        for b in range(1, 34):
            matrix[a][b] = co_occur[a].get(b, 0.0) / max(total_weight, 0.001)

    return matrix


# ==================== 特征分布拟合 ====================

def fit_sum_distribution(data):
    """拟合红球和值的正态分布参数"""
    sums = [sum(d["红球"]) for d in data]
    n = len(sums)
    mean_val = sum(sums) / n
    var_val = sum((s - mean_val) ** 2 for s in sums) / n
    return {"mean": mean_val, "std": math.sqrt(var_val), "min": min(sums), "max": max(sums)}


def fit_span_distribution(data):
    """拟合跨度分布: 经验离散分布"""
    spans = [max(d["红球"]) - min(d["红球"]) for d in data]
    span_counts = Counter(spans)
    n = len(data)
    return {s: c / n for s, c in span_counts.items()}


def fit_parity_distribution(data):
    """拟合奇偶比分布"""
    parity_counts = Counter()
    for d in data:
        odds = sum(1 for x in d["红球"] if x % 2 == 1)
        parity_counts[odds] += 1
    n = len(data)
    return {k: v / n for k, v in parity_counts.items()}


def fit_zone_distribution(data):
    """拟合区间分布 (z1, z2, z3)"""
    zone_counts = Counter()
    for d in data:
        reds = d["红球"]
        z1 = sum(1 for x in reds if 1 <= x <= 11)
        z2 = sum(1 for x in reds if 12 <= x <= 22)
        z3 = sum(1 for x in reds if 23 <= x <= 33)
        zone_counts[(z1, z2, z3)] += 1
    n = len(data)
    return {k: v / n for k, v in zone_counts.items()}


def fit_ac_distribution(data):
    """拟合AC值分布"""
    def ac_val(reds):
        r = sorted(reds)
        diffs = set()
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                diffs.add(r[j] - r[i])
        return len(diffs) - (len(r) - 1)

    ac_counts = Counter()
    for d in data:
        ac_counts[ac_val(d["红球"])] += 1
    n = len(data)
    return {k: v / n for k, v in ac_counts.items()}


def fit_position_distributions(data):
    """
    拟合每个排序位置 (pos_1 ~ pos_6) 的经验分布

    这是最强的单变量特征 — 位置分布非常不均匀
    例如 pos_1 (最小值) 从未超过 15, pos_6 (最大值) 从未低于 18
    """
    n = len(data)
    pos_values = {i: [] for i in range(1, 7)}
    for d in data:
        reds = sorted(d["红球"])
        for i, v in enumerate(reds):
            pos_values[i + 1].append(v)

    pos_dist = {}
    for pos, values in pos_values.items():
        cnt = Counter(values)
        pos_dist[pos] = {v: c / n for v, c in cnt.items()}

    return pos_dist


def fit_gap_distribution(data):
    """拟合间隔分布 (相邻号码差值)"""
    all_gaps = []
    for d in data:
        reds = sorted(d["红球"])
        for i in range(len(reds) - 1):
            all_gaps.append(reds[i + 1] - reds[i])

    gap_counts = Counter(all_gaps)
    total = len(all_gaps)
    return {k: v / total for k, v in gap_counts.items()}


def fit_all_distributions(data):
    """一次性拟合所有特征分布"""
    return {
        "sum": fit_sum_distribution(data),
        "span": fit_span_distribution(data),
        "parity": fit_parity_distribution(data),
        "zone": fit_zone_distribution(data),
        "ac": fit_ac_distribution(data),
        "position": fit_position_distributions(data),
        "gap": fit_gap_distribution(data),
    }


# ==================== 特征相关性 ====================

def compute_feature_correlation(feat_vectors, feat_names):
    """
    计算特征之间的 Pearson 相关系数矩阵

    Args:
        feat_vectors: (n_samples, n_features)
        feat_names: [str]

    Returns:
        {feat_name: {other_feat: correlation}}
    """
    n = len(feat_vectors)
    m = len(feat_names)

    # 计算每个特征的均值和标准差
    means = [0.0] * m
    stds = [0.0] * m
    for j in range(m):
        vals = [feat_vectors[i][j] for i in range(n)]
        means[j] = sum(vals) / n
        stds[j] = math.sqrt(sum((v - means[j]) ** 2 for v in vals) / n)

    corr = {}
    for j1 in range(m):
        corr[feat_names[j1]] = {}
        for j2 in range(m):
            if j1 == j2:
                corr[feat_names[j1]][feat_names[j2]] = 1.0
            elif j2 < j1:
                corr[feat_names[j1]][feat_names[j2]] = corr[feat_names[j2]][feat_names[j1]]
            else:
                if stds[j1] == 0 or stds[j2] == 0:
                    corr[feat_names[j1]][feat_names[j2]] = 0.0
                else:
                    cov = sum((feat_vectors[i][j1] - means[j1]) *
                              (feat_vectors[i][j2] - means[j2]) for i in range(n)) / n
                    corr[feat_names[j1]][feat_names[j2]] = cov / (stds[j1] * stds[j2])

    return corr


# ==================== 马尔可夫转移矩阵 ====================

def compute_markov_transition(data, order=1, half_life=200):
    """
    计算红球的马尔可夫转移概率 (基于相邻期)

    Args:
        data: 时间倒序列表
        order: 马尔可夫阶数 (1=只看上一期, 2=看前两期)
        half_life: 时间衰减半衰期

    Returns:
        trans: {prev_ball(s): {next_ball: probability}}
    """
    total_weight = 0.0
    trans = defaultdict(lambda: Counter())

    for i in range(len(data) - order):
        w = 1.0 if half_life <= 0 else math.exp(-math.log(2) * i / half_life)
        total_weight += w

        # 前 order 期的球
        prev_balls = set()
        for o in range(order):
            prev_balls.update(data[i + 1 + o]["红球"])

        current_balls = data[i]["红球"]

        for pb in prev_balls:
            for cb in current_balls:
                trans[pb][cb] += w

    # 归一化
    for pb in trans:
        total_pb = sum(trans[pb].values())
        if total_pb > 0:
            for cb in trans[pb]:
                trans[pb][cb] /= total_pb

    return dict(trans)


def compute_blue_transition(data, half_life=200):
    """蓝球马尔可夫转移概率"""
    total_weight = 0.0
    trans = defaultdict(lambda: Counter())

    for i in range(len(data) - 1):
        w = 1.0 if half_life <= 0 else math.exp(-math.log(2) * i / half_life)
        total_weight += w
        prev_blue = data[i + 1]["蓝球"]
        curr_blue = data[i]["蓝球"]
        trans[prev_blue][curr_blue] += w

    for pb in trans:
        total_pb = sum(trans[pb].values())
        if total_pb > 0:
            for cb in trans[pb]:
                trans[pb][cb] /= total_pb

    return dict(trans)
