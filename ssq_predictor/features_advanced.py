#!/usr/bin/env python3
"""
高级特征工程 — 连号、质数、尾数、扩展模、对称性、和值跨度比

每个特征维度都有明确的可解释含义。
"""

import math


# 红球范围 1-33 中的质数
PRIMES_1_33 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
PRIME_COUNT = len(PRIMES_1_33)  # 11


# ==================== 连号特征 ====================

def red_consecutive(reds):
    """
    连号检测:
    - 相邻号码差值为1的对数 (consecutive_pairs)
    - 最长连续序列长度 (max_run)
    - 是否有两组分开的连号 (has_double_pair)
    - 三连号及以上数量 (triple_plus)
    """
    reds = sorted(reds)
    gaps = [reds[i + 1] - reds[i] for i in range(len(reds) - 1)]
    consecutive_pairs = sum(1 for g in gaps if g == 1)

    max_run = 1
    current_run = 1
    for g in gaps:
        if g == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    # 检查是否有两组独立连号
    pair_runs = 0
    i = 0
    while i < len(gaps):
        if gaps[i] == 1:
            pair_runs += 1
            while i < len(gaps) and gaps[i] == 1:
                i += 1
        i += 1

    return {
        "consecutive_pairs": consecutive_pairs,
        "max_consecutive_run": max_run,
        "has_double_consecutive": 1 if pair_runs >= 2 else 0,
        "triple_consecutive": 1 if max_run >= 3 else 0,
        "consecutive_run_count": pair_runs,
    }


# ==================== 质数特征 ====================

def red_prime(reds):
    """质数特征: 1-33中有11个质数"""
    prime_count = sum(1 for x in reds if x in PRIMES_1_33)
    return {
        "prime_count": prime_count,
        "prime_ratio": prime_count / len(reds),
        "nonprime_count": len(reds) - prime_count,
    }


# ==================== 尾数特征 ====================

def red_tail(reds):
    """
    尾数(末位数字)分布:
    - unique_tails: 不同尾数的个数 (范围 1-6)
    - max_tail_freq: 同一尾数最大出现次数
    - 每个尾数0-9的出现次数
    """
    tails = [x % 10 for x in reds]
    unique = len(set(tails))
    tail_counts = {}
    for t in range(10):
        tail_counts[f"tail_{t}"] = tails.count(t)

    max_tail_freq = max(tail_counts.values())

    return {
        "unique_tails": unique,
        "max_tail_freq": max_tail_freq,
        "has_tail_triple": 1 if max_tail_freq >= 3 else 0,
        **tail_counts,
    }


# ==================== 扩展模运算特征 ====================

def red_mod_extended(reds):
    """mod 4/5/6 的余数分布"""
    features = {}

    for mod_val in [4, 5, 6]:
        counts = [0] * mod_val
        for x in reds:
            counts[x % mod_val] += 1
        for r in range(mod_val):
            features[f"mod{mod_val}_{r}"] = counts[r]

        # 余数均匀度 (反熵)
        expected = len(reds) / mod_val
        entropy = -sum((c / len(reds)) * math.log(max(c, 0.001) / len(reds))
                       for c in counts)
        max_entropy = math.log(mod_val)
        features[f"mod{mod_val}_entropy"] = entropy / max_entropy if max_entropy > 0 else 0

    return features


# ==================== 对称性特征 ====================

def red_symmetry(reds):
    """关于中心点17的对称性"""
    center = 17
    left = sum(1 for x in reds if x < center)
    right = sum(1 for x in reds if x > center)
    at_center = sum(1 for x in reds if x == center)
    return {
        "symmetry_left": left,
        "symmetry_right": right,
        "symmetry_center": at_center,
        "symmetry_diff": abs(left - right),
        "symmetry_balanced": 1 if abs(left - right) <= 1 else 0,
    }


# ==================== 跨度比例特征 ====================

def red_span_ratios(reds):
    """
    跨度内部比例:
    - 前1/3跨度 / 总跨度
    - 后1/3跨度 / 总跨度
    - 中段占比
    """
    reds = sorted(reds)
    total_span = reds[-1] - reds[0]
    if total_span == 0:
        return {
            "span_first_third": 0,
            "span_last_third": 0,
            "span_middle_ratio": 0,
        }

    n = len(reds)
    first_span = reds[n // 3] - reds[0]
    last_span = reds[-1] - reds[-(n // 3) - 1]

    return {
        "span_first_third": first_span / total_span,
        "span_last_third": last_span / total_span,
        "span_middle_ratio": (total_span - first_span - last_span) / total_span,
    }


# ==================== 质数间距特征 ====================

def red_prime_gaps(reds):
    """红球到最近质数的距离之和"""
    prime_list = sorted(PRIMES_1_33)
    total_dist = 0
    for x in reds:
        min_dist = min(abs(x - p) for p in prime_list)
        total_dist += min_dist
    return {
        "prime_distance_sum": total_dist,
        "prime_distance_mean": total_dist / len(reds),
    }


# ==================== 组合编码器 ====================

def encode_draw_advanced(reds):
    """将高级特征全部编码为一个扁平字典"""
    features = {}
    for fn in [
        red_consecutive, red_prime, red_tail, red_mod_extended,
        red_symmetry, red_span_ratios, red_prime_gaps,
    ]:
        result = fn(reds)
        features.update(result)
    return features
