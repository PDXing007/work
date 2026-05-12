#!/usr/bin/env python3
"""
基础特征工程 — 单期开奖号码 → 可解释特征向量

覆盖:
- 数学统计: sum, mean, median, min, max, span, std
- 分布: 奇偶比, 区间分布(小/中/大), 012路, AC值, 间隔
- 位置: pos_1~pos_6 (排序后位置, 每个位置有不同分布)
- 频率统计: 结合历史频率信息

设计原则:
- 每个特征维度有明确含义
- 不做复杂非线性变换
- 所有特征可逆或可解释
"""

import math
from collections import Counter


# ==================== 红球基础数学特征 ====================

def red_numeric(reds):
    """数学统计特征 (6维)"""
    reds = sorted(reds)
    n = len(reds)
    mean_val = sum(reds) / n
    return {
        "sum": sum(reds),
        "mean": mean_val,
        "median": (reds[n // 2 - 1] + reds[n // 2]) / 2 if n % 2 == 0 else reds[n // 2],
        "min": reds[0],
        "max": reds[-1],
        "span": reds[-1] - reds[0],
        "std": math.sqrt(sum((x - mean_val) ** 2 for x in reds) / n),
    }


def red_parity(reds):
    """奇偶比 (2维)"""
    odds = sum(1 for x in reds if x % 2 == 1)
    return {"odd_count": odds, "even_count": len(reds) - odds, "odd_ratio": odds / len(reds)}


def red_zones(reds):
    """三区间分布: 小(1-11), 中(12-22), 大(23-33) (3维)"""
    z1 = sum(1 for x in reds if 1 <= x <= 11)
    z2 = sum(1 for x in reds if 12 <= x <= 22)
    z3 = sum(1 for x in reds if 23 <= x <= 33)
    return {"zone_small": z1, "zone_mid": z2, "zone_large": z3}


def red_mod3(reds):
    """除3余数(012路)分布 (3维)"""
    r0 = sum(1 for x in reds if x % 3 == 0)
    r1 = sum(1 for x in reds if x % 3 == 1)
    r2 = sum(1 for x in reds if x % 3 == 2)
    return {"mod3_0": r0, "mod3_1": r1, "mod3_2": r2}


def red_gaps(reds):
    """相邻号码间隔特征 (4维)"""
    reds = sorted(reds)
    gaps = [reds[i + 1] - reds[i] for i in range(len(reds) - 1)]
    return {
        "max_gap": max(gaps) if gaps else 0,
        "min_gap": min(gaps) if gaps else 0,
        "mean_gap": sum(gaps) / len(gaps) if gaps else 0,
        "gap_range": (max(gaps) - min(gaps)) if gaps else 0,
    }


def ac_value(reds):
    """
    AC值 = 任意两号码差值的去重个数 - (号码数 - 1)
    范围 0~10，反映号码离散度
    """
    reds = sorted(reds)
    diffs = set()
    for i in range(len(reds)):
        for j in range(i + 1, len(reds)):
            diffs.add(reds[j] - reds[i])
    return len(diffs) - (len(reds) - 1)


def red_positional(reds):
    """排序后每个位置的值 (6维) — 位置分布是最强特征之一"""
    reds = sorted(reds)
    return {f"pos_{i + 1}": reds[i] for i in range(len(reds))}


# ==================== 蓝球特征 ====================

def blue_features(blue):
    """蓝球特征 (4维)"""
    return {
        "blue": blue,
        "blue_parity": 1 if blue % 2 == 1 else 0,
        "blue_mod3": blue % 3,
        "blue_zone": (0 if blue <= 4 else 1 if blue <= 8 else 2 if blue <= 12 else 3),
    }


# ==================== 频率相关特征 ====================

def red_frequency_features(reds, freq_red, total_draws):
    """基于历史频率的红球特征 (3维)"""
    avg_freq = sum(freq_red.get(r, 0) for r in reds) / len(reds)
    freq_list = [freq_red.get(r, 0) for r in reds]
    total_red_balls = total_draws * len(reds)
    return {
        "red_avg_freq": avg_freq,
        "red_max_freq": max(freq_list),
        "red_min_freq": min(freq_list),
        "red_freq_range": max(freq_list) - min(freq_list),
        "red_prob_product": math.prod(freq_red.get(r, 0) / max(total_red_balls, 1) for r in reds) * (1e15),
    }


def blue_freq_feature(blue, freq_blue, total_draws):
    """蓝球频率特征"""
    return {"blue_freq": freq_blue.get(blue, 0) / max(total_draws, 1)}


# ==================== 单期编码器 ====================

def encode_draw(reds, blue, freq_red=None, freq_blue=None, total_draws=None):
    """
    将一期开奖号码编码为完整特征向量

    Args:
        reds: [int] 6个红球
        blue: int 蓝球
        freq_red: {ball: count} 红球历史频率 (可选)
        freq_blue: {ball: count} 蓝球历史频率 (可选)
        total_draws: int 总期数 (可选)

    Returns:
        dict: 扁平化特征字典
    """
    features = {}

    # 红球基础特征
    for name, feat_fn in [
        ("red", red_numeric),
        ("red", red_parity),
        ("red", red_zones),
        ("red", red_mod3),
        ("red", red_gaps),
    ]:
        result = feat_fn(reds)
        for k, v in result.items():
            features[f"{name}_{k}"] = v

    features["red_ac"] = ac_value(reds)

    # 位置特征
    pos = red_positional(reds)
    features.update(pos)

    # 蓝球特征
    blue_feat = blue_features(blue)
    features.update(blue_feat)

    # 频率特征
    if freq_red and freq_blue and total_draws:
        freq_feat = red_frequency_features(reds, freq_red, total_draws)
        features.update(freq_feat)
        features.update(blue_freq_feature(blue, freq_blue, total_draws))

    return features


def feature_vector(feat_dict, feature_names=None):
    """将特征字典转为数值列表"""
    if feature_names:
        return [feat_dict.get(name, 0.0) for name in feature_names]
    return [v for v in feat_dict.values() if isinstance(v, (int, float))]


# ==================== 全局统计计算 ====================

def compute_global_frequencies(data):
    """
    从历史数据计算频率统计 (无时间衰减)

    Args:
        data: [{"红球": [int], "蓝球": int}, ...]

    Returns:
        freq_red: {1..33: count}
        freq_blue: {1..16: count}
        total: int
    """
    freq_red = Counter()
    freq_blue = Counter()
    for d in data:
        for r in d["红球"]:
            freq_red[r] += 1
        freq_blue[d["蓝球"]] += 1
    for i in range(1, 34):
        freq_red.setdefault(i, 0)
    for i in range(1, 17):
        freq_blue.setdefault(i, 0)
    return dict(freq_red), dict(freq_blue), len(data)
