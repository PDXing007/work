#!/usr/bin/env python3
"""
时序特征工程 — EMA趋势、波动率、多尺度窗口、时间衰减频率

设计原则:
- 所有统计均有时间衰减参数 half_life (半衰期, 单位: 期)
- 多尺度: 短期(5-13期), 中期(20-52期), 长期(100-200期)
- 每个特征在时间序列上计算，无未来数据泄露
"""

import math
from collections import Counter


# ==================== 时间衰减工具 ====================

def decay_weight(age, half_life):
    """exp(-ln(2) * age / half_life)"""
    if half_life <= 0:
        return 1.0
    return math.exp(-math.log(2) * age / half_life)


def decay_weights(n, half_life):
    """生成长度为n的衰减权重序列, weights[0] = 1 (最近期)"""
    return [decay_weight(i, half_life) for i in range(n)]


# ==================== 时间衰减频率 ====================

def compute_weighted_frequencies(data, half_life=200):
    """
    时间衰减频率统计

    Args:
        data: [{红球: [int], 蓝球: int}, ...] 时间倒序
        half_life: 半衰期(期数)

    Returns:
        freq_red: {1..33: weighted_count}
        freq_blue: {1..16: weighted_count}
        effective_n: 有效样本量 (权重和)
    """
    freq_red = Counter()
    freq_blue = Counter()
    total_weight = 0.0

    for i, record in enumerate(data):
        w = decay_weight(i, half_life)
        total_weight += w
        for r in record["红球"]:
            freq_red[r] += w
        freq_blue[record["蓝球"]] += w

    for i in range(1, 34):
        freq_red.setdefault(i, 0.0)
    for i in range(1, 17):
        freq_blue.setdefault(i, 0.0)

    return (dict(freq_red), dict(freq_blue), total_weight)


# ==================== EMA 特征 ====================

def compute_ema(data, idx, windows, feature_fn, alpha_factors=None):
    """
    计算多窗口指数移动平均

    Args:
        data: 时间倒序列表
        idx: 当前期索引 (0=最新)
        windows: [5, 13, 26, 52] 不同窗口大小
        feature_fn: (record) → dict of features
        alpha_factors: 每窗口的alpha系数 (默认=2/(window+1))

    Returns:
        {f"ema_{w}_{feat}": value, ...}
    """
    features = {}
    for wi, w in enumerate(windows):
        alpha = alpha_factors[wi] if alpha_factors else 2.0 / (w + 1)

        # 从 idx+1 开始 (不含当前期)
        lookback = min(w, len(data) - idx - 1)
        if lookback == 0:
            continue

        ema = None
        for i in range(lookback):
            record = data[idx + 1 + i]
            feat = feature_fn(record)
            if ema is None:
                ema = feat
            else:
                for k, v in feat.items():
                    ema[k] = alpha * v + (1 - alpha) * ema[k]

        if ema:
            for k, v in ema.items():
                features[f"ema_{w}_{k}"] = v

    return features


# ==================== 趋势特征 ====================

def compute_trend(data, idx, windows, feature_fn):
    """
    线性趋势 (最小二乘斜率)

    Returns:
        {f"trend_{w}_{feat}": slope, ...}
    """
    features = {}
    for w in windows:
        lookback = min(w, len(data) - idx - 1)
        if lookback < 3:
            continue

        xs = list(range(lookback))
        ys_dict = {}
        for i in range(lookback):
            record = data[idx + 1 + i]
            feat = feature_fn(record)
            for k, v in feat.items():
                if k not in ys_dict:
                    ys_dict[k] = [0.0] * lookback
                ys_dict[k][i] = v

        n = lookback
        mean_x = (n - 1) / 2.0
        ss_xx = sum((i - mean_x) ** 2 for i in range(n))

        if ss_xx > 0:
            for feat_name, ys in ys_dict.items():
                mean_y = sum(ys) / n
                slope = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / ss_xx
                features[f"trend_{w}_{feat_name}"] = slope

    return features


# ==================== 波动率特征 ====================

def compute_volatility(data, idx, windows, feature_fn):
    """
    滚动标准差 (波动率)

    Returns:
        {f"vol_{w}_{feat}": std, ...}
    """
    features = {}
    for w in windows:
        lookback = min(w, len(data) - idx - 1)
        if lookback < 2:
            continue

        ys_dict = {}
        for i in range(lookback):
            record = data[idx + 1 + i]
            feat = feature_fn(record)
            for k, v in feat.items():
                if k not in ys_dict:
                    ys_dict[k] = []
                ys_dict[k].append(v)

        for feat_name, ys in ys_dict.items():
            n = len(ys)
            mean_y = sum(ys) / n
            var_y = sum((y - mean_y) ** 2 for y in ys) / n
            features[f"vol_{w}_{feat_name}"] = math.sqrt(var_y)

    return features


# ==================== 近期变化特征 ====================

def compute_recent_changes(data, idx, feature_fn):
    """
    最近1/2/3期的变化量

    Returns:
        {f"delta_{lag}_{feat}": diff, ...}
    """
    features = {}
    for lag in [1, 2, 3]:
        if idx + lag >= len(data):
            continue
        current_feat = feature_fn(data[idx])
        past_feat = feature_fn(data[idx + lag])
        for k in current_feat:
            if k in past_feat:
                features[f"delta_{lag}_{k}"] = current_feat[k] - past_feat[k]
    return features


# ==================== 遗漏值 (增强版) ====================

def compute_missing_values(data, idx, max_lookback=200):
    """
    计算当前期每个号码的遗漏期数 (距上一次出现)
    同时返回: 平均遗漏、最大遗漏、遗漏偏度

    Returns:
        dict: 遗漏相关特征
    """
    current = data[idx]
    current_reds = set(current["红球"])
    current_blue = current["蓝球"]

    red_missing = {}
    blue_missing = 0
    n_past = min(max_lookback, len(data) - idx - 1)

    # 红球遗漏
    for r in range(1, 34):
        for gap in range(n_past):
            record = data[idx + 1 + gap]
            if r in record["红球"]:
                red_missing[r] = gap + 1
                break
        else:
            red_missing[r] = n_past + 1

    # 蓝球遗漏
    for gap in range(n_past):
        record = data[idx + 1 + gap]
        if record["蓝球"] == current_blue:
            blue_missing = gap + 1
            break
    else:
        blue_missing = n_past + 1

    missing_values = list(red_missing.values())
    mean_m = sum(missing_values) / 33
    var_m = sum((v - mean_m) ** 2 for v in missing_values) / 33

    return {
        "red_avg_missing": sum(red_missing[r] for r in current_reds) / 6,
        "red_max_missing": max(red_missing[r] for r in current_reds),
        "red_min_missing": min(red_missing[r] for r in current_reds),
        "blue_missing": blue_missing,
        "global_avg_missing": mean_m,
        "global_missing_std": math.sqrt(var_m),
        "missing_skew": (sum((v - mean_m) ** 3 for v in missing_values) / 33 /
                         max(math.sqrt(var_m) ** 3, 0.001)),
        "hot_red_count": sum(1 for r in current_reds if red_missing[r] <= 3),
        "cold_red_count": sum(1 for r in current_reds if red_missing[r] >= 20),
    }


# ==================== 重叠相似度 ====================

def compute_overlap_similarity(data, idx, windows):
    """
    当前期与近N期的红球重叠数和和值差

    Returns:
        {f"sim_{w}b_avg_overlap": ..., ...}
    """
    features = {}
    current_reds_set = set(data[idx]["红球"])
    current_reds_sum = sum(data[idx]["红球"])

    for w in windows:
        lookback = min(w, len(data) - idx - 1)
        if lookback == 0:
            continue
        overlaps = []
        sum_diffs = []
        for i in range(lookback):
            past_reds = data[idx + 1 + i]["红球"]
            overlaps.append(len(current_reds_set & set(past_reds)))
            sum_diffs.append(current_reds_sum - sum(past_reds))

        features[f"sim_{lookback}b_avg_overlap"] = sum(overlaps) / lookback
        features[f"sim_{lookback}b_max_overlap"] = max(overlaps)
        features[f"sim_{lookback}b_avg_sum_diff"] = sum(sum_diffs) / lookback
        features[f"sim_{lookback}b_has_identical"] = 1 if max(overlaps) >= 5 else 0

    return features


# ==================== 重号特征 ====================

def compute_repeat_features(data, idx):
    """与上一期的重号情况"""
    if idx + 1 >= len(data):
        return {"repeat_red_count": 0, "repeat_blue": 0, "repeat_ratio": 0}
    current_reds = set(data[idx]["红球"])
    prev_reds = set(data[idx + 1]["红球"])
    repeat_count = len(current_reds & prev_reds)
    return {
        "repeat_red_count": repeat_count,
        "repeat_ratio": repeat_count / 6,
        "repeat_blue": 1 if data[idx]["蓝球"] == data[idx + 1]["蓝球"] else 0,
    }


# ==================== 简单特征提取器(用于趋势/EMA) ====================

def extract_simple_features(record):
    """从一期数据提取简单的数值特征"""
    reds = record["红球"]
    return {
        "sum": sum(reds),
        "span": max(reds) - min(reds),
        "odd_count": sum(1 for x in reds if x % 2 == 1),
        "ac": _ac_simple(reds),
        "blue": record["蓝球"],
    }


def _ac_simple(reds):
    reds = sorted(reds)
    diffs = set()
    for i in range(len(reds)):
        for j in range(i + 1, len(reds)):
            diffs.add(reds[j] - reds[i])
    return len(diffs) - (len(reds) - 1)
