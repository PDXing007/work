#!/usr/bin/env python3
"""
动态冷热分析 — z-score 热度、遗漏分析、回补检测、显著性检验

核心理念:
- "热号": 近期出现频率显著高于期望 (z > 2)
- "冷号": 近期出现频率显著低于期望 (z < -2)
- 用 z-score 而非简单计数，因为考虑了样本量和随机波动
- 遗漏值分析配合回歸均值预期
"""

import math
import random
from collections import Counter, defaultdict


# ==================== 动态热度 z-score ====================

def compute_hotness_zscore(data, idx, half_life=50, min_effective_n=10):
    """
    计算每个号码的动态热度 z-score

    对于每个号码:
    recent_rate = 衰减加权近期频率 / 有效样本量
    expected_rate = 均匀概率 (1/33 for red, 1/16 for blue)
    std = sqrt(expected_rate * (1 - expected_rate) / effective_n)

    z = (recent_rate - expected_rate) / std

    Args:
        data: 时间倒序列表
        idx: 当前期索引
        half_life: 近期窗口的半衰期
        min_effective_n: 最小有效样本量

    Returns:
        red_zscore: {ball: z_score}
        blue_zscore: {ball: z_score}
        stats: {effective_n, ...}
    """
    # 收集近期数据 (不包括当前期)
    lookback = min(len(data) - idx - 1, half_life * 3)
    if lookback < min_effective_n:
        return {}, {}, {"effective_n": 0}

    # 计算有效样本量
    recent_data = []
    effective_n = 0.0
    for i in range(lookback):
        age = i
        w = math.exp(-math.log(2) * age / half_life)
        recent_data.append((data[idx + 1 + i], w))
        effective_n += w

    if effective_n < min_effective_n:
        return {}, {}, {"effective_n": effective_n}

    # 频率计数
    red_counts = Counter()
    blue_counts = Counter()
    for record, w in recent_data:
        for r in record["红球"]:
            red_counts[r] += w
        blue_counts[record["蓝球"]] += w

    # z-score for red balls
    expected_red_rate = 6.0 / 33.0
    std_red = math.sqrt(expected_red_rate * (1.0 - expected_red_rate) / effective_n)

    red_zscore = {}
    for b in range(1, 34):
        count = red_counts.get(b, 0.0)
        recent_rate = count / effective_n
        if std_red > 0:
            red_zscore[b] = (recent_rate - expected_red_rate) / std_red
        else:
            red_zscore[b] = 0.0

    # z-score for blue balls
    expected_blue_rate = 1.0 / 16.0
    std_blue = math.sqrt(expected_blue_rate * (1.0 - expected_blue_rate) / effective_n)

    blue_zscore = {}
    for b in range(1, 17):
        count = blue_counts.get(b, 0.0)
        recent_rate = count / effective_n
        if std_blue > 0:
            blue_zscore[b] = (recent_rate - expected_blue_rate) / std_blue
        else:
            blue_zscore[b] = 0.0

    stats = {
        "effective_n": effective_n,
        "half_life": half_life,
        "std_red": std_red,
        "std_blue": std_blue,
    }

    return red_zscore, blue_zscore, stats


# ==================== 冷热分类 ====================

def classify_hot_cold(zscores, hot_threshold=2.0, cold_threshold=-1.5):
    """
    基于 z-score 分类

    Returns:
        hot: [ball, ...] 显著热号
        warm: [ball, ...]
        neutral: [ball, ...]
        cool: [ball, ...]
        cold: [ball, ...] 显著冷号
    """
    hot, warm, neutral, cool, cold = [], [], [], [], []

    for ball, z in sorted(zscores.items()):
        if z >= hot_threshold:
            hot.append(ball)
        elif z >= 0.5:
            warm.append(ball)
        elif z >= -0.5:
            neutral.append(ball)
        elif z >= cold_threshold:
            cool.append(ball)
        else:
            cold.append(ball)

    return {
        "hot": hot, "warm": warm,
        "neutral": neutral, "cool": cool, "cold": cold,
        "hot_count": len(hot), "cold_count": len(cold),
    }


# ==================== 遗漏与回补 ====================

def compute_missing_analysis(data, idx, max_lookback=200):
    """
    遗漏分析:
    - 每个号码的遗漏期数
    - 平均出现间隔 (1/rate)
    - 遗漏/期望间隔 比值 (>1 = "超期未出", 但注意赌徒谬误)
    """
    lookback = min(max_lookback, len(data) - idx - 1)
    if lookback < 10:
        return {}

    # 每个红球的出现间隔
    red_intervals = defaultdict(list)
    red_last_seen = {}
    red_missing = {}

    for gap in range(lookback):
        record = data[idx + 1 + gap]
        reds = record["红球"]
        for r in reds:
            if r in red_last_seen:
                red_intervals[r].append(gap - red_last_seen[r])
            red_last_seen[r] = gap

    for r in range(1, 34):
        if r in red_last_seen:
            red_missing[r] = lookback - red_last_seen[r]
        else:
            red_missing[r] = lookback + 1

    # 蓝球遗漏
    blue_last_seen = None
    for gap in range(lookback):
        record = data[idx + 1 + gap]
        if record["蓝球"] == data[idx]["蓝球"]:
            blue_last_seen = gap
            break

    blue_missing = blue_last_seen + 1 if blue_last_seen is not None else lookback + 1

    # 平均间隔
    avg_intervals = {}
    for r, intervals in red_intervals.items():
        if intervals:
            avg_intervals[r] = sum(intervals) / len(intervals)
        else:
            avg_intervals[r] = 33.0 / 6.0  # 理论期望

    # 超期比
    overdue_ratio_red = {}
    for r in range(1, 34):
        avg_int = avg_intervals.get(r, 33.0 / 6.0)
        if avg_int > 0:
            overdue_ratio_red[r] = red_missing.get(r, 0) / avg_int

    avg_blue_interval = 16.0  # 理论期望
    overdue_ratio_blue = blue_missing / avg_blue_interval

    return {
        "red_missing": red_missing,
        "blue_missing": blue_missing,
        "avg_red_intervals": avg_intervals,
        "overdue_ratio_red": overdue_ratio_red,
        "overdue_ratio_blue": overdue_ratio_blue,
        "max_overdue_red": max(overdue_ratio_red.values()) if overdue_ratio_red else 0,
        "overdue_count_red": sum(1 for v in overdue_ratio_red.values() if v > 1.5),
    }


# ==================== 序列检测 ====================

def compute_streaks(data, idx):
    """检测当前期每个号码的连续出现/缺失序列"""
    current_reds = set(data[idx]["红球"])
    current_blue = data[idx]["蓝球"]

    red_streaks = {}
    for r in range(1, 34):
        streak = 0
        for gap in range(min(50, len(data) - idx - 1)):
            record = data[idx + 1 + gap]
            has_ball = r in record["红球"]
            if gap == 0:
                streak = 1 if has_ball else -1
            elif streak > 0 and has_ball:
                streak += 1
            elif streak < 0 and not has_ball:
                streak -= 1
            else:
                break
        red_streaks[r] = streak

    # 蓝球序列
    blue_streak = 0
    for gap in range(min(50, len(data) - idx - 1)):
        record = data[idx + 1 + gap]
        if gap == 0:
            blue_streak = 1 if record["蓝球"] == current_blue else -1
        elif blue_streak > 0 and record["蓝球"] == current_blue:
            blue_streak += 1
        elif blue_streak < 0 and record["蓝球"] != current_blue:
            blue_streak -= 1
        else:
            break

    return {
        "red_max_positive_streak": max(s for s in red_streaks.values() if s > 0) if any(s > 0 for s in red_streaks.values()) else 0,
        "red_max_negative_streak": abs(min(s for s in red_streaks.values() if s < 0)) if any(s < 0 for s in red_streaks.values()) else 0,
        "blue_streak": blue_streak,
        "current_hot_streak_sum": sum(1 for s in red_streaks.values() if s > 2),
        "current_cold_streak_sum": sum(1 for s in red_streaks.values() if s < -10),
    }


# ==================== 显著性检验 ====================

def permutation_test_hotness(data, idx, half_life=50, n_perm=1000):
    """
    排列检验: 打乱数据顺序后重新计算 z-score，
    比较观测到的最大 |z| 是否显著。

    Returns: p-value for the most extreme observed z-score
    """
    # 观测值
    obs_red_z, obs_blue_z, _ = compute_hotness_zscore(data, idx, half_life)
    obs_max_z = max(
        max(abs(z) for z in obs_red_z.values()) if obs_red_z else 0,
        max(abs(z) for z in obs_blue_z.values()) if obs_blue_z else 0,
    )

    # 排列检验
    count_extreme = 0
    lookback = min(len(data) - idx - 1, half_life * 3)
    if lookback < 20:
        return 1.0

    recent_segment = data[idx + 1:idx + 1 + lookback]

    for _ in range(n_perm):
        shuffled = list(recent_segment)
        random.shuffle(shuffled)
        # 重新计算 z-score
        perm_red_counts = Counter()
        perm_blue_counts = Counter()
        effective_n = 0.0
        for i, record in enumerate(shuffled):
            w = math.exp(-math.log(2) * i / half_life)
            effective_n += w
            for r in record["红球"]:
                perm_red_counts[r] += w
            perm_blue_counts[record["蓝球"]] += w

        expected_red = 6.0 / 33.0
        std_red = math.sqrt(expected_red * (1.0 - expected_red) / max(effective_n, 1))
        expected_blue = 1.0 / 16.0
        std_blue = math.sqrt(expected_blue * (1.0 - expected_blue) / max(effective_n, 1))

        perm_max_z = 0.0
        for b in range(1, 34):
            rate = perm_red_counts.get(b, 0.0) / max(effective_n, 1)
            z = (rate - expected_red) / max(std_red, 0.001)
            perm_max_z = max(perm_max_z, abs(z))
        for b in range(1, 17):
            rate = perm_blue_counts.get(b, 0.0) / max(effective_n, 1)
            z = (rate - expected_blue) / max(std_blue, 0.001)
            perm_max_z = max(perm_max_z, abs(z))

        if perm_max_z >= obs_max_z:
            count_extreme += 1

    return count_extreme / n_perm
