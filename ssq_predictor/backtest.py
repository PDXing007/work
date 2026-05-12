#!/usr/bin/env python3
"""
回测框架 — 前向验证 + 统计显著性检验

回答核心问题: 模型预测是否显著优于随机？
"""

import math
import random
from collections import Counter


# ==================== 前向回测 ====================

def backtest_walk_forward(data, predict_fn, n_folds=5, train_ratio=0.85):
    """
    前向回测: 滚动训练 + 验证

    Args:
        data: 时间倒序列表
        predict_fn: fn(train_data) → {"红球": [...], "蓝球": int}
        n_folds: 折数
        train_ratio: 每折的训练数据比例

    Returns:
        {
            "folds": [{...}, ...],
            "aggregate": {...},
            "cumulative_excess_red": [...],
        }
    """
    n = len(data)
    val_size = int(n * (1 - train_ratio) / n_folds)
    if val_size < 10:
        val_size = max(10, int(n * 0.02))

    results = []
    cumulative_red_hits = 0
    cumulative_total = 0
    cumulative_excess = []

    for fold in range(n_folds):
        # 切分点: 从最新数据逐渐往后
        val_start = fold * val_size
        val_end = val_start + val_size
        train_start = val_end

        if train_start >= n:
            break

        train_data = data[train_start:]
        val_data = data[val_start:val_end]

        # 预测
        prediction = predict_fn(train_data)
        if prediction is None:
            continue

        pred_reds = set(prediction.get("红球", []))
        pred_blue = prediction.get("蓝球")

        # 评估
        red_hits = 0
        blue_hits = 0
        match_3 = 0

        for record in val_data:
            actual_reds = set(record["红球"])
            hits = len(pred_reds & actual_reds)
            red_hits += hits
            if pred_blue == record["蓝球"]:
                blue_hits += 1
            if hits >= 3:
                match_3 += 1

        val_n = len(val_data)
        red_rate = red_hits / (val_n * 6) if val_n > 0 else 0
        blue_rate = blue_hits / val_n if val_n > 0 else 0

        cumulative_red_hits += red_hits
        cumulative_total += val_n * 6

        # 累积超额命中 (vs 随机期望 6/33 per ball)
        expected_hits = val_n * 6 * (6.0 / 33.0)
        excess = red_hits - expected_hits
        cumulative_excess.append(excess)

        results.append({
            "fold": fold,
            "n_train": len(train_data),
            "n_val": val_n,
            "red_hit_rate": red_rate,
            "blue_hit_rate": blue_rate,
            "red_3plus_rate": match_3 / val_n if val_n > 0 else 0,
            "excess_hits": excess,
        })

    # 汇总
    if not results:
        return {"error": "无有效回测结果"}

    red_rates = [r["red_hit_rate"] for r in results]
    blue_rates = [r["blue_hit_rate"] for r in results]

    aggregate = {
        "n_folds": len(results),
        "mean_red_hit_rate": sum(red_rates) / len(red_rates),
        "std_red_hit_rate": _std(red_rates),
        "mean_blue_hit_rate": sum(blue_rates) / len(blue_rates),
        "total_excess_hits": sum(r["excess_hits"] for r in results),
        "cumulative_excess": cumulative_excess,
        "consistency": sum(1 for r in red_rates if r > 6.0 / 33.0) / len(red_rates),
    }

    return {
        "folds": results,
        "aggregate": aggregate,
    }


# ==================== 统计显著性检验 ====================

def binomial_test(observed_hits, n_trials, p_random):
    """
    二项检验: 观测命中数是否显著高于随机期望

    H0: 命中率 = p_random
    H1: 命中率 > p_random

    Returns:
        p_value (越小越显著)
    """
    from math import comb
    p_value = 0.0
    for k in range(int(observed_hits), n_trials + 1):
        p_value += comb(n_trials, k) * (p_random ** k) * ((1 - p_random) ** (n_trials - k))
    return min(p_value, 1.0)


def permutation_test(data, predict_fn, n_permutations=500):
    """
    排列检验: 打乱开奖顺序后重新评估

    H0: 模型预测与随机无区别

    Returns:
        p_value: 观测结果在零假设下出现的概率
    """
    # 真实结果
    obs_result = backtest_walk_forward(data, predict_fn, n_folds=4)
    if "error" in obs_result:
        return 1.0
    obs_mean = obs_result["aggregate"]["mean_red_hit_rate"]

    # 排列检验
    count_better = 0
    n_valid = 0

    for _ in range(n_permutations):
        # 打乱数据顺序 (破坏时序结构)
        shuffled = list(data)
        random.shuffle(shuffled)

        perm_result = backtest_walk_forward(shuffled, predict_fn, n_folds=4)
        if "error" in perm_result:
            continue

        perm_mean = perm_result["aggregate"]["mean_red_hit_rate"]
        n_valid += 1
        if perm_mean >= obs_mean:
            count_better += 1

    if n_valid == 0:
        return 1.0

    return count_better / n_valid


# ==================== 基线对比 ====================

def random_baseline(n_val, n_trials=100):
    """
    随机基线: 模拟随机选号的期望表现

    Returns:
        expected_red_hit_rate, expected_blue_hit_rate, std_red
    """
    expected_red = 6.0 / 33.0
    expected_blue = 1.0 / 16.0

    # 标准差: sqrt(p*(1-p)/n) per trial
    std_red_per_draw = math.sqrt(expected_red * (1 - expected_red))
    std_blue_per_draw = math.sqrt(expected_blue * (1 - expected_blue))

    return {
        "expected_red_hit_rate": expected_red,
        "expected_blue_hit_rate": expected_blue,
        "std_red_hit_rate": std_red_per_draw / math.sqrt(n_val),
        "std_blue_hit_rate": std_blue_per_draw / math.sqrt(n_val),
        "red_3plus_expected": _expected_red_3plus_rate(),
    }


def _expected_red_3plus_rate():
    """随机选6个号中3+的理论概率 (超几何分布)"""
    # 简化: 用模拟估计
    import random as rnd
    total = 100000
    count = 0
    for _ in range(total):
        chosen = set(rnd.sample(range(33), 6))
        actual = set(rnd.sample(range(33), 6))
        if len(chosen & actual) >= 3:
            count += 1
    return count / total


def compare_to_baseline(backtest_result):
    """
    将回测结果与随机基线对比

    Returns:
        对比报告
    """
    if "error" in backtest_result:
        return backtest_result

    agg = backtest_result["aggregate"]
    baseline = random_baseline(100)  # rough estimate

    excess_red = agg["mean_red_hit_rate"] - baseline["expected_red_hit_rate"]
    excess_blue = agg["mean_blue_hit_rate"] - baseline["expected_blue_hit_rate"]

    # Sharpe-like ratio (信息比率)
    ir_red = excess_red / max(agg["std_red_hit_rate"], 0.001)

    return {
        **agg,
        "baseline_red": baseline["expected_red_hit_rate"],
        "baseline_blue": baseline["expected_blue_hit_rate"],
        "excess_red": excess_red,
        "excess_blue": excess_blue,
        "information_ratio_red": ir_red,
        "win_rate_vs_random": agg["consistency"],
        "assessment": _assess_performance(excess_red, ir_red, agg["consistency"]),
    }


def _assess_performance(excess_red, ir_red, consistency):
    """综合评估"""
    score = 0
    if excess_red > 0:
        score += 1
    if ir_red > 1.0:
        score += 1
    if consistency > 0.6:
        score += 1

    if score >= 3:
        return "显著优于随机 (值得关注)"
    elif score >= 2:
        return "轻微优于随机 (可能有微弱信号)"
    elif score >= 1:
        return "与随机无显著差异"
    else:
        return "未超过随机基线"


def _std(values):
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))
