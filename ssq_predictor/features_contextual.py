#!/usr/bin/env python3
"""
上下文特征 — 星期效应、销售额/奖池、出球顺序、季节性

这些特征捕捉开奖环境的上下文信息，可能影响号码分布。
"""

import math


# ==================== 星期特征 ====================

# 双色球只在 周日、周二、周四 开奖
DAY_MAP = {"周日": 0, "周二": 1, "周四": 2}
DAY_INDEX = {"周日": 0, "周二": 1, "周四": 2,
             "Sunday": 0, "Tuesday": 1, "Thursday": 2,
             "星期日": 0, "星期二": 1, "星期四": 2}


def encode_day_of_week(record):
    """
    星期编码:
    - 一周三期的序号 (0=周日, 1=周二, 2=周四)
    - 循环编码 sin/cos (2π * idx / 3)
    """
    day = record.get("星期", "")
    idx = DAY_INDEX.get(day, 0)

    return {
        "day_index": idx,
        "is_sunday": 1 if idx == 0 else 0,
        "is_tuesday": 1 if idx == 1 else 0,
        "is_thursday": 1 if idx == 2 else 0,
        "day_sin": math.sin(2 * math.pi * idx / 3),
        "day_cos": math.cos(2 * math.pi * idx / 3),
    }


# ==================== 销售额/奖池特征 ====================

def encode_financial(record, recent_sales_stats=None, recent_jackpot_stats=None):
    """
    销售额和奖池金额的 z-score 特征

    Args:
        record: 当前期数据
        recent_sales_stats: (mean, std) 近期销售额统计
        recent_jackpot_stats: (mean, std) 近期奖池金额统计

    Returns:
        dict: z-score 及相关特征
    """
    sales = int(record.get("销售额", 0))
    jackpot = int(record.get("奖池金额", 0))

    features = {
        "sales_raw": sales,
        "jackpot_raw": jackpot,
    }

    if recent_sales_stats:
        mean_s, std_s = recent_sales_stats
        if std_s > 0:
            features["sales_zscore"] = (sales - mean_s) / std_s
        else:
            features["sales_zscore"] = 0.0
    else:
        features["sales_zscore"] = 0.0

    if recent_jackpot_stats:
        mean_j, std_j = recent_jackpot_stats
        if std_j > 0:
            features["jackpot_zscore"] = (jackpot - mean_j) / std_j
        else:
            features["jackpot_zscore"] = 0.0
    else:
        features["jackpot_zscore"] = 0.0

    # 奖池/销售额比率
    if sales > 0:
        features["jackpot_sales_ratio"] = jackpot / sales
    else:
        features["jackpot_sales_ratio"] = 0.0

    return features


def compute_recent_financial_stats(data, idx, window=50):
    """计算近N期的销售额/奖池均值和标准差"""
    lookback = min(window, len(data) - idx - 1)
    if lookback < 2:
        return None, None

    sales_vals = []
    jackpot_vals = []
    for i in range(lookback):
        record = data[idx + 1 + i]
        s = int(record.get("销售额", 0))
        j = int(record.get("奖池金额", 0))
        if s > 0:
            sales_vals.append(s)
        if j > 0:
            jackpot_vals.append(j)

    def stats(vals):
        if len(vals) < 2:
            return 0, 1
        mean_val = sum(vals) / len(vals)
        var_val = sum((v - mean_val) ** 2 for v in vals) / len(vals)
        return mean_val, math.sqrt(var_val)

    return stats(sales_vals), stats(jackpot_vals)


# ==================== 出球顺序特征 ====================

def encode_draw_order(record):
    """
    出球顺序特征 — 红球顺序是机器的实际出球顺序

    特征:
    - 每个位置的出球顺序索引 (该球是第几个被抽出的)
    - 三个区间各自的平均出球顺序
    - 第一球和最后一球的号码
    - 顺序跨度 (第一个被抽出的球和最后一个被抽出的球的号码差)
    """
    reds_sorted = sorted(record["红球"])
    draw_order = record.get("红球顺序", record["红球"])

    if not draw_order or len(draw_order) != 6:
        return {"has_draw_order": 0}

    # 每个红球被抽出的序号
    pos_to_order = {}
    for order_idx, ball in enumerate(draw_order):
        pos_to_order[ball] = order_idx + 1  # 1-indexed

    # 按排序位置的出球序号
    order_features = {}
    for i, ball in enumerate(reds_sorted):
        order_features[f"order_pos_{i + 1}"] = pos_to_order.get(ball, i + 1)

    # 各区间的平均出球序号
    small_balls = [pos_to_order[b] for b in reds_sorted if 1 <= b <= 11]
    mid_balls = [pos_to_order[b] for b in reds_sorted if 12 <= b <= 22]
    large_balls = [pos_to_order[b] for b in reds_sorted if 23 <= b <= 33]

    order_features["order_small_avg"] = sum(small_balls) / len(small_balls) if small_balls else 0
    order_features["order_mid_avg"] = sum(mid_balls) / len(mid_balls) if mid_balls else 0
    order_features["order_large_avg"] = sum(large_balls) / len(large_balls) if large_balls else 0

    # 第一球和最后一球
    order_features["first_ball_drawn"] = draw_order[0]
    order_features["last_ball_drawn"] = draw_order[-1]
    order_features["first_ball_zone"] = (0 if draw_order[0] <= 11
                                         else 1 if draw_order[0] <= 22 else 2)

    # 顺序间隔: 相邻顺序出球之间的号码差
    order_gaps = [abs(draw_order[i + 1] - draw_order[i]) for i in range(5)]
    order_features["order_max_gap"] = max(order_gaps) if order_gaps else 0
    order_features["order_mean_gap"] = sum(order_gaps) / 5 if order_gaps else 0
    order_features["order_min_gap"] = min(order_gaps) if order_gaps else 0

    order_features["has_draw_order"] = 1

    return order_features


# ==================== 季节性特征 ====================

def encode_seasonal(record):
    """
    季节性循环编码 (使用 sin/cos 保证连续性)

    - 月份 (1-12), 季度 (1-4), 年内周数 (1-52)
    - 全部用 sin/cos 编码
    """
    date_str = record.get("开奖日期", "")
    if not date_str:
        return {"month_sin": 0, "month_cos": 0, "quarter_sin": 0, "quarter_cos": 0}

    try:
        parts = date_str.split("-")
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
    except (ValueError, IndexError):
        return {"month_sin": 0, "month_cos": 0, "quarter_sin": 0, "quarter_cos": 0}

    quarter = (month - 1) // 3 + 1

    # 年内的大致周数
    # 简化: 用月份和日期估算
    week_approx = month * 4.33 + day / 7.0

    return {
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        "quarter_sin": math.sin(2 * math.pi * quarter / 4),
        "quarter_cos": math.cos(2 * math.pi * quarter / 4),
        "week_sin": math.sin(2 * math.pi * week_approx / 52),
        "week_cos": math.cos(2 * math.pi * week_approx / 52),
        "month_raw": month,
        "quarter_raw": quarter,
    }


# ==================== 距上次一等奖特征 ====================

def encode_jackpot_features(data, idx, window=20):
    """
    奖池相关:
    - 距上次出一等奖的期数
    - 近N期一等奖总注数
    - 是否连续无一等奖
    """
    lookback = min(window, len(data) - idx - 1)
    jackpot_no_first = 0
    first_prize_counts = []

    for i in range(lookback):
        record = data[idx + 1 + i]
        count = int(record.get("一等奖注数", "0") or "0")
        first_prize_counts.append(count)

    for i in range(lookback):
        record = data[idx + 1 + i]
        count = int(record.get("一等奖注数", "0") or "0")
        if count == 0:
            jackpot_no_first += 1
        else:
            break

    return {
        "no_first_prize_streak": jackpot_no_first,
        "recent_first_prize_sum": sum(first_prize_counts),
        "recent_first_prize_mean": sum(first_prize_counts) / max(lookback, 1),
    }
