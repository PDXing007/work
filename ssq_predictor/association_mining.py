#!/usr/bin/env python3
"""
关联规则挖掘 — Apriori频繁项集 + Lift/Confidence评分

从历史开奖数据中挖掘红球之间的共现模式。
支持时间衰减版本以捕捉近期模式变化。
"""

import math
from collections import Counter, defaultdict


# ==================== Apriori 频繁项集 ====================

def apriori_frequent_itemsets(data, min_support=0.01, max_size=3, half_life=0):
    """
    Apriori 算法挖掘频繁项集

    Args:
        data: 时间倒序列表
        min_support: 最小支持度 (相对于总权重)
        max_size: 最大项集大小 (2或3)
        half_life: 时间衰减半衰期 (0=无衰减)

    Returns:
        {frozenset(items): support}
    """
    n = len(data)

    # 计算权重
    total_weight = 0.0
    transactions_weighted = []
    for i, record in enumerate(data):
        w = 1.0 if half_life <= 0 else math.exp(-math.log(2) * i / half_life)
        total_weight += w
        transactions_weighted.append((set(record["红球"]), w))

    if total_weight == 0:
        return {}

    # --- 1-itemsets ---
    item_counts = Counter()
    for items, w in transactions_weighted:
        for item in items:
            item_counts[item] += w

    L1 = {}
    for item, count in item_counts.items():
        support = count / total_weight
        if support >= min_support:
            L1[frozenset([item])] = support

    if max_size < 2:
        return L1

    # --- 2-itemsets ---
    L1_items = sorted(set(item for itemset in L1 for item in itemset))
    C2 = []
    for i in range(len(L1_items)):
        for j in range(i + 1, len(L1_items)):
            C2.append({L1_items[i], L1_items[j]})

    L2_counts = Counter()
    for items, w in transactions_weighted:
        items_list = list(items)
        for candidate in C2:
            cand_list = list(candidate)
            if all(c in items for c in cand_list):
                L2_counts[frozenset(cand_list)] += w

    L2 = {}
    for itemset, count in L2_counts.items():
        support = count / total_weight
        if support >= min_support:
            L2[itemset] = support

    L_all = {**L1, **L2}

    if max_size < 3:
        return L_all

    # --- 3-itemsets ---
    L2_list = [set(s) for s in L2]
    C3 = set()
    for i in range(len(L2_list)):
        for j in range(i + 1, len(L2_list)):
            union_set = L2_list[i] | L2_list[j]
            if len(union_set) == 3:
                C3.add(frozenset(union_set))

    L3_counts = Counter()
    for items, w in transactions_weighted:
        for candidate in C3:
            cand_list = list(candidate)
            if all(c in items for c in cand_list):
                L3_counts[frozenset(cand_list)] += w

    L3 = {}
    for itemset, count in L3_counts.items():
        support = count / total_weight
        if support >= min_support:
            L3[itemset] = support

    return {**L_all, **L3}


# ==================== 关联规则 ====================

def generate_association_rules(itemsets, total_draws):
    """
    从频繁项集生成关联规则

    For each itemset {A, B}:
      lift(A→B) = support(A,B) / (support(A) * support(B))
      confidence(A→B) = support(A,B) / support(A)

    Lift > 1: 正相关 (一起出现多于独立期望)
    Lift = 1: 独立
    Lift < 1: 负相关 (相斥)

    Returns:
        [{from: [items], to: [item], lift, confidence, support}, ...]
    """
    rules = []

    for itemset, support in itemsets.items():
        items = list(itemset)
        if len(items) < 2:
            continue

        # 对于每对，生成单向规则
        if len(items) == 2:
            a, b = items
            sup_a = itemsets.get(frozenset([a]), support)
            sup_b = itemsets.get(frozenset([b]), support)

            for from_item, to_item in [(a, b), (b, a)]:
                sup_from = itemsets.get(frozenset([from_item]), 0.001)
                sup_to = itemsets.get(frozenset([to_item]), 0.001)

                lift = support / max(sup_from * sup_to, 0.0001)
                confidence = support / max(sup_from, 0.0001)

                rules.append({
                    "from": [from_item],
                    "to": to_item,
                    "lift": lift,
                    "confidence": confidence,
                    "support": support,
                    "size": 2,
                })

        elif len(items) == 3:
            # 从2→1的规则
            for i, to_item in enumerate(items):
                from_items = items[:i] + items[i + 1:]
                sup_from = itemsets.get(frozenset(from_items), 0.001)
                sup_to = itemsets.get(frozenset([to_item]), 0.001)

                lift = support / max(sup_from * sup_to, 0.0001)
                confidence = support / max(sup_from, 0.0001)

                rules.append({
                    "from": from_items,
                    "to": to_item,
                    "lift": lift,
                    "confidence": confidence,
                    "support": support,
                    "size": 3,
                })

    rules.sort(key=lambda x: x["lift"], reverse=True)
    return rules


# ==================== 关联评分矩阵 ====================

def build_pair_scoring_matrix(rules, min_lift=1.05, n_top=50):
    """
    构建红球对的评分矩阵

    Args:
        rules: 关联规则列表
        min_lift: 最小 lift 阈值
        n_top: 取 top N 高 lift 规则

    Returns:
        pair_scores: {(a, b): lift_score}
        mean_pair_score: float
    """
    pair_scores = {}
    valid_rules = [r for r in rules if r["lift"] >= min_lift and r["size"] == 2]

    for rule in valid_rules[:n_top]:
        a, b = rule["from"][0], rule["to"]
        key = tuple(sorted([a, b]))
        if key not in pair_scores:
            pair_scores[key] = rule["lift"]
        else:
            pair_scores[key] = max(pair_scores[key], rule["lift"])

    scores = list(pair_scores.values())
    mean_score = sum(scores) / len(scores) if scores else 1.0

    return pair_scores, mean_score


def score_combination_pairs(reds, pair_scores):
    """
    给一组红球的所有内部pair打分

    Returns:
        mean_lift: 平均 lift
        min_lift: 最弱 pair 的 lift
        fraction_strong: lift > 1.1 的 pair 占比
    """
    reds = sorted(reds)
    lifts = []
    strong = 0
    total = 0

    for i in range(len(reds)):
        for j in range(i + 1, len(reds)):
            key = (reds[i], reds[j])
            lift = pair_scores.get(key, 1.0)
            lifts.append(lift)
            if lift > 1.1:
                strong += 1
            total += 1

    return {
        "mean_lift": sum(lifts) / len(lifts) if lifts else 1.0,
        "min_lift": min(lifts) if lifts else 1.0,
        "max_lift": max(lifts) if lifts else 1.0,
        "fraction_strong": strong / total if total else 0,
    }


# ==================== 时序稳定性检测 ====================

def check_temporal_stability(data, half_life_pairs, n_splits=3):
    """
    检测关联规则的时序稳定性:
    将数据按时间分成 n_splits 段，
    比较各段发现的高 lift 规则是否一致。

    稳定的规则更可能是真实的物理偏差，而非统计噪声。

    Returns:
        stable_rules: 在多个时间窗口都出现的规则
    """
    n = len(data)
    split_size = n // n_splits
    all_top_rules = []

    for s in range(n_splits):
        start = s * split_size
        end = start + split_size if s < n_splits - 1 else n
        segment = data[start:end]

        itemsets = apriori_frequent_itemsets(segment, min_support=0.01, max_size=2)
        rules = generate_association_rules(itemsets, len(segment))
        top_rules = [r for r in rules if r["lift"] > 1.2 and r["size"] == 2]
        top_rule_pairs = set()
        for r in top_rules:
            pair = tuple(sorted([r["from"][0], r["to"]]))
            top_rule_pairs.add(pair)
        all_top_rules.append(top_rule_pairs)

    # 计算在至少2个窗口都出现的规则
    pair_counts = Counter()
    for rule_set in all_top_rules:
        for pair in rule_set:
            pair_counts[pair] += 1

    stable_rules = {}
    for pair, count in pair_counts.items():
        if count >= 2:  # 至少2/3窗口
            stable_rules[pair] = count / n_splits

    return stable_rules
