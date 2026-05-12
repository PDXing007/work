#!/usr/bin/env python3
"""
两阶段预测管线

Stage 1: 候选生成
  - MCMC 采样 50000 组候选组合
  - 硬约束过滤 → ~5000 组

Stage 2: 评分排序
  - 时序自注意力模型: 概率评分
  - 关联规则矩阵: pair lift 评分
  - 衰减频率: 边际概率评分
  - 动态热度: z-score 评分
  - 特征分布: 能量评分 (低能量=高合理性)

综合评分 → Top-N 推荐输出
"""

import math
from collections import Counter

from stats_core import (
    compute_weighted_freq, compute_conditional_prob,
    fit_all_distributions, compute_markov_transition,
)
from cold_hot_analysis import compute_hotness_zscore, classify_hot_cold
from association_mining import (
    apriori_frequent_itemsets, generate_association_rules,
    build_pair_scoring_matrix, score_combination_pairs,
)
from energy_function import EnergyFunction
from mc_mcmc import MHSampler, rank_by_frequency, filter_by_constraints
from features_basic import compute_global_frequencies
from probability import ProbabilityOutput


class SSQPredictor:
    """
    双色球预测器 — 完整两阶段管线

    使用方式:
        predictor = SSQPredictor(data)
        predictor.prepare()           # 训练/拟合所有模型
        result = predictor.predict()  # 生成预测
    """

    def __init__(self, data, half_life=200):
        """
        Args:
            data: 时间倒序的历史数据列表
            half_life: 时间衰减半衰期
        """
        self.data = data
        self.half_life = half_life
        self.n_draws = len(data)

        # 拟合结果存储
        self.freq_red = None
        self.freq_blue = None
        self.cond_red = None
        self.freq_red_weighted = None
        self.freq_blue_weighted = None
        self.dists = None
        self.pair_scores = None
        self.hot_cold = None
        self.energy_fn = None
        self.nn_model = None
        self.ensemble = None

        # 预测配置
        self.mc_n_samples = 50000
        self.mc_top_n = 10
        self.filter_strictness = 0.95

    def prepare(self, train_mcmc=True, mine_associations=True,
                min_support=0.008):
        """
        准备阶段: 计算所有统计量和拟合模型

        Args:
            train_mcmc: 是否准备 MCMC 能量函数
            mine_associations: 是否挖掘关联规则
            min_support: 关联规则最小支持度
        """
        print("=" * 60)
        print("SSQ Predictor — 模型准备")
        print("=" * 60)

        # 1. 频率统计
        print("\n[1/5] 计算频率统计...")
        self.freq_red, self.freq_blue, _ = compute_global_frequencies(self.data)
        self.freq_red_weighted, self.freq_blue_weighted, _ = compute_weighted_freq(
            self.data, half_life=self.half_life
        )

        # 2. 条件概率
        print("[2/5] 计算条件概率矩阵...")
        self.cond_red, _ = compute_conditional_prob(
            self.data, half_life=self.half_life
        )

        # 3. 特征分布拟合
        print("[3/5] 拟合特征分布...")
        self.dists = fit_all_distributions(self.data)
        sum_dist = self.dists.get("sum", {})
        print(f"  和值: μ={sum_dist.get('mean', 'N/A'):.1f}, "
              f"σ={sum_dist.get('std', 'N/A'):.1f}")
        print(f"  位置分布: {len(self.dists.get('position', {}))} 个位置")

        # 4. 关联规则挖掘
        if mine_associations:
            print("[4/5] 挖掘关联规则...")
            itemsets = apriori_frequent_itemsets(
                self.data, min_support=min_support, max_size=2,
                half_life=self.half_life
            )
            rules = generate_association_rules(itemsets, self.n_draws)
            high_lift = [r for r in rules if r["lift"] > 1.1 and r["size"] == 2]
            print(f"  频繁项集: {len(itemsets)}, "
                  f"高Lift规则: {len(high_lift)}")
            if high_lift:
                top_rule = high_lift[0]
                print(f"  Top规则: {top_rule['from']}→{top_rule['to']}, "
                      f"lift={top_rule['lift']:.3f}")

            self.pair_scores, _ = build_pair_scoring_matrix(rules, min_lift=1.02)
        else:
            print("[4/5] 跳过关联规则挖掘")
            self.pair_scores = {}

        # 5. 能量函数
        if train_mcmc:
            print("[5/5] 构建 MCMC 能量函数...")
            self.energy_fn = EnergyFunction(
                self.dists,
                freq_red=self.freq_red_weighted,
                freq_blue=self.freq_blue_weighted,
                pair_scores=self.pair_scores,
                cond_red=self.cond_red,
            )

        print("\n准备完成!")

    def generate_candidates(self, n_samples=None):
        """
        Stage 1: MCMC 候选生成

        Returns:
            candidates: [{"红球": [...], "蓝球": int}, ...]
            diagnostics: MCMC 诊断信息
        """
        if self.energy_fn is None:
            raise ValueError("请先调用 prepare() 构建能量函数")

        n_samples = n_samples or self.mc_n_samples
        print(f"\n[Stage 1] MCMC 采样 ({n_samples} 组)...")

        sampler = MHSampler(self.energy_fn)
        samples, diagnostics = sampler.sample(
            n_samples=n_samples // 4,  # thinning 后获得指定数量
            T0=1.0, tau=3000,
            burn_in=1000, thin=2,
            verbose=True,
        )

        print(f"  生成: {len(samples)} 组唯一组合")
        print(f"  接受率: {diagnostics['acceptance_rate']:.3f} "
              f"(目标: 0.2-0.5)")

        # 硬约束过滤
        filtered = filter_by_constraints(samples, self.dists,
                                         strictness=self.filter_strictness)
        print(f"  过滤后: {len(filtered)} 组 "
              f"({len(filtered)/max(len(samples),1)*100:.1f}%)")

        return filtered, diagnostics

    def score_candidates(self, candidates):
        """
        Stage 2: 综合评分

        对每个候选组合:
        1. 频率评分: log P(balls | freq)
        2. 关联评分: mean pair lift
        3. 能量评分: -E (低能量=高合理性)
        4. 位置评分: log P(pos_i | pos_dist)

        Returns:
            排序后的候选列表 (带分数)
        """
        print(f"\n[Stage 2] 综合评分 ({len(candidates)} 组)...")

        total_red_w = sum(self.freq_red_weighted.values()) if self.freq_red_weighted else 1.0
        total_blue_w = sum(self.freq_blue_weighted.values()) if self.freq_blue_weighted else 1.0

        scored = []
        for i, cand in enumerate(candidates):
            reds = cand["红球"]
            blue = cand["蓝球"]

            # 1. 频率评分
            freq_score = 0.0
            for r in reds:
                freq_score += math.log(max(self.freq_red_weighted.get(r, 0.0) /
                                           max(total_red_w, 1), 0.0001))
            freq_score += math.log(max(self.freq_blue_weighted.get(blue, 0.0) /
                                       max(total_blue_w, 1), 0.0001))

            # 2. 关联评分
            if self.pair_scores:
                pair_result = score_combination_pairs(reds, self.pair_scores)
                assoc_score = pair_result["mean_lift"]
            else:
                assoc_score = 1.0

            # 3. 能量评分
            if self.energy_fn:
                energy = self.energy_fn.total_energy(reds, blue)
            else:
                energy = 0.0

            # 4. 综合
            combined_score = (freq_score +
                              math.log(max(assoc_score, 0.01)) * 0.5 -
                              energy * 0.1)

            scored.append({
                "红球": reds,
                "蓝球": blue,
                "freq_score": freq_score,
                "assoc_score": assoc_score,
                "energy": energy,
                "combined_score": combined_score,
            })

        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored

    def predict(self, top_n=10, n_mc_samples=None):
        """
        完整预测流程: Stage 1 + Stage 2

        Returns:
            {
                "recommendations": [{红球, 蓝球, scores}, ...],
                "diagnostics": {...},
                "marginal_probs": {red_probs, blue_probs},
            }
        """
        # Stage 1
        candidates, diagnostics = self.generate_candidates(n_mc_samples)

        # Stage 2
        scored = self.score_candidates(candidates)

        # Top-N
        top_recommendations = scored[:top_n]

        # 边际概率 (基于 MCMC 样本中每个球出现的频率)
        red_counts = Counter()
        blue_counts = Counter()
        for cand in candidates:
            for r in cand["红球"]:
                red_counts[r] += 1
            blue_counts[cand["蓝球"]] += 1

        total = len(candidates)
        red_probs = {i: red_counts.get(i, 0) / total for i in range(1, 34)}
        blue_probs = {i: blue_counts.get(i, 0) / total for i in range(1, 17)}

        # 频率排名 (备选方案)
        freq_ranked = rank_by_frequency(candidates, top_n=top_n)

        return {
            "recommendations": top_recommendations,
            "freq_ranked": freq_ranked,
            "diagnostics": diagnostics,
            "marginal_probs": {
                "red": red_probs,
                "blue": blue_probs,
            },
            "n_candidates_generated": len(candidates),
        }

    def print_report(self, result):
        """打印预测报告"""
        print("\n" + "=" * 60)
        print("SSQ 预测报告")
        print("=" * 60)

        # MCMC 诊断
        diag = result["diagnostics"]
        print(f"\n[MCMC 诊断]")
        print(f"  接受率: {diag['acceptance_rate']:.3f}")
        print(f"  候选组合数: {result['n_candidates_generated']}")

        # Top推荐
        print(f"\n[综合评分 Top-10]")
        for i, rec in enumerate(result["recommendations"], 1):
            reds_str = " ".join(f"{b:02d}" for b in rec["红球"])
            print(f"  #{i:2d}: 红球 [{reds_str}] + 蓝 {rec['蓝球']:02d}  "
                  f"(评分: {rec['combined_score']:.2f})")

        # 频率排名 Top-5
        print(f"\n[频率排名 Top-5]")
        for i, rec in enumerate(result["freq_ranked"][:5], 1):
            reds_str = " ".join(f"{b:02d}" for b in rec["红球"])
            print(f"  #{i}: 红球 [{reds_str}] + 蓝 {rec['蓝球']:02d}  "
                  f"(频率: {rec['frequency_pct']:.3f}%)")

        # 边际概率分布
        print(f"\n[红球边际概率 Top-12]")
        red_probs = result["marginal_probs"]["red"]
        red_sorted = sorted(red_probs.items(), key=lambda x: x[1], reverse=True)
        for ball, prob in red_sorted[:12]:
            bar = "█" * int(prob * 100)
            print(f"  {ball:02d}: {prob*100:5.2f}% {bar}")

        print(f"\n[蓝球边际概率 Top-8]")
        blue_probs = result["marginal_probs"]["blue"]
        blue_sorted = sorted(blue_probs.items(), key=lambda x: x[1], reverse=True)
        for ball, prob in blue_sorted[:8]:
            bar = "█" * int(prob * 100)
            print(f"  {ball:02d}: {prob*100:5.2f}% {bar}")
