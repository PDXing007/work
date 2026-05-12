#!/usr/bin/env python3
"""
MCMC 能量函数 — 特征联合概率模型的负对数似然

E(combination) = Σ feature_penalties - association_bonus - frequency_bonus

能量越低，组合越"合理"。MCMC 从中采样。
这是一个近似的对数概率模型，将各特征视为条件独立。
"""

import math


class EnergyFunction:
    """
    组合能量函数

    E(reds, blue) = w_sum * E_sum + w_span * E_span + w_parity * E_parity
                  + w_zone * E_zone + w_ac * E_ac + w_gap * E_gap
                  + w_pos * E_position + w_freq * E_freq - w_assoc * E_assoc

    每一项都是该特征在当前组合下的负对数似然。
    """

    def __init__(self, dists, freq_red=None, freq_blue=None,
                 pair_scores=None, cond_red=None):
        """
        Args:
            dists: fit_all_distributions() 的输出
            freq_red: {ball: weighted_count}
            freq_blue: {ball: weighted_count}
            pair_scores: {(a,b): lift} 关联评分矩阵
            cond_red: {a: {b: P(b|a)}} 条件概率矩阵
        """
        self.dists = dists
        self.freq_red = freq_red or {}
        self.freq_blue = freq_blue or {}
        self.pair_scores = pair_scores or {}
        self.cond_red = cond_red or {}

        # 权重 (可调)
        self.w_sum = 1.0
        self.w_span = 1.0
        self.w_parity = 1.5
        self.w_zone = 1.5
        self.w_ac = 0.5
        self.w_gap = 0.5
        self.w_pos = 2.0       # 位置分布是最强的约束
        self.w_freq = 0.3
        self.w_assoc = 0.2

    def energy_sum(self, reds):
        """和值能量: -log P(sum | N(μ,σ))"""
        dist = self.dists.get("sum")
        if not dist:
            return 0.0
        total = sum(reds)
        z = (total - dist["mean"]) / max(dist["std"], 0.01)
        return 0.5 * z * z  # -log N, 忽略常数项

    def energy_span(self, reds):
        """跨度能量"""
        span = max(reds) - min(reds)
        dist = self.dists.get("span", {})
        prob = dist.get(span, 0.001)
        return -math.log(prob)

    def energy_parity(self, reds):
        """奇偶比能量"""
        odds = sum(1 for x in reds if x % 2 == 1)
        dist = self.dists.get("parity", {})
        prob = dist.get(odds, 0.001)
        return -math.log(prob)

    def energy_zone(self, reds):
        """区间分布能量"""
        z1 = sum(1 for x in reds if 1 <= x <= 11)
        z2 = sum(1 for x in reds if 12 <= x <= 22)
        z3 = sum(1 for x in reds if 23 <= x <= 33)
        dist = self.dists.get("zone", {})
        prob = dist.get((z1, z2, z3), 0.001)
        return -math.log(prob)

    def energy_ac(self, reds):
        """AC值能量"""
        ac = self._compute_ac(reds)
        dist = self.dists.get("ac", {})
        prob = dist.get(ac, 0.001)
        return -math.log(prob)

    @staticmethod
    def _compute_ac(reds):
        r = sorted(reds)
        diffs = set()
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                diffs.add(r[j] - r[i])
        return len(diffs) - (len(r) - 1)

    def energy_gap(self, reds):
        """间隔能量: -Σ log P(gap_i)"""
        reds = sorted(reds)
        gap_dist = self.dists.get("gap", {})
        energy = 0.0
        for i in range(len(reds) - 1):
            gap = reds[i + 1] - reds[i]
            prob = gap_dist.get(gap, 0.001)
            energy += -math.log(prob)
        return energy

    def energy_position(self, reds):
        """位置分布能量: -Σ log P(pos_i | position_i)"""
        reds = sorted(reds)
        pos_dist = self.dists.get("position", {})
        energy = 0.0
        for i, v in enumerate(reds):
            dist_for_pos = pos_dist.get(i + 1, {})
            prob = dist_for_pos.get(v, 0.0001)
            energy += -math.log(prob)
        return energy

    def energy_frequency(self, reds, blue):
        """频率能量: -Σ log freq_normalized(ball)"""
        total_red_w = sum(self.freq_red.values()) if self.freq_red else 1.0
        total_blue_w = sum(self.freq_blue.values()) if self.freq_blue else 1.0

        energy = 0.0
        for r in reds:
            prob = self.freq_red.get(r, 0.0) / max(total_red_w, 1)
            energy += -math.log(max(prob, 0.0001))
        prob_b = self.freq_blue.get(blue, 0.0) / max(total_blue_w, 1)
        energy += -math.log(max(prob_b, 0.0001))
        return energy

    def energy_association(self, reds):
        """关联奖励 (负能量，降低总分): -mean_log_lift of all pairs"""
        if not self.pair_scores:
            return 0.0

        lifts = []
        for i in range(len(reds)):
            for j in range(i + 1, len(reds)):
                key = tuple(sorted([reds[i], reds[j]]))
                lift = self.pair_scores.get(key, 1.0)
                lifts.append(lift)

        mean_lift = sum(lifts) / len(lifts)
        if mean_lift > 1.0:
            return -math.log(mean_lift)
        return 0.0

    def energy_conditional(self, reds):
        """条件概率能量: 基于P(B|A)的联合一致性"""
        if not self.cond_red:
            return 0.0
        energy = 0.0
        for i in range(len(reds)):
            for j in range(i + 1, len(reds)):
                p_ij = self.cond_red.get(reds[i], {}).get(reds[j], 0.001)
                p_ji = self.cond_red.get(reds[j], {}).get(reds[i], 0.001)
                avg_p = (p_ij + p_ji) / 2.0
                energy += -math.log(max(avg_p, 0.0001))
        return energy

    def total_energy(self, reds, blue):
        """计算总能量"""
        e = 0.0
        e += self.w_sum * self.energy_sum(reds)
        e += self.w_span * self.energy_span(reds)
        e += self.w_parity * self.energy_parity(reds)
        e += self.w_zone * self.energy_zone(reds)
        e += self.w_ac * self.energy_ac(reds)
        e += self.w_gap * self.energy_gap(reds)
        e += self.w_pos * self.energy_position(reds)

        if self.freq_red:
            e += self.w_freq * self.energy_frequency(reds, blue)

        if self.pair_scores:
            e += self.w_assoc * self.energy_association(reds)

        if self.cond_red:
            e += 0.1 * self.energy_conditional(reds)

        return e

    def energy_components(self, reds, blue):
        """返回各分量的能量值 (用于诊断)"""
        return {
            "sum": self.w_sum * self.energy_sum(reds),
            "span": self.w_span * self.energy_span(reds),
            "parity": self.w_parity * self.energy_parity(reds),
            "zone": self.w_zone * self.energy_zone(reds),
            "ac": self.w_ac * self.energy_ac(reds),
            "gap": self.w_gap * self.energy_gap(reds),
            "position": self.w_pos * self.energy_position(reds),
            "frequency": self.w_freq * self.energy_frequency(reds, blue)
                          if self.freq_red else 0.0,
            "association": self.w_assoc * self.energy_association(reds)
                            if self.pair_scores else 0.0,
            "conditional": 0.1 * self.energy_conditional(reds)
                            if self.cond_red else 0.0,
            "total": self.total_energy(reds, blue),
        }
