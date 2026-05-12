#!/usr/bin/env python3
"""
Metropolis-Hastings MCMC 采样器 — 从特征联合分布中采样双色球组合

替换旧的简单加权随机采样。MCMC 保证采样自目标分布 π(x) ∝ exp(-E(x)/T)。

优势:
- 所有特征约束同时考虑（而非贪心逐个选球）
- 通过模拟退火探索-利用平衡
- 多链并行 + 诊断输出
"""

import math
import random
from collections import Counter


class MHSampler:
    """
    Metropolis-Hastings 采样器

    状态: (sorted_reds: [6 ints], blue: int)
    提议: 随机选一种扰动（单球替换/双球替换/蓝球替换）
    接受: min(1, exp(-(E_new - E_old) / T))
    """

    def __init__(self, energy_fn, seed=None):
        """
        Args:
            energy_fn: EnergyFunction 实例
            seed: 随机种子
        """
        self.energy_fn = energy_fn
        if seed is not None:
            random.seed(seed)

        # 提议分布权重
        self.proposal_weights = {
            "single_red_swap": 0.5,
            "double_red_swap": 0.3,
            "blue_swap": 0.2,
        }

        # 诊断
        self.acceptance_history = []
        self.energy_history = []

    def _random_initial_state(self):
        """生成随机初始状态"""
        reds = sorted(random.sample(range(1, 34), 6))
        blue = random.randint(1, 16)
        return reds, blue

    def _propose_single_red_swap(self, reds):
        """提议: 替换一个红球"""
        new_reds = list(reds)
        idx = random.randint(0, 5)
        old_ball = new_reds[idx]
        candidates = [b for b in range(1, 34) if b not in new_reds]
        new_ball = random.choice(candidates)
        new_reds[idx] = new_ball
        return sorted(new_reds)

    def _propose_double_red_swap(self, reds):
        """提议: 替换两个红球"""
        new_reds = list(reds)
        idx1, idx2 = random.sample(range(6), 2)
        old_balls = {new_reds[idx1], new_reds[idx2]}
        candidates = [b for b in range(1, 34) if b not in new_reds]
        if len(candidates) < 2:
            return sorted(new_reds)
        new_balls = random.sample(candidates, 2)
        new_reds[idx1] = new_balls[0]
        new_reds[idx2] = new_balls[1]
        return sorted(new_reds)

    def _propose_blue_swap(self, blue):
        """提议: 替换蓝球"""
        candidates = [b for b in range(1, 17) if b != blue]
        return random.choice(candidates)

    def _propose(self, reds, blue):
        """提议新状态"""
        r = random.random()
        if r < self.proposal_weights["single_red_swap"]:
            new_reds = self._propose_single_red_swap(reds)
            new_blue = blue
        elif r < self.proposal_weights["single_red_swap"] + \
                    self.proposal_weights["double_red_swap"]:
            new_reds = self._propose_double_red_swap(reds)
            new_blue = blue
        else:
            new_reds = reds
            new_blue = self._propose_blue_swap(blue)
        return new_reds, new_blue

    def sample(self, n_samples, T0=1.0, tau=5000, burn_in=2000, thin=10,
               verbose=False):
        """
        MCMC 采样

        Args:
            n_samples: 目标样本数 (thinning后)
            T0: 初始温度
            tau: 退火时间常数 (越大冷却越慢)
            burn_in: 预热期样本数
            thin: 稀释间隔 (每隔 thin 步保存一个样本)
            verbose: 是否输出进度

        Returns:
            samples: [{"红球": [...], "蓝球": int}, ...]
            diagnostics: dict
        """
        reds, blue = self._random_initial_state()
        E = self.energy_fn.total_energy(reds, blue)

        self.acceptance_history = []
        self.energy_history = []
        accepted = 0
        total_steps = burn_in + n_samples * thin
        samples = []
        unique_combos = set()

        for step in range(total_steps):
            # 温度退火
            T = T0 * math.exp(-step / tau) if tau > 0 and step < burn_in else T0

            new_reds, new_blue = self._propose(reds, blue)
            E_new = self.energy_fn.total_energy(new_reds, new_blue)

            # Metropolis 接受/拒绝
            delta_E = E_new - E
            if delta_E <= 0 or random.random() < math.exp(-delta_E / max(T, 0.001)):
                reds, blue = new_reds, new_blue
                E = E_new
                accepted += 1

            self.acceptance_history.append(1 if delta_E <= 0 or
                                           random.random() < math.exp(-delta_E / max(T, 0.001)) else 0)
            self.energy_history.append(E)

            # 预热期后保存
            if step >= burn_in and (step - burn_in) % thin == 0:
                key = (tuple(reds), blue)
                if key not in unique_combos:
                    unique_combos.add(key)
                    samples.append({"红球": list(reds), "蓝球": blue})

            if verbose and step % 5000 == 0 and step > 0:
                acc_rate = accepted / (step + 1)
                print(f"  Step {step}/{total_steps}: "
                      f"T={T:.4f}, E={E:.2f}, acc_rate={acc_rate:.3f}")

        acc_rate = accepted / max(total_steps, 1)

        diagnostics = {
            "acceptance_rate": acc_rate,
            "total_steps": total_steps,
            "burn_in": burn_in,
            "thin": thin,
            "n_unique_samples": len(samples),
            "T0": T0,
            "tau": tau,
            "mean_energy": sum(self.energy_history[burn_in:]) /
                           max(len(self.energy_history[burn_in:]), 1),
            "std_energy": self._std(self.energy_history[burn_in:]),
            "autocorr_lag1": self._autocorr_lag1(self.energy_history[burn_in:]),
        }

        return samples, diagnostics

    def sample_multichain(self, n_samples, n_chains=4, **kwargs):
        """多链并行采样"""
        all_samples = []
        all_diagnostics = []

        for c in range(n_chains):
            # 不同种子
            if kwargs.get("seed") is not None:
                kwargs["seed"] = kwargs["seed"] + c
            samples, diag = self.sample(n_samples, **kwargs)
            all_samples.extend(samples)
            all_diagnostics.append(diag)

            if kwargs.get("verbose"):
                print(f"  Chain {c+1}/{n_chains}: "
                      f"{len(samples)} samples, "
                      f"acc_rate={diag['acceptance_rate']:.3f}")

        return all_samples, all_diagnostics

    @staticmethod
    def _std(values):
        if not values:
            return 0.0
        mean_val = sum(values) / len(values)
        return math.sqrt(sum((v - mean_val) ** 2 for v in values) / len(values))

    @staticmethod
    def _autocorr_lag1(values):
        """滞后1自回归系数"""
        if len(values) < 2:
            return 0.0
        n = len(values)
        mean_val = sum(values) / n
        num = sum((values[i] - mean_val) * (values[i - 1] - mean_val)
                  for i in range(1, n))
        den = sum((v - mean_val) ** 2 for v in values)
        return num / max(den, 0.001)


# ==================== 频率排名 ====================

def rank_by_frequency(samples, top_n=10):
    """按出现频率排序 MCMC 样本"""
    counter = Counter()
    for s in samples:
        key = (tuple(s["红球"]), s["蓝球"])
        counter[key] += 1

    results = []
    for (reds, blue), count in counter.most_common(top_n):
        results.append({
            "红球": list(reds),
            "蓝球": blue,
            "count": count,
            "frequency_pct": count / len(samples) * 100,
        })
    return results


# ==================== 硬约束过滤 ====================

def filter_by_constraints(samples, dists, strictness=0.95):
    """
    硬约束过滤 (作为 MCMC 的补充安全检查)

    Args:
        samples: MCMC 样本列表
        dists: fit_all_distributions() 输出
        strictness: 保留比例

    Returns:
        过滤后的样本列表
    """
    # 允许的和值范围 (±3σ)
    sum_dist = dists.get("sum", {})
    sum_lo = sum_dist.get("mean", 100) - 3 * sum_dist.get("std", 20)
    sum_hi = sum_dist.get("mean", 100) + 3 * sum_dist.get("std", 20)

    # 允许的奇偶比 (累计占比 strictness)
    parity_dist = dists.get("parity", {})
    parity_sorted = sorted(parity_dist.items(), key=lambda x: x[1], reverse=True)
    cum = 0
    parity_allowed = set()
    for odds, prob in parity_sorted:
        cum += prob
        parity_allowed.add(odds)
        if cum >= strictness:
            break

    filtered = []
    for s in samples:
        reds = s["红球"]
        # 和值
        total = sum(reds)
        if total < sum_lo or total > sum_hi:
            continue
        # 奇偶比
        odds = sum(1 for x in reds if x % 2 == 1)
        if odds not in parity_allowed:
            continue
        filtered.append(s)

    return filtered
