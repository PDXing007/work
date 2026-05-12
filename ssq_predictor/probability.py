#!/usr/bin/env python3
"""
概率输出 + 不确定性量化

提供:
- 将模型 logits 转化为校准概率分布
- MC Dropout 不确定性估计
- 集成不确定性 (跨模型分歧)
- 概率校准 (ECE 评估)
"""

import math
import random
from collections import Counter


# ==================== 概率输出 ====================

class ProbabilityOutput:
    """将模型输出转化为概率分布"""

    @staticmethod
    def red_probs(red_logits, temperature=1.0):
        """红球概率: 33维 softmax"""
        scaled = [l / temperature for l in red_logits]
        return _softmax(scaled)

    @staticmethod
    def blue_probs(blue_logits, temperature=1.0):
        """蓝球概率: 16维 softmax"""
        scaled = [l / temperature for l in blue_logits]
        return _softmax(scaled)

    @staticmethod
    def top_k(probs, k=6):
        """返回概率最高的k个 (1-indexed)"""
        indexed = [(i + 1, p) for i, p in enumerate(probs)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:k]

    @staticmethod
    def combination_prob(reds, blue, red_probs, blue_probs):
        """组合近似概率 (假设条件独立)"""
        p = 1.0
        for r in reds:
            p *= max(red_probs[r - 1], 1e-10)
        p *= max(blue_probs[blue - 1], 1e-10)
        return p

    @staticmethod
    def entropy(probs):
        """预测熵: -Σ p*log(p)"""
        eps = 1e-10
        return -sum(p * math.log(max(p, eps)) for p in probs)

    @staticmethod
    def uncertainty_level(entropy, n_classes):
        """将熵映射为可读等级"""
        max_entropy = math.log(n_classes)
        ratio = entropy / max_entropy if max_entropy > 0 else 0
        if ratio > 0.8:
            return "高"
        elif ratio > 0.5:
            return "中"
        else:
            return "低"


# ==================== MC Dropout 不确定性 ====================

class MCDropoutUncertainty:
    """
    通过多次前向传播 (不同 dropout mask) 估计预测不确定性

    Args:
        dropout_rate: 神经元丢弃率
        n_iterations: 重复前向传播次数
    """

    def __init__(self, dropout_rate=0.1, n_iterations=30):
        self.dropout_rate = dropout_rate
        self.n_iterations = n_iterations

    def estimate(self, model, X):
        """
        估计预测不确定性

        Returns:
            mean_red: (33,) 平均红球概率
            mean_blue: (16,) 平均蓝球概率
            std_red: (33,) 红球概率标准差
            std_blue: (16,) 蓝球概率标准差
            red_entropy: 平均分布的熵
            blue_entropy: 平均分布的熵
        """
        all_red = []
        all_blue = []

        for _ in range(self.n_iterations):
            # 用不同随机种子模拟 dropout
            # 简化版: 通过向预测添加小噪声模拟不确定性
            rp, bp = model.predict_proba(X)

            # 加噪声模拟 dropout
            noisy_rp = [max(0.0, min(1.0, p + random.gauss(0, self.dropout_rate * p)))
                        for p in rp]
            # 重新归一化
            s = sum(noisy_rp)
            noisy_rp = [p / s for p in noisy_rp]

            noisy_bp = [max(0.0, min(1.0, p + random.gauss(0, self.dropout_rate * p)))
                        for p in bp]
            s = sum(noisy_bp)
            noisy_bp = [p / s for p in noisy_bp]

            all_red.append(noisy_rp)
            all_blue.append(noisy_bp)

        # 均值
        mean_red = [sum(all_red[k][i] for k in range(self.n_iterations)) / self.n_iterations
                    for i in range(33)]
        mean_blue = [sum(all_blue[k][i] for k in range(self.n_iterations)) / self.n_iterations
                     for i in range(16)]

        # 标准差
        std_red = [math.sqrt(sum((all_red[k][i] - mean_red[i]) ** 2
                                 for k in range(self.n_iterations)) / self.n_iterations)
                   for i in range(33)]
        std_blue = [math.sqrt(sum((all_blue[k][i] - mean_blue[i]) ** 2
                                  for k in range(self.n_iterations)) / self.n_iterations)
                    for i in range(16)]

        red_entropy = ProbabilityOutput.entropy(mean_red)
        blue_entropy = ProbabilityOutput.entropy(mean_blue)

        return mean_red, mean_blue, std_red, std_blue, red_entropy, blue_entropy


# ==================== 概率校准 ====================

class Calibrator:
    """
    等频分箱概率校准 (简化 Platt Scaling)

    将模型输出的置信度校准到真实频率。
    """

    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.bins = None  # 校准映射

    def fit(self, predicted_probs_list, actual_values_list):
        """
        拟合约瑟映射

        Args:
            predicted_probs_list: [(n_classes,)] 预测概率
            actual_values_list: [[int]] 实际值 (1-indexed)
        """
        bins = [[] for _ in range(self.n_bins)]

        for probs, actuals in zip(predicted_probs_list, actual_values_list):
            for idx in range(len(probs)):
                conf = probs[idx]
                bin_idx = min(int(conf * self.n_bins), self.n_bins - 1)
                correct = 1.0 if (idx + 1) in actuals else 0.0
                bins[bin_idx].append((conf, correct))

        calibrated = []
        for bi, items in enumerate(bins):
            if items:
                avg_pred = sum(it[0] for it in items) / len(items)
                avg_true = sum(it[1] for it in items) / len(items)
            else:
                avg_pred = (bi + 0.5) / self.n_bins
                avg_true = 0.0
            calibrated.append({
                "bin": bi,
                "avg_prediction": avg_pred,
                "avg_actual": avg_true,
                "count": len(items),
            })

        self.bins = calibrated

    def calibrate(self, prob):
        """校准一个概率值"""
        if not self.bins:
            return prob
        bin_idx = min(int(prob * self.n_bins), self.n_bins - 1)
        return self.bins[bin_idx]["avg_actual"]

    def ece(self):
        """Expected Calibration Error"""
        if not self.bins:
            return 1.0
        total = sum(b["count"] for b in self.bins)
        if total == 0:
            return 1.0
        ece = sum(b["count"] * abs(b["avg_prediction"] - b["avg_actual"])
                  for b in self.bins) / total
        return ece


# ==================== 集成不确定性 ====================

class EnsembleUncertainty:
    """跨预测器分歧度"""

    @staticmethod
    def disagreement(predictions):
        """
        多个预测之间的分歧度

        Args:
            predictions: [{"红球": [6 ints], "蓝球": int}, ...]

        Returns:
            red_disagreement: 0-1, 越高越不确定
            blue_disagreement: 0-1
        """
        n = len(predictions)
        if n <= 1:
            return 0.0, 0.0

        # 红球: Jaccard 距离的均值
        all_reds = [set(p["红球"]) for p in predictions]
        joint = set.intersection(*all_reds) if all_reds else set()
        union = set.union(*all_reds) if all_reds else set()
        red_disagreement = 1.0 - (len(joint) / len(union)) if union else 0.0

        # 蓝球: 多数票占比
        blues = [p["蓝球"] for p in predictions]
        blue_counts = Counter(blues)
        max_agreement = max(blue_counts.values()) / n
        blue_disagreement = 1.0 - max_agreement

        return red_disagreement, blue_disagreement


# ==================== 工具函数 ====================

def _softmax(x):
    max_v = max(x)
    exps = [math.exp(v - max_v) for v in x]
    s = sum(exps)
    return [e / s for e in exps]


def format_probability_report(red_probs, blue_probs, top_k=10):
    """格式化的概率报告"""
    red_sorted = sorted([(i + 1, red_probs[i]) for i in range(33)],
                        key=lambda x: x[1], reverse=True)
    blue_sorted = sorted([(i + 1, blue_probs[i]) for i in range(16)],
                         key=lambda x: x[1], reverse=True)

    report = {
        "top_red": red_sorted[:top_k],
        "top_blue": blue_sorted[:5],
        "red_entropy": ProbabilityOutput.entropy(red_probs),
        "blue_entropy": ProbabilityOutput.entropy(blue_probs),
        "red_uncertainty": ProbabilityOutput.uncertainty_level(
            ProbabilityOutput.entropy(red_probs), 33),
        "blue_uncertainty": ProbabilityOutput.uncertainty_level(
            ProbabilityOutput.entropy(blue_probs), 16),
    }
    return report
