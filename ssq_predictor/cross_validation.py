#!/usr/bin/env python3
"""
时间序列交叉验证框架

所有切分尊重时间顺序 — 绝不随机打乱。
支持: 扩展窗口、滚动窗口、锚定前向验证。
"""

import math
from collections import Counter


class TimeSeriesCV:
    """时间序列交叉验证"""

    def __init__(self, data):
        """
        Args:
            data: 时间倒序列表 (idx 0 = 最新)
        """
        self.data = data
        self.n = len(data)

    def expanding_window(self, n_train_min, n_val, n_step=None):
        """
        扩展窗口: 训练集不断增大

        fold 0: train=[total-val : total], val=[total-val-n_val : total-val]
        fold 1: train=[total-val-step : total], val=[...]
        ...
        """
        if n_step is None:
            n_step = n_val
        folds = []
        start = self.n - n_train_min - n_val
        while start >= 0:
            val_end = start + n_val
            train_start = val_end
            train_end = self.n
            folds.append((list(range(train_start, train_end)),
                          list(range(start, val_end))))
            start -= n_step
        return folds

    def rolling_window(self, n_train, n_val, n_step=None):
        """
        滚动窗口: 固定训练集大小向前滑动
        """
        if n_step is None:
            n_step = n_val
        folds = []
        start = self.n - n_train - n_val
        while start >= 0:
            val_start = start
            val_end = start + n_val
            train_start = val_end
            train_end = train_start + n_train
            folds.append((list(range(train_start, train_end)),
                          list(range(val_start, val_end))))
            start -= n_step
        return folds

    def anchored_walk_forward(self, anchors=None):
        """
        锚定前向验证: 在固定比例处切分
        """
        if anchors is None:
            anchors = [0.80, 0.85, 0.90, 0.95]
        folds = []
        for anchor in anchors:
            split = int(self.n * anchor)
            folds.append((list(range(split, self.n)),
                          list(range(0, split))))
        return folds

    def evaluate(self, fold, predict_fn):
        """
        在单折上评估

        Args:
            fold: (train_indices, val_indices)
            predict_fn: fn(train_data) → {"红球": [...], "蓝球": int}

        Returns:
            metrics dict
        """
        train_idx, val_idx = fold
        train_data = [self.data[i] for i in train_idx]
        val_data = [self.data[i] for i in val_idx]

        # 生成预测
        prediction = predict_fn(train_data)

        if prediction is None:
            return {"error": "预测失败"}

        pred_reds = set(prediction.get("红球", []))
        pred_blue = prediction.get("蓝球")

        total = len(val_data)
        if total == 0 or not pred_reds:
            return {"n_val": total, "error": "无效预测"}

        red_hits = 0
        blue_hits = 0
        match_3 = 0
        match_4 = 0
        match_5 = 0
        match_6 = 0

        for record in val_data:
            actual_reds = set(record["红球"])
            actual_blue = record["蓝球"]

            hits = len(pred_reds & actual_reds)
            red_hits += hits
            if actual_blue == pred_blue:
                blue_hits += 1
            if hits >= 3:
                match_3 += 1
            if hits >= 4:
                match_4 += 1
            if hits >= 5:
                match_5 += 1
            if hits == 6:
                match_6 += 1

        return {
            "n_train": len(train_idx),
            "n_val": total,
            "red_hit_rate": red_hits / (total * 6),
            "blue_hit_rate": blue_hits / total,
            "avg_red_hits": red_hits / total,
            "red_3plus_rate": match_3 / total,
            "red_4plus_rate": match_4 / total,
            "red_5plus_rate": match_5 / total,
            "red_6_rate": match_6 / total,
        }

    def run_cv(self, predict_fn, method="expanding", **kwargs):
        """
        运行完整CV

        Returns:
            {
                "folds": [metrics, ...],
                "summary": {mean, std, ...},
                "consistency": float (超随机基线的比例)
            }
        """
        if method == "expanding":
            folds = self.expanding_window(**kwargs)
        elif method == "rolling":
            folds = self.rolling_window(**kwargs)
        elif method == "anchored":
            folds = self.anchored_walk_forward(**kwargs)
        else:
            raise ValueError(f"未知CV方法: {method}")

        if not folds:
            return {"error": "无法生成CV切分"}

        results = []
        for i, fold in enumerate(folds):
            metrics = self.evaluate(fold, predict_fn)
            metrics["fold"] = i
            results.append(metrics)

        # 汇总
        red_rates = [r.get("red_hit_rate", 0) for r in results if "red_hit_rate" in r]
        blue_rates = [r.get("blue_hit_rate", 0) for r in results if "blue_hit_rate" in r]
        red_3plus = [r.get("red_3plus_rate", 0) for r in results if "red_3plus_rate" in r]

        n_valid = len(red_rates)
        if n_valid == 0:
            return {"error": "没有有效的CV结果"}

        # 随机基线
        random_red = 6.0 / 33.0      # 0.1818
        random_blue = 1.0 / 16.0     # 0.0625

        summary = {
            "n_folds": len(folds),
            "n_valid_folds": n_valid,
            "mean_red_hit_rate": sum(red_rates) / n_valid,
            "std_red_hit_rate": _std(red_rates),
            "mean_blue_hit_rate": sum(blue_rates) / n_valid,
            "std_blue_hit_rate": _std(blue_rates),
            "mean_red_3plus_rate": sum(red_3plus) / n_valid,
            "red_vs_random": (sum(red_rates) / n_valid) / random_red,
            "blue_vs_random": (sum(blue_rates) / n_valid) / random_blue,
            "consistency_red": sum(1 for r in red_rates if r > random_red) / n_valid,
            "consistency_blue": sum(1 for r in blue_rates if r > random_blue) / n_valid,
        }

        return {"folds": results, "summary": summary}


def _std(values):
    if len(values) < 2:
        return 0.0
    mean_val = sum(values) / len(values)
    return math.sqrt(sum((v - mean_val) ** 2 for v in values) / len(values))
