#!/usr/bin/env python3
"""
性能加权集成 — 多策略融合，按滚动窗口表现动态调整权重

集成成员:
1. MCMC 采样排名 (mc_mcmc)
2. 时序自注意力模型 (nn_attention)
3. 衰减频率基线 (stats_core)
4. 马尔可夫转移概率 (stats_core)
5. 热号策略 (cold_hot_analysis)
6. 冷号策略 (cold_hot_analysis)
7. 关联规则推荐 (association_mining)
"""

from collections import Counter


class RollingWeightedEnsemble:
    """
    滚动性能加权集成

    每个成员按近期表现获得权重:
    w_i(t+1) = beta * w_i(t) + (1-beta) * score_i(t)
    其中 score_i(t) = 该成员在最新一期上的红球命中数/6
    """

    def __init__(self, beta=0.9):
        """
        Args:
            beta: 权重平滑系数 (越大变化越慢)
        """
        self.beta = beta
        self.members = {}       # {name: predict_fn}
        self.weights = {}       # {name: weight}
        self.initial_weight = 1.0
        self.performance_history = {}  # {name: [scores]}

    def add_member(self, name, predict_fn, weight=None):
        """
        添加集成成员

        Args:
            name: 成员名称
            predict_fn: prediction_fn(features) → {"红球": [6], "蓝球": int}
            weight: 初始权重 (默认1.0)
        """
        self.members[name] = predict_fn
        w = weight if weight is not None else self.initial_weight
        self.weights[name] = w
        self.performance_history[name] = []

    def remove_member(self, name):
        """移除成员"""
        self.members.pop(name, None)
        self.weights.pop(name, None)
        self.performance_history.pop(name, None)

    def predict(self, context=None):
        """
        加权投票预测

        Args:
            context: 传递给各成员预测函数的上下文参数

        Returns:
            {"红球": [6], "蓝球": int, "confidence": float, "details": {...}}
        """
        if not self.members:
            return {"红球": [], "蓝球": None, "confidence": 0.0}

        red_votes = Counter()
        blue_votes = Counter()
        member_predictions = {}

        for name, fn in self.members.items():
            try:
                if context is not None:
                    result = fn(context)
                else:
                    result = fn()
                if result is None:
                    continue
                w = self.weights.get(name, 1.0)

                for r in result.get("红球", []):
                    red_votes[r] += w
                blue = result.get("蓝球")
                if isinstance(blue, int):
                    blue_votes[blue] += w
                elif isinstance(blue, list) and len(blue) > 0:
                    blue_votes[blue[0]] += w

                member_predictions[name] = result
            except Exception as e:
                print(f"  [WARNING] 成员 {name} 预测失败: {e}")
                continue

        if not red_votes:
            return {"红球": [], "蓝球": None, "confidence": 0.0}

        # Top-6 by weighted vote
        top_reds = [r for r, _ in red_votes.most_common(6)]
        top_blue = blue_votes.most_common(1)[0][0] if blue_votes else None

        # 置信度
        total_votes = sum(red_votes.values())
        confidence = sum(red_votes[r] for r in top_reds) / max(total_votes, 1)

        return {
            "红球": sorted(top_reds[:6]),
            "蓝球": top_blue,
            "confidence": confidence,
            "red_votes": dict(red_votes.most_common(10)),
            "blue_votes": dict(blue_votes.most_common(5)),
            "member_predictions": member_predictions,
        }

    def update_weights(self, actual_reds, actual_blue):
        """
        根据实际开奖结果更新权重

        Args:
            actual_reds: [int] 实际红球
            actual_blue: int 实际蓝球
        """
        actual_red_set = set(actual_reds)

        for name in self.members:
            if name not in self.member_predictions_cache:
                continue

            pred = self.member_predictions_cache.get(name, {})
            pred_reds = set(pred.get("红球", []))
            pred_blue = pred.get("蓝球")

            red_hits = len(pred_reds & actual_red_set)
            blue_hit = 1 if pred_blue == actual_blue else 0

            # 综合评分: 红球命中占70%, 蓝球占30%
            score = 0.7 * (red_hits / 6) + 0.3 * blue_hit

            self.weights[name] = (self.beta * self.weights[name] +
                                  (1 - self.beta) * score)
            self.performance_history[name].append(score)

    member_predictions_cache = {}

    def predict_and_cache(self, context=None):
        """预测并缓存各成员结果，用于后续 update_weights"""
        result = self.predict(context)
        self.member_predictions_cache = result.get("member_predictions", {})
        return result

    def get_weights_summary(self):
        """返回当前权重摘要"""
        if not self.weights:
            return {}
        total = sum(self.weights.values())
        return {
            name: {
                "weight": w,
                "normalized": w / total if total > 0 else 0,
                "recent_performance": (
                    sum(self.performance_history.get(name, [])[-10:]) / 10
                    if self.performance_history.get(name) else 0
                ),
            }
            for name, w in sorted(self.weights.items(),
                                  key=lambda x: x[1], reverse=True)
        }


# ==================== 策略函数工厂 ====================

def make_frequency_strategy(freq_red, freq_blue):
    """基于频率的最可能号码"""
    def predict():
        sorted_red = sorted(freq_red.items(), key=lambda x: x[1], reverse=True)
        sorted_blue = sorted(freq_blue.items(), key=lambda x: x[1], reverse=True)
        return {
            "红球": sorted([b for b, _ in sorted_red[:6]]),
            "蓝球": sorted_blue[0][0],
        }
    return predict


def make_markov_strategy(cond_red, freq_red, freq_blue, last_reds):
    """基于马尔可夫条件概率的预测"""
    def predict():
        selected = []
        candidates = list(range(1, 34))
        for _ in range(6):
            best_ball = None
            best_score = -1
            for c in candidates:
                if c in selected:
                    continue
                if selected:
                    avg_cond = sum(cond_red[s].get(c, 0.0) for s in selected) / len(selected)
                    score = avg_cond * freq_red.get(c, 0.0) / max(sum(freq_red.values()), 1)
                else:
                    score = freq_red.get(c, 0.0) / max(sum(freq_red.values()), 1)
                if score > best_score:
                    best_score = score
                    best_ball = c
            if best_ball is not None:
                selected.append(best_ball)
        return {"红球": sorted(selected[:6]), "蓝球": max(freq_blue, key=freq_blue.get) if freq_blue else 1}
    return predict


def make_hot_strategy(hot_reds, hot_blue, freq_red):
    """热号策略: 选最热的号码"""
    def predict():
        reds = list(hot_reds[:6]) if len(hot_reds) >= 6 else list(hot_reds)
        if len(reds) < 6:
            remaining = sorted([b for b in range(1, 34) if b not in reds],
                               key=lambda x: freq_red.get(x, 0), reverse=True)
            reds.extend(remaining[:6 - len(reds)])
        return {"红球": sorted(reds[:6]), "蓝球": hot_blue[0] if hot_blue else 1}
    return predict


def make_cold_strategy(cold_reds, cold_blue, freq_red):
    """冷号策略: 选最冷的号码 (赌回补)"""
    def predict():
        reds = list(cold_reds[:6]) if len(cold_reds) >= 6 else list(cold_reds)
        if len(reds) < 6:
            remaining = sorted([b for b in range(1, 34) if b not in reds],
                               key=lambda x: freq_red.get(x, 0))
            reds.extend(remaining[:6 - len(reds)])
        return {"红球": sorted(reds[:6]), "蓝球": cold_blue[0] if cold_blue else 1}
    return predict


def make_association_strategy(pair_scores, freq_red, freq_blue):
    """基于关联规则的策略: 选中lift最高的pair，补齐"""
    def predict():
        if not pair_scores:
            sorted_red = sorted(freq_red.items(), key=lambda x: x[1], reverse=True)
            return {"红球": sorted([b for b, _ in sorted_red[:6]]),
                    "蓝球": max(freq_blue, key=freq_blue.get) if freq_blue else 1}

        # 找lift最高的一对
        best_pair = max(pair_scores.items(), key=lambda x: x[1])
        a, b = best_pair[0]
        selected = [a, b]

        # 贪心补齐
        candidates = [x for x in range(1, 34) if x not in selected]
        for _ in range(4):
            best_ball = None
            best_score = -1
            for c in candidates:
                lifts = [pair_scores.get(tuple(sorted([c, s])), 1.0) for s in selected]
                score = sum(lifts) / len(lifts) * freq_red.get(c, 0) / max(sum(freq_red.values()), 1)
                if score > best_score:
                    best_score = score
                    best_ball = c
            if best_ball is not None:
                selected.append(best_ball)
                candidates.remove(best_ball)

        return {"红球": sorted(selected[:6]),
                "蓝球": max(freq_blue, key=freq_blue.get) if freq_blue else 1}
    return predict
