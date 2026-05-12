#!/usr/bin/env python3
"""
SSQ Predictor — 统一入口

使用方法:
    python main.py              # 完整预测管线
    python main.py --backtest   # 回测模式
    python main.py --quick      # 快速预测 (少量MCMC采样)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_history, load_parsed, get_data_summary
from predictor import SSQPredictor
from backtest import backtest_walk_forward, compare_to_baseline, random_baseline
from cross_validation import TimeSeriesCV


def cmd_predict(args):
    """完整预测"""
    print("=" * 60)
    print("SSQ Predictor — 双色球预测")
    print("=" * 60)

    # 加载数据
    data = load_parsed()
    summary = get_data_summary(load_history())
    print(f"\n数据: {summary['总期数']} 期")
    print(f"范围: {summary['日期范围']}")
    print(f"最新: 第{summary['最新一期']}期")

    # 准备模型
    predictor = SSQPredictor(data, half_life=args.half_life)
    predictor.mc_n_samples = args.mc_samples

    predictor.prepare(
        train_mcmc=not args.skip_mcmc,
        mine_associations=not args.skip_assoc,
        min_support=args.min_support,
    )

    # 预测
    result = predictor.predict(
        top_n=args.top_n,
        n_mc_samples=args.mc_samples,
    )

    predictor.print_report(result)

    # 如果启用NN，添加NN预测
    if not args.skip_nn:
        print("\n[NN模型] 请使用 train_nn.py 单独训练序列模型")

    return result


def cmd_backtest(args):
    """回测模式"""
    print("=" * 60)
    print("SSQ Predictor — 回测验证")
    print("=" * 60)

    data = load_parsed()
    summary = get_data_summary(load_history())
    print(f"\n数据: {summary['总期数']} 期")

    # 简单基线预测函数
    def simple_predict(train_data):
        predictor = SSQPredictor(
            train_data, half_life=args.half_life
        )
        predictor.prepare(
            train_mcmc=True,
            mine_associations=True,
            min_support=args.min_support,
        )
        predictor.mc_n_samples = args.mc_samples
        result = predictor.predict(top_n=1, n_mc_samples=args.mc_samples)
        if result["recommendations"]:
            best = result["recommendations"][0]
            return {"红球": best["红球"], "蓝球": best["蓝球"]}
        return None

    print(f"\n运行前向回测...")
    bt_result = backtest_walk_forward(
        data, simple_predict, n_folds=args.n_folds, train_ratio=0.85
    )

    if "error" in bt_result:
        print(f"回测失败: {bt_result['error']}")
        return

    # 对比基线
    comparison = compare_to_baseline(bt_result)
    baseline = random_baseline(100)

    print(f"\n[回测结果]")
    agg = bt_result["aggregate"]
    print(f"  折数: {agg['n_folds']}")
    print(f"  红球命中率: {agg['mean_red_hit_rate']:.4f} "
          f"(随机基线: {baseline['expected_red_hit_rate']:.4f})")
    print(f"  蓝球命中率: {agg['mean_blue_hit_rate']:.4f} "
          f"(随机基线: {baseline['expected_blue_hit_rate']:.4f})")
    print(f"  超额红球命中: {comparison.get('excess_red', 0):.4f}")
    print(f"  信息比率: {comparison.get('information_ratio_red', 0):.3f}")
    print(f"  一致性: {agg['consistency']:.2f} "
          f"({agg['consistency']*100:.0f}%折超过随机)")
    print(f"\n  评估: {comparison.get('assessment', 'N/A')}")

    print(f"\n[逐折详情]")
    for fold_result in bt_result["folds"]:
        print(f"  Fold {fold_result['fold']}: "
              f"红球={fold_result['red_hit_rate']:.4f}, "
              f"蓝球={fold_result['blue_hit_rate']:.4f}, "
              f"超额命中={fold_result['excess_hits']:.1f}")


def cmd_quick(args):
    """快速预测 (少量采样)"""
    args.mc_samples = 10000
    args.top_n = 5
    return cmd_predict(args)


def main():
    parser = argparse.ArgumentParser(description="SSQ Predictor")
    parser.add_argument("--backtest", action="store_true", help="回测模式")
    parser.add_argument("--quick", action="store_true", help="快速预测")
    parser.add_argument("--half-life", type=int, default=200, help="时间衰减半衰期(期)")
    parser.add_argument("--mc-samples", type=int, default=50000, help="MCMC采样数")
    parser.add_argument("--top-n", type=int, default=10, help="推荐方案数")
    parser.add_argument("--min-support", type=float, default=0.008, help="关联规则最小支持度")
    parser.add_argument("--n-folds", type=int, default=5, help="回测折数")
    parser.add_argument("--skip-mcmc", action="store_true", help="跳过MCMC")
    parser.add_argument("--skip-assoc", action="store_true", help="跳过关联规则")
    parser.add_argument("--skip-nn", action="store_true", help="跳过NN模型")

    args = parser.parse_args()

    if args.backtest:
        cmd_backtest(args)
    elif args.quick:
        cmd_quick(args)
    else:
        cmd_predict(args)


if __name__ == "__main__":
    main()
