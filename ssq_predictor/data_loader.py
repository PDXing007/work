#!/usr/bin/env python3
"""
统一数据加载 + 时间序列CV切分工厂

数据文件: ../ssq_全历史.json
字段: 期号, 开奖日期, 星期, 红球, 蓝球, 红球顺序, 销售额, 奖池金额

设计原则:
- 数据按时间排序（最新在前，index 0 = 最新一期）
- 所有切分尊重时间顺序，绝不随机打乱
"""

import json
import os
import sys
from typing import List, Dict, Tuple, Optional


def _find_data_file():
    """Find ssq_全历史.json across different platforms"""
    # Try current dir first (works on Android and when running from project root)
    if os.path.exists("ssq_全历史.json"):
        return "ssq_全历史.json"
    # Try relative to this source file
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in (".", "..", "../.."):
        path = os.path.join(here, rel, "ssq_全历史.json")
        if os.path.exists(path):
            return path
    # Try getcwd variants
    for sub in (".", "app", "files", "files/app"):
        path = os.path.join(os.getcwd(), sub, "ssq_全历史.json")
        if os.path.exists(path):
            return path
    # Default (will raise FileNotFoundError if missing)
    return os.path.join(os.getcwd(), "ssq_全历史.json")


DATA_PATH = _find_data_file()


def load_history(path: str = DATA_PATH) -> List[Dict]:
    """加载全部历史数据，返回按时间倒序的列表（最新在前）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def parse_record(record: Dict) -> Dict:
    """将原始记录解析为标准格式"""
    return {
        "期号": record["期号"],
        "日期": record.get("开奖日期", ""),
        "星期": record.get("星期", ""),
        "红球": [int(x) for x in record["红球"].split()],
        "蓝球": int(record["蓝球"]),
        "红球顺序": [int(x) for x in record.get("红球顺序", record["红球"]).split()],
        "销售额": int(record.get("销售额", 0)),
        "奖池金额": int(record.get("奖池金额", 0)),
    }


def load_parsed(path: str = DATA_PATH) -> List[Dict]:
    """加载并解析全部数据"""
    return [parse_record(r) for r in load_history(path)]


def get_data_summary(data: List[Dict]) -> Dict:
    """返回数据集摘要信息"""
    parsed = [parse_record(r) for r in data]
    n = len(parsed)
    return {
        "总期数": n,
        "最新一期": parsed[0]["期号"],
        "最早一期": parsed[-1]["期号"],
        "日期范围": f"{parsed[-1]['日期']} ~ {parsed[0]['日期']}",
        "红球范围": (1, 33),
        "蓝球范围": (1, 16),
    }


# ==================== 时间序列CV切分 ====================

def split_expanding(data: List, n_train_min: int, n_val: int, n_step: int) -> List[Tuple[List, List]]:
    """
    扩展窗口切分: 训练集不断增长，验证集向前滑动

    Args:
        data: 时间倒序列表 (idx 0 = 最新)
        n_train_min: 最小训练集大小
        n_val: 验证集大小
        n_step: 滑动步长

    Returns:
        [(train_indices, val_indices), ...] 从早到晚排列
    """
    folds = []
    total = len(data)
    # 从最早的数据开始构建切分
    start = total - n_train_min - n_val
    while start >= 0:
        val_end = start + n_val
        train_start = val_end
        train_end = min(train_start + n_train_min + (total - n_train_min - n_val - start), total)
        folds.append((
            list(range(train_start, train_end)),
            list(range(start, val_end)),
        ))
        start -= n_step
    return folds


def split_rolling(data: List, n_train: int, n_val: int, n_step: int) -> List[Tuple[List, List]]:
    """
    滚动窗口切分: 固定窗口大小的训练集向前滚动

    Args:
        data: 时间倒序列表
        n_train: 训练集大小
        n_val: 验证集大小
        n_step: 滚动步长

    Returns:
        [(train_indices, val_indices), ...]
    """
    folds = []
    total = len(data)
    start = total - n_train - n_val
    while start >= 0:
        val_start = start
        val_end = start + n_val
        train_start = val_end
        train_end = train_start + n_train
        folds.append((
            list(range(train_start, train_end)),
            list(range(val_start, val_end)),
        ))
        start -= n_step
    return folds


def split_anchored(data: List, anchors: List[float] = None) -> List[Tuple[List, List]]:
    """
    锚定前向验证: 在固定时间点切分，训练=该点之前，验证=该点之后

    Args:
        data: 时间倒序列表
        anchors: 锚点比例列表，如 [0.80, 0.85, 0.90, 0.95]

    Returns:
        [(train_indices, val_indices), ...]
    """
    if anchors is None:
        anchors = [0.80, 0.85, 0.90, 0.95]
    total = len(data)
    folds = []
    for anchor in anchors:
        split_point = int(total * anchor)
        folds.append((
            list(range(split_point, total)),  # 较早的数据做训练
            list(range(0, split_point)),       # 较新的数据做验证
        ))
    return folds


def split_by_date(data: List, cutoff_date: str) -> Tuple[List, List]:
    """
    按日期切分: cutoff_date 之前的数据训练，之后验证

    Args:
        data: 时间倒序列表
        cutoff_date: 切分日期 "YYYY-MM-DD"

    Returns:
        (train_indices, val_indices)
    """
    train_indices = []
    val_indices = []
    for i, record in enumerate(data):
        date = record.get("开奖日期", "")
        if not date:
            continue
        # data is newest-first. Newer dates (after cutoff) → val, older → train.
        if date > cutoff_date:
            val_indices.append(i)
        else:
            train_indices.append(i)
    return train_indices, val_indices
