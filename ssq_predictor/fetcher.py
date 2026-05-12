#!/usr/bin/env python3
"""
双色球开奖数据爬虫 — 增量更新

- 自动检测最新期号，仅抓取缺失的数据
- 按开奖日 (周日/周二/周四) 智能调度
- 支持命令行和定时任务两种模式
- 更新 ssq_全历史.json

用法:
  python fetcher.py              # 检查并抓取最新数据
  python fetcher.py --all        # 强制全量刷新
  python fetcher.py --daemon     # 守护模式，开奖日后自动抓取
"""

import json
import os
import sys
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import requests

# ==================== 配置 ====================

API_URL = "https://jc.zhcw.com/port/client_json.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.zhcw.com/kjxx/ssq/",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

# 代理配置: 绕过本地代理直接访问
PROXIES = {
    "http": None,
    "https": None,
}

# 数据文件路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "ssq_全历史.json")
BACKUP_DIR = os.path.join(BASE_DIR, "data_backups")

# 双色球开奖日: Python weekday Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
DRAW_WEEKDAYS = {1, 3, 6}  # Tue, Thu, Sun

# 请求间隔 (秒)
REQUEST_DELAY = 0.5
BATCH_DELAY = 2.0

def load_existing(path: str = None) -> List[Dict]:
    """加载现有数据"""
    if path is None:
        path = DATA_PATH
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, Exception) as e:
        print(f"[WARN] 数据文件读取失败: {e}")
        return []


def save_data(data: List[Dict], path: str = None):
    """保存数据 (带备份)"""
    if path is None:
        path = DATA_PATH
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 备份旧文件
    if os.path.exists(path):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"ssq_全历史_{timestamp}.json")
        try:
            with open(path, "r", encoding="utf-8") as src:
                with open(backup_path, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
        except Exception:
            pass  # 备份失败不阻塞

    # 按时间排序 (最新在前)
    data.sort(key=lambda x: x.get("期号", ""), reverse=True)

    # 写入
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {len(data)} 条记录 → {path}")


# ==================== API 调用 ====================

def api_fetch(params: Dict, timeout: int = 30) -> Optional[Dict]:
    """
    调用中彩网 API，带重试

    Args:
        params: API 参数
        timeout: 超时秒数

    Returns:
        响应 JSON 或 None
    """
    for attempt in range(3):
        try:
            session = requests.Session()
            session.trust_env = True  # 使用系统代理
            resp = session.get(API_URL, params=params, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("resCode") == "000000":
                return data
            else:
                msg = data.get("message", "未知错误")
                print(f"  [API ERR] {msg}")
                return None
        except requests.RequestException as e:
            print(f"  [RETRY {attempt+1}/3] {e}")
            time.sleep(2 ** attempt)
        except json.JSONDecodeError:
            print(f"  [RETRY {attempt+1}/3] JSON 解析失败")
            time.sleep(2 ** attempt)
    return None


def fetch_issue_list(count: int = 50) -> Optional[List[str]]:
    """获取最新 N 期的期号列表"""
    params = {
        "transactionType": "10001003",
        "lotteryId": "1",
        "count": str(count),
    }
    data = api_fetch(params)
    if data and "issue" in data:
        return data["issue"]
    return None


def fetch_issue_detail(issue: str) -> Optional[Dict]:
    """获取单期开奖详情"""
    params = {
        "transactionType": "10001002",
        "lotteryId": "1",
        "issue": issue,
    }
    return api_fetch(params)


def parse_issue(data: Dict) -> Optional[Dict]:
    """解析 API 返回的开奖数据为标准格式"""
    if not data:
        return None

    red_str = data.get("frontWinningNum", "")
    blue_str = data.get("backWinningNum", "")

    if not red_str or not blue_str:
        return None

    # 标准化红球格式
    red_list = [x.strip().zfill(2) for x in red_str.split()]

    return {
        "期号": data.get("issue", ""),
        "开奖日期": data.get("openTime", ""),
        "星期": data.get("week", ""),
        "红球": " ".join(red_list),
        "蓝球": blue_str.strip().zfill(2),
        "红球顺序": data.get("seqFrontWinningNum", ""),
        "销售额": data.get("saleMoney", "0"),
        "奖池金额": data.get("prizePoolMoney", "0"),
        "一等奖注数": _extract_prize_count(data, "1"),
        "一等奖奖金": _extract_prize_money(data, "1"),
    }


def _extract_prize_count(data: Dict, level: str) -> str:
    """提取某等奖注数"""
    for w in data.get("winnerDetails", []):
        if w.get("awardEtc") == level:
            return str(w.get("baseBetWinner", {}).get("awardNum", "0"))
    return "0"


def _extract_prize_money(data: Dict, level: str) -> str:
    """提取某等奖奖金"""
    for w in data.get("winnerDetails", []):
        if w.get("awardEtc") == level:
            return str(w.get("baseBetWinner", {}).get("awardMoney", "0"))
    return "0"


# ==================== 核心逻辑 ====================

def get_latest_issue(data: List[Dict]) -> Optional[str]:
    """获取已有数据中的最新期号"""
    if not data:
        return None
    # 数据按时间倒序，第一个是最新的
    return data[0].get("期号", None) if data else None


def get_missing_issues(existing_data: List[Dict], lookback: int = 30) -> List[str]:
    """
    比较已有数据和 API 返回的期号列表，找出缺失的期号

    Args:
        existing_data: 现有数据
        lookback: 向 API 查询多少期

    Returns:
        缺失的期号列表 (从早到晚排列，保证插入顺序正确)
    """
    existing_issues = {r.get("期号") for r in existing_data}
    api_issues = fetch_issue_list(lookback)

    if not api_issues:
        print("[WARN] 无法获取期号列表")
        return []

    missing = [iss for iss in api_issues if iss not in existing_issues]

    if missing:
        print(f"[INFO] 缺失 {len(missing)} 期: {missing}")
    else:
        print(f"[INFO] 数据已是最新 (已检查最近 {lookback} 期)")

    # API 返回从新到旧，反转为从旧到新 (方便按序追加)
    missing.reverse()
    return missing


def fetch_and_append(existing_data: List[Dict], missing_issues: List[str]) -> Tuple[List[Dict], int]:
    """
    抓取缺失的期号并追加到数据中

    Returns:
        (更新后的数据, 成功抓取数)
    """
    data = list(existing_data)
    success = 0

    for i, issue in enumerate(missing_issues, 1):
        print(f"  [{i}/{len(missing_issues)}] 第 {issue} 期 ... ", end="", flush=True)

        detail = fetch_issue_detail(issue)
        parsed = parse_issue(detail)

        if parsed:
            data.append(parsed)
            success += 1
            print(f"OK  红球={parsed['红球']} 蓝球={parsed['蓝球']}")
        else:
            print("FAIL")

        # 请求间隔
        if i < len(missing_issues):
            time.sleep(REQUEST_DELAY)

    return data, success


# ==================== 日期工具 ====================

def is_draw_day(dt: datetime = None) -> bool:
    """判断某天是否为双色球开奖日 (周二/周四/周日)"""
    if dt is None:
        dt = datetime.now()
    # weekday(): Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
    return dt.weekday() in {1, 3, 6}


def last_draw_date(dt: datetime = None) -> datetime:
    """获取最近的一个开奖日 (含今天)"""
    if dt is None:
        dt = datetime.now()
    while dt.weekday() not in {1, 3, 6}:
        dt = dt - timedelta(days=1)
    return dt


def next_draw_date(dt: datetime = None) -> datetime:
    """获取下一个开奖日 (不含今天)"""
    if dt is None:
        dt = datetime.now()
    dt = dt + timedelta(days=1)
    return last_draw_date(dt)


def seconds_until(target_dt: datetime) -> float:
    """计算距离开奖时间的秒数 (开奖时间: 晚上21:15)"""
    target = target_dt.replace(hour=21, minute=15, second=0, microsecond=0)
    return (target - datetime.now()).total_seconds()


# ==================== 主命令 ====================

def cmd_update(lookback: int = 50):
    """增量更新: 仅抓取缺失的数据"""
    print("=" * 60)
    print(f"双色球数据更新  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 加载已有数据
    print(f"\n[LOAD] {DATA_PATH}")
    data = load_existing()
    print(f"   已有 {len(data)} 期")

    if data:
        latest = data[0]
        print(f"   最新: 第{latest['期号']}期 ({latest['开奖日期']})  "
              f"红球={latest['红球']} 蓝球={latest['蓝球']}")

    # 2. 检查缺失
    print(f"\n[CHECK] 检查最近 {lookback} 期 ...")
    missing = get_missing_issues(data, lookback)

    if not missing:
        print("\n数据已是最新，无需更新。")
        return data

    # 3. 抓取
    print(f"\n[FETCH] 抓取 {len(missing)} 期 ...")
    data, success = fetch_and_append(data, missing)
    print(f"   成功: {success}/{len(missing)}")

    # 4. 保存
    if success > 0:
        print(f"\n[SAVE] 保存更新 ...")
        save_data(data)
        print(f"   {len(data)} 期 (新增 {success} 期)")

    return data


def cmd_full_refresh():
    """全量刷新: 重新抓取全部历史数据"""
    print("=" * 60)
    print(f"双色球全量刷新  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 获取全部期号 (SSQ 从 2003001 开始)
    print("\n[FETCH] 获取全部期号 ...")
    all_issues = fetch_issue_list(3500)
    if not all_issues:
        print("[ERROR] 无法获取期号列表")
        return []

    print(f"   获取到 {len(all_issues)} 期: {all_issues[0]} ~ {all_issues[-1]}")

    # 从旧到新抓取
    all_issues.reverse()
    data = []
    total = len(all_issues)
    errors = 0

    print(f"\n[FETCH] 逐期抓取 ({total} 期) ...")
    start_time = time.time()

    for i, issue in enumerate(all_issues, 1):
        if i % 200 == 0 or i == 1 or i == total:
            elapsed = time.time() - start_time
            speed = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / speed if speed > 0 else 0
            print(f"  [{i}/{total}] 当前:{issue}  "
                  f"已用:{elapsed:.0f}s  速度:{speed:.1f}期/s  预计剩余:{eta:.0f}s")

        detail = fetch_issue_detail(issue)
        parsed = parse_issue(detail)
        if parsed:
            data.append(parsed)
        else:
            errors += 1

        if i % 100 == 0:
            time.sleep(BATCH_DELAY)
        elif i < total:
            time.sleep(REQUEST_DELAY * 0.5)

    elapsed = time.time() - start_time
    print(f"\n[DONE] {len(data)} 期成功, {errors} 期失败, 耗时 {elapsed:.0f}s")

    # 保存
    save_data(data)
    return data


def cmd_daemon():
    """守护模式: 在开奖日后自动检查更新"""
    print("=" * 60)
    print("双色球数据守护进程")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n开奖日: 每周二、四、日 21:15")
    print("守护进程将在开奖日 21:30 自动检查并抓取数据")
    print("按 Ctrl+C 退出\n")

    last_processed = None

    while True:
        now = datetime.now()

        # 只在开奖日检查
        if now.weekday() in {1, 3, 6}:  # Tue, Thu, Sun
            # 在 21:30 之后检查 (开奖后15分钟)
            check_time = now.replace(hour=21, minute=30, second=0, microsecond=0)

            if now >= check_time:
                today_str = now.strftime("%Y%m%d")
                if last_processed != today_str:
                    print(f"\n{'='*60}")
                    print(f"[DAEMON] {now.strftime('%Y-%m-%d %H:%M:%S')}  检查新数据 ...")
                    print(f"{'='*60}")
                    try:
                        cmd_update(lookback=10)
                        last_processed = today_str
                    except Exception as e:
                        print(f"[ERROR] 更新失败: {e}")
                else:
                    pass  # 今天已处理过

        # 计算下次检查时间
        next_draw = next_draw_date(now)
        next_check = next_draw.replace(hour=21, minute=30, second=0, microsecond=0)

        if next_check <= now:
            next_check = next_check + timedelta(days=1)

        wait = (next_check - now).total_seconds()
        wait = min(wait, 3600)  # 最多等1小时

        # 显示等待状态
        next_draw_str = next_draw.strftime("%m-%d %a")
        print(f"\r[IDLE] 下一个开奖日: {next_draw_str}, "
              f"下次检查: {next_check.strftime('%H:%M')}  "
              f"(等待 {wait/60:.0f} 分钟)     ", end="", flush=True)

        time.sleep(min(wait, 300))  # 每5分钟刷新一次状态


def cmd_status():
    """显示数据状态"""
    data = load_existing()
    if not data:
        print("暂无数据")
        return

    latest = data[0]
    oldest = data[-1]
    n = len(data)

    # 检查是否遗漏
    all_issues = fetch_issue_list(30)
    existing_set = {r["期号"] for r in data}
    missing_online = [iss for iss in (all_issues or []) if iss not in existing_set]

    print("=" * 60)
    print("双色球数据状态")
    print("=" * 60)
    print(f"  数据文件: {DATA_PATH}")
    print(f"  总期数:   {n}")
    print(f"  最早:     第{oldest['期号']}期 ({oldest['开奖日期']})")
    print(f"  最新:     第{latest['期号']}期 ({latest['开奖日期']})")
    print(f"  最近开奖: 红球={latest['红球']}  蓝球={latest['蓝球']}")
    print(f"  在线缺失: {len(missing_online)} 期")
    if missing_online:
        print(f"             {missing_online}")

    # 计算覆盖天数
    try:
        d1 = datetime.strptime(oldest["开奖日期"], "%Y-%m-%d")
        d2 = datetime.strptime(latest["开奖日期"], "%Y-%m-%d")
        days = (d2 - d1).days
        print(f"  覆盖:      {days} 天 ({oldest['开奖日期']} ~ {latest['开奖日期']})")
    except Exception:
        pass

    # 预计还能获取的期数
    today = datetime.now()
    last_draw = last_draw_date(today)
    days_since = (today - datetime.strptime(latest["开奖日期"], "%Y-%m-%d")).days
    missing_est = max(0, days_since // 2)  # 每周3期, 约2天一期
    print(f"  距上次更新: {days_since} 天 (约遗漏 {missing_est} 期)")
    print(f"  下次开奖:   {next_draw_date(today).strftime('%Y-%m-%d %a')} 21:15")


# ==================== CLI ====================

def main():
    global DATA_PATH

    import argparse

    parser = argparse.ArgumentParser(description="双色球开奖数据爬虫")
    parser.add_argument("--update", "-u", action="store_true",
                        help="增量更新 (默认)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="全量刷新所有历史数据")
    parser.add_argument("--daemon", "-d", action="store_true",
                        help="守护模式，开奖日后自动抓取")
    parser.add_argument("--status", "-s", action="store_true",
                        help="查看数据状态")
    parser.add_argument("--lookback", "-n", type=int, default=50,
                        help="增量更新时检查的期数 (默认50)")
    parser.add_argument("--data-path", type=str, default="",
                        help="数据文件路径 (默认使用内置路径)")

    args = parser.parse_args()

    if args.data_path:
        DATA_PATH = args.data_path

    if args.all:
        cmd_full_refresh()
    elif args.daemon:
        cmd_daemon()
    elif args.status:
        cmd_status()
    else:
        cmd_update(lookback=args.lookback)


if __name__ == "__main__":
    main()
