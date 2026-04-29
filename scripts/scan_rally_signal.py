#!/usr/bin/env python3
"""
史诗级行情扫描器 v1.7 — 基于第二步"验证有行情"
每天扫描涨幅>=7%的股票，检查是否符合"有行情"信号

v1.6升级：启动日还原打分（针对4板以上股票）
- 问题：4板以上股票在山顶视角评分只有1-2分，但启动日角度可能得3-4分
- 解决：对4板以上股票，增加"启动日还原打分"（从真实启动日重新跑第二步信号）
- 触发条件：board_count >= 4
- 用途：识别"本来就不是黑马" vs "被山顶视角低估的黑马"

信号逻辑（第二步核心）：
1. 启动前背离：股价跌 / 融资余额增（持续至少5天）
2. 启动后持续：融资余额连续增加
3. 缩量加速：股价涨但量比下降
4. 连续涨停：2天+

性能优化：融资数据按日期批量查（每日1次API，不是每股票1次）

用法:
    python scan_rally_signal.py [日期YYYYMMDD]
    不带参数默认上一个交易日

依赖: tushare, pandas, numpy
安装: pip install tushare pandas numpy
"""

import sys
import os
import warnings
import time
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/yrflj/.hermes/hermes-agent/venv/lib/python3.11/site-packages')
import tushare as ts
import pandas as pd
import numpy as np

# ========== 配置 ==========
TOKEN = 'fe284a8656a25bf46f4b9092178d1a4dd4d63f26bdb8b83d7f461c47'
TRADE_DAYS_BACK = 30    # 往前查多少个交易日做背离分析
MIN_RISE_PCT = 7.0      # 涨幅门槛（%）
MARGIN_DIVERGENCE_DAYS = 5  # 背离需要持续多少天
CONTINUOUS_MARGIN_DAYS = 3  # 启动后融资余额需连续增加天数
LOOKUP_DAYS = 5         # 向后多查多少个交易日找启动日（默认5个）
BOARD_THRESHOLD = 4     # 触发启动日还原打分的板数门槛
# =========================

pro = ts.pro_api(TOKEN)


def get_trade_dates_desc(end_date, n=30):
    """获取end_date前n个交易日列表（降序，最新在前）"""
    start_dt = pd.to_datetime(end_date) - pd.Timedelta(days=60)
    cal = pro.trade_cal(exchange='SSE', start_date=start_dt.strftime('%Y%m%d'), end_date=end_date)
    dates = cal[cal['is_open'] == 1]['cal_date'].tolist()
    return sorted([d for d in dates if d <= end_date], reverse=True)[:n]


def get_trade_dates_asc(start_date, n=5):
    """获取start_date后n个交易日列表（升序）"""
    end_dt = pd.to_datetime(start_date) + pd.Timedelta(days=30)
    cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_dt.strftime('%Y%m%d'))
    dates = cal[cal['is_open'] == 1]['cal_date'].tolist()
    return sorted([d for d in dates if d >= start_date])[:n]


def load_all_margin_data(trade_dates):
    """一次性加载所有交易日的融资数据（批量查，每天1次API）"""
    all_margin = []
    for td in trade_dates:
        try:
            df = pro.margin_detail(trade_date=td)
            if len(df) > 0:
                df['trade_date'] = td
                all_margin.append(df)
            time.sleep(0.05)
        except Exception:
            pass
    if not all_margin:
        return pd.DataFrame()
    margin_df = pd.concat(all_margin, ignore_index=True)
    margin_df['rzye_yi'] = margin_df['rzye'] / 1e8
    return margin_df


def get_price_data(ts_code, trade_dates):
    """获取个股日线数据（trade_dates为降序列表，最新在前）"""
    start = trade_dates[-1]  # 最老的日期
    end = trade_dates[0]     # 最新的日期
    df = pro.daily(ts_code=ts_code, start_date=start, end_date=end,
                   fields='trade_date,close,pct_chg,vol,amount,open,high,low,pre_close')
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['vol_ma5'] = df['vol'].rolling(5).mean().shift(1)
    df['vol_ratio'] = df['vol'] / df['vol_ma5']
    return df


def find_true_launch_date(ts_code, scan_date, all_trade_dates):
    """
    v1.5新增：找到真正的启动日（连续涨停序列的第一板）

    问题：4/21扫到600961时，它已是连续第4个涨停，真正的启动日是4/16
    规则：
        1. 从scan_date向前，逐个检查是否是涨停板(>=9.5%)
        2. 如果是涨停板，继续向前找
        3. 直到找到"前一天<=5% 且 当天>=7%"的日子 → 这就是启动日
        4. 如果当天>=7%但<9.5%，也找到了（可能是反弹启动，不是连续板）

    返回: (启动日字符串, 第几板int) 或 (None, None)
    """
    # 获取scan_date之后几个交易日的日期（用于获取向后数据）
    forward_dates = get_trade_dates_asc(scan_date, LOOKUP_DAYS)
    # 历史交易日（降序）
    hist_dates = get_trade_dates_desc(scan_date, TRADE_DAYS_BACK)
    # 完整日历：历史 + 未来（用于找前一个交易日）
    all_calendar = sorted(set(hist_dates + forward_dates))

    # 批量获取价格数据
    # FIX bug1: start=earliest, end=latest (之前写反了)
    start = all_calendar[0]
    end = all_calendar[-1]
    df = pro.daily(ts_code=ts_code, start_date=start, end_date=end,
                   fields='trade_date,close,pct_chg,pre_close')
    if df.empty:
        return None, None

    df = df.sort_values('trade_date').reset_index(drop=True)
    price_dict = {row['trade_date']: row['pct_chg'] for _, row in df.iterrows()}

    # 按日期降序排列（在price_dict中存在的）
    dates_in_price = sorted([d for d in price_dict.keys() if d <= scan_date], reverse=True)

    for curr_date in dates_in_price:
        if curr_date not in price_dict:
            continue

        curr_pct = price_dict[curr_date]

        # 找前一个（更早的）交易日
        curr_idx = None
        for i, d in enumerate(all_calendar):
            if d == curr_date:
                curr_idx = i
                break
        if curr_idx is None or curr_idx == 0:
            continue
        # FIX bug2: 用[i-1]找前一个（更早的）交易日，而不是[i+1]
        prev_trade_day = all_calendar[curr_idx - 1]

        # 找前一个交易日的涨幅
        prev_pct = price_dict.get(prev_trade_day, None)

        # 判断是否是启动日
        if curr_pct >= 7 and prev_pct is not None and prev_pct <= 5:
            # 找到启动日！计算从启动日到scan_date的连续板数
            board_count = 0
            in_range = False
            for d in sorted(price_dict.keys(), reverse=True):
                if d == scan_date:
                    in_range = True
                if in_range:
                    if price_dict[d] >= 9.5 and d >= curr_date:
                        board_count += 1
                    if d == curr_date:
                        break
            return curr_date, board_count

        # 如果当天是涨停板(>=9.5%)，继续向前找
        if curr_pct >= 9.5:
            continue
        else:
            # 当天涨幅在7%~9.5%之间，不是连续板，也没找到启动日
            # 说明这是单独的大涨（非涨停），scan_date之前没有更早的启动日
            break

    # 没找到明确的启动日（可能是数据不连续）
    return None, None


def rescore_from_launch(ts_code, true_launch_date, all_trade_dates):
    """
    v1.6新增：从真实启动日角度重新打分（用于4板以上股票）
    区别于analyze_stock，本函数用完整的启动日前30天数据来计算背离，
    以及启动日到扫到日的完整数据来计算其他信号。

    返回: (还原总分, 信号字典)
    """
    # 获取启动日之前30天的历史数据（用于背离检测）
    hist_dates = get_trade_dates_desc(true_launch_date, TRADE_DAYS_BACK)
    # 获取启动日之后的数据（用于融资持续+缩量+涨停）
    # 最远查到all_trade_dates中最远的日期
    forward_dates = sorted([d for d in all_trade_dates if d >= true_launch_date])
    all_dates = sorted(set(hist_dates + forward_dates))

    # 加载价格数据
    price_df = pro.daily(ts_code=ts_code, start_date=all_dates[0], end_date=all_dates[-1],
                         fields='trade_date,close,pct_chg,vol')
    if price_df.empty:
        return 0, {}
    price_df = price_df.sort_values('trade_date').reset_index(drop=True)
    price_df['vol_ma5'] = price_df['vol'].rolling(5).mean().shift(1)
    price_df['vol_ratio'] = price_df['vol'] / price_df['vol_ma5']

    # 加载融资数据（批量）
    margin_records = []
    for td in hist_dates:
        try:
            df_m = pro.margin_detail(trade_date=td)
            if len(df_m) > 0:
                sub = df_m[df_m['ts_code'] == ts_code]
                if len(sub) > 0:
                    margin_records.append({'trade_date': td, 'rzye_yi': sub['rzye'].iloc[0] / 1e8})
            time.sleep(0.05)
        except:
            pass

    margin_df = pd.DataFrame(margin_records)

    # ── 信号1: 背离 ──
    price_before = price_df[price_df['trade_date'] < true_launch_date].copy()
    margin_before = margin_df[margin_df['trade_date'] < true_launch_date].copy() if not margin_df.empty else pd.DataFrame()

    divergence_days = 0
    price_chg = 0.0
    margin_chg = 0.0
    has_divergence = False

    if len(price_before) >= 10 and len(margin_before) >= 5:
        merged = price_before.merge(margin_before, on='trade_date', how='inner')
        merged = merged.sort_values('trade_date').reset_index(drop=True)

        if len(merged) >= 5:
            for i in range(len(merged) - 1):
                curr = merged.iloc[i]
                nxt = merged.iloc[i + 1]
                if nxt['close'] < curr['close'] and nxt['rzye_yi'] > curr['rzye_yi']:
                    divergence_days += 1

            mid = len(merged) // 2
            if mid < 1: mid = 1
            first_h = merged.iloc[:mid]
            second_h = merged.iloc[mid:]

            if len(first_h) >= 2 and len(second_h) >= 2:
                price_chg = (second_h['close'].iloc[-1] / first_h['close'].iloc[0] - 1) * 100
                margin_chg = (second_h['rzye_yi'].iloc[-1] / first_h['rzye_yi'].iloc[0] - 1) * 100
                has_divergence = (price_chg < 0) and (margin_chg > 0) and (divergence_days >= MARGIN_DIVERGENCE_DAYS)

    # ── 信号2: 启动后融资持续增加 ──
    # 重新从all_trade_dates获取启动日后的融资数据
    post_launch_margin = []
    for td in [d for d in all_trade_dates if d >= true_launch_date]:
        try:
            df_m = pro.margin_detail(trade_date=td)
            if len(df_m) > 0:
                sub = df_m[df_m['ts_code'] == ts_code]
                if len(sub) > 0:
                    post_launch_margin.append({'trade_date': td, 'rzye_yi': sub['rzye'].iloc[0] / 1e8})
            time.sleep(0.05)
        except:
            pass

    post_margin_df = pd.DataFrame(post_launch_margin)
    has_cont = False
    cont_days = 0
    margin_increase = 0.0

    if len(post_margin_df) >= CONTINUOUS_MARGIN_DAYS:
        post_margin_df = post_margin_df.sort_values('trade_date').reset_index(drop=True)
        consecutive = 0
        for i in range(1, len(post_margin_df)):
            if post_margin_df.iloc[i]['rzye_yi'] > post_margin_df.iloc[i-1]['rzye_yi']:
                consecutive += 1
            else:
                break
        cont_days = consecutive
        has_cont = consecutive >= CONTINUOUS_MARGIN_DAYS
        if consecutive > 0:
            margin_increase = (post_margin_df.iloc[consecutive]['rzye_yi'] /
                             post_margin_df.iloc[0]['rzye_yi'] - 1) * 100

    # ── 信号3: 缩量加速 ──
    price_after = price_df[price_df['trade_date'] >= true_launch_date].copy()
    has_shrink = False
    vol_ratio_change = 1.0

    if len(price_before) >= 3 and len(price_after) >= 2:
        vr_before = price_before['vol_ratio'].tail(3).mean()
        vr_after = price_after['vol_ratio'].head(3).mean()
        avg_pct_after = price_after['pct_chg'].head(3).mean()
        has_shrink = (avg_pct_after > 0) and (vr_after < vr_before)
        vol_ratio_change = vr_after / vr_before if vr_before > 0 else 1.0

    # ── 信号4: 连续涨停 ──
    launch_plus1 = price_after[price_after['trade_date'] > true_launch_date].head(5)
    recent = pd.concat([price_df[price_df['trade_date'] == true_launch_date], launch_plus1]).tail(6)
    limit_days = (recent['pct_chg'] >= 9.5).sum()
    has_limit = limit_days >= 2

    score = sum([has_divergence, has_cont, has_shrink, has_limit])

    return score, {
        'ts_code': ts_code,
        'launch_date': true_launch_date,
        'score': score,
        'divergence': has_divergence,
        'div_days': divergence_days,
        'price_chg': price_chg,
        'margin_chg': margin_chg,
        'margin_cont': has_cont,
        'cont_days': cont_days,
        'margin_increase': margin_increase,
        'shrink': has_shrink,
        'vol_ratio_change': vol_ratio_change,
        'limit_up': has_limit,
        'limit_days': limit_days,
    }


def analyze_stock(ts_code, name, pct_chg, scan_date, all_trade_dates, margin_df, price_df):
    """
    分析单只股票的第二步信号
    scan_date: 扫描日期（字符串，如'20260327'）
    all_trade_dates: 升序排列的完整日历（含历史+未来，用于find_true_launch_date）
    返回: 信号字典或None
    """
    # ── v1.5新增：找到真正的启动日 ──
    launch_date, board_count = find_true_launch_date(ts_code, scan_date, all_trade_dates)

    if launch_date is None:
        return None

    # 启动日涨幅
    launch_pct = None
    for _, row in price_df.iterrows():
        if row['trade_date'] == launch_date:
            launch_pct = row['pct_chg']
            break

    # ── 信号1: 启动前背离检测 ──
    # 背离分析：用launch_date之前的数据
    price_before = price_df[price_df['trade_date'] < launch_date].copy()
    margin_before_ts = margin_df[margin_df['trade_date'] < launch_date].copy()
    margin_before_ts = margin_before_ts[margin_before_ts['ts_code'] == ts_code]

    if len(price_before) < 10 or len(margin_before_ts) < 5:
        has_divergence = False
        divergence_days = 0
        price_chg = 0.0
        margin_chg = 0.0
    else:
        merged_before = price_before.merge(margin_before_ts[['trade_date', 'rzye_yi']], on='trade_date', how='inner')
        if len(merged_before) < 5:
            has_divergence = False
            divergence_days = 0
            price_chg = 0.0
            margin_chg = 0.0
        else:
            # 逐日背离天数
            divergence_days = 0
            for i in range(len(merged_before) - 1):
                curr = merged_before.iloc[i]
                nxt = merged_before.iloc[i + 1]
                if (nxt['close'] < curr['close']) and (nxt['rzye_yi'] > curr['rzye_yi']):
                    divergence_days += 1

            mid = len(merged_before) // 2
            if mid < 1: mid = 1
            first_h = merged_before.iloc[:mid]
            second_h = merged_before.iloc[mid:]

            if len(first_h) >= 2 and len(second_h) >= 2:
                price_chg = (second_h['close'].iloc[-1] / first_h['close'].iloc[0] - 1) * 100
                margin_chg = (second_h['rzye_yi'].iloc[-1] / first_h['rzye_yi'].iloc[0] - 1) * 100
                has_divergence = (price_chg < 0) and (margin_chg > 0) and (divergence_days >= MARGIN_DIVERGENCE_DAYS)
            else:
                has_divergence = False
                price_chg = 0.0
                margin_chg = 0.0

    # ── 信号2: 启动后融资余额持续增加 ──
    # 用launch_date之后的数据（而不是scan_date）
    price_after = price_df[price_df['trade_date'] >= launch_date].copy()
    margin_after_ts = margin_df[margin_df['trade_date'] >= launch_date].copy()
    margin_after_ts = margin_after_ts[margin_after_ts['ts_code'] == ts_code].reset_index(drop=True)

    if len(margin_after_ts) < CONTINUOUS_MARGIN_DAYS:
        has_cont, cont_days, margin_increase = False, 0, 0.0
    else:
        consecutive = 0
        for i in range(1, len(margin_after_ts)):
            if margin_after_ts.iloc[i]['rzye_yi'] > margin_after_ts.iloc[i-1]['rzye_yi']:
                consecutive += 1
            else:
                break
        cont_days = consecutive
        has_cont = consecutive >= CONTINUOUS_MARGIN_DAYS
        if consecutive > 0:
            margin_increase = (margin_after_ts.iloc[consecutive]['rzye_yi'] /
                              margin_after_ts.iloc[0]['rzye_yi'] - 1) * 100
        else:
            margin_increase = 0.0

    # ── 信号3: 缩量加速 ──
    if len(price_before) >= 3 and len(price_after) >= 2:
        vr_before = price_before['vol_ratio'].tail(3).mean()
        vr_after = price_after['vol_ratio'].head(3).mean()
        avg_pct_after = price_after['pct_chg'].head(3).mean()
        has_shrink = (avg_pct_after > 0) and (vr_after < vr_before)
        vol_ratio_change = vr_after / vr_before if vr_before > 0 else 1.0
    else:
        has_shrink, vol_ratio_change = False, 1.0

    # ── 信号4: 连续涨停 ──
    # 基于启动日后的数据判断
    launch_plus1 = price_after[price_after['trade_date'] > launch_date].head(5)
    recent = pd.concat([price_df[price_df['trade_date'] == launch_date], launch_plus1]).tail(6)
    limit_days = (recent['pct_chg'] >= 9.5).sum()
    has_limit = limit_days >= 2

    score = sum([has_divergence, has_cont, has_shrink, has_limit])

    return {
        'ts_code': ts_code,
        'name': name,
        'pct_chg': pct_chg,
        'scan_date': scan_date,
        'launch_date': launch_date,
        'launch_pct': launch_pct,
        'board_count': board_count,
        'score': score,
        'divergence': has_divergence,
        'div_days': divergence_days,
        'price_chg': price_chg,
        'margin_chg': margin_chg,
        'margin_cont': has_cont,
        'cont_days': cont_days,
        'margin_increase': margin_increase,
        'shrink': has_shrink,
        'vol_ratio_change': vol_ratio_change,
        'limit_up': has_limit,
        'limit_days': limit_days,
    }


def scan_date(target_date=None):
    """扫描指定日期的涨幅>=7%股票"""
    if target_date is None:
        cal = pro.trade_cal(exchange='SSE', start_date='20260420', end_date='20260430')
        cal = cal[cal['is_open'] == 1]['cal_date'].tolist()
        target_date = cal[-2]
    else:
        target_date = str(target_date)

    print(f"\n{'='*60}")
    print(f"📅 扫描日期: {target_date}  |  涨幅门槛: >={MIN_RISE_PCT}%")
    print(f"{'='*60}\n")

    # Step 1: 涨幅>=7%的股票
    df_today = pro.daily(trade_date=target_date)
    rise7 = df_today[df_today['pct_chg'] >= MIN_RISE_PCT].copy()
    rise7 = rise7.sort_values('pct_chg', ascending=False)

    if rise7.empty:
        print("今日无涨幅>=7%的股票")
        return

    print(f"涨幅>=7% 股票数: {len(rise7)}\n")
    print(f"{'代码':<12} {'涨幅%':>8} {'扫到日期':>10}")
    print('-' * 35)
    for _, row in rise7.head(15).iterrows():
        print(f"{row['ts_code']:<12} {row['pct_chg']:>8.2f} {target_date:>10}")

    # Step 2: 历史交易日
    trade_dates_desc = get_trade_dates_desc(target_date, TRADE_DAYS_BACK)
    # 完整日历（历史+未来），用于find_true_launch_date
    forward_dates = get_trade_dates_asc(target_date, LOOKUP_DAYS)
    all_calendar = sorted(set(trade_dates_desc + forward_dates))
    trade_dates_asc = sorted(trade_dates_desc, reverse=False)  # 纯历史（用于兼容margin_df）

    print(f"\n历史数据范围: {trade_dates_desc[-1]} ~ {trade_dates_desc[0]}")
    print(f"启动日识别：向后多查{LOOKUP_DAYS}个交易日")
    print(f"启动日还原打分：板数>={BOARD_THRESHOLD}时触发（4板以上）\n")

    # Step 3: 批量加载融资数据
    print("加载融资数据（批量查询中）...")
    margin_df = load_all_margin_data(trade_dates_desc)
    print(f"融资数据: {len(margin_df)} 条记录\n")

    # Step 4: 逐只分析
    print(f"{'='*60}")
    print(f"🔍 第二步信号检验中...")
    print(f"{'='*60}\n")

    results = []
    for idx, (_, stock) in enumerate(rise7.iterrows()):
        ts_code = stock['ts_code']
        name = stock.get('name', ts_code)

        if (idx + 1) % 20 == 0 or idx == len(rise7) - 1:
            print(f"  已处理 {idx+1}/{len(rise7)} 只股票...")

        try:
            price_df = get_price_data(ts_code, trade_dates_desc)
            if price_df.empty:
                continue

            result = analyze_stock(
                ts_code, name, stock['pct_chg'],
                target_date, all_calendar, margin_df, price_df
            )
            if result is not None:
                # ── v1.6新增：4板以上股票触发启动日还原打分 ──
                if result['board_count'] is not None and result['board_count'] >= BOARD_THRESHOLD:
                    rescore, rescore_detail = rescore_from_launch(
                        ts_code, result['launch_date'], all_calendar
                    )
                    result['rescore'] = rescore
                    result['rescore_detail'] = rescore_detail
                else:
                    result['rescore'] = None
                    result['rescore_detail'] = None

                results.append(result)
        except Exception:
            continue

    if not results:
        print("无足够数据完成分析")
        return

    df_result = pd.DataFrame(results).sort_values('score', ascending=False)

    # 打印汇总
    print(f"\n{'='*100}")
    print(f"{'代码':<12} {'名称':<8} {'评分':>4} {'背离':>4} {'融资持续':>8} {'缩量':>4} {'涨停':>4}  启动日    板数  综合信号")
    print('-' * 100)

    for _, r in df_result.iterrows():
        signals = []
        if r['divergence']: signals.append(f"背离{int(r['div_days'])}天")
        if r['margin_cont']: signals.append(f"融资连增{int(r['cont_days'])}天")
        if r['shrink']: signals.append("缩量加速")
        if r['limit_up']: signals.append(f"涨停{int(r['limit_days'])}天")
        sig_str = ", ".join(signals) if signals else "待观察"
        star = "⭐" * int(r['score'])
        board_str = f"{int(r['board_count'])}板" if r['board_count'] else "?"
        launch_str = r['launch_date'][4:] if r['launch_date'] else "?"

        # v1.6: 如果有还原打分，显示还原分
        rescore_str = ""
        if r.get('rescore') is not None and not pd.isna(r['rescore']):
            rescore_str = f" | 还原{int(r['rescore'])}分"

        print(f"{r['ts_code']:<12} {r['name']:<8} {int(r['score']):>4}  "
              f"{'✓' if r['divergence'] else '✗':>4} "
              f"{'✓' if r['margin_cont'] else '✗':>8} "
              f"{'✓' if r['shrink'] else '✗':>4} "
              f"{'✓' if r['limit_up'] else '✗':>4}  "
              f"{launch_str:>8}  {board_str:>4}  "
              f"{star} {sig_str}{rescore_str}")

    # 高分详情
    top = df_result[df_result['score'] >= 2]
    if not top.empty:
        print(f"\n{'='*60}")
        print(f"📊 高分股票详细分析（评分>=2，共{len(top)}只）")
        print(f"{'='*60}\n")
        for _, r in top.iterrows():
            board_note = f"，扫到时已是第{int(r['board_count'])}板" if r['board_count'] and r['board_count'] > 1 else ""
            print(f"[{r['ts_code']}] {r['name']}  扫到日期: {r['scan_date']}")
            print(f"  {'─'*40}")
            print(f"  ✅ 真正启动日: {r['launch_date']}（{r['launch_pct']:.2f}%{board_note}）")
            if r['divergence']:
                print(f"  ✅ 背离: 股价{r['price_chg']:+.1f}% / 融资余额{r['margin_chg']:+.1f}%（{int(r['div_days'])}天）")
            if r['margin_cont']:
                print(f"  ✅ 融资持续增加: 连续{int(r['cont_days'])}天，+{r['margin_increase']:.1f}%")
            if r['shrink']:
                print(f"  ✅ 缩量加速: 量比为前期的{r['vol_ratio_change']:.2f}倍")
            if r['limit_up']:
                print(f"  ✅ 连续涨停: {int(r['limit_days'])}天")

            # v1.6: 还原打分详情
            if r.get('rescore') is not None and not pd.isna(r['rescore']):
                rd = r['rescore_detail']
                price_chg_str = f"{rd['price_chg']:+.1f}%" if not pd.isna(rd['price_chg']) else "N/A"
                margin_chg_str = f"{rd['margin_chg']:+.1f}%" if not pd.isna(rd['margin_chg']) else "N/A"
                margin_inc_str = f"{rd['margin_increase']:+.1f}%" if not pd.isna(rd['margin_increase']) else "N/A"
                vol_rc_str = f"{rd['vol_ratio_change']:.2f}x" if not pd.isna(rd['vol_ratio_change']) else "N/A"
                print(f"  🔄 启动日还原打分（4板以上触发）:")
                print(f"     ① 背离: {'✓' if rd['divergence'] else '✗'} (股价{price_chg_str}, 融资{margin_chg_str}, {int(rd['div_days'])}天)")
                print(f"     ② 融资持续: {'✓' if rd['margin_cont'] else '✗'} (连续{int(rd['cont_days'])}天, {margin_inc_str})")
                print(f"     ③ 缩量: {'✓' if rd['shrink'] else '✗'} (量比变化{vol_rc_str})")
                print(f"     ④ 涨停: {'✓' if rd['limit_up'] else '✗'} ({int(rd['limit_days'])}天)")
                verdict = '✅ 强信号' if int(r['rescore']) >= 3 else '⚠️ 待观察' if int(r['rescore']) >= 2 else '❌ 低分'
                print(f"     → 还原总分: {int(r['rescore'])}分 {verdict}")

                if r['rescore'] < r['score']:
                    print(f"     ⚠️ 还原分低于山顶分：说明从启动日角度看信号更弱")
                elif r['rescore'] > r['score']:
                    print(f"     💡 还原分高于山顶分：说明从启动日角度看该股本来就应该得高分，只是被山顶视角低估")

            print(f"  {'🎯' if r['score'] >= 3 else '👀'} 结论: {'强信号，三步验证接近完成' if r['score'] >= 3 else '中强信号，等待第三步确认'}")
            print()

    print(f"\n{'='*60}")
    print(f"评分规则: 背离(1) + 融资持续(1) + 缩量加速(1) + 连续涨停(1)")
    print(f"背离: 启动日前，股价跌+融资余额增，持续{MARGIN_DIVERGENCE_DAYS}+天")
    print(f"融资持续: 启动日后连续{CONTINUOUS_MARGIN_DAYS}天融资余额增加")
    print(f"缩量加速: 股价涨但量比下降")
    print(f"连续涨停: 启动日+次日均有涨停")
    print(f"启动日定义: 前一天涨幅≤5% 且 当天涨幅≥7%")
    print(f"启动日还原打分: 4板以上股票额外从启动日角度重新打分")
    print(f"{'='*60}")


if __name__ == '__main__':
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    scan_date(date_arg)
