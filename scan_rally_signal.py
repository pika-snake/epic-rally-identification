#!/usr/bin/env python3
"""
史诗级行情扫描器 — 第二步"验证有行情" + 第三步"验证史诗级"
每天扫描涨幅>=7%的股票，检查是否符合第二步+第三步信号

信号逻辑：
  第二步核心（三维背离验证）：
    1. 背离验证：整体斜率背离 OR 后半段背离 OR 逐日背离≥5天（三选二）
    2. 启动后持续：融资余额连续增加
    3. 缩量加速：股价涨但量比下降
    4. 连续涨停：2天+
  第三步（催化因素）：
    1. 年报催化：ROE出现V型反转（最新>上期×1.2）
    2. 融资增速 > 股价增速
    3. 估值信号：低估或合理（PE<60）

性能优化：融资数据按日期批量查（每日1次API，不是每股票1次）

用法:
    python scan_rally_signal.py [日期YYYYMMDD]
    不带参数默认上一个交易日
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
# =========================

pro = ts.pro_api(TOKEN)

def get_trade_dates(end_date, n=30):
    """获取end_date前n个交易日的日期列表（按日期升序，API查询用）"""
    start_dt = pd.to_datetime(end_date) - pd.Timedelta(days=60)
    cal = pro.trade_cal(exchange='SSE', start_date=start_dt.strftime('%Y%m%d'), end_date=end_date)
    dates = cal[cal['is_open'] == 1]['cal_date'].tolist()
    # 过滤并升序排列（API需要 start < end）
    filtered = sorted([d for d in dates if d <= end_date])
    return filtered[-n:]  # 返回最近的n个，升序

def get_trade_dates_desc(end_date, n=30):
    """获取end_date前n个交易日列表（降序，用于进度显示）"""
    start_dt = pd.to_datetime(end_date) - pd.Timedelta(days=60)
    cal = pro.trade_cal(exchange='SSE', start_date=start_dt.strftime('%Y%m%d'), end_date=end_date)
    dates = cal[cal['is_open'] == 1]['cal_date'].tolist()
    filtered = sorted([d for d in dates if d <= end_date], reverse=True)
    return filtered[:n]

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
    # 计算量比（当日成交量/前5日均量）
    df['vol_ma5'] = df['vol'].rolling(5).mean().shift(1)
    df['vol_ratio'] = df['vol'] / df['vol_ma5']
    return df

def linear_slope(y):
    """计算一维数组的线性回归斜率"""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.array(y, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    return num / den if den != 0 else 0.0

def analyze_stock(ts_code, name, pct_chg, target_date, trade_dates_desc, margin_df, price_df):
    """
    分析单只股票的第二步信号
    target_date: 扫描日期（字符串，如'20260327'）
    trade_dates_desc: 降序日期列表（最新在前，供日线查询用）
    """
    # ── 信号1: 启动前背离检测（优化版：三维验证） ──
    price_before = price_df[price_df['trade_date'] < target_date].copy()
    margin_before_ts = margin_df[margin_df['trade_date'] < target_date].copy()
    margin_before_ts = margin_before_ts[margin_before_ts['ts_code'] == ts_code]

    if len(price_before) < 10 or len(margin_before_ts) < 5:
        return None  # 数据不足

    # 合并后作为分析数据
    data = price_before.merge(margin_before_ts[['trade_date', 'rzye_yi']], on='trade_date', how='inner')
    if len(data) < 10:
        return None

    # ── 维度1: 整体斜率背离（线性回归） ──
    price_slope = linear_slope(data['close'].values)
    margin_slope = linear_slope(data['rzye_yi'].values)
    dim1_overall = (price_slope < 0) and (margin_slope > 0)

    # ── 维度2: 后半段背离（更接近启动日，更关键） ──
    mid = len(data) // 2
    if mid < 2:
        return None
    second = data.iloc[mid:].copy().reset_index(drop=True)
    p_s, p_e = second['close'].iloc[0], second['close'].iloc[-1]
    m_s, m_e = second['rzye_yi'].iloc[0], second['rzye_yi'].iloc[-1]
    dim2_second_half = ((p_e / p_s - 1) * 100 < 0) and ((m_e / m_s - 1) * 100 > 0)

    # ── 维度3: 逐日背离天数 ──
    divergence_days = 0
    for i in range(len(data) - 1):
        curr = data.iloc[i]
        nxt = data.iloc[i + 1]
        if (nxt['close'] < curr['close']) and (nxt['rzye_yi'] > curr['rzye_yi']):
            divergence_days += 1

    # 综合判断：3个维度中满足>=2个即为背离
    has_divergence = sum([dim1_overall, dim2_second_half, divergence_days >= 5]) >= 2

    # 计算后半段变化率（用于输出）
    price_chg = (p_e / p_s - 1) * 100 if p_s != 0 else 0.0
    margin_chg = (m_e / m_s - 1) * 100 if m_s != 0 else 0.0

    # ── 信号2: 启动后融资余额持续增加 ──
    price_after = price_df[price_df['trade_date'] >= target_date].copy()
    margin_after_ts = margin_df[margin_df['trade_date'] >= target_date].copy()
    margin_after_ts = margin_after_ts[margin_after_ts['ts_code'] == ts_code].reset_index(drop=True)

    if len(margin_after_ts) < CONTINUOUS_MARGIN_DAYS:
        has_cont = False
        cont_days = 0
        margin_increase = 0.0
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
        vol_ratio_before = price_before['vol_ratio'].tail(3).mean()
        vol_ratio_after = price_after['vol_ratio'].head(3).mean()
        avg_pct_after = price_after['pct_chg'].head(3).mean()
        has_shrink = (avg_pct_after > 0) and (vol_ratio_after < vol_ratio_before)
        vol_ratio_change = vol_ratio_after / vol_ratio_before if vol_ratio_before > 0 else 1.0
    else:
        has_shrink = False
        vol_ratio_change = 1.0

    # ── 信号4: 连续涨停 ──
    # 看今日+前日
    recent = price_df.tail(3)
    if len(recent) >= 2:
        recent = recent.tail(2)
        limit_days = (recent['pct_chg'] >= 9.5).sum()
        has_limit = limit_days >= 2
    else:
        has_limit = False
        limit_days = 0

    # 综合评分
    score = sum([has_divergence, has_cont, has_shrink, has_limit])

    return {
        'ts_code': ts_code,
        'name': name,
        'pct_chg': pct_chg,
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


def analyze_step3(ts_code, target_date, trade_dates_desc, price_df):
    """
    第三步：验证"史诗级"（催化因素）
    返回 dict:
      - annual_catalyst: bool    # 年报催化（ROE反转）
      - roe_now: float           # 最新ROE
      - roe_prev: float          # 上期ROE
      - profit_growth: float      # 净利润增速
      - margin_growth_rate: float # 融资余额增速（30天）
      - price_growth_rate: float  # 股价增速（30天）
      - margin_gt_price: bool    # 融资增速 > 股价增速
      - pe_current: float        # 当前PE（TTM）
      - valuation_signal: str     # '低估'/'合理'/'高估'
      - score_step3: int         # 第三步得分 0~3
    """
    result = {
        'annual_catalyst': False,
        'roe_now': None,
        'roe_prev': None,
        'profit_growth': None,
        'margin_growth_rate': None,
        'price_growth_rate': None,
        'margin_gt_price': False,
        'pe_current': None,
        'valuation_signal': '无法判断',
        'score_step3': 0,
    }

    try:
        # ── 1. 年报催化（fina_indicator）────────────────────────
        fi = pro.fina_indicator(ts_code=ts_code, start_date='20250101', end_date=target_date)
        if fi is not None and len(fi) > 0:
            fi = fi.sort_values('end_date', ascending=False).head(2)
            if len(fi) >= 1:
                result['roe_now'] = round(float(fi.iloc[0]['roe']), 2) if pd.notna(fi.iloc[0]['roe']) else None
            if len(fi) >= 2:
                result['roe_prev'] = round(float(fi.iloc[1]['roe']), 2) if pd.notna(fi.iloc[1]['roe']) else None
            if result['roe_now'] and result['roe_prev'] and result['roe_prev'] > 0:
                # ROE提升20%以上视为V型反转
                if result['roe_now'] > result['roe_prev'] * 1.2:
                    result['annual_catalyst'] = True

        # ── 2. 净利润增速（income 表）──────────────────────────
        inc = pro.income(ts_code=ts_code, start_date='20250101', end_date=target_date)
        if inc is not None and len(inc) > 0:
            inc = inc.sort_values('end_date', ascending=False).head(2)
            if len(inc) >= 2:
                cur, prev = inc.iloc[0], inc.iloc[1]
                if (pd.notna(cur['n_income']) and pd.notna(prev['n_income'])
                        and prev['n_income'] > 0):
                    result['profit_growth'] = round(
                        (cur['n_income'] - prev['n_income']) / prev['n_income'] * 100, 1)

        # ── 3. 当前PE（daily_basic）─────────────────────────────
        try:
            db = pro.daily_basic(ts_code=ts_code, trade_date=target_date)
            if db is not None and len(db) > 0 and pd.notna(db.iloc[0]['pe_ttm']):
                result['pe_current'] = round(float(db.iloc[0]['pe_ttm']), 1)
        except Exception:
            pass

        # ── 4. 股价增速（30天区间）────────────────────────────
        if price_df is not None and len(price_df) >= 5:
            oldest = float(price_df.iloc[-1]['close'])
            newest = float(price_df.iloc[0]['close'])
            if oldest > 0:
                result['price_growth_rate'] = round((newest - oldest) / oldest * 100, 1)

        # ── 5. 融资增速（30天两端）────────────────────────────
        # 只查两端，各1次API，不是逐日查
        try:
            m_new = pro.margin_detail(ts_code=ts_code, trade_date=trade_dates_desc[0])
            m_old = pro.margin_detail(ts_code=ts_code, trade_date=trade_dates_desc[-1])
            if (m_new is not None and len(m_new) > 0 and pd.notna(m_new.iloc[0]['rzye'])
                    and m_old is not None and len(m_old) > 0 and pd.notna(m_old.iloc[0]['rzye'])):
                b_new = float(m_new.iloc[0]['rzye'])
                b_old = float(m_old.iloc[0]['rzye'])
                if b_old > 0:
                    result['margin_growth_rate'] = round((b_new - b_old) / b_old * 100, 1)
                    result['margin_gt_price'] = (
                        result['margin_growth_rate'] is not None
                        and result['price_growth_rate'] is not None
                        and result['margin_growth_rate'] > result['price_growth_rate']
                    )
        except Exception:
            pass

        # ── 6. 估值信号 ──────────────────────────────────────
        if result['pe_current'] is not None and result['pe_current'] > 0:
            pe = result['pe_current']
            result['valuation_signal'] = '低估' if pe < 30 else ('合理' if pe < 60 else '高估')

        # ── 7. 第三步评分（0~3）───────────────────────────────
        score = 0
        if result['annual_catalyst']:
            score += 1
        if result['margin_gt_price']:
            score += 1
        if result['valuation_signal'] in ('低估', '合理'):
            score += 1
        result['score_step3'] = score

    except Exception:
        pass

    return result


def scan_date(target_date=None):
    """扫描指定日期的涨幅>=7%股票"""
    if target_date is None:
        cal = pro.trade_cal(exchange='SSE', start_date='20260420', end_date='20260430')
        cal = cal[cal['is_open'] == 1]['cal_date'].tolist()
        target_date = cal[-2]  # 倒数第二个交易日
    else:
        target_date = str(target_date)

    print(f"\n{'='*60}")
    print(f"📅 扫描日期: {target_date}  |  涨幅门槛: >={MIN_RISE_PCT}%")
    print(f"{'='*60}\n")

    # Step 1: 获取涨幅>=7%的股票
    df_today = pro.daily(trade_date=target_date)
    rise7 = df_today[df_today['pct_chg'] >= MIN_RISE_PCT].copy()
    rise7 = rise7.sort_values('pct_chg', ascending=False)

    if rise7.empty:
        print("今日无涨幅>=7%的股票")
        return

    print(f"涨幅>=7% 股票数: {len(rise7)}\n")
    print(f"{'代码':<12} {'涨幅%':>8}")
    print('-' * 25)
    for _, row in rise7.head(15).iterrows():
        print(f"{row['ts_code']:<12} {row['pct_chg']:>8.2f}")

    # Step 2: 获取历史交易日列表
    trade_dates_asc = get_trade_dates(target_date, TRADE_DAYS_BACK)
    trade_dates_desc = get_trade_dates_desc(target_date, TRADE_DAYS_BACK)
    print(f"\n历史数据范围: {trade_dates_desc[-1]} ~ {trade_dates_desc[0]}")

    # Step 3: 批量加载所有融资数据（按日期，每天1次API）
    print("加载融资数据（批量查询中）...")
    margin_df = load_all_margin_data(trade_dates_desc)
    print(f"融资数据: {len(margin_df)} 条记录\n")

    # Step 4: 逐只股票分析
    print(f"{'='*60}")
    print(f"🔍 第二步信号检验中...")
    print(f"{'='*60}\n")

    results = []
    step3_results = {}  # ts_code -> step3 result dict

    for idx, (_, stock) in enumerate(rise7.iterrows()):
        ts_code = stock['ts_code']
        name = stock.get('name', ts_code)

        # 打印进度
        if (idx + 1) % 20 == 0 or idx == len(rise7) - 1:
            print(f"  已处理 {idx+1}/{len(rise7)} 只股票...")

        try:
            # 用降序的日期列表（最新在前）传给分析函数
            price_df = get_price_data(ts_code, trade_dates_desc)
            if price_df.empty:
                continue

            result = analyze_stock(
                ts_code, name, stock['pct_chg'],
                target_date, trade_dates_desc, margin_df, price_df
            )
            if result is not None:
                results.append(result)
        except Exception:
            continue

    if not results:
        print("无足够数据完成分析")
        return

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values('score', ascending=False)

    # ── 对高分股票（第二步>=2）执行第三步 ──
    top_ts_codes = df_result[df_result['score'] >= 2]['ts_code'].tolist()
    if top_ts_codes:
        print(f"\n{'='*60}")
        print(f"🔍 第三步催化因素分析中（第二步高分股票 {len(top_ts_codes)} 只）...")
        print(f"{'='*60}\n")

        for ts_code in top_ts_codes:
            price_df = get_price_data(ts_code, trade_dates_desc)
            if price_df.empty:
                continue
            try:
                s3 = analyze_step3(ts_code, target_date, trade_dates_desc, price_df)
                step3_results[ts_code] = s3
            except Exception:
                pass

    # 把第三步得分合并到结果表
    df_result['score_step3'] = df_result['ts_code'].map(
        lambda x: step3_results.get(x, {}).get('score_step3', 0)
    )
    df_result['total_score'] = df_result['score'] + df_result['score_step3']

    # ── 打印汇总表 ──
    print(f"\n{'='*90}")
    print(f"{'代码':<12} {'名称':<8} {'第二步':>5} {'第三步':>5} {'总分':>5}  第二步信号                                           第三步信号")
    print('-' * 90)

    for _, r in df_result.iterrows():
        # 第二步信号
        s2_signals = []
        if r['divergence']: s2_signals.append(f"背离{int(r['div_days'])}天")
        if r['margin_cont']: s2_signals.append(f"融资连增{int(r['cont_days'])}天")
        if r['shrink']: s2_signals.append("缩量加速")
        if r['limit_up']: s2_signals.append(f"涨停{int(r['limit_days'])}天")
        s2_str = ", ".join(s2_signals) if s2_signals else "待观察"

        # 第三步信号
        s3 = step3_results.get(r['ts_code'], {})
        s3_signals = []
        if s3.get('annual_catalyst'): s3_signals.append(f"年报催化(ROE{s3.get('roe_now','?')}%)")
        if s3.get('margin_gt_price'): s3_signals.append(f"融资增速>{s3.get('price_growth_rate','?')}%")
        if s3.get('valuation_signal') in ('低估', '合理'):
            s3_signals.append(f"PE{s3.get('pe_current','?')}({s3.get('valuation_signal')})")
        s3_str = ", ".join(s3_signals) if s3_signals else "待确认"

        star = "⭐" * int(r['total_score'])
        print(f"{r['ts_code']:<12} {r['name']:<8} {int(r['score']):>5} {int(r['score_step3']):>5} {int(r['total_score']):>5}  {s2_str:<44} {s3_str}")

    # ── 高分股票详情 ──
    top = df_result[df_result['total_score'] >= 5]
    if not top.empty:
        print(f"\n{'='*60}")
        print(f"📊 重点关注（总分>=5，共{len(top)}只）")
        print(f"{'='*60}\n")

        for _, r in top.iterrows():
            s3 = step3_results.get(r['ts_code'], {})
            print(f"[{r['ts_code']}] {r['name']}  今日涨幅 {r['pct_chg']:.2f}%  总分 {int(r['total_score'])}/{'⭐'*int(r['total_score'])}")
            print(f"  {'─'*44}")
            if r['divergence']:
                print(f"  ✅ 背离: 股价{r['price_chg']:+.1f}% / 融资余额{r['margin_chg']:+.1f}%（{int(r['div_days'])}天背离）")
            if r['margin_cont']:
                print(f"  ✅ 融资持续增加: 连续{int(r['cont_days'])}天，增幅{r['margin_increase']:.1f}%")
            if r['shrink']:
                print(f"  ✅ 缩量加速: 量比为前期的{r['vol_ratio_change']:.2f}倍")
            if r['limit_up']:
                print(f"  ✅ 连续涨停: {int(r['limit_days'])}天")
            if s3.get('annual_catalyst'):
                print(f"  ✅ 年报催化: ROE {s3.get('roe_prev','?')}% → {s3.get('roe_now','?')}%（V型反转）")
                if s3.get('profit_growth') is not None:
                    print(f"     净利润增速: {s3.get('profit_growth'):+.1f}%")
            if s3.get('margin_gt_price'):
                print(f"  ✅ 融资增速>股价增速: 融资{s3.get('margin_growth_rate','?')}% > 股价{s3.get('price_growth_rate','?')}%")
            if s3.get('pe_current') is not None:
                print(f"  ✅ 估值: PE={s3.get('pe_current')}（{s3.get('valuation_signal')}）")
            if r['total_score'] >= 6:
                print(f"  🎯 结论: 三步齐备！史诗级行情确认，坚定持有")
            elif r['total_score'] >= 5:
                print(f"  👀 结论: 强信号，行情大概率还没走完，持有观察")
            print()

    # ── 说明 ──
    print(f"{'='*60}")
    print(f"评分规则:")
    print(f"  第二步（0~4分）:")
    print(f"    背离(1分): 三维背离验证（三选二）")
    print(f"    融资持续(1分): 今日起连续{CONTINUOUS_MARGIN_DAYS}天融资余额增加")
    print(f"    缩量加速(1分): 股价在涨但量比在降")
    print(f"    连续涨停(1分): 今日+前日涨幅均>=9.5%")
    print(f"  第三步（0~3分）:")
    print(f"    年报催化(1分): ROE V型反转（最新>上期×1.2）")
    print(f"    融资增速>股价增速(1分)")
    print(f"    估值合理(1分): PE<60")
    print(f"  总分 7 分满分，总分>=5 可重点关注")
    print(f"{'='*60}")

if __name__ == '__main__':
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    scan_date(date_arg)
