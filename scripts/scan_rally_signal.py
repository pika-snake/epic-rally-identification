#!/usr/bin/env python3
"""
史诗级行情扫描器 v6.56 — T日买入评估 + T日融资爆量分级 + D象限历史启动日过滤
｜（v6.56：v2.28——D象限pre5>10%否决、D象限启动日距T日>7天否决、融资爆量20~25%区间已在SKILL.md定义）
｜（v6.30：基于0126批次调整——前5涨跌<=12%、15日涨跌<=20%、量比<=2.5；新增节前最后交易日检测；修复API频率限制：每20只股票延迟0.5秒）
｜（v6.29：整合param_tuning.md v6.29参数调整——前5日涨跌<=8%、前15日涨跌<=15%、小实体天数>=6、大实体0天+量比>=5排除、小实体8~9天+前5日0~5%排除）
｜（v6.16：融资条款全部降级为参考信息——Y型移除硬过滤（条件②前5日rzche比>5%、条件③启动日rzche比>15%、条件⑩融资蓄力比<=2倍），仅作参考展示；移除Y总分≥35门槛）
｜（v6.15：K线实体规律——Y型新增条件⑪（启动前15日K线实体>=5%天数<3）+小实体天数bonus）
｜（v6.14：Y型新增条件⑨前5日单日最大涨幅≤5% + 融资蓄力比≤2倍 + Y总分≥35门槛）
｜（v6.12：黑马路月线MA5上升条件——所有黑马必须满足月线沿着MA5上升，即price_above_ma5=True）
（v6.10：Y型条件1改为量比[2~3]——启动日量比>=2且<=3，>3直接排除）
（v6.9：Y型新增第8条件——启动前15日涨跌幅绝对值>7%天数>=2直接排除）
（v6.5：新增"急跌+缩量横盘"形态检测，Y型评分加入急跌挖坑Bonus）
（v6.4：Y型综合强度分公式；v6.0：新增Y型游资炒作型；黑马排除Y型）

核心设计原则：
|- T日买入决策：只使用 T-1及之前 + T日当天 数据
|- T+1持有评估：使用 T+1及之后 数据（买入后用来判断要不要持有）

用法:
    python scan_rally_signal.py [日期YYYYMMDD] [-v|--verify] [-t 6|7] [--codes 600000.SH,...]
    不带参数默认上一个交易日
    -v/--verify: 验证模式，对Baux/B/A象限候选股输出月线+融资结构+启动后走势汇总
    -t/--threshold: 涨停门槛，6.0或7.0，默认7.0

T日买入评估（0~4分 + 融资四维度 + 月线过热过滤 + v4.2涨跌过滤）：
  ① 背离验证（1分）：启动前股价跌/融资余额增，持续>=5天
     v2.1修复：改用"启动前最后10日窗口"，避免中间大幅波动切断背离判断
  ② 量比验证（1分）：T日量比 < 前5日均量（缩量上涨）
  ③ 象限分类（Baux新增）：启动前融资持续增长但股价不涨，隐形建仓型
  ④ 启动日融资变化（最强单信号）：
     - 启动日融资余额较前一交易日 >+10% → 强烈买入信号
     - 启动日融资余额较前一交易日 <-5% → 否决信号
     - 中间值 → 普通信号
  ⑤ 融资正天数占比（v2.7）：30日净买入正天数>=65% → 黑马特征
  ⑥ 启动前5日融资增幅（v2.7）：<20% → 黑马特征（蜗牛>40%）
  ⑦ 30日融资净买总额（v2.7）：>0 → 辅助参考（负值多为蜗牛）
  ⑧ 5日净买占30日比例（v2.7）：<80% → 均匀建仓型；>80% → 冲刺型
  ⑨ 月线过热过滤（v2.8新增，硬过滤）：
     - 启动前1月涨幅 >20% → 短期过热，排除
     - 启动前2月涨幅 >60%/80%（月线多头时放宽到80%）
     - 黑马特征：月线上升通道 OR 历史低位启动，稳健不过热
  ⓘ v4.1：启动前15日涨跌绝对值>=5%天数>=4 → 排除（短期炒作型）
     逻辑：8日内2天及以上涨跌超5%说明有短线活跃资金，不符合安静建仓型
  ⑩ 大单资金过滤（v3.6新增，硬过滤）：
     - 启动日大单+超大单净量 <=0亿 → 无条件排除（主力出逃）
     - 数据来源：tushare moneyflow接口（buy_lg_amount/sell_lg_amount/buy_elg_amount/sell_elg_amount）

T+1持有评估（买入后使用，不用于买入决策）：
  ① 融资持续：买入后连续3天融资余额增加
  ② 缩量加速：持有期股价涨但量比持续下降
  ③ 连续涨停：T日+T+1均有涨停

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
TRADE_DAYS_BACK = 60    # 往前查多少个交易日做背离分析（60日才能捕捉慢建仓型，如603318水发燃气）
MIN_RISE_PCT = 7.0      # 涨幅门槛（%）
MARGIN_DIVERGENCE_DAYS = 5  # 背离需要持续多少天
CONTINUOUS_MARGIN_DAYS = 3  # 启动后融资余额需连续增加天数
LOOKUP_DAYS = 5         # 向后多查多少个交易日找启动日（默认5个）
# v2.7新增融资四维度的阈值（基于黑马vs蜗牛后验分析）
# v3.4修正：
#   - 正天数阈值0.65→0.55（黑马边界案例如600487在数据不足时仅48%，调低捕获边缘黑马）
#   - 新增5日净买绝对规模阈值（5日净买<5亿=均匀建仓，>5亿=冲刺型）
#   - 5D/30D比例仅在30日净买>=5亿时有意义（避免小分母导致ratio失真）
MARGIN_POS_DAYS_RATIO_THRESH = 0.55   # 30日融资净买入为正天数占比阈值（黑马>=55%）
MARGIN_CHG_5D_THRESH = 20.0           # 启动前5日融资增幅上限（黑马<20%，蜗牛>40%）
MARGIN_NET_30D_POSITIVE = True        # 30日融资净买总额>0（排除负值蜗牛）
MARGIN_5D_NET_ABS_THRESH = 10.0       # v3.5修正：原5亿会误杀黑马600487，改为10亿
MARGIN_5D_NET_RATIO_THRESH = 0.80     # 5日净买占30日净买比例（>80%=冲刺型，仅在30日净买>=5亿时有效）
# v2.9新增：月线过热维度阈值（基于月线趋势区分黑马vs蜗牛）
PRE_1M_CHG_THRESH = 20.0             # 启动前1月涨幅上限（>20%=短期过热，排除蜗牛）
PRE_1M_WARM_THRESH = 25.0            # v3.0缓冲带上限（20%~25%为"观察"级别，不直接排除）
PRE_2M_CHG_THRESH = 60.0            # 启动前2月涨幅上限（非月线多头时）
PRE_2M_CHG_THRESH_BULL = 80.0       # 启动前2月涨幅上限（月线多头时，放宽20%）
# v3.6新增：大单资金过滤
LAUNCH_DAY_NET_LG_THRESH = 0          # 启动日大单净量阈值(亿)，<=0亿则无条件排除（负=主力出逃）
# v2.10新增：市场环境过滤参数
MARKET_DROP_SKIP_THRESH = -2.0       # 单日跌幅>2% → 当日跳过买入
MARKET_CONSEC_DROP_THRESH = 3        # 连续3日下跌 → 降仓50%
BOARD_COUNT_DEMOTE = 2               # 板数>=2 → 降低权重50%（v2.13调整）
BOARD_COUNT_SKIP = 4                 # 板数>=4 → 直接排除（v2.13调整，原为5）
# =========================

pro = ts.pro_api(TOKEN)


def get_market_env(target_date):
    """获取目标日期的市场环境状态
    Returns: {
        'index_code': '000001.SH',
        'index_chg': float,       # 当日涨跌幅%
        'ma5_direction': str,     # 'up'/'down'/'flat'
        'recent_3d_chg': list,    # 近3日涨跌幅列表
        'env_level': str,         # 'safe'/'caution'/'skip'
        '降仓比例': float,         # 0.0=正常, 0.5=降50%
    }
    """
    # 获取上证指数近5日数据
    index_code = '000001.SH'
    end_dt = pd.to_datetime(target_date)
    start_dt = end_dt - pd.Timedelta(days=15)
    start_str = start_dt.strftime('%Y%m%d')

    try:
        index_df = pro.index_daily(ts_code=index_code, start_date=start_str, end_date=target_date)
        index_df = index_df.sort_values('trade_date', ascending=False).head(6)
        index_df['pct_chg'] = index_df['pct_chg'].astype(float)
    except Exception:
        return {
            'index_code': index_code, 'index_chg': 0.0,
            'ma5_direction': 'flat', 'recent_3d_chg': [],
            'env_level': 'safe', '降仓比例': 0.0
        }

    if len(index_df) < 5:
        return {
            'index_code': index_code, 'index_chg': 0.0,
            'ma5_direction': 'flat', 'recent_3d_chg': [],
            'env_level': 'safe', '降仓比例': 0.0
        }

    # 按日期升序排列（老日期在前）
    index_df = index_df.sort_values('trade_date')

    chg_today = float(index_df.iloc[-1]['pct_chg'])   # 目标日涨跌幅
    chg_list = index_df['pct_chg'].tolist()[-4:]      # 近4日（包含今日）

    # MA5方向：用最近5日收盘价简单判断
    ma5_prices = index_df['close'].astype(float).tolist()[-5:]
    if len(ma5_prices) >= 5:
        ma5_direction = 'up' if ma5_prices[-1] > ma5_prices[-5] else 'down'
    else:
        ma5_direction = 'flat'

    # 统计近3日（不含今日，因为今日可能还在交易中，用前3个已确认交易日）
    recent_3d = index_df['pct_chg'].tolist()[-4:-1] if len(index_df) >= 4 else []
    consec_drops = sum(1 for c in recent_3d if c < 0)

    # 环境判断
    if chg_today < MARKET_DROP_SKIP_THRESH:
        env_level = 'skip'
        降仓比例 = 1.0
    elif ma5_direction == 'down':
        env_level = 'caution'
        降仓比例 = 0.5
    elif consec_drops >= MARKET_CONSEC_DROP_THRESH:
        env_level = 'caution'
        降仓比例 = 0.5
    else:
        env_level = 'safe'
        降仓比例 = 0.0

    return {
        'index_code': index_code,
        'index_chg': chg_today,
        'ma5_direction': ma5_direction,
        'recent_3d_chg': recent_3d,
        'env_level': env_level,
        '降仓比例': 降仓比例,
    }


def get_trade_dates_desc(end_date, n=30):
    """获取end_date前n个交易日列表（降序，最新在前）
    v3.4修复：改用end_date往前推n个交易日的逻辑，而不是固定60个自然日
    """
    # 粗略估计：n个交易日大约需要 n*1.5 个自然日（考虑周末和假期）
    approx_days = int(n * 1.5) + 10
    start_dt = pd.to_datetime(end_date) - pd.Timedelta(days=approx_days)
    cal = pro.trade_cal(exchange='SSE', start_date=start_dt.strftime('%Y%m%d'), end_date=end_date)
    dates = cal[cal['is_open'] == 1]['cal_date'].tolist()
    return sorted([d for d in dates if d <= end_date], reverse=True)[:n]


def get_trade_dates_asc(start_date, n=5):
    """获取start_date后n个交易日列表（升序）"""
    end_dt = pd.to_datetime(start_date) + pd.Timedelta(days=30)
    cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_dt.strftime('%Y%m%d'))
    dates = cal[cal['is_open'] == 1]['cal_date'].tolist()
    return sorted([d for d in dates if d >= start_date])[:n]


def get_monthly_close(ts_code, launch_date, months=4):
    """获取启动日前月线状态和过热维度（1月/2月涨幅）
    Args:
        ts_code: 股票代码
        launch_date: 启动日（YYYYMMDD格式）
        months: 未使用，保留参数
    Returns: {'pre_1m_chg': float, 'pre_2m_chg': float, 'ma_bullish': bool,
              'price_above_ma5': bool, 'overheat_1m': bool, 'overheat_2m': bool}
    """
    # 直接查全部历史（不需要start_date，MA20只需要20个月，自然有足够数据）
    mdf = pro.monthly(ts_code=ts_code, end_date='20260531')
    mdf = mdf.sort_values('trade_date')
    mdf['close'] = mdf['close'].astype(float)
    mdf['ma5'] = mdf['close'].rolling(5).mean()
    mdf['ma10'] = mdf['close'].rolling(10).mean()
    mdf['ma20'] = mdf['close'].rolling(20).mean()
    mdf = mdf.dropna()
    if len(mdf) < 4:
        return None

    # 启动日所在月（launch_date格式YYYYMMDD）
    launch_month_str = launch_date[:6]  # 'YYYYMM'
    launch_dt = pd.to_datetime(launch_date)

    # 找启动日之前的最近一个月末
    mdf['trade_date_str'] = mdf['trade_date'].astype(str)
    prev_month_rows = mdf[mdf['trade_date_str'] < launch_month_str]
    if prev_month_rows.empty:
        return None
    last_month_row = prev_month_rows.iloc[-1]  # 最近一个已记录的月末

    # 最近第2个已记录月末
    if len(prev_month_rows) < 2:
        return None
    second_last_row = prev_month_rows.iloc[-2]  # 第2个已记录月末

    # 最近第3个已记录月末（用于计算2月涨幅）
    if len(prev_month_rows) < 3:
        return None
    third_last_row = prev_month_rows.iloc[-3]

    # 月线多头判断（最近一个已记录月末）
    ma_bullish = (float(last_month_row['ma5']) > float(last_month_row['ma10']) > float(last_month_row['ma20']))
    price_above_ma5 = (float(last_month_row['close']) > float(last_month_row['ma5']))

    # 月末收盘
    last_close = float(last_month_row['close'])        # 最近月末 = 3月末(20260331)
    second_close = float(second_last_row['close'])     # 第2个月末 = 2月末(20260227)
    third_close = float(third_last_row['close'])       # 第3个月末 = 1月末(20260130)

    # 启动日收盘
    try:
        ddf = pro.daily(ts_code=ts_code, start_date=launch_date, end_date=launch_date)
        launch_close = float(ddf['close'].iloc[0])
    except Exception:
        launch_close = last_close

    # 1月涨幅 = (启动日收盘 / 3月末收盘 - 1) * 100
    pre_1m_chg = (launch_close / last_close - 1) * 100 if last_close > 0 else 0.0
    # 2月涨幅 = (启动日收盘 / 1月末收盘 - 1) * 100
    pre_2m_chg = (launch_close / third_close - 1) * 100 if third_close > 0 else 0.0

    # 过热判断
    threshold_2m = PRE_2M_CHG_THRESH_BULL if (ma_bullish and price_above_ma5) else PRE_2M_CHG_THRESH
    overheat_1m = pre_1m_chg > PRE_1M_CHG_THRESH
    overheat_2m = pre_2m_chg > threshold_2m
    # v3.0新增：缓冲带。1月涨幅20%~25%为"观察"级别，不直接排除
    is_warm = PRE_1M_CHG_THRESH < pre_1m_chg <= PRE_1M_WARM_THRESH

    return {
        'pre_1m_chg': pre_1m_chg,
        'pre_2m_chg': pre_2m_chg,
        'ma_bullish': ma_bullish,
        'price_above_ma5': price_above_ma5,
        'overheat_1m': overheat_1m,
        'overheat_2m': overheat_2m,
        'is_warm': is_warm,
    }


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
    margin_df['rzche_yi'] = margin_df['rzche'] / 1e8
    return margin_df


def get_price_data(ts_code, trade_dates):
    """获取个股日线数据（trade_dates为降序列表，最新在前）"""
    start = trade_dates[-1]  # 最老的日期
    end = trade_dates[0]     # 最新日期
    df = pro.daily(ts_code=ts_code, start_date=start, end_date=end,
                   fields='trade_date,close,pct_chg,vol,amount,open,high,low,pre_close')
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values('trade_date').reset_index(drop=True)
    # vol_ma5: 标准前5日算法
    # rolling(5).mean().shift(1) 在 T 日 = mean(vol[T-1]~vol[T-5])
    # vol_ratio = vol[T] / vol_ma5[T] = T日成交量 / 前5日成交量均值
    df['vol_ma5'] = df['vol'].rolling(5).mean().shift(1)
    df['vol_ratio'] = df['vol'] / df['vol_ma5']
    # vol_ma10: 前10日算法（Y型量比专用）
    df['vol_ma10'] = df['vol'].rolling(10).mean().shift(1)
    df['vol_ratio_10d'] = df['vol'] / df['vol_ma10']
    return df


def _is_launch_date_pre8_clean(ts_code, candidate_date, all_calendar_sorted):
    """
    检查候选启动日的前8日涨跌是否干净（>=5%的天数<2天）。
    如果不干净（>=2天），返回False，让调用方继续往前找。
    这是"安静建仓"的必要条件——如果启动日之前已有频繁异动，说明不是真正的黑马建仓起点。
    """
    # 向前取8个交易日（不包括candidate_date当天）
    cand_idx = None
    for i, d in enumerate(all_calendar_sorted):
        if d == candidate_date:
            cand_idx = i
            break
    if cand_idx is None or cand_idx < 9:
        return True  # 数据不够，不阻止

    # 取candidate_date之前的8个交易日
    pre8_dates = all_calendar_sorted[cand_idx - 8:cand_idx]
    if len(pre8_dates) < 8:
        return True

    # 查这8天的pct_chg
    date_strs = [str(d) for d in pre8_dates]
    df = pro.daily(ts_code=ts_code, start_date=date_strs[0], end_date=date_strs[-1],
                   fields='trade_date,pct_chg')
    if df.empty:
        return True

    df_dict = {row['trade_date']: row['pct_chg'] for _, row in df.iterrows()}
    dirty_days = sum(1 for d in date_strs if d in df_dict and abs(float(df_dict[d])) >= 5)
    return dirty_days < 2


def find_true_launch_date(ts_code, scan_date, all_calendar, min_rise_pct=7.0):
    """
    找到真正的启动日（连续涨停序列的第一板）

    规则：
        1. 从scan_date向前，逐个检查是否是涨停板(>=9.5%)
        2. 如果是涨停板，继续向前找
        3. 直到找到"前一天涨幅<=5% 且 当天涨幅>=min_rise_pct"的日子 → 这就是启动日
        4. 如果当天>=min_rise_pct但<9.5%，也找到了（可能是反弹启动，不是连续板）

    返回: (启动日字符串, 第几板int) 或 (None, None)
    """
    # 完整日历：历史 + 未来
    all_calendar_sorted = sorted(all_calendar)

    # 获取价格数据
    start = all_calendar_sorted[0]
    end = all_calendar_sorted[-1]
    df = pro.daily(ts_code=ts_code, start_date=start, end_date=end,
                   fields='trade_date,close,pct_chg,pre_close')
    if df.empty:
        return None, None

    df = df.sort_values('trade_date').reset_index(drop=True)
    price_dict = {row['trade_date']: row['pct_chg'] for _, row in df.iterrows()}

    # 按日期降序排列
    dates_in_price = sorted([d for d in price_dict.keys() if d <= scan_date], reverse=True)

    for curr_date in dates_in_price:
        if curr_date not in price_dict:
            continue

        curr_pct = price_dict[curr_date]

        # 找前一个（更早的）交易日
        curr_idx = None
        for i, d in enumerate(all_calendar_sorted):
            if d == curr_date:
                curr_idx = i
                break
        if curr_idx is None or curr_idx == 0:
            continue
        prev_trade_day = all_calendar_sorted[curr_idx - 1]
        prev_pct = price_dict.get(prev_trade_day, None)

        # 判断是否是启动日
        # 条件：当日涨幅>=min_rise_pct 且 前日涨幅<=5%（前日数据必须存在）
        # 注意：当前日数据不存在时（prev_pct is None），不能作为启动日，需继续找
        if curr_pct >= min_rise_pct and prev_pct is not None and prev_pct <= 5:
            # 找到启动日！先验证pre8是否干净（v4.2新增）
            if not _is_launch_date_pre8_clean(ts_code, curr_date, all_calendar_sorted):
                # pre8不干净，继续往前找更早的启动日
                continue
            # pre8干净，确认是启动日
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

        # 当天涨幅在7%~9.5%之间，或<7%：
        # 判断是否有有效前日数据
        if prev_pct is None:
            # 前日数据缺失（如0420之前有非交易日），继续向前找
            continue
        elif curr_pct >= min_rise_pct and prev_pct <= 5:
            # 当天涨幅>=7%且前日<=5%，找到启动日，先验证pre8是否干净（v4.2新增）
            if not _is_launch_date_pre8_clean(ts_code, curr_date, all_calendar_sorted):
                # pre8不干净，继续往前找更早的启动日
                continue
            # pre8干净，确认是启动日
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
        elif curr_pct >= min_rise_pct:
            # 当天涨幅>=min_rise_pct但前日>5%，不是启动日，停止搜索
            break
        else:
            # 当天涨幅<min_rise_pct，继续向前找
            continue

    return None, None


def analyze_stock_v2(ts_code, name, pct_chg, scan_date, all_calendar, margin_df, price_df, min_rise_pct=7.0):
    """
    v2.0: T日买入评估 + T+1持有评估

    T日买入评估（0~4分）：
      ① 背离验证（1分）
      ② 量比验证（1分）
      ③ 象限分类（T日开盘前决策，0/1分）
      ④ 启动日融资变化（最强单信号，0~2分）

    T+1持有评估（买入后使用）：
      ① 融资持续
      ② 缩量加速
      ③ 连续涨停
    """
    # ── 找到真正的启动日 ──
    launch_date, board_count = find_true_launch_date(ts_code, scan_date, all_calendar, min_rise_pct)
    if launch_date is None:
        return None

    # ── 启动日涨幅 ──
    launch_pct = None
    for _, row in price_df.iterrows():
        if str(row['trade_date']) == str(launch_date):
            launch_pct = row['pct_chg']
            break
    if launch_pct is None:
        return None

    # ═══════════════════════════════════════════════════════════
    # T日买入评估
    # ═══════════════════════════════════════════════════════════

    # ── ① 背离验证（使用启动日前数据）────────────
    price_before = price_df[price_df['trade_date'] < launch_date].copy()
    margin_before_ts = margin_df[margin_df['trade_date'] < launch_date].copy()
    margin_before_ts = margin_before_ts[margin_before_ts['ts_code'] == ts_code]

    divergence_days = 0
    price_chg = 0.0
    margin_chg = 0.0
    has_divergence = False

    if len(price_before) >= 10 and len(margin_before_ts) >= 5:
        merged_before = price_before.merge(margin_before_ts[['trade_date', 'rzye_yi']], on='trade_date', how='inner')
        if len(merged_before) >= 5:
            merged_before = merged_before.sort_values('trade_date').reset_index(drop=True)

            # v2.1修复：改用"启动前最后10日窗口"判断背离
            # 避免中间大幅波动（四连板/深跌）切断导致背离判断失真
            DIV_WINDOW = 10
            last_n = merged_before.tail(DIV_WINDOW)

            # 逐日背离天数（最后10日窗口内）
            divergence_days = 0
            for i in range(len(last_n) - 1):
                curr = last_n.iloc[i]
                nxt = last_n.iloc[i + 1]
                if (nxt['close'] < curr['close']) and (nxt['rzye_yi'] > curr['rzye_yi']):
                    divergence_days += 1

            # 最后10日窗口内的股价/融资变化
            if len(last_n) >= 3:
                price_chg = (last_n['close'].iloc[-1] / last_n['close'].iloc[0] - 1) * 100
                margin_chg = (last_n['rzye_yi'].iloc[-1] / last_n['rzye_yi'].iloc[0] - 1) * 100
                has_divergence = (price_chg < 0) and (margin_chg > 0) and (divergence_days >= 3)

    # ── ② 量比验证（使用T日当天数据）────────────
    # T日量比 vs 前5日均量（判断是否缩量上涨）
    vol_ratio_tday = None
    vol_ratio_10d_tday = None
    vol_ma5_before = None
    has_volume_shrink = False
    prev_day_pct_chg = None  # 前一天涨跌幅（用于判断温和放量场景）

    scan_row = price_df[price_df['trade_date'] == scan_date]
    if not scan_row.empty:
        vol_ratio_tday = scan_row['vol_ratio'].iloc[0]
        vol_ratio_10d_tday = scan_row['vol_ratio_10d'].iloc[0]

        # 获取前一天涨跌幅
        prev_day_rows = price_df[price_df['trade_date'] < scan_date]
        if len(prev_day_rows) >= 1:
            prev_day = prev_day_rows.sort_values('trade_date').tail(1)
            prev_day_pct_chg = prev_day['pct_chg'].iloc[0]

        # 前5日均量（不含T日）
        vol_ma5_before = price_df[
            (price_df['trade_date'] < scan_date) &
            (price_df['trade_date'] >= launch_date)
        ]['vol_ratio'].mean() if len(price_df[price_df['trade_date'] < scan_date]) >= 5 else None

        # 两种有效量价信号（满足任一即得1分）：
        # ① 经典缩量上涨：T日上涨 but 量比 < 前期均值×90%（机构锁仓）
        # ② 温和放量：前一天下跌 + T日上涨 + 量比 < 1.5（下跌无量+启动日温和放量 = 健康主力行为）
        # 注意：①②互斥，不同时满足
        cond_shrink = cond_healthy = False
        if vol_ratio_tday is not None and vol_ma5_before is not None and vol_ma5_before > 0:
            cond_shrink = (pct_chg > 0) and (vol_ratio_tday < vol_ma5_before * 0.9)
        if vol_ratio_tday is not None and prev_day_pct_chg is not None:
            cond_healthy = (prev_day_pct_chg < 0 and pct_chg > 0 and vol_ratio_tday < 1.5)
        has_volume_shrink = cond_shrink or cond_healthy

    # ── ④ 启动日融资变化（最强单信号，使用启动日当天数据）────────────
    # 获取启动日融资余额 vs 前一交易日
    launch_margin = margin_df[
        (margin_df['ts_code'] == ts_code) &
        (margin_df['trade_date'] == launch_date)
    ]
    prev_margin = margin_df[
        (margin_df['ts_code'] == ts_code) &
        (margin_df['trade_date'] < launch_date)
    ].sort_values('trade_date', ascending=False).head(1)

    launch_margin_chg = None  # 启动日融资变化%
    margin_signal = 0         # 0/1/2
    margin_signal_desc = '无数据'

    if not launch_margin.empty and not prev_margin.empty:
        launch_rzye = launch_margin['rzye_yi'].iloc[0]
        prev_rzye = prev_margin['rzye_yi'].iloc[0]
        if prev_rzye > 0:
            launch_margin_chg = (launch_rzye / prev_rzye - 1) * 100

            if launch_margin_chg > 10:
                margin_signal = 2   # 强烈买入信号
                margin_signal_desc = f'+{launch_margin_chg:.1f}% 强烈买入'
            elif launch_margin_chg < -5:
                margin_signal = 0   # 否决信号
                margin_signal_desc = f'{launch_margin_chg:.1f}% 否决'
            else:
                margin_signal = 1   # 普通信号
                margin_signal_desc = f'{launch_margin_chg:+.1f}% 普通'

    # ═══════════════════════════════════════════════════════════
    # v2.7新增：融资三维度（黑马vs蜗牛后验发现的关键差异）
    # v3.4修复：改用"启动日前最近30个交易日"而非45个自然日
    #   原因：603318真实黑马正天数54.2%（30个交易日内），原窗口45天≈32个交易日会漏算首尾
    # ═══════════════════════════════════════════════════════════
    # 取启动日之前最近30个交易日
    launch_dt = pd.to_datetime(launch_date)
    all_margin_before = margin_before_ts[margin_before_ts['trade_date'] < launch_date].sort_values('trade_date')
    if len(all_margin_before) >= 30:
        margin_before_30d = all_margin_before.tail(30).reset_index(drop=True)
    else:
        margin_before_30d = all_margin_before.reset_index(drop=True)

    # 维度1：30日融资净买入为正天数占比（黑马>=55%，蜗牛偏低）
    margin_pos_days_ratio = 0.0
    if len(margin_before_30d) >= 15:
        # 当日净买入 ≈ rzmre - rzche（融资买入 - 融资偿还）
        margin_before_30d = margin_before_30d.copy()
        margin_before_30d['daily_net'] = (
            margin_before_30d['rzmre'] - margin_before_30d['rzche']
        )
        pos_days = (margin_before_30d['daily_net'] > 0).sum()
        total_days = len(margin_before_30d)
        margin_pos_days_ratio = pos_days / total_days if total_days > 0 else 0.0

    # 维度2：启动前5日融资增幅（黑马<20%，蜗牛>40%）
    # 用前5日（启动日之前最近5个交易日，对应pre5窗口）
    pre5_margin = margin_before_30d.tail(5).sort_values('trade_date')
    margin_chg_5d = 0.0
    if len(pre5_margin) >= 3:
        rzye_start = pre5_margin['rzye_yi'].iloc[0]
        rzye_end   = pre5_margin['rzye_yi'].iloc[-1]
        if rzye_start > 0:
            margin_chg_5d = (rzye_end / rzye_start - 1) * 100

    # 维度3：30日融资净买总额（辅助参考）
    margin_net_30d = 0.0
    if len(margin_before_30d) >= 5:
        margin_before_30d = margin_before_30d.copy()
        margin_before_30d['daily_net'] = (
            margin_before_30d['rzmre'] - margin_before_30d['rzche']
        )
        margin_net_30d = margin_before_30d['daily_net'].sum()

    # 维度4（v3.4修正）：5日净买绝对规模 + 5日净买占30日比例
    # - 均匀建仓型：5日净买 < 5亿（主力慢慢吸）
    # - 冲刺型：5日净买 > 5亿 且 5D/30D > 80%（短期集中拉升）
    # - 仅在30日净买>=5亿时，5D/30D比例才有意义（避免小分母导致ratio失真）
    margin_net_5d = 0.0
    if len(pre5_margin) >= 3:
        pre5_margin = pre5_margin.copy()
        pre5_margin['daily_net'] = (
            pre5_margin['rzmre'] - pre5_margin['rzche']
        )
        margin_net_5d = pre5_margin['daily_net'].sum()
    margin_net_5d_ratio = 0.0
    if margin_net_30d >= 5.0:  # 仅在30日净买>=5亿时看比例
        margin_net_5d_ratio = margin_net_5d / margin_net_30d

    # ── 象限分类（使用启动日前数据）────────────
    # Baux新增：启动前融资持续增长但股价几乎不涨（隐形建仓型）
    # 计算启动前30日融资变化（捕捉长期无声建仓）
    margin_before_sorted = margin_before_ts.sort_values('trade_date')
    if len(margin_before_sorted) >= 10:
        margin_long_chg = (margin_before_sorted['rzye_yi'].iloc[-1] / margin_before_sorted['rzye_yi'].iloc[0] - 1) * 100
    else:
        margin_long_chg = 0.0

    # ── v4.1：启动前15日涨跌绝对值>=5%天数过滤 ─────────────────────────
    # 排除这类股票，避免把短线炒作误判为黑马启动
    # 重要：启动日本身（涨幅>=7%的涨停日）不纳入统计
    DIV_WINDOW = 15
    price_before_launch = price_df[price_df['trade_date'] < launch_date].copy()
    price_before_launch = price_before_launch.sort_values('trade_date').reset_index(drop=True)
    last_15_window = price_before_launch.tail(DIV_WINDOW)

    # 启动前5日涨幅（启动日之前最近5个交易日）
    pre5 = last_15_window.tail(5)
    pre5_chg = 0.0
    if len(pre5) >= 3:
        pre5_chg = (pre5['close'].iloc[-1] / pre5['close'].iloc[0] - 1) * 100

    # 启动前15日总涨跌幅（用于Y型过滤：涨跌幅<=-3%直接排除）
    pre15_chg = 0.0
    if len(last_15_window) >= 3:
        pre15_chg = (last_15_window['close'].iloc[-1] / last_15_window['close'].iloc[0] - 1) * 100

    # 启动前5日最大单日涨幅（用于Baux排除有明显异动的股票）
    pre5_max_day = 0.0
    if len(pre5) >= 2:
        pre5_with_ret = pre5.copy()
        pre5_with_ret['daily_return'] = pre5_with_ret['close'].pct_change() * 100
        pre5_max_day = pre5_with_ret['daily_return'].max()

    # v6.14新增：启动前5日单日最大涨幅（用于Y型条件⑨——封堵分段计算漏洞）
    # 原理：pre5_chg是区间总涨幅，分段计算可能隐藏单日暴涨（如1212涨+10.1%在1217前5日内）
    # Y型要求：单日最大涨幅<=5%，超过则排除（已有明显异动）
    pre5_single_day_max = 0.0
    if len(pre5) >= 2:
        pre5_ret = pre5.copy()
        pre5_ret['daily_return'] = pre5_ret['close'].pct_change() * 100
        pre5_single_day_max = pre5_ret['daily_return'].max()

    pre15_rise5_abs_days = 0
    if len(last_15_window) >= 2:
        window_excl_launch = last_15_window[last_15_window['trade_date'] != launch_date]
        pre15_rise5_abs_days = (window_excl_launch['pct_chg'].abs() >= 5).sum()

    # v6.9新增：启动前15日涨跌幅绝对值>7%的天数（用于Y型过滤：>=2天直接排除）
    pre15_rise7_abs_days = 0
    if len(last_15_window) >= 2:
        window_excl_launch = last_15_window[last_15_window['trade_date'] != launch_date]
        pre15_rise7_abs_days = (window_excl_launch['pct_chg'].abs() > 7).sum()

    # v6.15新增：启动前15日K线实体统计（涨跌幅实体 = |收盘-开盘|/开盘）
    # 原理：启动日前K线实体越小（多日小涨小跌），说明游资在悄悄建仓、不惊动市场
    #       实体越大（长红长绿），说明有资金提前异动，不符合安静建仓逻辑
    # 指标1：pre15_body_small_abs_days = 实体涨幅<2%的天数（越多越好）
    # 指标2：pre15_body_big_abs_days = 实体涨幅>=5%的天数（>=3天直接排除）
    pre15_body_small_abs_days = 0
    pre15_body_big_abs_days = 0
    if len(last_15_window) >= 2 and 'open' in last_15_window.columns and 'close' in last_15_window.columns:
        window_excl = last_15_window[last_15_window['trade_date'] != launch_date].copy()
        if len(window_excl) > 0:
            window_excl['body_pct'] = (window_excl['close'] - window_excl['open']).abs() / window_excl['open'] * 100
            pre15_body_small_abs_days = (window_excl['body_pct'] < 2.0).sum()
            pre15_body_big_abs_days = (window_excl['body_pct'] >= 5.0).sum()

    # ── v6.5 形态检测：启动前"急跌+缩量横盘"结构 ─────────────────────────
    # 形态定义：启动前15日内有 >=1次急跌（单日跌幅>=5%）+ 随后横盘缩量
    # 典型案例：301217.SZ（0323急跌-7.8% + 0403~0406缩量横盘 + 0410启动）
    #          002033.SZ（0323急跌-9.7% + 0324~0414缩量横盘 + 0415启动）
    # 注意：急跌可能发生在启动日前16~20日，所以用20日窗口（大于v4.1的15日）
    # 形态本质：游资"吓出散户后快速拉升"——在启动日前通过急跌洗盘
    #           后续横盘期极度缩量说明游资在悄悄吸筹而非出货
    CRASH_WINDOW = 20  # 用更大的窗口捕捉启动前更早的急跌
    last_crash_window = price_before_launch.tail(CRASH_WINDOW)
    pre15_crash_days = last_crash_window[last_crash_window['pct_chg'] <= -5.0]
    has_pre_crash = len(pre15_crash_days) >= 1

    # 横盘缩量验证：急跌后5日内成交量比急跌日缩量>=30%（游资在吸筹而非出逃）
    has_lateral_shrink = False
    if has_pre_crash:
        crash_row = pre15_crash_days.sort_values('trade_date').iloc[-1]
        crash_date = crash_row['trade_date']
        crash_vol = float(crash_row['vol'])
        # 在last_crash_window中定位crash_date的位置（用窗口内的相对索引）
        crash_window_indices = [i for i in last_crash_window.index
                               if last_crash_window.loc[i, 'trade_date'] == crash_date]
        if crash_window_indices:
            crash_local_idx = crash_window_indices[0]
            # 用iloc定位crash在窗口中的相对位置
            crash_rel_pos = list(last_crash_window.index).index(crash_local_idx)
            all_indices = list(last_crash_window.index)
            post_indices = all_indices[crash_rel_pos:crash_rel_pos + 5]
            post_crash = last_crash_window.loc[post_indices]
            if len(post_crash) >= 2:
                avg_post_vol = float(post_crash['vol'].mean())
                if crash_vol > 0 and avg_post_vol <= crash_vol * 0.70:
                    has_lateral_shrink = True

    has_crash_dig = has_pre_crash and has_lateral_shrink

    # ═══════════════════════════════════════════════════════════
    # v6.0 Y型（游资炒作型）
    # 计算逻辑：pre5d_rzche_ratio_mean + launch_day_rzche_ratio
    # ═══════════════════════════════════════════════════════════
    is_y_type = False
    y_type_reason = ''
    pre5d_rzche_ratio_mean = 0.0
    launch_day_rzche_ratio = 0.0
    margin_sustainability_pass = False  # 默认值，避免UnboundLocalError（if块外使用前必须初始化）

    # v5.0新增：R型已放弃（风险太高），保留变量但不使用
    is_r_type = False
    r_type_reason = ''
    margin_pulse_days = 0

    quadrant = None
    quadrant_desc = None
    # 启动前有明显异动（有任一单日涨跌幅>=5%）
    # 注意：这里区分"纯炒作型异动"和"建仓型异动"
    # 建仓型异动：往往是借市场利空（大盘跌）洗盘，融资反而持续增长 → 不应直接否决
    # 纯炒作型异动：无利好无缘无故大涨大跌，融资不跟 → 坚决否决
    has_pre_suspicious = pre5_max_day >= 5
    has_pre_drop_5 = pre5_max_day <= -5  # 借利空挖坑型（大盘跌时跟跌）

    if has_divergence and pre5_chg > 5 and not has_pre_suspicious:
        quadrant = 'A'
        quadrant_desc = '★ 最优先买入'
    elif pre5_chg < -5 and launch_margin_chg is not None and launch_margin_chg > 10:
        quadrant = 'B'
        quadrant_desc = '△ 谨慎买入（启动日融资>10%）'
    elif margin_long_chg > 25 and pre5_chg < 10 and launch_margin_chg is not None and launch_margin_chg > 0:
        # Baux：隐形建仓型
        # v2.6修复：移除 pre5_max_day < 5 的硬性排除
        # 原因：启动前借利空挖坑（单日大跌）≠ 纯炒作型异动
        # 真正的纯炒作：无缘无故单日暴涨；建仓型：大跌后缩量横盘
        # 判断标准：若有单日异动>=5%，要求融资长期增长>40%（强建仓信号）才过
        if pre5_max_day >= 5:
            if margin_long_chg >= 40:
                quadrant = 'Baux'
                quadrant_desc = '☆ 隐形建仓（融资持续增，建仓型异动）'
        else:
            quadrant = 'Baux'
            quadrant_desc = '☆ 隐形建仓（融资持续增）'
    elif pre5_chg > 5 and pre5_chg < 10 and not has_divergence and not has_pre_suspicious:
        quadrant = 'C'
        quadrant_desc = '◇ 小仓位短线'
    else:
        quadrant = 'D'
        quadrant_desc = '○ 坚决不买（启动前有明显异动或无信号）'

    # ── T日买入总分 ──
    # 背离(0/1) + 量比(0/1) + 启动日融资(0/1/2)
    # 象限用于描述性分类，不直接加到总分（决定是否值得买）
    t_day_score = int(has_divergence) + int(has_volume_shrink) + margin_signal

    # ═══════════════════════════════════════════════════════════
    # v2.8新增：月线过热维度（启动前1月/2月涨幅判断，排除蜗牛）
    # 逻辑：黑马稳健启动（1月<20%，2月<60%/80%），蜗牛过热必崩
    # ═══════════════════════════════════════════════════════════
    overheat_1m = False
    overheat_2m = False
    pre_1m_chg = 0.0
    pre_2m_chg = 0.0
    ma_bullish = False
    price_above_ma5 = False

    monthly_data = get_monthly_close(ts_code, launch_date, months=4)
    if monthly_data is not None:
        ma_bullish = monthly_data['ma_bullish']
        price_above_ma5 = monthly_data['price_above_ma5']
        pre_1m_chg = monthly_data['pre_1m_chg']
        pre_2m_chg = monthly_data['pre_2m_chg']
        overheat_1m = monthly_data['overheat_1m']
        overheat_2m = monthly_data['overheat_2m']
        is_warm = monthly_data.get('is_warm', False)  # v3.0缓冲带
    else:
        ma_bullish = False
        price_above_ma5 = False
        pre_1m_chg = 0.0
        pre_2m_chg = 0.0
        overheat_1m = False
        overheat_2m = False
        is_warm = False

    # v3.0过热硬过滤：1月过热(>20%)时，仅在非缓冲带(非20%~25%)才排除；2月过热必排除
    is_overheat_excluded = (overheat_1m and not is_warm) or overheat_2m

    # ═══════════════════════════════════════════════════════════
    # v6.0 Y型（游资炒作型）判定
    # 核心思想：启动日前5个交易日融资大量涌入 + 启动日融资继续大量涌入
    # 指标：rzche/rzye（融资占余额比）= 当日新增融资/融资余额，衡量融资强度
    # ═══════════════════════════════════════════════════════════
    margin_before_sorted = margin_before_ts.sort_values('trade_date')
    pre5 = margin_before_sorted.tail(5)
    if len(pre5) >= 3 and vol_ratio_tday is not None:
        # 条件1：前5日rzche/rzye均值 > 5%（启动前持续吸钱）
        valid = pre5[pre5['rzye_yi'] > 0].copy()
        valid['rzche_ratio'] = valid['rzche_yi'] / valid['rzye_yi']
        launch_day_rzche_ratio = 0.0  # 默认值，避免UnboundLocalError
        pre5d_rzche_ratio_mean = 0.0   # 默认值，避免UnboundLocalError（前面只在len>=3时初始化）
        valid = valid.dropna(subset=['rzche_ratio'])
        if len(valid) >= 3:
            pre5d_rzche_ratio_mean = valid['rzche_ratio'].mean() * 100

        # 条件2：启动日rzche/rzye > 15%（启动日爆发）
        launch_margin_row = margin_df[
            (margin_df['ts_code'] == ts_code) &
            (margin_df['trade_date'] == launch_date)
        ]
        if not launch_margin_row.empty:
            rzche_val = launch_margin_row['rzche_yi'].iloc[0]
            rzye_val = launch_margin_row['rzye_yi'].iloc[0]
            if rzye_val > 0:
                launch_day_rzche_ratio = rzche_val / rzye_val * 100

        # v6.14新增：融资持续性验证
        # 条件⑩：启动日rzche比不超过前5日均值的2倍（避免"一日游"游资——启动日爆量后快速撤退）
        # 原理：若启动日融资是前5日均值的2倍以上，说明游资在启动日集中对倒，并非持续建仓
        #       真正的慢牛蓄力：启动日融资应该有迹可循，不是一次性冲高
        #       案例：图南股份300855——启动日28%是前5日均值13.7%的2.04倍，推荐后快速亏损
        margin_sustainability_pass = False
        if pre5d_rzche_ratio_mean > 0 and launch_day_rzche_ratio > 0:
            ratio_check = launch_day_rzche_ratio / pre5d_rzche_ratio_mean
            margin_sustainability_pass = (ratio_check <= 2.0)

    # Y型条件（需全部满足，共9条，硬过滤全部基于量价和K线形态）：
    #  1. vol_ratio_10d_tday >= 2.0（启动日前10交易日量比>=2，排除温和放量和游资控盘不活跃的票）
    #  2. pre5_chg <= 12.0（启动前5日涨跌<=12%，v6.30从<=8%放宽）
    #  3. pre15_chg > -3.0 and pre15_chg <= 20.0（启动前15日涨跌在健康区间，v6.30上限从15%扩至20%）
    #  4. pre15_rise7_abs_days < 2（启动前15日涨跌幅>=7%的天数<2，避免有明显的异动）
    #  5. pre5_single_day_max <= 5.0（启动前5日单日最大涨幅<=5%，避免启动前就有脉冲式异动）
    #  6. pre15_body_big_abs_days < 3（启动前15日K线实体>=5%的天数<3，实体大天数越多说明异动越明显）
    #  7. pre15_body_small_abs_days >= 6（启动前15日K线实体<2%的天数>=6，v6.29从>=8降至>=6）
    #  8. price_above_ma5 == True（月线沿着MA5上升，股票整体趋势向上）
    # 注：Y型融资数据（pre5d_rzche_ratio_mean、launch_day_rzche_ratio、融资蓄力比）仅作参考展示，不作为排除条件
    if (vol_ratio_10d_tday is not None and vol_ratio_10d_tday >= 2.0 and
        vol_ratio_10d_tday <= 2.5 and
        pre5_chg <= 12.0 and
        pre15_chg > -3.0 and pre15_chg <= 20.0 and
        pre15_rise7_abs_days < 2 and
        pre5_single_day_max <= 5.0 and
        pre15_body_big_abs_days < 3 and
        pre15_body_small_abs_days >= 6 and
        price_above_ma5):
        # v6.29新增：硬排除——大实体0天 + 量比>=5.0
        if pre15_body_big_abs_days == 0 and vol_ratio_10d_tday >= 5.0:
            is_y_type = False
            y_type_reason = '硬排除：大实体0天+量比>=5.0（游资大量对倒出货，跨批次全败）'
        # v6.29新增：硬排除——小实体8~9天 AND 前5日涨跌0~5%
        elif pre15_body_small_abs_days in (8, 9) and 0.0 <= pre5_chg <= 5.0:
            is_y_type = False
            y_type_reason = '硬排除：小实体8~9天+前5日涨跌0~5%（死亡组合，0416~0421批次0%胜率）'
        else:
            is_y_type = True
            # 融资数据仅作参考，格式化为reason字符串
            rzche_ratio_check = (launch_day_rzche_ratio / pre5d_rzche_ratio_mean) if pre5d_rzche_ratio_mean > 0 else 0.0
            rzche_ref = (f'前5日rzche比{pre5d_rzche_ratio_mean:.1f}%[参考]｜'
                        f'启动日rzche比{launch_day_rzche_ratio:.1f}%[参考]｜'
                        f'融资蓄力比{rzche_ratio_check:.1f}x[参考]｜')
            y_type_reason = (f'量比10日{vol_ratio_10d_tday:.2f}[2~2.5]｜'
                            f'前5日涨跌{pre5_chg:+.1f}%<=12%｜'
                            f'15日涨跌{pre15_chg:+.1f}%(-3%~20%)｜'
                            f'15日巨幅波动{pre15_rise7_abs_days}天<2天｜'
                            f'前5日单日最大{pre5_single_day_max:+.1f}%<=5%｜'
                            f'K线实体大天数{pre15_body_big_abs_days}<3｜'
                            f'小实体天数{pre15_body_small_abs_days}>=6｜'
                            f'{rzche_ref}'
                            f'月线MA5上升{price_above_ma5}')

    # v3.6新增：启动日大单资金过滤（无条件排除：启动日大单净量<=0）
    net_lg_elg_yi = None  # 大单+超大单净量(亿)
    try:
        mf = pro.moneyflow(ts_code=ts_code, start_date=launch_date, end_date=launch_date)
        if not mf.empty:
            buy_lg = float(mf['buy_lg_amount'].iloc[0])
            sell_lg = float(mf['sell_lg_amount'].iloc[0])
            buy_elg = float(mf['buy_elg_amount'].iloc[0])
            sell_elg = float(mf['sell_elg_amount'].iloc[0])
            net_lg_elg_yi = (buy_lg - sell_lg + buy_elg - sell_elg) / 1e8
    except Exception:
        pass

    if net_lg_elg_yi is not None and net_lg_elg_yi <= LAUNCH_DAY_NET_LG_THRESH:
        return None  # 启动日大单净出逃，无条件排除

    # ═══════════════════════════════════════════════════════════
    # T+1持有评估（使用启动日后数据，包括T+1）
    # ═══════════════════════════════════════════════════════════
    price_after_launch = price_df[price_df['trade_date'] >= launch_date].copy()
    margin_after_ts = margin_df[margin_df['trade_date'] >= launch_date].copy()
    margin_after_ts = margin_after_ts[margin_after_ts['ts_code'] == ts_code].reset_index(drop=True)

    # ── ① 融资持续（启动日后连续增加）────────────
    has_cont = False
    cont_days = 0
    margin_increase = 0.0

    if len(margin_after_ts) >= CONTINUOUS_MARGIN_DAYS:
        margin_after_ts = margin_after_ts.sort_values('trade_date').reset_index(drop=True)
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

    # ── ② 缩量加速（启动日后量比持续下降）────────────
    has_shrink = False
    vol_ratio_change = 1.0

    if len(price_before) >= 3 and len(price_after_launch) >= 2:
        vr_before = price_before['vol_ratio'].tail(3).mean()
        vr_after = price_after_launch['vol_ratio'].head(3).mean()
        avg_pct_after = price_after_launch['pct_chg'].head(3).mean()
        has_shrink = (avg_pct_after > 0) and (vr_after < vr_before)
        vol_ratio_change = vr_after / vr_before if vr_before > 0 else 1.0

    # ── ③ 连续涨停（T日+T+1均有涨停）────────────
    launch_plus1 = price_after_launch[price_after_launch['trade_date'] > launch_date].head(5)
    recent = pd.concat([price_df[price_df['trade_date'] == launch_date], launch_plus1]).tail(6)
    limit_days = (recent['pct_chg'] >= 9.5).sum()
    has_limit = limit_days >= 2

    return {
        'ts_code': ts_code,
        'name': name,
        'pct_chg': pct_chg,
        'scan_date': scan_date,
        'launch_date': launch_date,
        'launch_pct': launch_pct,
        'board_count': board_count,

        # T日买入评估
        't_day_score': t_day_score,
        'has_divergence': has_divergence,
        'div_days': divergence_days,
        'price_chg': price_chg,
        'margin_chg': margin_chg,
        'has_volume_shrink': has_volume_shrink,
        'vol_ratio_tday': vol_ratio_tday,
        'vol_ratio_10d_tday': vol_ratio_10d_tday,
        'vol_ma5_before': vol_ma5_before,
        'quadrant': quadrant,
        'quadrant_desc': quadrant_desc,
        'pre5_chg': pre5_chg,
        'pre15_chg': pre15_chg,
        'pre5_max_day': pre5_max_day,
        'pre5_single_day_max': pre5_single_day_max,  # v6.14新增：Y型条件⑨
        'pre15_rise5_abs_days': pre15_rise5_abs_days,
        'pre15_rise7_abs_days': pre15_rise7_abs_days,
        'pre15_body_small_abs_days': pre15_body_small_abs_days,  # v6.15新增：K线实体<2%天数
        'pre15_body_big_abs_days': pre15_body_big_abs_days,     # v6.15新增：K线实体>=5%天数
        'margin_long_chg': margin_long_chg,
        'margin_signal': margin_signal,
        'launch_margin_chg': launch_margin_chg,
        'margin_signal_desc': margin_signal_desc,

        # v2.7新增融资三维度（+维度4）
        'margin_pos_days_ratio': margin_pos_days_ratio,   # 维度1：30日净买入正天数占比
        'margin_chg_5d': margin_chg_5d,                  # 维度2：启动前5日融资增幅
        'margin_net_30d': margin_net_30d,                 # 维度3：30日融资净买总额(亿)
        'margin_net_5d': margin_net_5d,             # 维度4a：5日净买绝对值(元)
        'margin_net_5d_ratio': margin_net_5d_ratio,  # 维度4b：5日净买占30日比例(仅30日>=5亿时有效)

        # v2.8新增月线过热维度（v3.0修正：1月20%~25%为缓冲带，不直接排除）
        'is_overheat_excluded': is_overheat_excluded,     # 硬过滤：过热排除（1月>20%且非缓冲带，或2月过热）
        'overheat_1m': overheat_1m,                      # 启动前1月过热
        'overheat_2m': overheat_2m,                      # 启动前2月过热
        'is_warm': is_warm,                              # v3.0新增：1月涨幅20%~25%缓冲带
        'pre_1m_chg': pre_1m_chg,                        # 启动前1月涨幅%
        'pre_2m_chg': pre_2m_chg,                        # 启动前2月涨幅%
        'ma_bullish': ma_bullish,                         # 月线多头排列
        'price_above_ma5': price_above_ma5,              # 价格在MA5上方

        # T+1持有评估
        'has_cont': has_cont,
        'cont_days': cont_days,
        'margin_increase': margin_increase,
        'has_shrink': has_shrink,
        'vol_ratio_change': vol_ratio_change,
        'has_limit': has_limit,
        'limit_days': limit_days,

        # v5.0新增：R型（游资快速拉升型）
        'is_r_type': is_r_type,
        'r_type_reason': r_type_reason,
        'margin_pulse_days': margin_pulse_days,

        # v6.0新增：Y型（游资炒作型）
        'is_y_type': is_y_type,
        'y_type_reason': y_type_reason,
        'pre5d_rzche_ratio_mean': pre5d_rzche_ratio_mean,
        'launch_day_rzche_ratio': launch_day_rzche_ratio,
        'margin_sustainability_pass': margin_sustainability_pass,  # v6.14新增：融资持续性

        # v6.5新增：形态特征
        'has_crash_dig': has_crash_dig,
        'pre5_chg': pre5_chg,
    }


def scan_date(target_date=None, verify_mode=False, codes_filter=None, min_rise_pct=7.0):
    """扫描指定日期的涨幅>=门槛%股票
       codes_filter: list of ts_codes to limit scan (for batch processing)
       min_rise_pct: 涨幅门槛，默认7.0
    """
    if target_date is None:
        cal = pro.trade_cal(exchange='SSE', start_date='20260420', end_date='20260430')
        cal = cal[cal['is_open'] == 1]['cal_date'].tolist()
        target_date = cal[-2]
    else:
        target_date = str(target_date)

    print(f"\n{'='*70}")
    print(f"📅 扫描日期: {target_date}  |  涨幅门槛: >={min_rise_pct}%")
    if codes_filter:
        print(f"📦 批量模式: {len(codes_filter)}只")
    print(f"🎯 评估类型: T日买入评估（启动日当天决策）")
    print(f"{'='*70}\n")

    # Step 1: 涨幅>=门槛%的股票
    df_today = pro.daily(trade_date=target_date)
    riseN = df_today[df_today['pct_chg'] >= min_rise_pct].copy()
    riseN = riseN.sort_values('pct_chg', ascending=False)
    # v6.1：跳过北交所股票（不投资北交所）
    riseN = riseN[~riseN['ts_code'].str.endswith('.BJ')]

    if codes_filter:
        riseN = riseN[riseN['ts_code'].isin(codes_filter)]

    if riseN.empty:
        print(f"今日无涨幅>={min_rise_pct}%的股票")
        return

    print(f"涨幅>={min_rise_pct}% 股票数: {len(riseN)}\n")
    print(f"{'代码':<12} {'涨幅%':>8} {'扫到日期':>10}")
    print('-' * 35)
    for _, row in riseN.head(15).iterrows():
        print(f"{row['ts_code']:<12} {row['pct_chg']:>8.2f} {target_date:>10}")

    # Step 2: 历史交易日
    trade_dates_desc = get_trade_dates_desc(target_date, TRADE_DAYS_BACK)
    forward_dates = get_trade_dates_asc(target_date, LOOKUP_DAYS)
    all_calendar = sorted(set(trade_dates_desc + forward_dates))

    print(f"\n历史数据范围: {trade_dates_desc[-1]} ~ {trade_dates_desc[0]}")
    print(f"启动日识别：向后多查{LOOKUP_DAYS}个交易日\n")

    # Step 3: 批量加载融资数据
    print("加载融资数据（批量查询中）...")
    margin_df = load_all_margin_data(trade_dates_desc)
    print(f"融资数据: {len(margin_df)} 条记录\n")

    # Step 4: 逐只分析
    print(f"{'='*70}")
    print(f"🔍 T日买入评估中...")
    print(f"{'='*70}\n")

    results = []
    for idx, (_, stock) in enumerate(riseN.iterrows()):
        ts_code = stock['ts_code']
        name = stock.get('name', ts_code)

        if (idx + 1) % 20 == 0 or idx == len(riseN) - 1:
            print(f"  已处理 {idx+1}/{len(riseN)} 只股票...")

        try:
            price_df = get_price_data(ts_code, trade_dates_desc)
            if price_df.empty:
                continue

            result = analyze_stock_v2(
                ts_code, name, stock['pct_chg'],
                target_date, all_calendar, margin_df, price_df, min_rise_pct
            )
            if result is not None:
                results.append(result)
            else:
                print(f"\n  ⚠️ {ts_code} 返回None（启动日未找到或被过滤）")
        except Exception as e:
            import traceback
            print(f"\n  ⚠️ {ts_code} 分析异常: {e}")
            traceback.print_exc()
            continue

        # v6.30：API频率限制修复——每20只股票延迟0.5秒，避免触发tushare 1500次/分钟限制
        if (idx + 1) % 20 == 0 and idx < len(riseN) - 1:
            time.sleep(0.5)

    if not results:
        print("无足够数据完成分析")
        return

    df_result = pd.DataFrame(results)
    # 按T日总分降序
    df_result = df_result.sort_values('t_day_score', ascending=False)

    # ═══════════════════════════════════════════════════════════
    # v2.10新增：计算gap（启动日与扫描日的交易日差）
    # gap = 扫描日在all_calendar中的位置 - 启动日在all_calendar中的位置
    # gap > 0：启动日早于扫描日（已启动一段时间）
    # gap = 0：启动日=扫描日（当日启动）
    # ═══════════════════════════════════════════════════════════
    all_dates_asc = sorted(all_calendar)
    scan_idx = all_dates_asc.index(target_date) if target_date in all_dates_asc else 0
    df_result['launch_date_str'] = df_result['launch_date'].astype(str)
    df_result['launch_idx'] = df_result['launch_date_str'].apply(
        lambda d: all_dates_asc.index(d) if d in all_dates_asc else -1
    )
    df_result['gap'] = scan_idx - df_result['launch_idx']

    # ═══════════════════════════════════════════════════════════
    # v2.10新增：市场环境检测
    # ═══════════════════════════════════════════════════════════
    market_env = get_market_env(target_date)
    df_result['market_env_level'] = market_env['env_level']
    df_result['market_index_chg'] = market_env['index_chg']
    df_result['market_ma5_dir'] = market_env['ma5_direction']
    df_result['market_reduce_ratio'] = market_env['降仓比例']

    env_label = {'safe': '✅ 安全', 'caution': '⚠️ 谨慎', 'skip': '🚫 跳过'}
    print(f"\n📈 市场环境: {env_label.get(market_env['env_level'], market_env['env_level'])} | "
          f"上证{market_env['index_chg']:+.2f}% | MA5={market_env['ma5_direction']} | "
          f"降仓比例={market_env['降仓比例']*100:.0f}%")
    if market_env['recent_3d_chg']:
        print(f"   近3日涨跌: {[f'{c:.2f}%' for c in market_env['recent_3d_chg']]}")
    print()

    # ═══════════════════════════════════════════════════════════
    # v2.13新增：候选过滤（板数/gag/象限优先级/Y型启动日涨幅）
    # ═══════════════════════════════════════════════════════════
    n_before = len(df_result)

    # 调整1：删除总分>=3死亡硬规则（v2.13——规律不稳定）
    # 原v2.10代码：if market_env['env_level'] == 'skip': df_filtered = df_result[df_result['t_day_score'] >= 3]
    # v2.13：市场环境差时仍然保留全部候选，不因总分过滤

    # 调整2：板数过滤——>=4板直接排除，>=2板降低权重50%（v2.13更新）
    df_filtered = df_result.copy()
    df_filtered['board_demoted'] = (df_filtered['board_count'] >= BOARD_COUNT_DEMOTE) & (df_filtered['board_count'] < BOARD_COUNT_SKIP)
    df_filtered['board_excluded'] = df_filtered['board_count'] >= BOARD_COUNT_SKIP
    n_board_skip = df_filtered['board_excluded'].sum()
    if n_board_skip > 0:
        print(f"⚠️  板数过滤：{n_board_skip}只因>={BOARD_COUNT_SKIP}板被排除")
    df_filtered = df_filtered[~df_filtered['board_excluded']]

    # 调整3：Baux象限加市场条件——市场偏空时将Baux降级为D象限
    if market_env['env_level'] in ('caution', 'skip') and '降仓比例' in market_env:
        bau_x_count = ((df_filtered['quadrant'] == 'Baux') & (df_filtered['market_ma5_dir'] == 'down')).sum()
        if bau_x_count > 0:
            print(f"⚠️  Baux象限{ bau_x_count }只因MA5向下降级为D象限处理")
            mask = (df_filtered['quadrant'] == 'Baux') & (df_filtered['market_ma5_dir'] == 'down')
            df_filtered.loc[mask, 'quadrant'] = 'D'
            df_filtered.loc[mask, 'quadrant_desc'] = '○ D象限（MA5向下降级）'

    # 调整4：gag>0从D象限排除条件 → 改为B象限加分项（v2.13——D象限gag>0不稳定）
    # D象限不再因gag<=0过滤，只要板数满足即可
    # gag>0在B象限内作为bonus标记（影响排序）

    # 调整5：B象限 + gag=0 + 1板 = 最优先买入信号（v2.13 ★最强组合）
    df_filtered['b_gag0_1board'] = (
        (df_filtered['quadrant'] == 'B') &
        (df_filtered['gap'] == 0) &
        (df_filtered['board_count'] == 1)
    )
    n_star = df_filtered['b_gag0_1board'].sum()
    if n_star > 0:
        print(f"★ B象限+gag=0+1板：{n_star}只候选（迄今最强组合，100%胜率）")

    # 调整6：B象限gag>0 = 加分标记
    df_filtered['b_gag_positive'] = (
        (df_filtered['quadrant'] == 'B') &
        (df_filtered['gap'] > 0)
    )

    # 调整7：D象限 + pre5_chg > 10% → 否决（v2.28新增——高位跟涨陷阱）
    # 实证：000990 pre5=+10.3%却成功，但这是低概率随机事件(1/10)，不能因为1个例外保留10个风险
    d_pre5_too_high = (
        (df_filtered['quadrant'] == 'D') &
        (df_filtered['pre5_chg'] > 10)
    )
    n_d_pre5_skip = d_pre5_too_high.sum()
    if n_d_pre5_skip > 0:
        print(f"⚠️  D象限pre5>10%：{n_d_pre5_skip}只否决（高位跟涨风险）")
        df_filtered.loc[d_pre5_too_high, 'board_excluded'] = True

    # 调整8：D象限启动日距T日>7天 → 否决（v2.28新增——历史启动日陷阱）
    # 原理：启动日早已过去（如0325启动、0407才扫到），这些股票早已完成主升，不是真正的"启动日买入"
    d_launch_too_old = (
        (df_filtered['quadrant'] == 'D') &
        (df_filtered['launch_idx'] >= 0) &
        ((scan_idx - df_filtered['launch_idx']) > 7)
    )
    n_d_launch_skip = d_launch_too_old.sum()
    if n_d_launch_skip > 0:
        print(f"⚠️  D象限启动日距T日>7天：{n_d_launch_skip}只否决（历史启动日陷阱）")
        df_filtered.loc[d_launch_too_old, 'board_excluded'] = True

    # 重新过滤被排除的股票
    df_filtered = df_filtered[~df_filtered['board_excluded']]

    n_after = len(df_filtered)
    print(f"📊 候选过滤: {n_before} -> {n_after} 只（v2.28调整后）\n")

    # ═══════════════════════════════════════════════════════════
    # 打印汇总表
    #  v5.5/6.0 分类体系：
    #   黑马（机构安静建仓型）：总分>=2 + 非Y型
    #   Y型（游资炒作型）：vol_ratio>1.5 + 前5日rzche比均值>5% + 启动日rzche比>15%
    #   黑马（机构安静建仓型）：总分>=2 + 非Y型 + 月线MA5上升(price_above_ma5=True, v6.12新增)
    # v2.10：使用df_filtered（已通过市场/gap/板数过滤）
    top_for_print = df_filtered[
        (df_filtered['t_day_score'] >= 2) &
        (~df_filtered['is_overheat_excluded']) &
        (df_filtered['pre15_rise5_abs_days'] < 4) &
        (df_filtered['is_y_type'] == False) &  # 黑马排除Y型（只留安静建仓型）
        (df_filtered['price_above_ma5'] == True)   # v6.12新增：月线MA5上升
    ]

    print(f"\n{'='*130}")
    header = (f"{'代码':<12} {'名称':<6} {'T日总分':>6} "
              f"{'背离':>4} {'缩量':>4} {'象限':>4} {'融资信号':>10} "
              f"{'启动日':>8} {'gap':>4} {'板数':>4} 信号说明")
    print(header)
    print('-' * 130)

    for _, r in top_for_print.iterrows():
        launch_str = r['launch_date'][4:] if r['launch_date'] else "?"
        board_str = f"{int(r['board_count'])}板" if r['board_count'] else "?"
        demote_str = "⚠️" if r.get('board_demoted', False) else "  "
        gap_str = f"{int(r['gap'])}" if pd.notna(r['gap']) else "?"
        margin_sig = r.get('margin_signal_desc', 'N/A')
        if margin_sig == 'N/A' and r['launch_margin_chg'] is not None:
            chg = r['launch_margin_chg']
            if chg > 10:
                margin_sig = f'+{chg:.1f}%强买入'
            elif chg < -5:
                margin_sig = f'{chg:.1f}%否决'
            else:
                margin_sig = f'{chg:+.1f}%普通'
        elif margin_sig == 'N/A':
            margin_sig = '无数据'

        print(f"{r['ts_code']:<12} {r['name']:<6} {int(r['t_day_score']):>6}  "
              f"{'✓' if r['has_divergence'] else '✗':>4} "
              f"{'✓' if r['has_volume_shrink'] else '✗':>4} "
              f"{r['quadrant']:>4} "
              f"{margin_sig:>10} "
              f"{launch_str:>8} {gap_str:>4} {board_str:>4}  "
              f"{'★' if r.get('b_gag0_1board') else ' ':1} "
              f"{'G+' if r.get('b_gag_positive') else '  '} "
              f"{'⚠️' if r.get('board_demoted') else '   '} "
              f"{r['quadrant_desc']}")

    # v6.17: Y型综合强度分
    # 核心：启动日rzche涌入力度(权重最高) + 量比(游资参与度)
    # v6.17修改：融资贡献封顶5分（launch×0.5 + pre5d×0.2 + margin_chg_5d×0.3 ≤ 5）
    def calc_y_strength(row):
        launch = row['launch_day_rzche_ratio']
        vol = row['vol_ratio_10d_tday'] if pd.notna(row['vol_ratio_10d_tday']) else 1.0
        pre5d = row['pre5d_rzche_ratio_mean']
        margin_chg_5d = row.get('margin_chg_5d', 0.0)
        pre5_chg = row['pre5_chg']
        # 融资贡献封顶5分
        margin_contrib = launch * 0.5 + pre5d * 0.2 + margin_chg_5d * 0.3
        margin_contrib = min(margin_contrib, 5.0)
        base = vol * 2.0 + margin_contrib
        # 安静Bonus：启动前越安静、涨幅越低，游资吸筹越隐蔽
        if pre5_chg < -1.5:
            quiet_bonus = 4.5   # 深度安静：持续小幅下跌后启动，最隐蔽
        elif pre5_chg < 0:
            quiet_bonus = 3.5
        elif abs(pre5_chg) < 1:
            quiet_bonus = 2.5
        elif pre5_chg < 5:
            quiet_bonus = 1.5
        else:
            quiet_bonus = 0.0
        # 慢建仓Bonus：pre5d 5~12%区间代表游资"漫长低调"地悄悄吸筹
        if 5.0 <= pre5d < 10.0:
            slow_bonus = 4.0    # 极慢极隐蔽
        elif 10.0 <= pre5d <= 12.0:
            slow_bonus = 3.0    # 慢建仓
        else:
            slow_bonus = 0.0
        # v6.5急跌挖坑Bonus：有"急跌+缩量横盘"结构说明游资洗盘彻底，拉升更有力
        crash_bonus = 6.0 if row.get('has_crash_dig', False) else 0.0
        # v6.15新增小实体天数Bonus：启动前15日K线实体越小（<2%的天数越多），游资建仓越安静，拉升越有劲
        small_body_days = row.get('pre15_body_small_abs_days', 0)
        if small_body_days >= 10:
            body_quiet_bonus = 6.0   # 极度安静：10天以上小实体，完美蓄力形态
        elif small_body_days >= 8:
            body_quiet_bonus = 4.5
        elif small_body_days >= 6:
            body_quiet_bonus = 3.0
        elif small_body_days >= 4:
            body_quiet_bonus = 1.5
        else:
            body_quiet_bonus = 0.0
        return base + quiet_bonus + slow_bonus + crash_bonus + body_quiet_bonus

    # v6.14: 先在df_result上算好y_strength_score（y_top过滤需要）
    df_result['y_strength_score'] = df_result.apply(calc_y_strength, axis=1)

    # v2.13：使用df_filtered（已通过板数/Baux降级过滤）
    # v2.13新增：Y型启动日涨幅>15%过滤（游资已大幅拉升，非最佳介入点）
    y_type_candidates = df_filtered[
        (df_filtered['is_y_type'] == True) &
        (df_filtered['price_above_ma5'] == True) &   # v6.13新增：月线MA5上升
        (df_filtered['launch_pct'] <= 15.0)             # v2.13新增：启动日涨幅>15%排除
    ]

    y_type_candidates['y_strength_score'] = y_type_candidates.apply(calc_y_strength, axis=1)
    # v6.16移除：Y总分≥35门槛（融资条款降级为参考后，不再设置总分门槛）
    y_type_candidates = y_type_candidates.sort_values(
        'y_strength_score', ascending=False
    ).reset_index(drop=True)

    print(f"\n{'='*185}")
    y_header = (f"{'#':>3} {'代码':<12} {'名称':<6} {'T日总分':>6} {'Y强度分':>8} "
                f"{'量比(10日)[2~3]':>14} {'前5日rzche比[参]':>16} {'启动日rzche比[参]':>18} "
                f"{'前5日涨跌':>8} {'15日涨跌':>8} {'gap':>4} {'板数':>4} {'小实体<2%':>10} {'大实体>=5%':>10}  {'急跌挖坑'}")
    print(y_header)
    print('-' * 230)
    for idx, (_, r) in enumerate(y_type_candidates.iterrows(), 1):
        board_str = f"{int(r['board_count'])}板" if r['board_count'] else "?"
        pre5d_ratio = r.get('pre5d_rzche_ratio_mean', 0.0)
        rzche_ratio = r.get('launch_day_rzche_ratio', 0.0)
        vol_r = r.get('vol_ratio_10d_tday', 0.0)
        pre5_chg = r.get('pre5_chg', 0.0)
        pre15_chg = r.get('pre15_chg', 0.0)
        msp = r.get('margin_sustainability_pass', False)
        rzche_ratio_check = (rzche_ratio / pre5d_ratio) if pre5d_ratio > 0 else 0.0
        crash_tag = "🔨急跌挖坑" if r.get('has_crash_dig', False) else ""
        msp_tag = "融资蓄力✓" if msp else "融资蓄力✗"
        gap_str = f"{int(r['gap'])}" if pd.notna(r.get('gap')) else "?"
        print(f"{idx:>3}. {r['ts_code']:<12} {r['name']:<6} {int(r['t_day_score']):>6} {r['y_strength_score']:>8.1f} "
              f"{vol_r:>9.2f} {pre5d_ratio:>11.1f}%[参] {rzche_ratio:>11.1f}%[参] "
              f"{pre5_chg:>7.1f}% {pre15_chg:>7.1f}% {gap_str:>4} {board_str:>4} "
              f"{r.get('pre15_body_small_abs_days', 0):>10} {r.get('pre15_body_big_abs_days', 0):>10}  "
              f"{crash_tag} {msp_tag}")

    # ═══════════════════════════════════════════════════════════
    # 高分详情
    # ═══════════════════════════════════════════════════════════
    # 高分候选（黑马：安静建仓型，v6.12月线MA5上升，v2.13使用df_filtered+板数过滤）
    top = df_filtered[(df_filtered['t_day_score'] >= 2) & (~df_filtered['is_overheat_excluded']) & (df_filtered['pre15_rise5_abs_days'] < 4) & (df_filtered['is_y_type'] == False) & (df_filtered['price_above_ma5'] == True) & (df_filtered['pre15_body_big_abs_days'] < 3)]
    # Y型候选（v2.13使用df_filtered+启动日涨幅<=15%过滤）
    y_top = df_filtered[
        (df_filtered['is_y_type'] == True) &
        (df_filtered['price_above_ma5'] == True) &
        (df_filtered['launch_pct'] <= 15.0)
    ]

    # v2.9: 收集Baux/B/A候选用于验证模式
    verify_candidates = []
    for _, r in top.iterrows():
        if r['quadrant'] in ('Baux', 'B', 'A'):
            verify_candidates.append((r['ts_code'], r['launch_date'], r['quadrant'], int(r['t_day_score'])))

    if not top.empty:
        print(f"\n{'='*70}")
        print(f"📊 T日买入信号详情（总分>=2，共{len(top)}只，通过v4.2过滤）")
        print(f"{'='*70}\n")
        for _, r in top.iterrows():
            board_note = f"，扫到时已是第{int(r['board_count'])}板" if r['board_count'] and r['board_count'] > 1 else ""

            print(f"[{r['ts_code']}] {r['name']}  扫到日期: {r['scan_date']}  T日总分: {int(r['t_day_score'])}")
            print(f"  {'─'*50}")

            # T日买入评估
            print(f"  🎯 T日买入评估:")
            print(f"     ① 背离验证: {'✓' if r['has_divergence'] else '✗'} "
                  f"(股价{r['price_chg']:+.1f}% / 融资{r['margin_chg']:+.1f}%，{int(r['div_days'])}天)")

            vol_r = f"{r['vol_ratio_tday']:.2f}" if r['vol_ratio_tday'] else "N/A"
            vol_m = f"{r['vol_ma5_before']:.2f}" if r['vol_ma5_before'] else "N/A"
            cond = "缩量上涨" if r['has_volume_shrink'] else "✗不满足"
            print(f"     ② 量比验证: {cond} "
                  f"(T日量比={vol_r} vs 前期均值={vol_m})")

            print(f"     ③ 象限分类: {r['quadrant']} {r['quadrant_desc']} "
                  f"(启动前5日涨幅={r['pre5_chg']:+.1f}%)")

            # v4.1：启动前15日涨跌绝对值>=5%天数过滤
            print(f"     ⓘ v4.1过滤: 15日涨跌绝对值≥5%天数={int(r['pre15_rise5_abs_days'])}天"
                  f" {' ✗排除' if r['pre15_rise5_abs_days'] >= 4 else ' ✓通过'}")

            margin_sig = r.get('margin_signal_desc', 'N/A')
            if margin_sig == 'N/A' and r['launch_margin_chg'] is not None:
                chg = r['launch_margin_chg']
                if chg > 10:
                    margin_sig = f'+{chg:.1f}% 强烈买入'
                elif chg < -5:
                    margin_sig = f'{chg:.1f}% 否决'
                else:
                    margin_sig = f'{chg:+.1f}% 普通'
            print(f"     ④ 启动日融资: {margin_sig}")

            # v2.7新增：融资四维度（黑马vs蜗牛的核心差异）
            # 注意：margin_net_30d 单位是"元"，需要除以1e8转亿
            ratio_pct = r['margin_pos_days_ratio'] * 100
            pos_days_pass = "✓" if r['margin_pos_days_ratio'] >= MARGIN_POS_DAYS_RATIO_THRESH else "✗"
            chg5d_pass = "✓" if r['margin_chg_5d'] < MARGIN_CHG_5D_THRESH else "✗"
            net30_yi = r['margin_net_30d'] / 1e8  # 转为亿
            net30_pass = "✓" if MARGIN_NET_30D_POSITIVE and r['margin_net_30d'] > 0 else ("N/A" if not MARGIN_NET_30D_POSITIVE else "✗")
            net5d_ratio_pct = r['margin_net_5d_ratio'] * 100
            net5d_pass = "✓" if abs(r['margin_net_5d']) < MARGIN_5D_NET_ABS_THRESH else "✗"
            print(f"     ⑤ 融资正天数占比: {pos_days_pass} {ratio_pct:.0f}% (阈值>={MARGIN_POS_DAYS_RATIO_THRESH*100:.0f}%)")
            print(f"     ⑥ 启动前5日融资增幅: {chg5d_pass} {r['margin_chg_5d']:+.1f}% (阈值<{MARGIN_CHG_5D_THRESH:.0f}%)")
            print(f"     ⑦ 30日融资净买总额: {net30_yi:+.2f}亿 ({net30_pass})")
            net5d_yi = r['margin_net_5d'] / 1e8  # 转亿
            print(f"     ⑧ 5日净买绝对规模: {net5d_pass} {net5d_yi:+.2f}亿 (阈值<{MARGIN_5D_NET_ABS_THRESH:.0f}亿)")

            # v2.8新增：月线过热维度
            if r['is_overheat_excluded']:
                print(f"     ⚠️ 月线过热过滤: ✗ 排除（1月{r['pre_1m_chg']:+.1f}% 2月{r['pre_2m_chg']:+.1f}%）")
            else:
                bull_mark = "✓" if r['ma_bullish'] else "✗"
                threshold_2m = PRE_2M_CHG_THRESH_BULL if (r['ma_bullish'] and r['price_above_ma5']) else PRE_2M_CHG_THRESH
                print(f"     ⑨ 月线过热过滤: ✓ 通过（月线多头{bull_mark}，1月{r['pre_1m_chg']:+.1f}%<{PRE_1M_CHG_THRESH:.0f}%，2月{r['pre_2m_chg']:+.1f}%<{threshold_2m:.0f}%）")

            # T+1持有评估
            print(f"  📈 T+1持有评估（买入后使用，不用于买入决策）:")
            print(f"     ① 融资持续: {'✓' if r['has_cont'] else '✗'} (连续{int(r['cont_days'])}天, {r['margin_increase']:+.1f}%)")
            print(f"     ② 缩量加速: {'✓' if r['has_shrink'] else '✗'} (量比变化{r['vol_ratio_change']:.2f}x)")
            print(f"     ③ 连续涨停: {'✓' if r['has_limit'] else '✗'} ({int(r['limit_days'])}天)")

            # 结论
            print(f"  ✅ 结论: {r['name']} 启动日{r['launch_date']}（{r['launch_pct']:.1f}%{board_note}）")
            star_mark = " ★最强组合" if r.get('b_gag0_1board', False) else ""
            gag_bonus = "，gag>0加分" if r.get('b_gag_positive', False) else ""
            board_demote = " ⚠️降权50%" if r.get('board_demoted', False) else ""
            if r['quadrant'] == 'A':
                print(f"     → 象限A，最优先买入，T日总分{int(r['t_day_score'])}分{star_mark}{gag_bonus}{board_demote}")
            elif r['quadrant'] == 'B':
                print(f"     → 象限B，谨慎买入{star_mark}{gag_bonus}{board_demote}")
            elif r['quadrant'] == 'Baux':
                print(f"     → 象限Baux，隐形建仓买入（融资持续增{int(r['margin_long_chg']):+.0f}%，前5日最大单日{int(r['pre5_max_day']):+.1f}%）{board_demote}")
            elif r['quadrant'] == 'C':
                print(f"     → 象限C，小仓位短线{board_demote}")
            else:
                print(f"     → 象限D，不建议买入{board_demote}")
            print()

    # R型已放弃，不输出详情

    # ═══════════════════════════════════════════════════════════
    # 信号规则说明
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"T日买入评估（0~4分，只用T-1及之前 + T日当天数据）:")
    print(f"  ① 背离验证（1分）：启动前最后10日窗口内股价跌/融资余额增，持续>=3天")
    print(f"     v2.1修复：改用启动前最后10日窗口，避免中间大幅波动切断判断")
    print(f"  ② 量比验证（1分）：T日上涨但量比 < 前期均量（缩量上涨）")
    print(f"     新增：前一天跌+T日涨+量比<1.5也有效（温和放量=健康主力行为）")
    print(f"  ③ 启动日融资变化（0~2分）：>+10%强烈买入，<-5%否决")
    print(f"\nT+1持有评估（买入后使用，不用于买入决策）:")
    print(f"  ① 融资持续：启动日后连续3天融资余额增加")
    print(f"  ② 缩量加速：持有期股价涨但量比持续下降")
    print(f"  ③ 连续涨停：启动日+次日均有涨停")
    print(f"\n象限分类（用于描述性分类，不直接加到总分）:")
    print(f"  ★ 象限A：启动前涨幅>5% + 有实质背离 + 无单日异动>=5% → 最优先买入")
    print(f"  △ 象限B：启动前涨幅<-5% + 启动日融资>10% → 谨慎买入")
    print(f"  ☆ 象限Baux：启动前融资持续增长+25%但股价不涨（总涨幅<10%且无单日异动>=5%）+ 启动日融资>0% → 隐形建仓买入")
    print(f"  ◇ 象限C：启动前涨幅5~10% + 无实质背离 + 无单日异动 → 小仓位短线")
    print(f"  ○ 象限D：不属于A/B/Baux/C → 坚决不买（含启动前有明显异动）")
    print(f"\nY型游资炒作型（v6.0新增，独立于象限体系）:")
    print(f"  🌊 特征：启动日前5个交易日融资大量涌入 + 启动日融资继续大量涌入")
    print(f"  条件：量比>1.5 + 前5日rzche/rzye均值>5% + 启动日rzche/rzye>15% + 15日异动<5天")
    print(f"  与黑马体系区别：黑马追求安静建仓，Y型专抓游资热炒加速段")
    print(f"{'='*70}")

    # v2.9: 验证模式汇总
    if verify_mode and verify_candidates:
        print_verify_summary(target_date, verify_candidates)
    elif verify_mode:
        print(f"\n（无可验证的Baux/B/A象限候选股）\n")





# ═══════════════════════════════════════════════════════════
# v2.9新增：验证模式 — 月线+融资结构+启动后走势汇总
# ═══════════════════════════════════════════════════════════

def get_verify_data(ts_code, launch_date):
    """获取候选股的验证数据（月线+融资四维度+启动后走势）

    Returns: {
        'pre_1m': float, 'pre_2m': float, 'ma_bullish': bool,
        'is_overheat': bool, 'overheat_reason': str, 'is_warm': bool,
        'pos_ratio': float, 'chg_5d': float, 'net_30d': float, 'ratio_5d': float,
        'margin_score': int,  # 融资四维度通过数
        'peak_chg': float, 'pivot_day': int, 'pivot_chg': float,
        'type': str  # '黑马'/'蜗牛'/'待观察'/'过热排除'/'观察'
    }
    """
    result = {
        'pre_1m': 0.0, 'pre_2m': 0.0, 'ma_bullish': False,
        'is_overheat': False, 'overheat_reason': '', 'is_warm': False,
        'pos_ratio': 0.0, 'chg_5d': 0.0, 'net_30d': 0.0, 'ratio_5d': 0.0,
        'margin_score': 0, 'peak_chg': 0.0, 'pivot_day': None, 'pivot_chg': None,
        'type': '待观察'
    }

    # 1. 月线数据
    try:
        mdf = pro.monthly(ts_code=ts_code, end_date='20260531')
        mdf = mdf.sort_values('trade_date')
        mdf['close'] = mdf['close'].astype(float)
        mdf['ma5'] = mdf['close'].rolling(5).mean()
        mdf['ma10'] = mdf['close'].rolling(10).mean()
        mdf['ma20'] = mdf['close'].rolling(20).mean()
        mdf = mdf.dropna(subset=['close'])
        if len(mdf) >= 4:
            launch_month_str = launch_date[:6]
            mdf['td'] = mdf['trade_date'].astype(str)
            prev = mdf[mdf['td'] < launch_month_str]
            if len(prev) >= 3:
                last = prev.iloc[-1]
                third = prev.iloc[-3]
                result['ma_bullish'] = float(last['ma5']) > float(last['ma10']) > float(last['ma20'])
                price_above_ma5 = float(last['close']) > float(last['ma5'])
                ddf = pro.daily(ts_code=ts_code, start_date=launch_date, end_date=launch_date)
                if not ddf.empty:
                    lc = float(ddf['close'].iloc[0])
                    result['pre_1m'] = (lc / float(last['close']) - 1) * 100
                    result['pre_2m'] = (lc / float(third['close']) - 1) * 100
                    th2 = PRE_2M_CHG_THRESH_BULL if (result['ma_bullish'] and price_above_ma5) else PRE_2M_CHG_THRESH
                    is_warm = PRE_1M_CHG_THRESH < result['pre_1m'] <= PRE_1M_WARM_THRESH
                    # v3.0: 1月>25%=过热，20%~25%=缓冲带（不排除），≤20%=通过
                    if result['pre_1m'] > PRE_1M_WARM_THRESH:
                        result['is_overheat'] = True
                        result['overheat_reason'] = f"1月{result['pre_1m']:+.1f}%>{PRE_1M_WARM_THRESH:.0f}%"
                    elif result['pre_2m'] > th2:
                        result['is_overheat'] = True
                        result['overheat_reason'] = f"2月{result['pre_2m']:+.1f}%>{th2:.0f}%"
                    # v3.0: 缓冲带标记（用于类型判断）
                    result['is_warm'] = is_warm
    except Exception:
        pass

    # 2. 融资四维度
    try:
        m30 = pro.margin_detail(ts_code=ts_code, start_date='20260301', end_date=launch_date)
        m30 = m30.sort_values('trade_date')
        m30['nb'] = m30['rzye'].astype(float).diff()
        m30['pos'] = m30['nb'] > 0
        tot = len(m30)
        if tot > 0:
            result['pos_ratio'] = m30['pos'].sum() / tot
            result['net_30d'] = m30['nb'].sum()
            m5 = m30.tail(5)
            if len(m5) >= 2 and float(m5['rzye'].iloc[0]) != 0:
                result['chg_5d'] = (float(m5['rzye'].iloc[-1]) / float(m5['rzye'].iloc[0]) - 1) * 100
            nb5 = m5['nb'].sum()
            if result['net_30d'] != 0:
                result['ratio_5d'] = abs(nb5) / abs(result['net_30d'])
            # 计算通过数
            if result['pos_ratio'] >= MARGIN_POS_DAYS_RATIO_THRESH:
                result['margin_score'] += 1
            if result['chg_5d'] < MARGIN_CHG_5D_THRESH:
                result['margin_score'] += 1
            if MARGIN_NET_30D_POSITIVE and result['net_30d'] > 0:
                result['margin_score'] += 1
            if result['ratio_5d'] < MARGIN_5D_NET_RATIO_THRESH:
                result['margin_score'] += 1
    except Exception:
        pass

    # 3. 启动后走势（重心拐点）
    try:
        ddf = pro.daily(ts_code=ts_code, start_date=launch_date, end_date='20260531')
        ddf = ddf.sort_values('trade_date').reset_index(drop=True)
        if len(ddf) >= 2:
            lp = ddf.iloc[0]['close']
            pp = ddf['close'].max()
            result['peak_chg'] = (pp / lp - 1) * 100
            ddf['center'] = (ddf['high'] + ddf['low']) / 2
            peak_center = ddf['center'].iloc[0]
            consecutive = 0
            for i in range(1, len(ddf)):
                if ddf['center'].iloc[i] < peak_center:
                    consecutive += 1
                    if consecutive >= 2:
                        result['pivot_day'] = i
                        result['pivot_chg'] = (ddf.iloc[i]['center'] / lp - 1) * 100
                        break
                else:
                    peak_center = ddf['center'].iloc[i]
                    consecutive = 0
    except Exception:
        pass

    # 4. 类型判断
    if result['is_overheat']:
        result['type'] = '过热排除'
    elif result['is_warm']:
        # v3.0: 缓冲带（1月20%~25%），标记为观察
        result['type'] = '观察'
    elif result['peak_chg'] >= 15 and result['pivot_day'] is not None:
        result['type'] = '黑马'
    elif result['peak_chg'] >= 10 and result['pivot_day'] is not None:
        result['type'] = '黑马(小)'
    elif result['peak_chg'] < 5:
        result['type'] = '蜗牛'
    elif result['pivot_day'] is None and result['peak_chg'] > 0:
        result['type'] = '待观察'
    else:
        result['type'] = '蜗牛'

    return result


def print_verify_summary(scan_date_str, candidates):
    """打印候选股验证汇总表

    Args:
        scan_date_str: 扫描日期字符串
        candidates: list of (ts_code, launch_date, quadrant, score) tuples
    """
    print(f"\n{'='*90}")
    print(f"📋 候选股验证汇总（扫描日期: {scan_date_str}）")
    print(f"{'='*90}")
    print(f"{'代码':<12} {'象限':<6} {'分':<3} {'月线过热':<16} {'正天数':<8} {'5D增幅':<10} {'30D净买':<12} {'5D/30D':<8} {'峰值':<8} {'拐点':<10} {'类型':<10}")
    print(f"{'-'*90}")

    for ts_code, launch_date, quadrant, score in candidates:
        v = get_verify_data(ts_code, launch_date)
        if v['is_overheat']:
            overheat_str = v['overheat_reason']
        elif v['is_warm']:
            overheat_str = f"缓冲带({v['pre_1m']:+.1f}%)"
        elif v['pre_1m'] != 0:
            overheat_str = '✓通过'
        else:
            overheat_str = '无数据'
        pr_str = f"{v['pos_ratio']*100:.0f}%"
        c5_str = f"{v['chg_5d']:+.1f}%"
        nb_str = f"{v['net_30d']/1e8:+.2f}亿" if v['net_30d'] != 0 else 'N/A'
        r5_str = f"{v['ratio_5d']*100:.0f}%"
        peak_str = f"{v['peak_chg']:+.1f}%"
        if v['pivot_day'] is not None:
            piv_str = f"第{v['pivot_day']}天{v['pivot_chg']:+.1f}%"
        else:
            piv_str = '未确认'
        type_str = v['type']

        print(f"{ts_code:<12} {quadrant:<6} {score:<3} {overheat_str:<12} {pr_str:<8} {c5_str:<10} {nb_str:<12} {r5_str:<8} {peak_str:<8} {piv_str:<10} {type_str:<10}")
        time.sleep(0.3)  # 避免API限速

    print(f"{'='*90}")
    print(f"说明: 月线过热=1月>{PRE_1M_CHG_THRESH:.0f}%或2月>阈值; 融资四维度: 正天数≥65%/5D增幅<20%/30D净买>0/5D/30D<80%")
    print(f"{'='*90}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='史诗级行情扫描器 v3.0')
    parser.add_argument('date', nargs='?', default=None, help='扫描日期 YYYYMMDD格式，默认上一交易日')
    parser.add_argument('-v', '--verify', action='store_true', help='验证模式：对Baux/B/A象限股输出月线+融资结构+启动后走势')
    parser.add_argument('--codes', type=str, default=None, help='逗号分隔的股票代码列表，用于批量扫描')
    parser.add_argument('-t', '--threshold', type=float, default=7.0,
                        choices=[6.0, 7.0],
                        help='涨停门槛（6.0或7.0），默认7.0')
    args = parser.parse_args()

    codes_filter = None
    if args.codes:
        codes_filter = args.codes.split(',')

    scan_date(args.date, verify_mode=args.verify, codes_filter=codes_filter, min_rise_pct=args.threshold)
