#!/home/yrflj/.hermes/hermes-agent/venv/bin/python3
import sys
sys.path.insert(0, "/home/yrflj/.hermes/skills/investment-research/epic-rally-identification/scripts")
import tushare as ts
import pandas as pd

pro = ts.pro_api("fe284a8656a25bf46f4b9092178d1a4dd4d63f26bdb8b83d7f461c47")

def get_trade_date_n_days_ago(pro, end_date, n):
    df = pro.trade_cal(end_date=end_date, is_open=1)
    dates = df["cal_date"].tolist()
    if end_date in dates:
        idx = dates.index(end_date)
    else:
        return dates[-1]
    return dates[max(0, idx - n)]

launch_date = "20260423"

for code, name in [("002606.SZ", "大连电瓷"), ("301302.SZ", "华安鑫创"), ("600956.SH", "新赛股份")]:
    print(f"\n{'='*60}")
    print(f"深度分析: {code} {name}")
    print(f"{'='*60}")
    
    # 价格数据 - 用正确的日期范围
    price_df = pro.daily(ts_code=code, start_date="2026-01-05", end_date=launch_date)
    price_df = price_df.sort_values("trade_date").reset_index(drop=True)
    print(f"价格数据: {len(price_df)}条, 从{price_df.iloc[0]['trade_date']}到{price_df.iloc[-1]['trade_date']}")
    
    ld = launch_date
    launch_row = price_df[price_df["trade_date"] == ld]
    
    # 月线
    monthly = pro.monthly(ts_code=code, start_date="202501", end_date="202605")
    monthly = monthly.sort_values("trade_date").reset_index(drop=True)
    # monthly字段: trade_date, close, open, high, low, vol, amount, pct_chg
    # 需要计算MA5/MA10
    monthly['ma5'] = monthly['close'].rolling(5).mean()
    monthly['ma10'] = monthly['close'].rolling(10).mean()
    print(f"月线(最近3月): {monthly.tail(3)[['trade_date','close','ma5','ma10']].to_dict('records')}")
    
    # 量比(10日)
    pre10 = price_df[price_df["trade_date"] < ld].tail(10)
    vol_ma10 = pre10["vol"].mean()
    tday_vol = launch_row.iloc[0]["vol"] if len(launch_row) > 0 else 0
    vol_ratio = tday_vol / vol_ma10 if vol_ma10 > 0 else 0
    print(f"前10日均成交量: {vol_ma10:,.0f} | 启动日成交量: {tday_vol:,.0f} | 量比(10日): {vol_ratio:.2f}")
    
    # 15日涨跌
    pre15 = price_df[price_df["trade_date"] < ld].tail(15)
    if len(pre15) >= 2:
        pre15_chg = (pre15.iloc[-1]["close"] / pre15.iloc[0]["close"] - 1) * 100
        print(f"15日涨跌: {pre15_chg:.2f}%")
    
    # 前5日涨跌
    pre5p = price_df[price_df["trade_date"] < ld].tail(5)
    if len(pre5p) >= 2:
        pre5_chg = (pre5p.iloc[-1]["close"] / pre5p.iloc[0]["close"] - 1) * 100
        print(f"前5日涨跌: {pre5_chg:.2f}%")
    
    # 15日>7%天数
    pre15_pct = pre15["pct_chg"].dropna()
    rise7_days = (pre15_pct.abs() > 7).sum()
    print(f"15日涨跌绝对值>7%天数: {int(rise7_days)}天")
    
    # 15日>=5%天数
    rise5_days = (pre15_pct.abs() >= 5).sum()
    print(f"15日涨跌绝对值>=5%天数: {int(rise5_days)}天")
    
    # 融资数据
    margin_df = pro.margin_detail(ts_code=code, start_date="2026-01-05", end_date=launch_date)
    margin_df = margin_df.sort_values("trade_date").reset_index(drop=True)
    print(f"融资数据: {len(margin_df)}条")
    
    # 前5日rzche/rzye均值
    pre5_margin = margin_df[margin_df["trade_date"] < ld].tail(5)
    valid_pre5 = pre5_margin[pre5_margin["rzye"] > 0]
    if len(valid_pre5) >= 3:
        rzche_ratio_mean = (valid_pre5["rzche"] / valid_pre5["rzye"] * 100).mean()
        print(f"前5日rzche/rzye均值: {rzche_ratio_mean:.2f}%")
    else:
        print(f"前5日有效融资样本: {len(valid_pre5)} (需要>=3)")
    
    # 启动日rzche/rzye
    ld_m = margin_df[margin_df["trade_date"] == ld]
    if len(ld_m) > 0:
        rzye_t = ld_m.iloc[0]["rzye"]
        rzche_t = ld_m.iloc[0]["rzche"]
        if rzye_t > 0:
            rzche_ratio_t = rzche_t / rzye_t * 100
            print(f"启动日rzche/rzye: {rzche_ratio_t:.2f}%")
    else:
        print("启动日无融资数据")
    
    # 启动日融资变化
    prev_trade = pre5_margin.iloc[-1] if len(pre5_margin) > 0 else None
    if len(ld_m) > 0 and prev_trade is not None and prev_trade["rzye"] > 0:
        margin_chg = (ld_m.iloc[0]["rzye"] / prev_trade["rzye"] - 1) * 100
        print(f"启动日融资变化: {margin_chg:.2f}%")

print("\n\n查询完毕")
