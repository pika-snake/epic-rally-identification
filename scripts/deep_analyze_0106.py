#!/home/yrflj/.hermes/hermes-agent/venv/bin/python3
import tushare as ts

pro = ts.pro_api("fe284a8656a25bf46f4b9092178d1a4dd4d63f26bdb8b83d7f461c47")

launch_date = "20260106"

for code, name in [
    ("300821.SZ", "300821"),
    ("002160.SZ", "002160"),
    ("601615.SH", "601615"),
    ("600707.SH", "600707"),
    ("600869.SH", "600869"),
]:
    print(f"\n{'='*60}")
    print(f"深度分析: {code} {name}")
    print(f"{'='*60}")
    
    # 价格数据
    price_df = pro.daily(ts_code=code, start_date="2025-10-01", end_date=launch_date)
    price_df = price_df.sort_values("trade_date").reset_index(drop=True)
    
    ld = launch_date
    launch_row = price_df[price_df["trade_date"] == ld]
    
    # 月线
    monthly = pro.monthly(ts_code=code, start_date="202501", end_date="202605")
    monthly = monthly.sort_values("trade_date").reset_index(drop=True)
    monthly['ma5'] = monthly['close'].rolling(5).mean()
    monthly['ma10'] = monthly['close'].rolling(10).mean()
    print(f"月线(最近4月):")
    for _, row in monthly.tail(4).iterrows():
        print(f"  {row['trade_date']}: 收盘={row['close']}, MA5={row['ma5']:.2f}")
    
    # 月线MA5方向
    if len(monthly) >= 3:
        m_current = monthly.iloc[-1]
        m_prev = monthly.iloc[-2]
        ma5_up = m_current['ma5'] > m_prev['ma5']
        price_above_ma5 = m_current['close'] > m_current['ma5']
        print(f"月线MA5方向: {'上升' if ma5_up else '下降'} | 当期MA5={m_current['ma5']:.2f} | 当期收盘={m_current['close']} {'✓站上MA5' if price_above_ma5 else '✗低于MA5'}")
        print(f"  上期MA5={m_prev['ma5']:.2f} | 上期收盘={m_prev['close']} {'✓站上MA5' if m_prev['close'] > m_prev['ma5'] else '✗低于MA5'}")
    
    # 量比(10日)
    pre10 = price_df[price_df["trade_date"] < ld].tail(10)
    vol_ma10 = pre10["vol"].mean()
    tday_vol = launch_row.iloc[0]["vol"] if len(launch_row) > 0 else 0
    vol_ratio = tday_vol / vol_ma10 if vol_ma10 > 0 else 0
    print(f"\n前10日均成交量: {vol_ma10:,.0f} | 启动日成交量: {tday_vol:,.0f} | 量比(10日): {vol_ratio:.2f}")
    
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
    margin_df = pro.margin_detail(ts_code=code, start_date="2025-10-01", end_date=launch_date)
    margin_df = margin_df.sort_values("trade_date").reset_index(drop=True)
    print(f"融资数据: {len(margin_df)}条")
    
    # 前5日rzche/rzye均值
    pre5_margin = margin_df[margin_df["trade_date"] < ld].tail(5)
    valid_pre5 = pre5_margin[pre5_margin["rzye"] > 0]
    if len(valid_pre5) >= 3:
        rzche_ratio_mean = (valid_pre5["rzche"] / valid_pre5["rzye"] * 100).mean()
        print(f"前5日rzche/rzye均值: {rzche_ratio_mean:.2f}%")
        for _, mrow in valid_pre5.iterrows():
            ratio = mrow["rzche"] / mrow["rzye"] * 100 if mrow["rzye"] > 0 else 0
            print(f"  {mrow['trade_date']}: rzye={mrow['rzye']/1e4:.0f}万, rzche={mrow['rzche']/1e4:.0f}万, 比={ratio:.2f}%")
    else:
        print(f"前5日有效融资样本: {len(valid_pre5)} (需要>=3)")
    
    # 启动日rzche/rzye
    ld_m = margin_df[margin_df["trade_date"] == ld]
    if len(ld_m) > 0:
        rzye_t = ld_m.iloc[0]["rzye"]
        rzche_t = ld_m.iloc[0]["rzche"]
        if rzye_t > 0:
            rzche_ratio_t = rzche_t / rzye_t * 100
            print(f"启动日rzche/rzye: {rzche_ratio_t:.2f}% (rzye={rzye_t/1e4:.0f}万, rzche={rzche_t/1e4:.0f}万)")
    else:
        print("启动日无融资数据")

print("\n\n查询完毕")
