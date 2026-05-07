#!/home/yrflj/.hermes/hermes-agent/venv/bin/python3
import tushare as ts
pro = ts.pro_api("fe284a8656a25bf46f4b9092178d1a4dd4d63f26bdb8b83d7f461c47")

launch = "20260109"
y_codes = ["600143.SH", "301231.SZ", "301280.SZ", "300433.SZ", "600637.SH"]
baux_codes = ["300058.SZ", "301171.SZ", "301088.SZ"]
d_codes = ["002364.SZ", "000960.SZ", "301387.SZ"]

# 1. 查名称
print("=== Y型候选 ===")
for c in y_codes:
    name = pro.stock_basic(ts_code=c, fields='name').iloc[0]['name']
    print(f"{c}: {name}")

# 2. Y型月线MA5检查
print("\n=== Y型月线MA5 ===")
for c in y_codes:
    name = pro.stock_basic(ts_code=c, fields='name').iloc[0]['name']
    monthly = pro.monthly(ts_code=c, start_date="202504", end_date="202605")
    monthly = monthly.sort_values("trade_date").reset_index(drop=True)
    monthly['ma5'] = monthly['close'].rolling(5).mean()
    last3 = monthly.tail(3)
    if len(last3) >= 2:
        cur = last3.iloc[-1]
        prv = last3.iloc[-2]
        ma5_up = cur['ma5'] > prv['ma5']
        above = cur['close'] > cur['ma5']
        print(f"{c} {name}: MA5={'上升✓' if ma5_up else '下降✗'} | 收盘{cur['close']} {'✓站上MA5' if above else '✗低于MA5'} | MA5={cur['ma5']:.2f}")

# 3. Y型8条件深度分析
print("\n\n=== Y型8条件深度分析 ===")
for code in y_codes:
    name = pro.stock_basic(ts_code=code, fields='name').iloc[0]['name']
    print(f"\n{'='*50}")
    print(f"{code} {name}")
    print(f"{'='*50}")

    price_df = pro.daily(ts_code=code, start_date="2025-10-01", end_date=launch)
    price_df = price_df.sort_values("trade_date").reset_index(drop=True)

    # 量比(10日)
    pre10 = price_df[price_df["trade_date"] < launch].tail(10)
    vol_ma10 = pre10["vol"].mean()
    launch_row = price_df[price_df["trade_date"] == launch]
    tday_vol = launch_row.iloc[0]["vol"] if len(launch_row) > 0 else 0
    vol_ratio = tday_vol / vol_ma10 if vol_ma10 > 0 else 0
    print(f"量比(10日): {vol_ratio:.2f} | 前10日均:{vol_ma10:,.0f} | 启动日:{tday_vol:,.0f}")

    # 15日涨跌
    pre15 = price_df[price_df["trade_date"] < launch].tail(15)
    if len(pre15) >= 2:
        pre15_chg = (pre15.iloc[-1]["close"] / pre15.iloc[0]["close"] - 1) * 100
        print(f"15日涨跌: {pre15_chg:.2f}%")

    # 前5日涨跌
    pre5p = price_df[price_df["trade_date"] < launch].tail(5)
    if len(pre5p) >= 2:
        pre5_chg = (pre5p.iloc[-1]["close"] / pre5p.iloc[0]["close"] - 1) * 100
        print(f"前5日涨跌: {pre5_chg:.2f}%")

    # 15日>7%天数
    pre15_pct = pre15["pct_chg"].dropna()
    rise7_days = (pre15_pct.abs() > 7).sum()
    print(f"15日涨跌绝对值>7%天数: {int(rise7_days)}天")

    # 融资
    margin_df = pro.margin_detail(ts_code=code, start_date="2025-10-01", end_date=launch)
    margin_df = margin_df.sort_values("trade_date").reset_index(drop=True)

    pre5_margin = margin_df[margin_df["trade_date"] < launch].tail(5)
    valid_pre5 = pre5_margin[pre5_margin["rzye"] > 0]
    if len(valid_pre5) >= 3:
        rzche_ratio_mean = (valid_pre5["rzche"] / valid_pre5["rzye"] * 100).mean()
        print(f"前5日rzche/rzye均值: {rzche_ratio_mean:.2f}%")
        for _, mrow in valid_pre5.iterrows():
            ratio = mrow["rzche"] / mrow["rzye"] * 100 if mrow["rzye"] > 0 else 0
            print(f"  {mrow['trade_date']}: rzye={mrow['rzye']/1e4:.0f}万, rzche={mrow['rzche']/1e4:.0f}万, 比={ratio:.2f}%")

    ld_m = margin_df[margin_df["trade_date"] == launch]
    if len(ld_m) > 0:
        rzye_t = ld_m.iloc[0]["rzye"]
        rzche_t = ld_m.iloc[0]["rzche"]
        if rzye_t > 0:
            rzche_ratio_t = rzche_t / rzye_t * 100
            print(f"启动日rzche/rzye: {rzche_ratio_t:.2f}% (rzye={rzye_t/1e4:.0f}万, rzche={rzche_t/1e4:.0f}万)")

    # 月线
    monthly = pro.monthly(ts_code=code, start_date="202504", end_date="202605")
    monthly = monthly.sort_values("trade_date").reset_index(drop=True)
    monthly['ma5'] = monthly['close'].rolling(5).mean()
    last3 = monthly.tail(3)
    if len(last3) >= 2:
        cur = last3.iloc[-1]
        prv = last3.iloc[-2]
        ma5_up = cur['ma5'] > prv['ma5']
        above = cur['close'] > cur['ma5']
        print(f"月线: MA5={'上升✓' if ma5_up else '下降✗'} | {'✓站上MA5' if above else '✗低于MA5'}(收盘{cur['close']}>MA5{cur['ma5']:.2f})")

# 4. 黑马月线检查(象限Baux)
print("\n\n=== 黑马月线检查(象限Baux) ===")
for code in baux_codes:
    name = pro.stock_basic(ts_code=code, fields='name').iloc[0]['name']
    monthly = pro.monthly(ts_code=code, start_date="202504", end_date="202605")
    monthly = monthly.sort_values("trade_date").reset_index(drop=True)
    monthly['ma5'] = monthly['close'].rolling(5).mean()
    last3 = monthly.tail(3)
    if len(last3) >= 2:
        cur = last3.iloc[-1]
        prv = last3.iloc[-2]
        ma5_up = cur['ma5'] > prv['ma5']
        above = cur['close'] > cur['ma5']
        print(f"{code} {name}: MA5={'上升✓' if ma5_up else '下降✗'} | 收盘{cur['close']} {'✓站上MA5' if above else '✗低于MA5'}")

# 5. D类月线检查(象限D)
print("\n\n=== 黑马月线检查(象限D - 高分候选) ===")
for code in d_codes:
    name = pro.stock_basic(ts_code=code, fields='name').iloc[0]['name']
    monthly = pro.monthly(ts_code=code, start_date="202504", end_date="202605")
    monthly = monthly.sort_values("trade_date").reset_index(drop=True)
    monthly['ma5'] = monthly['close'].rolling(5).mean()
    last3 = monthly.tail(3)
    if len(last3) >= 2:
        cur = last3.iloc[-1]
        prv = last3.iloc[-2]
        ma5_up = cur['ma5'] > prv['ma5']
        above = cur['close'] > cur['ma5']
        print(f"{code} {name}: MA5={'上升✓' if ma5_up else '下降✗'} | 收盘{cur['close']} {'✓站上MA5' if above else '✗低于MA5'}")
