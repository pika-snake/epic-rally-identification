#!/home/yrflj/.hermes/hermes-agent/venv/bin/python3
import tushare as ts

pro = ts.pro_api("fe284a8656a25bf46f4b9092178d1a4dd4d63f26bdb8b83d7f461c47")

code = "688416.SH"

# 1. 日线数据 - 查启动日前后
print("="*60)
print("688416 日线数据（1225~0110）")
daily = pro.daily(ts_code=code, start_date="20251220", end_date="20260110")
daily = daily.sort_values("trade_date")
for _, row in daily.iterrows():
    print(f"  {row['trade_date']}: close={row['close']}, pct={row['pct_chg']:.2f}%, vol={row['vol']:,}")

# 2. 月线数据 - 确认202512收盘
print("\n月线数据:")
monthly = pro.monthly(ts_code=code, start_date="202510", end_date="202605")
monthly = monthly.sort_values("trade_date")
for _, row in monthly.iterrows():
    print(f"  {row['trade_date']}: close={row['close']}, vol={row['vol']}")

# 3. 计算MA5 at scan date (只用202512及之前的月线数据)
print("\n202512月线收盘: 71.24")
print("计算MA5(5个月收盘均值):")
closes_5 = monthly[monthly['trade_date'] <= '20251231']['close'].tail(5).tolist()
print(f"  最近5个月收盘: {closes_5}")
if len(closes_5) >= 5:
    ma5_dec = sum(closes_5) / 5
    print(f"  MA5(202512) = {ma5_dec:.2f}")
    print(f"  202512收盘71.24 vs MA5={ma5_dec:.2f}: {'✓站上MA5' if 71.24 > ma5_dec else '✗低于MA5'}")

# 4. 查20251231那天的月线数据
print("\n202512月线数据:")
dec_data = pro.monthly(ts_code=code, start_date="20251201", end_date="20251231")
print(dec_data.to_dict('records'))

print("\n\n300082 日线数据（1225~0110）")
daily2 = pro.daily(ts_code="300082.SZ", start_date="20251220", end_date="20260110")
daily2 = daily2.sort_values("trade_date")
for _, row in daily2.iterrows():
    print(f"  {row['trade_date']}: close={row['close']}, pct={row['pct_chg']:.2f}%, vol={row['vol']:,}")
