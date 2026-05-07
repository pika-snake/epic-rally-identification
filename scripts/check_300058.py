#!/home/yrflj/.hermes/hermes-agent/venv/bin/python3
import tushare as ts
pro = ts.pro_api("fe284a8656a25bf46f4b9092178d1a4dd4d63f26bdb8b83d7f461c47")

# 检查300058象限分类问题
code = "300058.SZ"
monthly = pro.monthly(ts_code=code, start_date="202504", end_date="202605")
monthly = monthly.sort_values("trade_date").reset_index(drop=True)
monthly['ma5'] = monthly['close'].rolling(5).mean()
print(f"300058 蓝色光标 月线:")
for _, r in monthly.tail(4).iterrows():
    print(f"  {r['trade_date']}: 收盘={r['close']}, MA5={r['ma5']:.2f}")

# 300058启动日是1230，但这里象限显示Baux - 需要查融资结构
# 启动日1230的前5日涨跌=+6.3%（超过10%阈值），所以不是真正的Baux
# 真正的Baux要求前5日总涨幅<10%且无单日异动>=5%
print("\n300058 启动日1230前5日融资数据:")
margin_df = pro.margin_detail(ts_code=code, start_date="2025-12-01", end_date="20251230")
margin_df = margin_df.sort_values("trade_date").reset_index(drop=True)
pre5 = margin_df[margin_df["trade_date"] <= "20251230"].tail(6)
for _, m in pre5.iterrows():
    ratio = m["rzche"]/m["rzye"]*100 if m["rzye"] > 0 else 0
    print(f"  {m['trade_date']}: rzye={m['rzye']/1e4:.0f}万, rzche={m['rzche']/1e4:.0f}万, 比={ratio:.2f}%")

# 象限Baux的真实要求确认
print("\n确认Baux的融资持续增+68%数据:")
# 启动日0109的前5日(0102~0108)
pre5m = pro.margin_detail(ts_code=code, start_date="2025-12-25", end_date="20260108")
pre5m = pre5m.sort_values("trade_date").reset_index(drop=True)
print(f"前5日融资数据(0102~0108):")
for _, m in pre5m.iterrows():
    ratio = m["rzche"]/m["rzye"]*100 if m["rzye"] > 0 else 0
    print(f"  {m['trade_date']}: rzye={m['rzye']/1e4:.0f}万, rzche={m['rzche']/1e4:.0f}万, 比={ratio:.2f}%")
