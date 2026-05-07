#!/home/yrflj/.hermes/hermes-agent/venv/bin/python3
import tushare as ts
pro = ts.pro_api("fe284a8656a25bf46f4b9092178d1a4dd4d63f26bdb8b83d7f461c47")

code = "600172.SH"
name = pro.stock_basic(ts_code=code, fields='name').iloc[0]['name']
print(f"{code} {name}")

price_df = pro.daily(ts_code=code, start_date="2025-12-01", end_date="20260108")
price_df = price_df.sort_values("trade_date").reset_index(drop=True)

pre10 = price_df[price_df["trade_date"] < "20260108"].tail(10)
vol_ma10 = pre10["vol"].mean()
tday = price_df[price_df["trade_date"] == "20260108"]
tday_vol = tday.iloc[0]["vol"]
print(f"量比(10日): {tday_vol/vol_ma10:.2f}")

pre15 = price_df[price_df["trade_date"] < "20260108"].tail(15)
pre15_chg = (pre15.iloc[-1]["close"]/pre15.iloc[0]["close"]-1)*100
print(f"15日涨跌: {pre15_chg:.2f}%")

pre5p = price_df[price_df["trade_date"] < "20260108"].tail(5)
pre5_chg = (pre5p.iloc[-1]["close"]/pre5p.iloc[0]["close"]-1)*100
print(f"前5日涨跌: {pre5_chg:.2f}%")

pre15_pct = pre15["pct_chg"].dropna()
rise7 = (pre15_pct.abs() > 7).sum()
print(f"15日>7%天数: {int(rise7)}天")

margin_df = pro.margin_detail(ts_code=code, start_date="2025-12-01", end_date="20260108")
margin_df = margin_df.sort_values("trade_date").reset_index(drop=True)
pre5m = margin_df[margin_df["trade_date"] < "20260108"].tail(5)
valid = pre5m[pre5m["rzye"] > 0]
if len(valid) >= 3:
    ratio = (valid["rzche"]/valid["rzye"]*100).mean()
    print(f"前5日rzche比均值: {ratio:.2f}%")

ldm = margin_df[margin_df["trade_date"] == "20260108"]
if len(ldm) > 0:
    rzye = ldm.iloc[0]["rzye"]
    rzche = ldm.iloc[0]["rzche"]
    if rzye > 0:
        print(f"启动日rzche比: {rzche/rzye*100:.2f}%")
