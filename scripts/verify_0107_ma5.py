#!/home/yrflj/.hermes/hermes-agent/venv/bin/python3
import tushare as ts
pro = ts.pro_api("fe284a8656a25bf46f4b9092178d1a4dd4d63f26bdb8b83d7f461c47")

launch = "20260107"
results = {}

for code in ["600475.SH", "688480.SH", "002112.SZ", "600637.SH"]:
    monthly = pro.monthly(ts_code=code, start_date="202504", end_date="202605")
    monthly = monthly.sort_values("trade_date").reset_index(drop=True)
    monthly['ma5'] = monthly['close'].rolling(5).mean()
    
    # 最近3个月
    last3 = monthly.tail(3)
    print(f"\n{code} 月线MA5:")
    for _, r in last3.iterrows():
        print(f"  {r['trade_date']}: 收盘={r['close']}, MA5={r['ma5']:.2f}")
    
    if len(last3) >= 2:
        cur = last3.iloc[-1]
        prv = last3.iloc[-2]
        ma5_up = cur['ma5'] > prv['ma5']
        above = cur['close'] > cur['ma5']
        results[code] = {"ma5_up": ma5_up, "above_ma5": above, "ma5": cur['ma5'], "close": cur['close']}
        print(f"  → MA5方向: {'上升✓' if ma5_up else '下降✗'} | 收盘{cur['close']} {'✓站上MA5' if above else '✗低于MA5'}")

print("\n\n=== Y型8条件全部满足情况 ===")
for code, v in results.items():
    print(f"{code}: MA5上升={v['ma5_up']}, 站上MA5={v['above_ma5']}")

# 检查黑马月线
print("\n\n=== 黑马月线检查(总分>=2) ===")
black_candidates = ["003031.SZ", "688205.SZ", "000657.SZ", "600330.SH", "002324.SZ"]
for code in black_candidates:
    monthly = pro.monthly(ts_code=code, start_date="202504", end_date="202605")
    monthly = monthly.sort_values("trade_date").reset_index(drop=True)
    monthly['ma5'] = monthly['close'].rolling(5).mean()
    last3 = monthly.tail(3)
    if len(last3) >= 2:
        cur = last3.iloc[-1]
        prv = last3.iloc[-2]
        ma5_up = cur['ma5'] > prv['ma5']
        above = cur['close'] > cur['ma5']
        print(f"{code}: MA5方向={'上升✓' if ma5_up else '下降✗'}, 收盘{cur['close']} {'✓站上MA5' if above else '✗低于MA5'}")
