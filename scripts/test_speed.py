#!/home/yrflj/.hermes/hermes-agent/venv/bin/python3
import tushare as ts
pro = ts.pro_api('fe284a8656a25bf46f4b9092178d1a4dd4d63f26bdb8b83d7f461c47')
import time

t0 = time.time()
m = pro.monthly(ts_code='600143.SH', start_date='202504', end_date='202605')
print(f'monthly query took {time.time()-t0:.1f}s, {len(m)} rows')

t0 = time.time()
n = pro.stock_basic(ts_code='600143.SH', fields='name').iloc[0]['name']
print(f'stock_basic took {time.time()-t0:.1f}s: {n}')
