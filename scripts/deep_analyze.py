#!/home/yrflj/.hermes/hermes-agent/venv/bin/python3
"""
深度分析脚本 - 史诗级行情黑马识别系统 v1.5
==========================================
用法: 
  # 标准两步走（推荐）
  python3 scan_rally_signal.py 20260109 -t 7 > /tmp/scan_20260109.log 2>&1
  python3 deep_analyze.py 20260109 --log /tmp/scan_20260109.log

功能:
  1. 解析扫描日志中的Y型候选、象限Baux、D/C类高分候选
  2. 查Tushare确认股票名称（必须步骤，防止name列错误）
  3. 核查Y型9条件 + 月线MA5过滤 + 硬排除规则
  4. 核查象限Baux黑马月线MA5状态
  5. 输出推荐股票池

版本历史:
  v1.6 - 20260507 - 同步v6.30参数调整：前5日涨跌<=12%、前15日涨跌<=20%、量比<=2.5
  v1.4 - 20260509 - 修复Y型解析bug（分隔线误判、.SZ截断）；修复Baux解析误捕符号
"""

import sys
import re
import tushare as ts

TUSHARE_TOKEN = "fe284a8656a25bf46f4b9092178d1a4dd4d63f26bdb8b83d7f461c47"
LAUNCH_DATE = None
LOG_FILE = None

for arg in sys.argv[1:]:
    if arg.startswith("20"):
        LAUNCH_DATE = arg
    elif arg == "--reuse":
        pass  # backward compat
    elif arg.startswith("--log="):
        LOG_FILE = arg[6:]
    elif arg.startswith("--"):
        pass  # ignore unknown flags

if not LAUNCH_DATE:
    print("用法: python3 deep_analyze.py <扫描日期> [--log=/path/to/scan.log]")
    sys.exit(1)

SCRIPT_DIR = "/home/yrflj/.hermes/skills/investment-research/epic-rally-identification/scripts"
DEFAULT_LOG = f"/tmp/scan_{LAUNCH_DATE}.log"

if LOG_FILE is None:
    LOG_FILE = DEFAULT_LOG

pro = ts.pro_api(TUSHARE_TOKEN)

# ============================================================
# 工具函数
# ============================================================

def get_name(code):
    try:
        return pro.stock_basic(ts_code=code, fields='name').iloc[0]['name']
    except:
        return "未找到"


def get_monthly_ma5(code):
    monthly = pro.monthly(ts_code=code, start_date="202504", end_date="202605")
    monthly = monthly.sort_values("trade_date").reset_index(drop=True)
    monthly['ma5'] = monthly['close'].rolling(5).mean()
    last3 = monthly.tail(3)
    if len(last3) >= 2:
        cur = last3.iloc[-1]
        prv = last3.iloc[-2]
        return {
            "ma5_up": cur['ma5'] > prv['ma5'],
            "above_ma5": cur['close'] > cur['ma5'],
            "ma5": cur['ma5'],
            "close": cur['close'],
        }
    return None


def get_y_type_conditions(code, launch_date):
    price_df = pro.daily(ts_code=code, start_date="2025-10-01", end_date=launch_date)
    price_df = price_df.sort_values("trade_date").reset_index(drop=True)

    pre10 = price_df[price_df["trade_date"] < launch_date].tail(10)
    vol_ma10 = pre10["vol"].mean()
    launch_row = price_df[price_df["trade_date"] == launch_date]
    tday_vol = launch_row.iloc[0]["vol"] if len(launch_row) > 0 else 0
    vol_ratio = tday_vol / vol_ma10 if vol_ma10 > 0 else 0

    pre15 = price_df[price_df["trade_date"] < launch_date].tail(15)
    pre15_chg = (pre15.iloc[-1]["close"] / pre15.iloc[0]["close"] - 1) * 100 if len(pre15) >= 2 else None

    pre5p = price_df[price_df["trade_date"] < launch_date].tail(5)
    pre5_chg = (pre5p.iloc[-1]["close"] / pre5p.iloc[0]["close"] - 1) * 100 if len(pre5p) >= 2 else None

    pre15_pct = pre15["pct_chg"].dropna()
    rise7_days = int((pre15_pct.abs() > 7).sum())
    body_big_abs_days = int((pre15_pct.abs() >= 5).sum())  # 大实体：单日涨跌幅>=5%
    body_small_abs_days = int((pre15_pct.abs() < 2).sum())  # 小实体：单日涨跌幅<2%

    margin_df = pro.margin_detail(ts_code=code, start_date="2025-10-01", end_date=launch_date)
    margin_df = margin_df.sort_values("trade_date").reset_index(drop=True)

    pre5_margin = margin_df[margin_df["trade_date"] < launch_date].tail(5)
    valid_pre5 = pre5_margin[pre5_margin["rzye"] > 0]
    rzche_ratio_mean = (valid_pre5["rzche"] / valid_pre5["rzye"] * 100).mean() if len(valid_pre5) >= 3 else None

    ld_m = margin_df[margin_df["trade_date"] == launch_date]
    rzche_ratio_t = None
    if len(ld_m) > 0:
        rzye_t = ld_m.iloc[0]["rzye"]
        rzche_t = ld_m.iloc[0]["rzche"]
        if rzye_t > 0:
            rzche_ratio_t = rzche_t / rzye_t * 100

    monthly_info = get_monthly_ma5(code)

    return {
        "vol_ratio": vol_ratio,
        "vol_ma10": vol_ma10,
        "tday_vol": tday_vol,
        "pre15_chg": pre15_chg,
        "pre5_chg": pre5_chg,
        "rise7_days": rise7_days,
        "body_big_abs_days": body_big_abs_days,
        "body_small_abs_days": body_small_abs_days,
        "rzche_ratio_mean": rzche_ratio_mean,
        "rzche_ratio_t": rzche_ratio_t,
        "monthly": monthly_info,
    }


def check_y_type_9conditions(data):
    """v6.29 Y型9条件检查（融资条款已降级为参考，此处仅检查量价条件）"""
    c1 = data["vol_ratio"] >= 2.0
    c2 = data["vol_ratio"] <= 2.5  # v6.30: <=2.5
    c3 = data["pre5_chg"] is not None and data["pre5_chg"] <= 12.0  # v6.30: <=12%
    c4 = data["pre15_chg"] is not None and -3.0 < data["pre15_chg"] <= 20.0  # v6.30: <=20%
    c5 = data["rise7_days"] < 2
    c6 = data.get("body_big_abs_days", 999) < 3  # v6.29: 大实体<3天
    c7 = data.get("body_small_abs_days", 0) >= 6  # v6.29: 小实体>=6天（原>=8天降至>=6）
    c8 = data["monthly"] is not None and data["monthly"]["ma5_up"] and data["monthly"]["above_ma5"]

    # v6.29新增硬排除（条件满足时直接排除，不进入all_pass）
    hard_exclude = False
    hard_exclude_reason = ""
    # 硬排除1：大实体0天 + 量比>=5.0
    if data.get("body_big_abs_days", 999) == 0 and data["vol_ratio"] >= 5.0:
        hard_exclude = True
        hard_exclude_reason = "大实体0天+量比>=5.0"
    # 硬排除2：小实体8~9天 + 前5日涨跌0~5%
    if data.get("body_small_abs_days", 0) in (8, 9) and data["pre5_chg"] is not None and 0.0 <= data["pre5_chg"] <= 5.0:
        hard_exclude = True
        hard_exclude_reason = "小实体8~9天+前5日0~5%"

    return {
        "c1_min": c1, "c2_max": c2,
        "c3_pre5chg": c3, "c4_pre15chg": c4,
        "c5_rise7": c5, "c6_big_body": c6,
        "c7_small_body": c7, "c8_ma5": c8,
        "all_pass": all([c1, c2, c3, c4, c5, c6, c7, c8]) and not hard_exclude,
        "hard_exclude": hard_exclude,
        "hard_exclude_reason": hard_exclude_reason,
    }


def parse_scan_output(scan_output):
    y_codes, baux_codes, all_score2 = [], [], []
    in_y_table = False  # 是否在Y型候选表格体内
    past_sep = False  # 是否已越过Y型表格后的分隔线

    for line in scan_output.split("\n"):
        # 检测Y型候选表头：进入Y型表格
        if not in_y_table and ("量比(10日)" in line or "前5日rzche比" in line):
            in_y_table = True
            continue

        if in_y_table:
            stripped = line.strip()
            # Y型表格结束：遇到T日买入信号详情标记
            if "T日买入信号详情" in line:
                in_y_table = False
                past_sep = True
                continue  # 不break，继续解析后面的Baux/总分表
            # 跳过Y型表头后的分隔线（===== 或 -----）
            if re.match(r'^[-=]{10,}$', stripped):
                continue  # 跳过，继续解析
            if stripped == "":
                continue  # 跳过空行
            m = re.match(r"\s*(\d+)\.\s+(\d{6}\.(?:SZ|SH))", line)
            if m and not m.group(2).startswith("92"):
                y_codes.append(m.group(2))
        # 象限Baux：匹配 "600143.SH ... Baux" 格式（股票代码后面跟Baux标记）
        m_baux = re.search(r'(\d{6}\.(?:SZ|SH)).*?Baux', line)
        if m_baux and not m_baux.group(1).startswith("92"):
            baux_codes.append(m_baux.group(1))
        # 总分>=2
        parts = line.split()
        if len(parts) >= 3:
            code = parts[0]
            if re.match(r"\d{6}\.[SZ]H?", code) and not code.startswith("92"):
                try:
                    score = int(parts[2])
                    if score >= 2:
                        all_score2.append(code)
                except:
                    pass
    
    y_codes = list(dict.fromkeys(y_codes))
    baux_codes = list(dict.fromkeys(baux_codes))
    all_score2 = list(dict.fromkeys(all_score2))
    d_codes = [c for c in all_score2 if c not in y_codes and c not in baux_codes]
    return y_codes, baux_codes, d_codes


# ============================================================
# 主逻辑
# ============================================================

def main():
    print(f"深度分析 {LAUNCH_DATE} - v1.3")
    print("=" * 60)

    # 读取扫描日志
    try:
        with open(LOG_FILE, "r") as f:
            scan_output = f.read()
        print(f"读取扫描日志: {LOG_FILE}")
    except FileNotFoundError:
        print(f"扫描日志不存在: {LOG_FILE}")
        print(f"请先运行: python3 scan_rally_signal.py {LAUNCH_DATE} -t 7 > {LOG_FILE} 2>&1")
        sys.exit(1)

    # 解析候选
    y_codes, baux_codes, d_codes = parse_scan_output(scan_output)
    print(f"\nY型候选: {len(y_codes)}只 - {y_codes}")
    print(f"象限Baux: {len(baux_codes)}只 - {baux_codes}")
    print(f"象限D/C(高分): {len(d_codes)}只")

    # 查名称
    print("\n【第一步：股票名称确认】")
    y_names = {c: get_name(c) for c in y_codes}
    baux_names = {c: get_name(c) for c in baux_codes}
    d_names = {c: get_name(c) for c in d_codes}

    print("\nY型候选:")
    for c in y_codes:
        print(f"  {c}: {y_names.get(c, '未知')}")
    print("\n象限Baux:")
    for c in baux_codes:
        print(f"  {c}: {baux_names.get(c, '未知')}")
    print("\n象限D/C(高分):")
    for c in d_codes:
        print(f"  {c}: {d_names.get(c, '未知')}")

    # Y型8条件
    print("\n【第二步：Y型8条件核验】")
    y_pass, y_fail = [], []

    for code in y_codes:
        name = y_names.get(code, "未知")
        data = get_y_type_conditions(code, LAUNCH_DATE)
        checks = check_y_type_9conditions(data)
        m = data['monthly']

        print(f"\n{code} {name}")
        print(f"  ①量比={data['vol_ratio']:.2f} {'✓' if checks['c1_min'] else '✗'} {'✓' if checks['c2_max'] else '✗'} (范围[2~2.5])")
        print(f"  ②前5涨跌={data['pre5_chg']:.2f}% {'✓' if checks['c3_pre5chg'] else '✗'} (<=12%)")
        print(f"  ③15日涨跌={data['pre15_chg']:.2f}% {'✓' if checks['c4_pre15chg'] else '✗'} (-3%~20%)")
        print(f"  ④15日>7%={data['rise7_days']}天 {'✓' if checks['c5_rise7'] else '✗'} (<2)")
        print(f"  ⑤大实体={data.get('body_big_abs_days', 'N/A')}天 {'✓' if checks['c6_big_body'] else '✗'} (<3)")
        print(f"  ⑥小实体={data.get('body_small_abs_days', 'N/A')}天 {'✓' if checks['c7_small_body'] else '✗'} (>=6)")
        if m:
            print(f"  ⑦月线MA5={'上升✓' if m['ma5_up'] else '下降✗'} | ⑧收盘{m['close']} {'✓站上' if m['above_ma5'] else '✗低于'}MA5{m['ma5']:.2f}")
        else:
            print(f"  ⑦月线=无法获取")

        if checks['hard_exclude']:
            print(f"  → ❌ 硬排除：{checks['hard_exclude_reason']}")
            y_fail.append(code)
        elif checks['all_pass']:
            print(f"  → ✅ 全部通过")
            y_pass.append(code)
        else:
            fails = []
            if not checks['c1_min']: fails.append("①<2.0")
            if not checks['c2_max']: fails.append("②>3.0")
            if not checks['c3_pre5chg']: fails.append("③前5涨跌>8%")
            if not checks['c4_pre15chg']: fails.append("④15日超出-3%~15%")
            if not checks['c5_rise7']: fails.append("⑤>7%天数≥2")
            if not checks['c6_big_body']: fails.append("⑥大实体≥3天")
            if not checks['c7_small_body']: fails.append("⑦小实体<6天")
            if not checks['c8_ma5']: fails.append("⑧月线")
            print(f"  → ❌ {', '.join(fails)}")
            y_fail.append(code)

    # 月线检查
    print("\n【第三步：黑马月线MA5检查】")
    monthly_status = {}
    for code_list, names, label in [
        (baux_codes, baux_names, "象限Baux"),
        (d_codes, d_names, "象限D/C")
    ]:
        print(f"\n{label}:")
        for code in code_list:
            name = names.get(code, "未知")
            m = get_monthly_ma5(code)
            monthly_status[code] = m
            if m:
                print(f"  {code} {name}: MA5={'上升✓' if m['ma5_up'] else '下降✗'} | 收盘{m['close']} {'✓站上' if m['above_ma5'] else '✗低于'}MA5{m['ma5']:.2f}")
            else:
                print(f"  {code} {name}: 月线数据不足")

    # 推荐股票池
    print("\n" + "=" * 60)
    print("【推荐股票池】")
    print("=" * 60)

    if y_pass:
        print(f"\n✅ Y型通过({len(y_pass)}只):")
        for code in y_pass:
            name = y_names.get(code, "未知")
            data = get_y_type_conditions(code, LAUNCH_DATE)
            m = data['monthly']
            print(f"\n  ★ {code} {name}")
            print(f"    量比={data['vol_ratio']:.2f} | 前5涨跌={data['pre5_chg']:.2f}% | 15日涨跌={data['pre15_chg']:.2f}%")
            print(f"    大实体={data.get('body_big_abs_days', 'N/A')}天 | 小实体={data.get('body_small_abs_days', 'N/A')}天 | 15日>7%={data['rise7_days']}天")
            if m:
                print(f"    月线: {'上升✓' if m['ma5_up'] else '下降✗'} | {'✓站上' if m['above_ma5'] else '✗低于'}MA5({m['ma5']:.2f})")
    else:
        print("\n  Y型候选: 0只通过")

    baux_ok = [c for c in baux_codes
               if monthly_status.get(c)
               and monthly_status[c]['ma5_up']
               and monthly_status[c]['above_ma5']]
    if baux_ok:
        print(f"\n黑马观察({len(baux_ok)}只 - 象限Baux+月线满足):")
        for code in baux_ok:
            print(f"  {code} {baux_names.get(code, '未知')} ← 象限Baux+月线MA5上升+站上MA5")
    else:
        print("\n  黑马观察: 0只")

    print(f"\n⚠️ 仅供参考，不构成投资建议")
    print(f"扫描日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
