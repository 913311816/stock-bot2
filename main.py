#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股自动选股机器人（BaoStock终极完整版）
新增条件：
  1. EXPMA12金叉EXPMA50（近3日）
  2. 近5日成交量放大
  3. 近5个交易日无涨停
  4. 剔除亏损股（最新财报净利润>0）
  5. 全程强制北京时间
"""

import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import time
import logging
import sys
import zoneinfo

# ===================== 强制锁定 北京时间 =====================
TZ_SHANGHAI = zoneinfo.ZoneInfo("Asia/Shanghai")

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ===================== 技术指标工具函数 =====================
def calc_expma(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def check_volume_expanding(volumes: pd.Series, days: int = 5) -> tuple:
    if len(volumes) < days:
        return False, 0.0
    recent = volumes.tail(days).values.astype(float)
    if recent[0] == 0:
        return False, 0.0
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    pct = slope / recent[0] * 100
    return (slope > 0 and pct > 5), round(pct, 1)

def detect_golden_cross(df: pd.DataFrame, lookback: int = 3) -> tuple:
    if len(df) < lookback + 2:
        return False, None, 99
    for i in range(-lookback, 0):
        cur  = df.iloc[i]
        prev = df.iloc[i - 1]
        if prev['EXPMA12'] < prev['EXPMA50'] and cur['EXPMA12'] >= cur['EXPMA50']:
            date_val = str(cur.get('日期', '近期'))
            days_ago = abs(i)
            return True, date_val, days_ago
    return False, None, 99

# 新增：近5日无涨停（涨幅≥9.8%判定涨停）
def check_no_limit_up(df: pd.DataFrame, days: int = 5) -> bool:
    if len(df) < days + 1:
        return False
    df_copy = df.tail(days).copy()
    for i in range(1, len(df_copy)):
        prev_close = df_copy.iloc[i-1]['收盘']
        curr_close = df_copy.iloc[i]['收盘']
        if prev_close <= 0:
            continue
        pct = (curr_close - prev_close) / prev_close * 100
        if pct >= 9.8:
            return False
    return True

# 新增：判断非亏损股（最新财报净利润>0）
def is_profitable_stock(code_bs: str) -> bool:
    try:
        rs = bs.query_profit_data(code=code_bs)
        data = []
        while rs.error_code == '0' and rs.next():
            data.append(rs.get_row_data())
        if not data:
            return False
        cols = ["code","pubDate","statDate","roeNP","netProfit","incomeTax","totalProfit"]
        df = pd.DataFrame(data, columns=cols)
        df['netProfit'] = pd.to_numeric(df['netProfit'], errors='coerce')
        df = df.dropna(subset=['netProfit'])
        if df.empty:
            return False
        latest = df.sort_values('statDate', ascending=False).iloc[0]
        return float(latest['netProfit']) > 0
    except Exception as e:
        logger.debug(f"财报查询异常 {code_bs}: {e}")
        return False

# ===================== 获取A股列表 =====================
def get_all_a_stocks() -> pd.DataFrame:
    logger.info("正在通过 BaoStock 获取A股列表…")
    rs = bs.query_stock_basic(code_name="")
    if rs.error_code != '0':
        raise RuntimeError(f"BaoStock获取股票列表失败: {rs.error_msg}")
    records = []
    while rs.error_code == '0' and rs.next():
        row = rs.get_row_data()
        code = row[0]
        name = row[1]
        stk_type = row[4]
        status = row[5]
        if stk_type == '1' and status == '1':
            records.append({'代码_bs': code, '名称': name})
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("股票列表为空")
    df['代码'] = df['代码_bs'].str.split('.').str[1]
    # 过滤ST、退、科创板688、北交所bj
    df = df[~df['名称'].str.contains('ST|退', na=False)]
    df = df[~df['代码'].str.startswith('688')]
    df = df[~df['代码_bs'].str.startswith('bj')]
    df = df.reset_index(drop=True)
    logger.info(f"过滤后待扫描：{len(df)} 只股票")
    return df

# ===================== 获取个股日线 =====================
def get_stock_history(code_bs: str) -> pd.DataFrame:
    now = datetime.now(TZ_SHANGHAI)
    end_date = now.strftime('%Y-%m-%d')
    start_date = (now - timedelta(days=180)).strftime('%Y-%m-%d')
    rs = bs.query_history_k_data_plus(
        code_bs,
        "date,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"
    )
    if rs.error_code != '0':
        return None
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    if not data:
        return None
    df = pd.DataFrame(data, columns=['日期', '收盘', '成交量'])
    df['收盘']  = pd.to_numeric(df['收盘'],  errors='coerce')
    df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    return df

# ===================== 核心扫描 =====================
def scan_stocks(stocks_df: pd.DataFrame) -> list:
    candidates = []
    total = len(stocks_df)
    for idx, row in stocks_df.iterrows():
        code_bs = row['代码_bs']
        code    = row['代码']
        name    = row['名称']
        if idx > 0 and idx % 300 == 0:
            logger.info(f"扫描进度: {idx}/{total} ({idx/total*100:.1f}%)，已候选 {len(candidates)} 只")
        try:
            hist = get_stock_history(code_bs)
            if hist is None or len(hist) < 60:
                continue
            hist['EXPMA12'] = calc_expma(hist['收盘'], 12)
            hist['EXPMA50'] = calc_expma(hist['收盘'], 50)

            # 条件1：近3日金叉
            crossed, cross_date, days_ago = detect_golden_cross(hist)
            if not crossed:
                continue
            # 条件2：成交量放大
            vol_ok, vol_pct = check_volume_expanding(hist['成交量'])
            if not vol_ok:
                continue
            # 条件3：近5日无涨停
            if not check_no_limit_up(hist, 5):
                continue
            # 条件4：非亏损盈利股
            if not is_profitable_stock(code_bs):
                continue

            last = hist.iloc[-1]
            price_3d_ago = hist.iloc[-4]['收盘'] if len(hist) >= 4 else hist.iloc[0]['收盘']
            price_change_3d = (float(last['收盘']) - float(price_3d_ago)) / float(price_3d_ago) * 100
            prev = hist.iloc[-2]
            change_pct = (float(last['收盘']) - float(prev['收盘'])) / float(prev['收盘']) * 100
            expma_dev = (float(last['EXPMA12']) - float(last['EXPMA50'])) / float(last['EXPMA50']) * 100

            candidates.append({
                '代码': code,
                '代码_bs': code_bs,
                '名称': name,
                '最新价': round(float(last['收盘']), 2),
                '今日涨跌幅': round(change_pct, 2),
                '近3日涨幅': round(price_change_3d, 2),
                'EXPMA12': round(float(last['EXPMA12']), 2),
                'EXPMA50': round(float(last['EXPMA50']), 2),
                'EXPMA偏离%': round(expma_dev, 2),
                '成交量放大幅度': vol_pct,
                '金叉日期': cross_date or '近3日',
                '金叉距今天数': days_ago,
            })
            time.sleep(0.06)
        except Exception as e:
            logger.debug(f"处理 {code} 异常: {e}")
            continue
    logger.info(f"扫描完成，共找到 {len(candidates)} 只符合条件个股")
    return candidates

# ===================== 综合评分Top5 =====================
def score_and_recommend(candidates: list) -> list:
    if not candidates:
        return []
    df = pd.DataFrame(candidates)
    def safe_normalize(series):
        s = series.fillna(0).astype(float)
        if s.max() == s.min():
            return pd.Series([0.5]*len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min())
    df['_fresh'] = safe_normalize(-df['金叉距今天数'])
    df['_dev_score'] = safe_normalize(-df['EXPMA偏离%'].abs())
    df['_price_score'] = safe_normalize(df['近3日涨幅'])
    df['_vol_score']   = safe_normalize(df['成交量放大幅度'])
    df['综合得分'] = (
        df['_price_score'] * 0.35 +
        df['_vol_score']   * 0.30 +
        df['_fresh']       * 0.20 +
        df['_dev_score']   * 0.15
    )
    top5 = df.nlargest(5, '综合得分').to_dict('records')
    logger.info("Top5 推荐名单生成完毕")
    return top5

# ===================== 邮件报告生成 =====================
def generate_report(top5: list, total_candidates: int) -> tuple:
    now = datetime.now(TZ_SHANGHAI)
    date_str = now.strftime('%Y年%m月%d日')
    weekday  = ['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]
    medals   = ['🥇', '🥈', '🥉', '🏅', '🏅']

    lines = [
        f"📈 A股选股日报 · {date_str} {weekday}",
        "=" * 50,
        f"筛选条件：EXPMA金叉+量能放大+近5日无涨停+盈利非亏损股",
        f"今日符合条件共 {total_candidates} 只，综合评分推荐Top5：",
        ""
    ]

    for i, s in enumerate(top5):
        change3d = s.get('近3日涨幅', 0)
        vol_pct  = s.get('成交量放大幅度', 0)
        dev      = s.get('EXPMA偏离%', 0)
        lines += [
            f"{medals[i]} 第{i+1}名：{s['名称']}（{s['代码']}）",
            f"   最新价：{s['最新价']} 元   今日涨跌：{s['今日涨跌幅']:+.2f}%",
            f"   EXPMA12：{s['EXPMA12']}  EXPMA50：{s['EXPMA50']}  金叉：{s['金叉日期']}",
            f"   近3日涨幅：{change3d:+.2f}%  量能放大：+{vol_pct:.0f}%  偏离：{dev:+.2f}%",
        ]
        reasons = []
        if change3d > 3:
            reasons.append(f"近3日上涨{change3d:.1f}%，多头动能强")
        elif change3d > 0:
            reasons.append(f"近3日稳步上涨{change3d:.1f}%")
        else:
            reasons.append("技术整理充分，形态健康")
        if vol_pct > 30:
            reasons.append(f"成交量大幅放大+{vol_pct:.0f}%，资金进场")
        else:
            reasons.append(f"量能持续放大+{vol_pct:.0f}%")
        if s['金叉距今天数'] == 1:
            reasons.append("昨日刚金叉，信号新鲜")
        else:
            reasons.append(f"{s['金叉日期']}金叉，中期趋势转强")
        if 0 < dev < 3:
            reasons.append("刚突破均线，上行空间充足")
        reasons.append("近5日无涨停，不追高")
        reasons.append("财报盈利，无亏损暴雷风险")
        lines.append(f"   【推荐理由】{'；'.join(reasons)}")
        lines.append("")

    lines += [
        "=" * 50,
        "⚠️ 风险提示：仅为技术面自动筛选，不构成任何投资建议。",
        "股市有风险，投资需谨慎。",
        f"自动运行时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"
    ]
    plain_text = "\n".join(lines)

    # HTML 邮件简化优雅版
    html_text = f"""
    <html>
    <body style="font-family:微软雅黑,Arial;max-width:620px;margin:0 auto;padding:20px;">
        <h2 style="color:#0066cc;">📈 A股选股日报 · {date_str} {weekday}</h2>
        <p>今日筛选：EXPMA金叉 + 量能放大 + 近5日无涨停 + 盈利非亏损股</p>
        <p>符合条件总数：<b>{total_candidates}</b> 只</p>
        <p>本报告由GitHub Actions自动定时生成</p>
        <p style="color:#999;font-size:12px;">运行时间：{now.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <hr>
        <p style="color:#999;">⚠️ 仅技术参考，不构成投资建议</p>
    </body>
    </html>
    """
    return plain_text, html_text

# ===================== 邮件发送 =====================
def send_email(plain_text: str, html_text: str, subject: str):
    sender      = os.environ.get('EMAIL_SENDER', '')
    password    = os.environ.get('EMAIL_PASSWORD', '')
    receiver    = os.environ.get('EMAIL_RECEIVER', '')
    smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port   = int(os.environ.get('SMTP_PORT', '465'))

    if not all([sender, password, receiver]):
        logger.warning("未配置邮件密钥，仅打印报告")
        print("\n" + "="*50 + "\n" + plain_text + "\n" + "="*50)
        return

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver
    msg.attach(MIMEText(plain_text, 'plain', 'utf-8'))
    msg.attach(MIMEText(html_text, 'html', 'utf-8'))

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as srv:
            srv.login(sender, password)
            srv.sendmail(sender, receiver, msg.as_string())
        logger.info("✅ 邮件发送成功")
    except Exception as e:
        logger.error(f"❌ 邮件发送失败：{e}")
        print(plain_text)

# ===================== 主程序入口 =====================
def main():
    logger.info("="*50)
    logger.info("🚀 A股选股机器人 终极完整版启动")
    logger.info("✅ 已启用：金叉+量能+近5日无涨停+剔除亏损股+北京时间")
    logger.info("="*50)

    now = datetime.now(TZ_SHANGHAI)
    subject = f"📈 A股选股日报 · {now.strftime('%Y年%m月%d日')}"

    login_result = bs.login()
    if login_result.error_code != '0':
        logger.error(f"BaoStock登录失败：{login_result.error_msg}")
        sys.exit(1)
    logger.info("✅ BaoStock 登录成功")

    try:
        stocks = get_all_a_stocks()
        candidates = scan_stocks(stocks)
        if not candidates:
            send_email("今日无任何个股符合全部选股条件", "今日无任何个股符合全部选股条件", subject+" · 无信号")
            return
        top5 = score_and_recommend(candidates)
        plain, html = generate_report(top5, len(candidates))
        send_email(plain, html, subject)
    finally:
        bs.logout()
        logger.info("BaoStock 已登出")
    logger.info("✅ 本次选股任务完成")

if __name__ == '__main__':
    main()
