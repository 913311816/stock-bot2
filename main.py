#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股自动选股机器人（BaoStock版）
================================
数据源：BaoStock（海外GitHub服务器可正常访问）

筛选逻辑：
  1. EXPMA12日线上穿EXPMA50日线（近3日内发生金叉）
  2. 成交量近5日呈放大趋势

推荐逻辑：
  综合以下4项指标打分，推荐Top5：
  - 近3日价格涨幅（资金流入代理指标）
  - 成交量放大强度
  - 金叉新鲜度（越近越好）
  - EXPMA偏离程度（刚突破最佳）
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

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================
# 技术指标计算
# ============================================================

def calc_expma(series: pd.Series, n: int) -> pd.Series:
    """计算EXPMA（指数移动平均线）"""
    return series.ewm(span=n, adjust=False).mean()


def check_volume_expanding(volumes: pd.Series, days: int = 5) -> tuple:
    """
    检查近N日成交量是否呈放大趋势
    返回：(是否放大: bool, 增幅百分比: float)
    """
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
    """
    检测近N天内是否出现EXPMA12上穿EXPMA50的金叉
    返回：(是否金叉: bool, 金叉日期: str, 距今天数: int)
    """
    if len(df) < lookback + 2:
        return False, None, 99
    for i in range(-lookback, 0):
        cur  = df.iloc[i]
        prev = df.iloc[i - 1]
        if prev['EXPMA12'] < prev['EXPMA50'] and cur['EXPMA12'] >= cur['EXPMA50']:
            date_val = str(cur.get('日期', '近期'))
            days_ago = abs(i)  # 1=昨天，2=前天，3=大前天
            return True, date_val, days_ago
    return False, None, 99


# ============================================================
# BaoStock 数据获取
# ============================================================

def get_all_a_stocks() -> pd.DataFrame:
    """获取所有A股列表（通过BaoStock），过滤ST、退市、科创板、北交所"""
    logger.info("正在通过 BaoStock 获取A股列表…")
    
    rs = bs.query_stock_basic(code_name="")
    if rs.error_code != '0':
        raise RuntimeError(f"BaoStock获取股票列表失败: {rs.error_msg}")
    
    records = []
    while (rs.error_code == '0') and rs.next():
        row = rs.get_row_data()
        # row: [code, code_name, ipoDate, outDate, type, status]
        code      = row[0]  # 格式: sh.600000 / sz.000001
        name      = row[1]
        stk_type  = row[4]  # '1'=股票
        status    = row[5]  # '1'=上市
        if stk_type == '1' and status == '1':
            records.append({'代码_bs': code, '名称': name})
    
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("BaoStock返回股票列表为空，请检查登录状态")
    
    # 提取纯数字代码，方便过滤
    df['代码'] = df['代码_bs'].str.split('.').str[1]
    
    # 过滤ST、退市
    df = df[~df['名称'].str.contains('ST|退', na=False)]
    # 过滤科创板 (688)
    df = df[~df['代码'].str.startswith('688')]
    # 过滤北交所 (BJ前缀)
    df = df[~df['代码_bs'].str.startswith('bj')]
    
    df = df.reset_index(drop=True)
    logger.info(f"过滤后待扫描：{len(df)} 只股票")
    return df


def get_stock_history(code_bs: str) -> pd.DataFrame:
    """通过BaoStock获取个股日线历史数据（前复权）"""
    end_date   = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    rs = bs.query_history_k_data_plus(
        code_bs,
        "date,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"   # 2=前复权
    )
    
    if rs.error_code != '0':
        return None
    
    data = []
    while (rs.error_code == '0') and rs.next():
        data.append(rs.get_row_data())
    
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=['日期', '收盘', '成交量'])
    df['收盘']  = pd.to_numeric(df['收盘'],  errors='coerce')
    df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    return df


# ============================================================
# 核心扫描逻辑
# ============================================================

def scan_stocks(stocks_df: pd.DataFrame) -> list:
    """
    遍历所有股票，找出同时满足以下条件的候选：
      - 近3日内EXPMA12上穿EXPMA50（金叉）
      - 近5日成交量放大
    """
    candidates = []
    total = len(stocks_df)

    for idx, row in stocks_df.iterrows():
        code_bs = row['代码_bs']
        code    = row['代码']
        name    = row['名称']

        if idx > 0 and idx % 300 == 0:
            logger.info(f"扫描进度: {idx}/{total} ({idx/total*100:.1f}%)，"
                        f"已找到候选 {len(candidates)} 只")

        try:
            hist = get_stock_history(code_bs)
            if hist is None or len(hist) < 60:
                continue

            hist['EXPMA12'] = calc_expma(hist['收盘'], 12)
            hist['EXPMA50'] = calc_expma(hist['收盘'], 50)

            # 条件1：EXPMA金叉（近3日内）
            crossed, cross_date, days_ago = detect_golden_cross(hist)
            if not crossed:
                continue

            # 条件2：成交量放大
            vol_ok, vol_pct = check_volume_expanding(hist['成交量'])
            if not vol_ok:
                continue

            last = hist.iloc[-1]
            # 近3日价格涨幅
            price_3d_ago = hist.iloc[-4]['收盘'] if len(hist) >= 4 else hist.iloc[0]['收盘']
            price_change_3d = (float(last['收盘']) - float(price_3d_ago)) / float(price_3d_ago) * 100

            # 今日涨跌幅
            prev = hist.iloc[-2]
            change_pct = (float(last['收盘']) - float(prev['收盘'])) / float(prev['收盘']) * 100

            # EXPMA偏离度（12线高于50线的百分比）
            expma_dev = (float(last['EXPMA12']) - float(last['EXPMA50'])) / float(last['EXPMA50']) * 100

            candidates.append({
                '代码':         code,
                '代码_bs':      code_bs,
                '名称':         name,
                '最新价':       round(float(last['收盘']), 2),
                '今日涨跌幅':   round(change_pct, 2),
                '近3日涨幅':    round(price_change_3d, 2),
                'EXPMA12':      round(float(last['EXPMA12']), 2),
                'EXPMA50':      round(float(last['EXPMA50']), 2),
                'EXPMA偏离%':   round(expma_dev, 2),
                '成交量放大幅度': vol_pct,
                '金叉日期':     cross_date or '近3日',
                '金叉距今天数': days_ago,
                '市场':         'sh' if code.startswith('6') else 'sz',
            })

            time.sleep(0.05)  # BaoStock较快，稍作延迟即可

        except Exception as e:
            logger.debug(f"处理 {code} 出错: {e}")
            continue

    logger.info(f"扫描完成，共找到 {len(candidates)} 只候选")
    return candidates


# ============================================================
# 评分与推荐
# ============================================================

def score_and_recommend(candidates: list) -> list:
    """
    综合评分（无需资金流接口，用价格+量能指标代替）：
      近3日价格涨幅     35%  （资金流入代理）
      成交量放大强度    30%  （买盘积极度）
      金叉新鲜度        20%  （越新越强）
      EXPMA偏离适中     15%  （刚突破最理想）
    """
    if not candidates:
        return []

    df = pd.DataFrame(candidates)

    def safe_normalize(series):
        s = series.fillna(0).astype(float)
        if s.max() == s.min():
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    # 金叉新鲜度：天数越少越好，取反归一化
    df['_fresh'] = safe_normalize(-df['金叉距今天数'])

    # EXPMA偏离：偏高惩罚（超过5%扣分），用负的绝对偏差的归一化
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


# ============================================================
# 报告生成（纯文本 + HTML）
# ============================================================

def generate_report(top5: list, total_candidates: int) -> tuple:
    now      = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    weekday  = ['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]
    medals   = ['🥇', '🥈', '🥉', '🏅', '🏅']

    # -------- 纯文本 --------
    lines = [
        f"📈 A股选股日报 · {date_str} {weekday}",
        "=" * 46,
        f"今日扫描主板+创业板，符合【EXPMA12金叉EXPMA50 + 量能放大】共 {total_candidates} 只",
        "综合近3日价格动能和量能强度，重点推荐以下5只：",
        ""
    ]

    for i, s in enumerate(top5):
        change3d = s.get('近3日涨幅', 0)
        vol_pct  = s.get('成交量放大幅度', 0)
        dev      = s.get('EXPMA偏离%', 0)

        lines += [
            f"{medals[i]} 第{i+1}名：{s['名称']}（{s['代码']}）",
            f"   当前价格：{s['最新价']} 元    今日涨跌：{s.get('今日涨跌幅', 0):+.2f}%",
            f"   EXPMA12: {s['EXPMA12']}  |  EXPMA50: {s['EXPMA50']}  |  金叉日期: {s['金叉日期']}",
            f"   近3日涨幅：{change3d:+.2f}%    成交量放大：+{vol_pct:.0f}%    EXPMA偏离：{dev:+.2f}%",
        ]

        reasons = []
        if change3d > 3:
            reasons.append(f"近3日累计上涨{change3d:.1f}%，价格动能强劲，主力资金在持续推升")
        elif change3d > 0:
            reasons.append(f"近3日小幅上涨{change3d:.1f}%，稳健上行")
        else:
            reasons.append(f"价格整理中，技术形态良好")

        if vol_pct > 30:
            reasons.append(f"近5日成交量大幅放大（+{vol_pct:.0f}%），买盘涌入积极")
        else:
            reasons.append(f"成交量稳步放大（+{vol_pct:.0f}%），资金有序进场")

        if s['金叉距今天数'] == 1:
            reasons.append(f"昨日刚刚发生EXPMA12上穿EXPMA50金叉，信号最为新鲜")
        else:
            reasons.append(f"{s['金叉日期']}出现EXPMA金叉，中期趋势确认转强")

        if 0 < dev < 3:
            reasons.append(f"EXPMA12仅偏离EXPMA50约{dev:.1f}%，刚刚突破位置，上涨空间充足")

        lines.append(f"   【推荐理由】{'；'.join(reasons)}")
        lines.append("")

    lines += [
        "=" * 46,
        "⚠️  风险提示：以上结果为纯技术面自动筛选，不构成投资建议。",
        "   股市有风险，投资需谨慎，请结合基本面和个人判断操作。",
        f"   本邮件由 GitHub Actions 自动发送 · {now.strftime('%H:%M')}"
    ]
    plain_text = "\n".join(lines)

    # -------- HTML --------
    colors  = ['#e8f5e9','#e3f2fd','#fff3e0','#f3e5f5','#fce4ec']
    borders = ['#43a047','#1e88e5','#fb8c00','#8e24aa','#e53935']

    cards_html = ""
    for i, s in enumerate(top5):
        change3d = s.get('近3日涨幅', 0)
        vol_pct  = s.get('成交量放大幅度', 0)
        dev      = s.get('EXPMA偏离%', 0)

        reasons = []
        if change3d > 3:
            reasons.append(f"近3日累计上涨<b>{change3d:.1f}%</b>，价格动能强劲")
        elif change3d > 0:
            reasons.append(f"近3日稳健上涨<b>{change3d:.1f}%</b>")
        else:
            reasons.append("技术形态良好，等待突破")

        if vol_pct > 30:
            reasons.append(f"成交量<b>大幅放大+{vol_pct:.0f}%</b>，买盘涌入")
        else:
            reasons.append(f"成交量稳步放大<b>+{vol_pct:.0f}%</b>")

        if s['金叉距今天数'] == 1:
            reasons.append("<b>昨日刚发生EXPMA金叉</b>，信号最新鲜")
        else:
            reasons.append(f"{s['金叉日期']}<b>EXPMA金叉确认</b>，中期趋势转强")

        if 0 < dev < 3:
            reasons.append(f"EXPMA12偏离仅{dev:.1f}%，<b>刚突破位置，上涨空间充足</b>")

        today_color = '#c62828' if s.get('今日涨跌幅', 0) < 0 else '#2e7d32'
        c3d_color   = '#c62828' if change3d < 0 else '#2e7d32'

        cards_html += f"""
        <div style="background:{colors[i]};border-left:5px solid {borders[i]};
                    padding:16px 20px;margin:14px 0;border-radius:6px;">
          <h3 style="margin:0 0 10px;color:#222;font-size:1.05em;">
            {medals[i]}&nbsp;{s['名称']}&nbsp;
            <span style="color:#666;font-weight:normal;font-size:0.85em;">({s['代码']})</span>
          </h3>
          <table style="font-size:0.88em;color:#444;border-collapse:collapse;width:100%;">
            <tr>
              <td style="padding:3px 10px 3px 0;">💰 当前价格</td>
              <td style="padding:3px 16px 3px 0;"><b>{s['最新价']} 元</b></td>
              <td style="padding:3px 10px 3px 0;">📊 今日涨跌</td>
              <td style="padding:3px 0;color:{today_color}">
                <b>{s.get('今日涨跌幅',0):+.2f}%</b></td>
            </tr>
            <tr>
              <td style="padding:3px 10px 3px 0;">📈 EXPMA12</td>
              <td style="padding:3px 16px 3px 0;">{s['EXPMA12']}</td>
              <td style="padding:3px 10px 3px 0;">📉 EXPMA50</td>
              <td style="padding:3px 0;">{s['EXPMA50']}</td>
            </tr>
            <tr>
              <td style="padding:3px 10px 3px 0;">🔔 金叉日期</td>
              <td style="padding:3px 16px 3px 0;">{s['金叉日期']}</td>
              <td style="padding:3px 10px 3px 0;">📦 量能放大</td>
              <td style="padding:3px 0;color:#2e7d32;"><b>+{vol_pct:.0f}%</b></td>
            </tr>
            <tr>
              <td style="padding:3px 10px 3px 0;" colspan="2">📅 近3日涨幅</td>
              <td style="padding:3px 0;color:{c3d_color};" colspan="2">
                <b>{change3d:+.2f}%</b></td>
            </tr>
          </table>
          <div style="margin-top:10px;padding:8px 10px;background:rgba(255,255,255,0.65);
                      border-radius:4px;font-size:0.85em;color:#333;line-height:1.7;">
            💡 <b>推荐理由：</b>{'；'.join(reasons)}
          </div>
        </div>"""

    html_text = f"""
    <html><body style="font-family:'PingFang SC',Arial,sans-serif;max-width:620px;
                        margin:0 auto;padding:24px 20px;color:#222;">
      <h2 style="color:#1565c0;border-bottom:3px solid #1565c0;padding-bottom:10px;margin-bottom:4px;">
        📈 A股选股日报 · {date_str} {weekday}
      </h2>
      <p style="color:#555;font-size:0.9em;margin-top:4px;">
        今日扫描主板+创业板，符合条件共 <b>{total_candidates}</b> 只 ·
        综合近3日价格动能和量能强度，推荐Top5
      </p>
      {cards_html}
      <div style="background:#fff8e1;padding:12px 16px;border-radius:6px;
                  font-size:0.82em;color:#6d4c41;margin-top:18px;line-height:1.7;">
        ⚠️ <b>风险提示：</b>以上结果为技术指标自动筛选，不构成投资建议。
        股市有风险，投资需谨慎，请结合基本面自行决策。<br>
        <span style="color:#aaa;">本邮件由 GitHub Actions 自动发送 · {now.strftime('%H:%M')}</span>
      </div>
    </body></html>"""

    return plain_text, html_text


# ============================================================
# 邮件发送
# ============================================================

def send_email(plain_text: str, html_text: str, subject: str):
    sender      = os.environ.get('EMAIL_SENDER', '')
    password    = os.environ.get('EMAIL_PASSWORD', '')
    receiver    = os.environ.get('EMAIL_RECEIVER', '')
    smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port   = int(os.environ.get('SMTP_PORT', '465'))

    if not all([sender, password, receiver]):
        logger.warning("⚠️  邮件环境变量未配置，将报告打印到控制台")
        print("\n" + "=" * 50)
        print(plain_text)
        print("=" * 50)
        return

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = sender
    msg['To']      = receiver
    msg.attach(MIMEText(plain_text, 'plain', 'utf-8'))
    msg.attach(MIMEText(html_text,  'html',  'utf-8'))

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as srv:
            srv.login(sender, password)
            srv.sendmail(sender, receiver, msg.as_string())
        logger.info(f"✅ 邮件已发送至 {receiver}")
    except Exception as e:
        logger.error(f"❌ 邮件发送失败: {e}")
        print(plain_text)


# ============================================================
# 主入口
# ============================================================

def main():
    logger.info("=" * 50)
    logger.info("🚀 A股选股机器人启动（BaoStock版）")
    logger.info("=" * 50)

    now      = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    subject  = f"📈 A股选股日报 · {date_str}"

    # 登录 BaoStock
    login_result = bs.login()
    if login_result.error_code != '0':
        logger.error(f"BaoStock登录失败: {login_result.error_msg}")
        sys.exit(1)
    logger.info("✅ BaoStock 登录成功")

    try:
        # 1. 获取股票列表
        stocks = get_all_a_stocks()

        # 2. 扫描技术条件
        candidates = scan_stocks(stocks)

        if not candidates:
            msg = (f"{date_str} 今日未发现符合条件的股票\n"
                   "（EXPMA12金叉EXPMA50 且 量能放大），市场可能处于调整阶段。")
            send_email(msg, msg, subject + " · 今日无信号")
            logger.info("今日无信号，已发送空报告")
            return

        # 3. 综合打分，选出Top5
        top5 = score_and_recommend(candidates)

        if not top5:
            logger.error("评分异常，top5为空")
            return

        # 4. 生成报告
        plain_text, html_text = generate_report(top5, len(candidates))

        # 5. 发送邮件
        send_email(plain_text, html_text, subject)

    finally:
        bs.logout()
        logger.info("BaoStock 已登出")

    logger.info("✅ 选股机器人运行完成")


if __name__ == '__main__':
    main()
