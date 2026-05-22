#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股自动选股机器人
=================
筛选逻辑：
  1. EXPMA12日线上穿EXPMA50日线（近3日内发生金叉）
  2. 成交量近5日呈放大趋势

推荐逻辑：
  结合近3日资金流向（主力净流入）和行业板块热度综合打分，
  从候选池中推荐最优5只，并说明理由。
"""

import akshare as ak
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
    """计算EXPMA（指数移动平均线），等同于EMA"""
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
    # 斜率为正且每日平均涨幅超过5%才视为放大
    return (slope > 0 and pct > 5), round(pct, 1)


def detect_golden_cross(df: pd.DataFrame, lookback: int = 3) -> tuple:
    """
    检测近N天内是否出现EXPMA12上穿EXPMA50的金叉
    返回：(是否金叉: bool, 金叉日期: str)
    """
    if len(df) < lookback + 2:
        return False, None
    for i in range(-lookback, 0):
        cur  = df.iloc[i]
        prev = df.iloc[i - 1]
        if prev['EXPMA12'] < prev['EXPMA50'] and cur['EXPMA12'] >= cur['EXPMA50']:
            date_val = cur.get('日期', '近期')
            return True, str(date_val)
    return False, None


# ============================================================
# 数据获取
# ============================================================

def get_all_a_stocks() -> pd.DataFrame:
    """获取所有A股列表，过滤ST、退市、科创板、北交所"""
    logger.info("正在获取A股全量列表…")
    df = ak.stock_zh_a_spot_em()

    # 过滤：ST / 退市
    df = df[~df['名称'].str.contains('ST|退', na=False)]
    # 过滤：科创板(688)、北交所(83/87/43开头)
    df = df[~df['代码'].str.startswith(('688', '83', '87', '43'))]
    # 过滤：成交额过低（可能停牌，低于500万）
    if '成交额' in df.columns:
        df = df[df['成交额'].fillna(0) > 5e6]

    df = df[['代码', '名称']].reset_index(drop=True)
    logger.info(f"过滤后待扫描：{len(df)} 只股票")
    return df


def get_stock_history(code: str, days: int = 150) -> pd.DataFrame:
    """获取个股日线历史数据（前复权）"""
    end_date   = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    df = ak.stock_zh_a_hist(
        symbol=code, period="daily",
        start_date=start_date, end_date=end_date,
        adjust="qfq"
    )
    return df


def get_fund_flow(code: str, market: str) -> dict:
    """获取个股近3日资金流向数据"""
    try:
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        if df is None or df.empty:
            return {}
        recent3 = df.tail(3)
        result = {}
        if '主力净流入-净额' in recent3.columns:
            result['主力净流入3日'] = recent3['主力净流入-净额'].sum() / 1e4
            result['今日主力净流入']  = recent3.iloc[-1]['主力净流入-净额'] / 1e4
        if '超大单净流入-净额' in recent3.columns:
            result['超大单净流入3日'] = recent3['超大单净流入-净额'].sum() / 1e4
        return result
    except Exception as e:
        logger.debug(f"资金流向获取失败 {code}: {e}")
        return {}


def get_hot_sectors() -> dict:
    """获取今日热门行业板块及热度得分（排名越高分越高）"""
    try:
        df = ak.stock_board_industry_fund_flow_rank(symbol="今日")
        sector_heat = {}
        for rank, (_, row) in enumerate(df.iterrows()):
            name = row.get('名称', row.get('行业', ''))
            if name:
                sector_heat[name] = max(0, 30 - rank)
        logger.info(f"获取到 {len(sector_heat)} 个板块热度数据")
        return sector_heat
    except Exception as e:
        logger.error(f"板块热度获取失败: {e}")
        return {}


def get_stock_industry(code: str) -> str:
    """获取股票所属行业"""
    try:
        info = ak.stock_individual_info_em(symbol=code)
        if info is not None and not info.empty:
            row = info[info['item'] == '行业']
            if not row.empty:
                return str(row.iloc[0]['value'])
    except Exception:
        pass
    return '未知'


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
        code = row['代码']
        name = row['名称']

        if idx > 0 and idx % 200 == 0:
            logger.info(f"扫描进度: {idx}/{total} ({idx/total*100:.1f}%)，"
                        f"已找到候选 {len(candidates)} 只")

        try:
            hist = get_stock_history(code)
            if hist is None or len(hist) < 60:
                continue

            hist = hist.reset_index(drop=True)
            hist['EXPMA12'] = calc_expma(hist['收盘'], 12)
            hist['EXPMA50'] = calc_expma(hist['收盘'], 50)

            # 条件1：EXPMA金叉
            crossed, cross_date = detect_golden_cross(hist)
            if not crossed:
                continue

            # 条件2：成交量放大
            vol_ok, vol_pct = check_volume_expanding(hist['成交量'])
            if not vol_ok:
                continue

            last = hist.iloc[-1]
            prev = hist.iloc[-2]
            change_pct = (float(last['收盘']) - float(prev['收盘'])) / float(prev['收盘']) * 100

            candidates.append({
                '代码':         code,
                '名称':         name,
                '最新价':       round(float(last['收盘']), 2),
                '涨跌幅':       round(change_pct, 2),
                'EXPMA12':      round(float(last['EXPMA12']), 2),
                'EXPMA50':      round(float(last['EXPMA50']), 2),
                'EXPMA偏离%':   round((float(last['EXPMA12']) - float(last['EXPMA50'])) /
                                       float(last['EXPMA50']) * 100, 2),
                '成交量放大幅度': vol_pct,
                '金叉日期':     cross_date or '近3日',
                '市场':         'sh' if code.startswith('6') else 'sz',
            })

            time.sleep(0.15)  # 控制请求频率，避免被限流

        except Exception as e:
            logger.debug(f"处理 {code} 出错: {e}")
            time.sleep(0.1)
            continue

    logger.info(f"扫描完成，共找到 {len(candidates)} 只候选")
    return candidates


# ============================================================
# 评分与推荐
# ============================================================

def score_and_recommend(candidates: list, sector_heat: dict) -> list:
    """
    对候选股打综合分，权重：
      资金流向 45% | 板块热度 30% | 量能放大 25%
    """
    if not candidates:
        return []

    logger.info("正在对候选股票进行资金流向查询及综合评分…")

    for stock in candidates:
        ff = get_fund_flow(stock['代码'], stock['市场'])
        stock.update(ff)

        # 获取行业并匹配板块热度
        stock['行业'] = get_stock_industry(stock['代码'])
        heat = 0
        matched = stock['行业']
        for sector_name, score in sector_heat.items():
            if sector_name in stock['行业'] or stock['行业'] in sector_name:
                heat = score
                matched = sector_name
                break
        stock['板块热度得分'] = heat
        stock['匹配板块']   = matched

        time.sleep(0.3)

    df = pd.DataFrame(candidates)

    def safe_normalize(col):
        if col not in df.columns:
            return pd.Series([0.0] * len(df), index=df.index)
        s = df[col].fillna(0).astype(float)
        if s.max() == s.min():
            return pd.Series([0.5] * len(df), index=df.index)
        return (s - s.min()) / (s.max() - s.min())

    df['_资金分'] = safe_normalize('主力净流入3日')
    df['_量能分'] = safe_normalize('成交量放大幅度')
    df['_板块分'] = safe_normalize('板块热度得分')

    df['综合得分'] = (
        df['_资金分'] * 0.45 +
        df['_板块分'] * 0.30 +
        df['_量能分'] * 0.25
    )

    top5 = df.nlargest(5, '综合得分').to_dict('records')
    logger.info("推荐名单生成完毕")
    return top5


# ============================================================
# 报告生成（纯文本 + HTML 双版本）
# ============================================================

def generate_report(top5: list, total_candidates: int) -> tuple:
    """返回 (纯文本报告, HTML报告)"""
    now      = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    weekday  = ['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]
    medals   = ['🥇', '🥈', '🥉', '🏅', '🏅']

    # -------- 纯文本 --------
    lines = [
        f"📈 A股选股日报 · {date_str} {weekday}",
        "=" * 46,
        f"今日扫描全市场，符合【EXPMA12金叉EXPMA50 + 量能放大】条件共 {total_candidates} 只",
        "综合近3日资金流向和行业板块热度，重点推荐以下5只：",
        ""
    ]

    for i, s in enumerate(top5):
        net = s.get('主力净流入3日', 0)
        lines += [
            f"{medals[i]} 第{i+1}名：{s['名称']}（{s['代码']}）",
            f"   当前价格：{s['最新价']} 元    今日涨跌：{s.get('涨跌幅', 0):+.2f}%",
            f"   EXPMA12: {s['EXPMA12']}  |  EXPMA50: {s['EXPMA50']}  |  金叉日期: {s['金叉日期']}",
            f"   3日主力净流入：{net:+.0f} 万元    所属行业：{s.get('行业','未知')}",
        ]

        reasons = []
        if net > 0:
            reasons.append(f"近3日主力持续净流入{net:.0f}万元，资金积极进场")
        elif net < 0:
            reasons.append(f"注意：近3日资金有净流出（{abs(net):.0f}万元），以技术信号为主参考")
        if s.get('板块热度得分', 0) > 5:
            reasons.append(f"所属「{s.get('匹配板块', s.get('行业',''))}」板块今日资金流入排名靠前，处于市场热点")
        reasons.append(f"近5日量能持续放大（斜率+{s.get('成交量放大幅度',0):.0f}%），市场活跃度提升")
        reasons.append(f"EXPMA12于{s['金叉日期']}上穿EXPMA50，中期趋势明确转强")

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
        net = s.get('主力净流入3日', 0)
        flow_color = '#2e7d32' if net >= 0 else '#c62828'
        flow_arrow = '▲' if net >= 0 else '▼'

        reasons = []
        if net > 0:
            reasons.append(f"近3日主力净流入<b>{net:.0f}万元</b>，机构在持续买入")
        elif net < 0:
            reasons.append(f"注意：近3日资金净流出<b>{abs(net):.0f}万元</b>，以技术信号为主参考")
        if s.get('板块热度得分', 0) > 5:
            reasons.append(f"所属「<b>{s.get('匹配板块', s.get('行业',''))}</b>」板块今日处于热点前列")
        reasons.append(f"近5日量能趋势性放大（+{s.get('成交量放大幅度',0):.0f}%），买盘积极")
        reasons.append(f"EXPMA12于{s['金叉日期']}上穿EXPMA50，<b>中期趋势确认转强</b>")

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
              <td style="padding:3px 0;color:{'#c62828' if s.get('涨跌幅',0)<0 else '#2e7d32'}">
                <b>{s.get('涨跌幅',0):+.2f}%</b></td>
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
              <td style="padding:3px 10px 3px 0;">🏭 所属行业</td>
              <td style="padding:3px 0;">{s.get('行业','未知')}</td>
            </tr>
            <tr>
              <td style="padding:3px 10px 3px 0;" colspan="2">💵 3日主力净流入</td>
              <td style="padding:3px 0;color:{flow_color};" colspan="2">
                <b>{flow_arrow} {abs(net):.0f} 万元</b></td>
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
        今日扫描全市场，符合条件共 <b>{total_candidates}</b> 只 ·
        综合3日资金流向和板块热度，重点推荐5只
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
    """通过SMTP发送HTML邮件（支持Gmail / QQ邮箱 / 163）"""
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
    logger.info("🚀 A股选股机器人启动")
    logger.info("=" * 50)

    now      = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    subject  = f"📈 A股选股日报 · {date_str}"

    # 1. 获取股票列表
    stocks = get_all_a_stocks()

    # 2. 扫描技术条件（EXPMA金叉 + 量能放大）
    candidates = scan_stocks(stocks)

    if not candidates:
        msg = (f"{date_str} 今日未发现符合条件的股票\n"
               "（EXPMA12金叉EXPMA50 且 量能放大），市场可能处于调整阶段。")
        send_email(msg, msg, subject + " · 今日无信号")
        logger.info("今日无信号，已发送空报告")
        return

    # 3. 获取热门板块
    sector_heat = get_hot_sectors()

    # 4. 综合打分，选出Top5
    top5 = score_and_recommend(candidates, sector_heat)

    if not top5:
        logger.error("评分异常，top5为空")
        return

    # 5. 生成报告
    plain_text, html_text = generate_report(top5, len(candidates))

    # 6. 发送邮件
    send_email(plain_text, html_text, subject)

    logger.info("✅ 选股机器人运行完成")


if __name__ == '__main__':
    main()
