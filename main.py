#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股自动选股机器人（BaoStock版 v3）
====================================
筛选条件：
  1. EXPMA12日线近3日内上穿EXPMA50日线（金叉）
  2. 近5日成交量呈放大趋势
  3. 剔除亏损股（最近季报净利润 < 0）
  4. 近5个交易日内无涨停板

推荐逻辑：
  综合近3日价格动能、量能放大强度、金叉新鲜度打分，推荐Top5
  每只股票附带K线截图（MA5/MA10/MA28 + 成交量柱）
"""

import io
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from matplotlib.lines import Line2D

import baostock as bs
import matplotlib
matplotlib.use('Agg')  # 非交互模式，适合服务器环境
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 中文字体配置（需系统安装 fonts-wqy-microhei）
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# 技术指标计算
# ============================================================

def calc_expma(series: pd.Series, n: int) -> pd.Series:
    """EXPMA（指数移动平均线）"""
    return series.ewm(span=n, adjust=False).mean()


def check_volume_expanding(volumes: pd.Series, days: int = 5) -> tuple:
    """
    近N日成交量是否呈放大趋势
    返回: (是否放大: bool, 斜率增幅%: float)
    """
    if len(volumes) < days:
        return False, 0.0
    recent = volumes.tail(days).values.astype(float)
    if recent[0] == 0:
        return False, 0.0
    slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
    pct = slope / recent[0] * 100
    return (slope > 0 and pct > 5), round(pct, 1)


def detect_golden_cross(df: pd.DataFrame, lookback: int = 3) -> tuple:
    """
    近N天内是否出现 EXPMA12 上穿 EXPMA50 的金叉
    返回: (是否金叉: bool, 金叉日期: str, 距今天数: int)
    """
    if len(df) < lookback + 2:
        return False, None, 99
    for i in range(-lookback, 0):
        cur, prev = df.iloc[i], df.iloc[i - 1]
        if prev['EXPMA12'] < prev['EXPMA50'] and cur['EXPMA12'] >= cur['EXPMA50']:
            return True, str(cur.get('日期', '近期')), abs(i)
    return False, None, 99


def check_no_limit_up(hist: pd.DataFrame, code: str) -> bool:
    """
    近5个交易日内是否无涨停板
    主板涨停阈值：9.5%，创业板(300/301)：19.5%
    返回 True 表示无涨停（通过筛选）
    """
    if len(hist) < 6:
        return True
    recent = hist.tail(6)
    threshold = 0.195 if code.startswith('3') else 0.095
    closes = recent['收盘'].values.astype(float)
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            chg = (closes[i] - closes[i - 1]) / closes[i - 1]
            if chg >= threshold:
                return False
    return True


# ============================================================
# BaoStock 数据获取
# ============================================================

def get_all_a_stocks() -> pd.DataFrame:
    """获取所有A股列表，过滤ST、科创板、北交所"""
    logger.info("正在通过 BaoStock 获取A股列表…")
    rs = bs.query_stock_basic(code_name="")
    if rs.error_code != '0':
        raise RuntimeError(f"BaoStock获取股票列表失败: {rs.error_msg}")

    records = []
    while (rs.error_code == '0') and rs.next():
        row = rs.get_row_data()
        code, name, stk_type, status = row[0], row[1], row[4], row[5]
        if stk_type == '1' and status == '1':
            records.append({'代码_bs': code, '名称': name})

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("BaoStock返回股票列表为空")

    df['代码'] = df['代码_bs'].str.split('.').str[1]
    df = df[~df['名称'].str.contains('ST|退', na=False)]
    df = df[~df['代码'].str.startswith('688')]       # 科创板
    df = df[~df['代码_bs'].str.startswith('bj')]     # 北交所
    df = df.reset_index(drop=True)
    logger.info(f"过滤后待扫描：{len(df)} 只股票")
    return df


def get_stock_history(code_bs: str, days: int = 180) -> pd.DataFrame:
    """获取个股日线OHLCV数据（前复权）"""
    end_date   = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    rs = bs.query_history_k_data_plus(
        code_bs,
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"  # 前复权
    )
    if rs.error_code != '0':
        return None

    data = []
    while (rs.error_code == '0') and rs.next():
        data.append(rs.get_row_data())
    if not data:
        return None

    df = pd.DataFrame(data, columns=['日期', '开盘', '最高', '最低', '收盘', '成交量'])
    for col in ['开盘', '最高', '最低', '收盘', '成交量']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['收盘', '成交量']).reset_index(drop=True)
    return df


def is_profitable(code_bs: str) -> bool:
    """
    检查最近季报净利润是否为正（剔除亏损股）
    按时间倒序尝试最近3个季度，任一有数据即判断
    """
    now = datetime.now()
    year, month = now.year, now.month

    # 推算最近已披露季度
    if month <= 3:
        quarters = [(year - 1, 3), (year - 1, 2), (year - 1, 1)]
    elif month <= 4:
        quarters = [(year - 1, 4), (year - 1, 3), (year - 1, 2)]
    elif month <= 8:
        quarters = [(year, 1), (year - 1, 4), (year - 1, 3)]
    elif month <= 10:
        quarters = [(year, 2), (year, 1), (year - 1, 4)]
    else:
        quarters = [(year, 3), (year, 2), (year, 1)]

    for y, q in quarters:
        rs = bs.query_profit_data(code=code_bs, year=str(y), quarter=str(q))
        if rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            try:
                # row[4] = netProfit
                net_profit = float(row[4]) if row[4] else None
                if net_profit is not None:
                    return net_profit > 0
            except (ValueError, IndexError):
                pass
        time.sleep(0.1)

    logger.debug(f"无法获取 {code_bs} 盈利数据，默认保留")
    return True  # 无数据时不过滤


# ============================================================
# K线图生成
# ============================================================

def generate_kline_chart(hist: pd.DataFrame, code: str, name: str) -> bytes:
    """
    生成近60日K线图（红涨绿跌，含MA5/MA10/MA28 + 成交量柱）
    返回 PNG 字节流
    """
    df = hist.tail(60).copy().reset_index(drop=True)
    df.index = pd.DatetimeIndex(df['日期'])

    ohlcv = pd.DataFrame({
        'Open':   df['开盘'].astype(float),
        'High':   df['最高'].astype(float),
        'Low':    df['最低'].astype(float),
        'Close':  df['收盘'].astype(float),
        'Volume': df['成交量'].astype(float),
    }, index=df.index)

    ma5  = ohlcv['Close'].rolling(5).mean()
    ma10 = ohlcv['Close'].rolling(10).mean()
    ma28 = ohlcv['Close'].rolling(28).mean()

    apds = [
        mpf.make_addplot(ma5,  panel=0, color='#1565C0', width=1.3, label='MA5'),
        mpf.make_addplot(ma10, panel=0, color='#E65100', width=1.3, label='MA10'),
        mpf.make_addplot(ma28, panel=0, color='#6A1B9A', width=1.3, label='MA28'),
    ]

    mc = mpf.make_marketcolors(
        up='#e53935', down='#43a047',
        wick={'up': '#e53935', 'down': '#43a047'},
        edge={'up': '#e53935', 'down': '#43a047'},
        volume={'up': '#e53935', 'down': '#43a047'},
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='--',
        gridcolor='#eeeeee',
        facecolor='white',
        figcolor='white',
    )

    fig, axes = mpf.plot(
        ohlcv,
        type='candle',
        style=style,
        addplot=apds,
        volume=True,
        title=f'\n{name}（{code}）  近60日K线',
        figsize=(13, 7),
        returnfig=True,
        warn_too_much_data=1000,
        datetime_format='%m-%d',
        xrotation=30,
    )

    # 手动添加图例（mplfinance addplot label兼容性问题的保险处理）
    legend_elements = [
        Line2D([0], [0], color='#1565C0', linewidth=1.5, label='MA5'),
        Line2D([0], [0], color='#E65100', linewidth=1.5, label='MA10'),
        Line2D([0], [0], color='#6A1B9A', linewidth=1.5, label='MA28'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper left',
                   fontsize=9, framealpha=0.7)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# 核心扫描逻辑
# ============================================================

def scan_stocks(stocks_df: pd.DataFrame) -> list:
    """
    遍历所有股票，筛选同时满足以下条件的候选：
      - 近3日内EXPMA12上穿EXPMA50（金叉）
      - 近5日成交量放大
      - 近5日内无涨停板
    """
    candidates = []
    total = len(stocks_df)

    for idx, row in stocks_df.iterrows():
        code_bs = row['代码_bs']
        code    = row['代码']
        name    = row['名称']

        if idx > 0 and idx % 300 == 0:
            logger.info(f"扫描进度: {idx}/{total} ({idx/total*100:.1f}%)，"
                        f"候选 {len(candidates)} 只")
        try:
            hist = get_stock_history(code_bs)
            if hist is None or len(hist) < 60:
                continue

            hist['EXPMA12'] = calc_expma(hist['收盘'], 12)
            hist['EXPMA50'] = calc_expma(hist['收盘'], 50)

            # 条件1：EXPMA金叉
            crossed, cross_date, days_ago = detect_golden_cross(hist)
            if not crossed:
                continue

            # 条件2：成交量放大
            vol_ok, vol_pct = check_volume_expanding(hist['成交量'])
            if not vol_ok:
                continue

            # 条件3：近5日无涨停板
            if not check_no_limit_up(hist, code):
                continue

            last  = hist.iloc[-1]
            prev  = hist.iloc[-2]
            price_4d_ago   = hist.iloc[-5]['收盘'] if len(hist) >= 5 else prev['收盘']
            change_pct     = (float(last['收盘']) - float(prev['收盘'])) / float(prev['收盘']) * 100
            price_change_3d = (float(last['收盘']) - float(price_4d_ago)) / float(price_4d_ago) * 100
            expma_dev      = (float(last['EXPMA12']) - float(last['EXPMA50'])) / float(last['EXPMA50']) * 100

            candidates.append({
                '代码':          code,
                '代码_bs':       code_bs,
                '名称':          name,
                '最新价':        round(float(last['收盘']), 2),
                '今日涨跌幅':    round(change_pct, 2),
                '近3日涨幅':     round(price_change_3d, 2),
                'EXPMA12':       round(float(last['EXPMA12']), 2),
                'EXPMA50':       round(float(last['EXPMA50']), 2),
                'EXPMA偏离%':    round(expma_dev, 2),
                '成交量放大幅度': vol_pct,
                '金叉日期':      cross_date or '近3日',
                '金叉距今天数':  days_ago,
                '市场':          'sh' if code.startswith('6') else 'sz',
                '_hist':         hist,   # 保留历史数据用于画图
            })

            time.sleep(0.05)

        except Exception as e:
            logger.debug(f"处理 {code} 出错: {e}")
            continue

    logger.info(f"技术面扫描完成，候选 {len(candidates)} 只")
    return candidates


def filter_profitable(candidates: list) -> list:
    """对候选股逐一查询季报，剔除亏损股"""
    if not candidates:
        return []
    logger.info("正在剔除亏损股（查询季报数据）…")
    result = []
    for s in candidates:
        if is_profitable(s['代码_bs']):
            result.append(s)
        else:
            logger.info(f"剔除亏损股：{s['名称']}（{s['代码']}）")
        time.sleep(0.15)
    logger.info(f"盈利过滤后剩余 {len(result)} 只")
    return result


# ============================================================
# 评分与推荐
# ============================================================

def score_and_recommend(candidates: list) -> list:
    """
    综合评分权重：
      近3日价格涨幅   35%  （资金流入代理）
      成交量放大幅度  30%  （买盘积极度）
      金叉新鲜度      20%  （越新越强）
      EXPMA偏离适中   15%  （刚突破最佳）
    """
    if not candidates:
        return []

    df = pd.DataFrame(candidates)

    def safe_norm(series):
        s = series.fillna(0).astype(float)
        return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else pd.Series([0.5]*len(s), index=s.index)

    df['_price'] = safe_norm(df['近3日涨幅'])
    df['_vol']   = safe_norm(df['成交量放大幅度'])
    df['_fresh'] = safe_norm(-df['金叉距今天数'])
    df['_dev']   = safe_norm(-df['EXPMA偏离%'].abs())

    df['综合得分'] = (
        df['_price'] * 0.35 +
        df['_vol']   * 0.30 +
        df['_fresh'] * 0.20 +
        df['_dev']   * 0.15
    )

    return df.nlargest(5, '综合得分').to_dict('records')


# ============================================================
# 报告 + K线图生成
# ============================================================

def build_report_and_charts(top5: list, total_candidates: int) -> tuple:
    """
    返回: (纯文本: str, HTML: str, charts: dict{code: bytes})
    charts 是 {股票代码: PNG字节} 的字典
    """
    now      = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    weekday  = ['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]
    medals   = ['🥇','🥈','🥉','🏅','🏅']

    # ---- 生成K线图 ----
    charts = {}
    for s in top5:
        logger.info(f"生成K线图：{s['名称']}（{s['代码']}）")
        try:
            img_bytes = generate_kline_chart(s['_hist'], s['代码'], s['名称'])
            charts[s['代码']] = img_bytes
        except Exception as e:
            logger.error(f"K线图生成失败 {s['代码']}: {e}")

    # ---- 纯文本 ----
    lines = [
        f"📈 A股选股日报 · {date_str} {weekday}",
        "=" * 48,
        f"今日扫描主板+创业板，符合条件（EXPMA金叉+量能放大+盈利+无涨停）共 {total_candidates} 只",
        "综合近3日价格动能和量能强度，重点推荐以下5只：",
        ""
    ]
    for i, s in enumerate(top5):
        vol_pct  = s.get('成交量放大幅度', 0)
        change3d = s.get('近3日涨幅', 0)
        dev      = s.get('EXPMA偏离%', 0)
        lines += [
            f"{medals[i]} 第{i+1}名：{s['名称']}（{s['代码']}）",
            f"   当前价格：{s['最新价']} 元    今日涨跌：{s.get('今日涨跌幅',0):+.2f}%",
            f"   近3日涨幅：{change3d:+.2f}%    量能放大：+{vol_pct:.0f}%    EXPMA偏离：{dev:+.2f}%",
            f"   金叉日期：{s['金叉日期']}    EXPMA12: {s['EXPMA12']}  EXPMA50: {s['EXPMA50']}",
        ]
        reasons = _build_reasons(s)
        lines.append(f"   【推荐理由】{'；'.join(reasons)}")
        lines.append("")

    lines += [
        "=" * 48,
        "⚠️  风险提示：以上结果为纯技术面自动筛选，不构成投资建议。",
        "   股市有风险，投资需谨慎，请结合基本面和个人判断操作。",
        f"   本邮件由 GitHub Actions 自动发送 · {now.strftime('%H:%M')}"
    ]
    plain_text = "\n".join(lines)

    # ---- HTML ----
    colors  = ['#e8f5e9','#e3f2fd','#fff3e0','#f3e5f5','#fce4ec']
    borders = ['#43a047','#1e88e5','#fb8c00','#8e24aa','#e53935']

    cards_html = ""
    for i, s in enumerate(top5):
        vol_pct  = s.get('成交量放大幅度', 0)
        change3d = s.get('近3日涨幅', 0)
        dev      = s.get('EXPMA偏离%', 0)
        today_c  = '#c62828' if s.get('今日涨跌幅', 0) < 0 else '#2e7d32'
        c3d_c    = '#c62828' if change3d < 0 else '#2e7d32'
        reasons  = _build_reasons(s, html=True)
        has_chart = s['代码'] in charts

        cards_html += f"""
        <div style="background:{colors[i]};border-left:5px solid {borders[i]};
                    padding:16px 20px;margin:14px 0;border-radius:8px;">
          <h3 style="margin:0 0 10px;color:#222;font-size:1.05em;">
            {medals[i]}&nbsp;{s['名称']}&nbsp;
            <span style="color:#666;font-weight:normal;font-size:0.85em;">({s['代码']})</span>
          </h3>
          <table style="font-size:0.88em;color:#444;border-collapse:collapse;width:100%;">
            <tr>
              <td style="padding:3px 10px 3px 0;">💰 当前价格</td>
              <td style="padding:3px 16px 3px 0;"><b>{s['最新价']} 元</b></td>
              <td style="padding:3px 10px 3px 0;">📊 今日涨跌</td>
              <td style="color:{today_c};"><b>{s.get('今日涨跌幅',0):+.2f}%</b></td>
            </tr>
            <tr>
              <td style="padding:3px 10px 3px 0;">📅 近3日涨幅</td>
              <td style="padding:3px 16px 3px 0;color:{c3d_c};"><b>{change3d:+.2f}%</b></td>
              <td style="padding:3px 10px 3px 0;">📦 量能放大</td>
              <td style="color:#2e7d32;"><b>+{vol_pct:.0f}%</b></td>
            </tr>
            <tr>
              <td style="padding:3px 10px 3px 0;">🔔 金叉日期</td>
              <td style="padding:3px 16px 3px 0;">{s['金叉日期']}</td>
              <td style="padding:3px 10px 3px 0;">📈 EXPMA偏离</td>
              <td><b>{dev:+.2f}%</b></td>
            </tr>
          </table>
          <div style="margin-top:10px;padding:8px 10px;background:rgba(255,255,255,0.65);
                      border-radius:4px;font-size:0.85em;color:#333;line-height:1.7;">
            💡 <b>推荐理由：</b>{'；'.join(reasons)}
          </div>
          {'<div style="margin-top:12px;"><img src="cid:chart_' + s["代码"] + '" style="max-width:100%;border-radius:6px;border:1px solid #ddd;"></div>' if has_chart else ''}
        </div>"""

    html_text = f"""
    <html><body style="font-family:'PingFang SC',Arial,sans-serif;max-width:680px;
                        margin:0 auto;padding:24px 20px;color:#222;">
      <h2 style="color:#1565c0;border-bottom:3px solid #1565c0;padding-bottom:10px;margin-bottom:4px;">
        📈 A股选股日报 · {date_str} {weekday}
      </h2>
      <p style="color:#555;font-size:0.9em;margin-top:4px;">
        扫描主板+创业板，条件：EXPMA金叉 + 量能放大 + 盈利股 + 无涨停 · 符合 <b>{total_candidates}</b> 只 · 推荐Top5
      </p>
      {cards_html}
      <div style="background:#fff8e1;padding:12px 16px;border-radius:6px;
                  font-size:0.82em;color:#6d4c41;margin-top:18px;line-height:1.7;">
        ⚠️ <b>风险提示：</b>以上结果为技术指标自动筛选，不构成投资建议。
        股市有风险，投资需谨慎，请结合基本面自行决策。<br>
        <span style="color:#aaa;">本邮件由 GitHub Actions 自动发送 · {now.strftime('%H:%M')}</span>
      </div>
    </body></html>"""

    return plain_text, html_text, charts


def _build_reasons(s: dict, html: bool = False) -> list:
    """根据股票数据生成推荐理由列表"""
    reasons = []
    change3d = s.get('近3日涨幅', 0)
    vol_pct  = s.get('成交量放大幅度', 0)
    dev      = s.get('EXPMA偏离%', 0)

    if change3d > 5:
        t = f"近3日<b>累计上涨{change3d:.1f}%</b>，价格动能强劲" if html else f"近3日累计上涨{change3d:.1f}%，价格动能强劲"
    elif change3d > 0:
        t = f"近3日<b>上涨{change3d:.1f}%</b>，稳健上行" if html else f"近3日上涨{change3d:.1f}%，稳健上行"
    else:
        t = "价格整理充分，技术形态良好"
    reasons.append(t)

    if vol_pct > 30:
        t = f"近5日成交量<b>大幅放大+{vol_pct:.0f}%</b>，买盘涌入" if html else f"近5日成交量大幅放大+{vol_pct:.0f}%，买盘涌入"
    else:
        t = f"成交量稳步放大<b>+{vol_pct:.0f}%</b>，资金有序进场" if html else f"成交量稳步放大+{vol_pct:.0f}%，资金有序进场"
    reasons.append(t)

    if s.get('金叉距今天数', 99) == 1:
        t = "<b>昨日刚发生EXPMA12金叉</b>，信号最新鲜" if html else "昨日刚发生EXPMA12金叉，信号最新鲜"
    else:
        t = f"{s['金叉日期']}<b>EXPMA金叉确认</b>，中期趋势转强" if html else f"{s['金叉日期']}EXPMA金叉确认，中期趋势转强"
    reasons.append(t)

    if 0 < dev < 3:
        t = f"EXPMA12仅偏离{dev:.1f}%，<b>刚突破位置，上涨空间充足</b>" if html else f"EXPMA12仅偏离{dev:.1f}%，刚突破位置，上涨空间充足"
        reasons.append(t)

    return reasons


# ============================================================
# 邮件发送（含内嵌K线图）
# ============================================================

def send_email(plain_text: str, html_text: str, charts: dict, subject: str):
    """发送含内嵌K线截图的HTML邮件"""
    sender      = os.environ.get('EMAIL_SENDER', '')
    password    = os.environ.get('EMAIL_PASSWORD', '')
    receiver    = os.environ.get('EMAIL_RECEIVER', '')
    smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port   = int(os.environ.get('SMTP_PORT', '465'))

    if not all([sender, password, receiver]):
        logger.warning("⚠️  邮件环境变量未配置，将报告打印到控制台")
        print("\n" + "=" * 50 + "\n" + plain_text + "\n" + "=" * 50)
        return

    # 构造邮件：mixed > related > alternative(plain+html) + 图片
    msg_mixed = MIMEMultipart('mixed')
    msg_mixed['Subject'] = subject
    msg_mixed['From']    = sender
    msg_mixed['To']      = receiver

    msg_related = MIMEMultipart('related')
    msg_alt     = MIMEMultipart('alternative')
    msg_alt.attach(MIMEText(plain_text, 'plain', 'utf-8'))
    msg_alt.attach(MIMEText(html_text,  'html',  'utf-8'))
    msg_related.attach(msg_alt)

    # 内嵌K线图
    for code, img_bytes in charts.items():
        img = MIMEImage(img_bytes, 'png')
        img.add_header('Content-ID', f'<chart_{code}>')
        img.add_header('Content-Disposition', 'inline',
                       filename=f'kline_{code}.png')
        msg_related.attach(img)

    msg_mixed.attach(msg_related)

    import smtplib
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as srv:
            srv.login(sender, password)
            srv.sendmail(sender, receiver, msg_mixed.as_string())
        logger.info(f"✅ 邮件已发送至 {receiver}，含 {len(charts)} 张K线图")
    except Exception as e:
        logger.error(f"❌ 邮件发送失败: {e}")
        print(plain_text)


# ============================================================
# 主入口
# ============================================================

def main():
    logger.info("=" * 50)
    logger.info("🚀 A股选股机器人启动 v3")
    logger.info("=" * 50)

    now      = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    subject  = f"📈 A股选股日报 · {date_str}"

    login_result = bs.login()
    if login_result.error_code != '0':
        logger.error(f"BaoStock登录失败: {login_result.error_msg}")
        sys.exit(1)
    logger.info("✅ BaoStock 登录成功")

    try:
        # 1. 获取股票列表
        stocks = get_all_a_stocks()

        # 2. 技术面扫描（EXPMA金叉 + 量能放大 + 无涨停）
        candidates = scan_stocks(stocks)

        # 3. 基本面过滤（剔除亏损股）
        candidates = filter_profitable(candidates)

        if not candidates:
            msg = (f"{date_str} 今日未发现符合全部条件的股票，"
                   "市场可能处于调整阶段。")
            send_email(msg, msg, {}, subject + " · 今日无信号")
            return

        # 4. 综合评分，选出Top5
        top5 = score_and_recommend(candidates)

        # 5. 生成报告 + K线图
        plain_text, html_text, charts = build_report_and_charts(top5, len(candidates))

        # 6. 发送邮件
        send_email(plain_text, html_text, charts, subject)

    finally:
        bs.logout()
        logger.info("BaoStock 已登出")

    logger.info("✅ 选股机器人运行完成")


if __name__ == '__main__':
    main()
