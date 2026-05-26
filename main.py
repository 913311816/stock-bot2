#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股自动选股机器人 v4 · Tushare Pro + Supabase
===============================================
筛选条件：
  1. EXPMA12日线近3日内上穿EXPMA50日线（金叉）
  2. 近5日成交量呈放大趋势
  3. 近5日内无涨停板
  4. 剔除亏损股

推荐评分权重：
  主力净流入3日  40% | 成交量放大  25% | 北向资金  15% | 金叉新鲜度  20%
"""

import io, os, sys, time, logging, smtplib
from datetime import datetime, timedelta
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from matplotlib.lines import Line2D

import tushare as ts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from supabase import create_client

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

_pro = None
_supabase = None

def get_pro():
    global _pro
    if _pro is None:
        token = os.environ.get('TUSHARE_TOKEN', '')
        if not token:
            raise RuntimeError("缺少环境变量 TUSHARE_TOKEN")
        ts.set_token(token)
        _pro = ts.pro_api()
    return _pro

def get_supabase():
    global _supabase
    if _supabase is None:
        url = os.environ.get('SUPABASE_URL', '')
        key = os.environ.get('SUPABASE_KEY', '')
        if not url or not key:
            logger.warning("SUPABASE 未配置，本次不保存推荐记录")
            return None
        _supabase = create_client(url, key)
    return _supabase

# ============================================================
# 技术指标
# ============================================================

def calc_expma(series, n):
    return series.ewm(span=n, adjust=False).mean()

def check_volume_expanding(volumes, days=5):
    if len(volumes) < days:
        return False, 0.0
    recent = volumes.tail(days).values.astype(float)
    if recent[0] == 0:
        return False, 0.0
    slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
    pct = slope / recent[0] * 100
    return (slope > 0 and pct > 5), round(pct, 1)

def detect_golden_cross(df, lookback=3):
    if len(df) < lookback + 2:
        return False, None, 99
    for i in range(-lookback, 0):
        cur, prev = df.iloc[i], df.iloc[i - 1]
        if prev['EXPMA12'] < prev['EXPMA50'] and cur['EXPMA12'] >= cur['EXPMA50']:
            return True, str(cur.get('日期', '近期')), abs(i)
    return False, None, 99

def check_no_limit_up(hist, code):
    if len(hist) < 6:
        return True
    recent = hist.tail(6)
    threshold = 0.195 if code.startswith('3') else 0.095
    closes = recent['收盘'].values.astype(float)
    for i in range(1, len(closes)):
        if closes[i-1] > 0 and (closes[i] - closes[i-1]) / closes[i-1] >= threshold:
            return False
    return True

# ============================================================
# Tushare 数据获取
# ============================================================

def get_all_a_stocks():
    logger.info("正在获取A股列表…")
    pro = get_pro()
    df = pro.stock_basic(exchange='', list_status='L',
                         fields='ts_code,symbol,name,industry,market')
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['symbol'].str.startswith('688')]
    df = df[~df['ts_code'].str.endswith('.BJ')]
    df = df.reset_index(drop=True)
    logger.info(f"过滤后待扫描：{len(df)} 只")
    return df

def get_stock_history(ts_code, days=200):
    pro = get_pro()
    end   = datetime.now().strftime('%Y%m%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        return None
    df = df.sort_values('trade_date').reset_index(drop=True)
    df = df.rename(columns={'trade_date':'日期','open':'开盘','high':'最高',
                             'low':'最低','close':'收盘','vol':'成交量'})
    for c in ['开盘','最高','最低','收盘','成交量']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['收盘','成交量']).reset_index(drop=True)

def get_fund_flow(ts_code, days=3):
    try:
        pro   = get_pro()
        end   = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=days+5)).strftime('%Y%m%d')
        df = pro.moneyflow(ts_code=ts_code, start_date=start, end_date=end,
                           fields='trade_date,net_mf_amount')
        if df is None or df.empty:
            return 0.0
        return round(df.sort_values('trade_date').tail(days)['net_mf_amount'].sum(), 2)
    except Exception as e:
        logger.debug(f"资金流向失败 {ts_code}: {e}")
        return 0.0

def get_north_flow_trend(ts_code):
    try:
        pro   = get_pro()
        end   = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        df = pro.hk_hold(ts_code=ts_code, start_date=start, end_date=end,
                         fields='trade_date,ratio')
        if df is None or len(df) < 2:
            return None
        df = df.sort_values('trade_date')
        df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
        df = df.dropna().tail(3)
        if len(df) < 2:
            return None
        return round(float(df.iloc[-1]['ratio']) - float(df.iloc[0]['ratio']), 4)
    except Exception:
        return None

def is_profitable(ts_code):
    try:
        pro = get_pro()
        now = datetime.now()
        y   = now.year
        periods = []
        for yr in [y, y-1]:
            for q in ['1231','0930','0630','0331']:
                periods.append(f"{yr}{q}")
        for period in periods[:5]:
            df = pro.fina_indicator(ts_code=ts_code, period=period,
                                    fields='ts_code,end_date,netprofit')
            if df is not None and not df.empty:
                val = pd.to_numeric(df.iloc[0].get('netprofit', None), errors='coerce')
                if val is not None and not np.isnan(float(val)):
                    return float(val) > 0
            time.sleep(0.1)
    except Exception as e:
        logger.debug(f"盈利查询失败 {ts_code}: {e}")
    return True

# ============================================================
# 扫描 & 过滤
# ============================================================

def scan_stocks(stocks_df):
    candidates = []
    total = len(stocks_df)
    for idx, row in stocks_df.iterrows():
        ts_code  = row['ts_code']
        code     = row['symbol']
        name     = row['name']
        industry = row.get('industry', '未知')

        if idx > 0 and idx % 300 == 0:
            logger.info(f"扫描进度: {idx}/{total} ({idx/total*100:.1f}%) · 候选 {len(candidates)} 只")
        try:
            hist = get_stock_history(ts_code)
            if hist is None or len(hist) < 60:
                continue
            hist['EXPMA12'] = calc_expma(hist['收盘'], 12)
            hist['EXPMA50'] = calc_expma(hist['收盘'], 50)

            crossed, cross_date, days_ago = detect_golden_cross(hist)
            if not crossed:
                continue
            vol_ok, vol_pct = check_volume_expanding(hist['成交量'])
            if not vol_ok:
                continue
            if not check_no_limit_up(hist, code):
                continue

            last  = hist.iloc[-1]
            prev  = hist.iloc[-2]
            p4    = hist.iloc[-5]['收盘'] if len(hist) >= 5 else prev['收盘']
            chg   = (float(last['收盘']) - float(prev['收盘'])) / float(prev['收盘']) * 100
            chg3d = (float(last['收盘']) - float(p4))          / float(p4)            * 100
            dev   = (float(last['EXPMA12']) - float(last['EXPMA50'])) / float(last['EXPMA50']) * 100

            candidates.append({
                'ts_code': ts_code, '代码': code, '名称': name, '行业': industry,
                '最新价': round(float(last['收盘']), 2),
                '今日涨跌幅': round(chg, 2), '近3日涨幅': round(chg3d, 2),
                'EXPMA12': round(float(last['EXPMA12']), 2),
                'EXPMA50': round(float(last['EXPMA50']), 2),
                'EXPMA偏离%': round(dev, 2),
                '成交量放大幅度': vol_pct,
                '金叉日期': cross_date or '近3日',
                '金叉距今天数': days_ago,
                '_hist': hist,
            })
            time.sleep(0.2)
        except Exception as e:
            logger.debug(f"处理 {code} 出错: {e}")
            time.sleep(0.1)

    logger.info(f"技术面扫描完成，候选 {len(candidates)} 只")
    return candidates

def enrich_and_filter(candidates):
    if not candidates:
        return []
    logger.info("补充资金流向 / 北向资金 / 盈利过滤…")
    result = []
    for s in candidates:
        if not is_profitable(s['ts_code']):
            logger.info(f"剔除亏损股：{s['名称']}（{s['代码']}）")
            time.sleep(0.1)
            continue
        s['主力净流入3日'] = get_fund_flow(s['ts_code'], days=3)
        time.sleep(0.25)
        s['北向变化'] = get_north_flow_trend(s['ts_code'])
        time.sleep(0.25)
        result.append(s)
    logger.info(f"过滤后剩余 {len(result)} 只")
    return result

def score_and_recommend(candidates):
    if not candidates:
        return []
    df = pd.DataFrame(candidates)

    def safe_norm(s):
        s = s.fillna(0).astype(float)
        return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() \
               else pd.Series([0.5]*len(s), index=s.index)

    has_north = df['北向变化'].notna().sum() > len(df) * 0.3
    df['_flow']  = safe_norm(df['主力净流入3日'])
    df['_vol']   = safe_norm(df['成交量放大幅度'])
    df['_fresh'] = safe_norm(-df['金叉距今天数'])

    if has_north:
        df['北向变化'] = df['北向变化'].fillna(0)
        df['_north']  = safe_norm(df['北向变化'])
        df['综合得分'] = df['_flow']*0.40 + df['_vol']*0.25 + df['_north']*0.15 + df['_fresh']*0.20
    else:
        df['综合得分'] = df['_flow']*0.40 + df['_vol']*0.40 + df['_fresh']*0.20

    return df.nlargest(5, '综合得分').to_dict('records')

# ============================================================
# K线图
# ============================================================

def generate_kline_chart(hist, code, name):
    df = hist.tail(60).copy().reset_index(drop=True)
    df.index = pd.DatetimeIndex(df['日期'])
    ohlcv = pd.DataFrame({
        'Open': df['开盘'].astype(float), 'High': df['最高'].astype(float),
        'Low':  df['最低'].astype(float), 'Close': df['收盘'].astype(float),
        'Volume': df['成交量'].astype(float)}, index=df.index)

    ma5  = ohlcv['Close'].rolling(5).mean()
    ma10 = ohlcv['Close'].rolling(10).mean()
    ma28 = ohlcv['Close'].rolling(28).mean()

    apds = [
        mpf.make_addplot(ma5,  panel=0, color='#1565C0', width=1.3),
        mpf.make_addplot(ma10, panel=0, color='#E65100', width=1.3),
        mpf.make_addplot(ma28, panel=0, color='#6A1B9A', width=1.3),
    ]
    mc    = mpf.make_marketcolors(up='#e53935', down='#43a047',
                wick={'up':'#e53935','down':'#43a047'},
                edge={'up':'#e53935','down':'#43a047'},
                volume={'up':'#e53935','down':'#43a047'})
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle='--',
                gridcolor='#eeeeee', facecolor='white', figcolor='white')
    fig, axes = mpf.plot(ohlcv, type='candle', style=style, addplot=apds,
        volume=True, title=f'\n{name}（{code}）  近60日K线',
        figsize=(13, 7), returnfig=True, warn_too_much_data=1000,
        datetime_format='%m-%d', xrotation=30)
    axes[0].legend(handles=[
        Line2D([0],[0], color='#1565C0', linewidth=1.5, label='MA5'),
        Line2D([0],[0], color='#E65100', linewidth=1.5, label='MA10'),
        Line2D([0],[0], color='#6A1B9A', linewidth=1.5, label='MA28'),
    ], loc='upper left', fontsize=9, framealpha=0.7)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ============================================================
# Supabase 存储
# ============================================================

def save_to_supabase(top5, report_date):
    sb = get_supabase()
    if sb is None:
        return
    try:
        records = [{
            'report_date':    report_date,
            'ts_code':        s['ts_code'],
            'code':           s['代码'],
            'name':           s['名称'],
            'industry':       s.get('行业', ''),
            'close_price':    s['最新价'],
            'rank_in_report': rank,
            'score':          round(s.get('综合得分', 0), 4),
            'fund_flow_3d':   s.get('主力净流入3日', 0),
            'north_change':   s.get('北向变化', None),
            'vol_expand_pct': s.get('成交量放大幅度', 0),
            'expma12':        s['EXPMA12'],
            'expma50':        s['EXPMA50'],
            'cross_date':     s['金叉日期'],
        } for rank, s in enumerate(top5, 1)]
        sb.table('stock_recommendations').insert(records).execute()
        logger.info(f"✅ 已保存 {len(records)} 条推荐记录（{report_date}）")
    except Exception as e:
        logger.error(f"Supabase 写入失败: {e}")

# ============================================================
# 报告生成
# ============================================================

def _reasons(s, html=False):
    reasons = []
    ff    = s.get('主力净流入3日', 0)
    north = s.get('北向变化', None)
    vol   = s.get('成交量放大幅度', 0)
    dev   = s.get('EXPMA偏离%', 0)

    if ff > 5000:
        r = f"近3日主力<b>大幅净流入{ff:.0f}万元</b>" if html else f"近3日主力大幅净流入{ff:.0f}万元"
    elif ff > 0:
        r = f"近3日主力净流入<b>{ff:.0f}万元</b>" if html else f"近3日主力净流入{ff:.0f}万元"
    else:
        r = f"近3日资金净流出{abs(ff):.0f}万元，以技术信号为主参考"
    reasons.append(r)

    if north is not None:
        r = (f"北向3日{'增持' if north>=0 else '减持'}<b>{abs(north):.4f}%</b>" if html
             else f"北向3日{'增持' if north>=0 else '减持'}{abs(north):.4f}%")
        reasons.append(r)

    r = (f"近5日成交量<b>放大+{vol:.0f}%</b>" if html else f"成交量放大+{vol:.0f}%")
    reasons.append(r)

    days = s.get('金叉距今天数', 99)
    if days == 1:
        r = "<b>昨日刚发生EXPMA金叉</b>，信号最新鲜" if html else "昨日刚发生EXPMA金叉"
    else:
        r = f"{s['金叉日期']}<b>确认EXPMA金叉</b>，趋势转强" if html else f"{s['金叉日期']}确认EXPMA金叉"
    reasons.append(r)

    if 0 < dev < 3:
        r = (f"EXPMA12偏离仅{dev:.1f}%，<b>刚突破位置</b>" if html
             else f"EXPMA12偏离{dev:.1f}%，刚突破")
        reasons.append(r)
    return reasons

def build_report_and_charts(top5, total):
    now      = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    weekday  = ['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]
    medals   = ['🥇','🥈','🥉','🏅','🏅']

    charts = {}
    for s in top5:
        logger.info(f"生成K线图：{s['名称']}")
        try:
            charts[s['代码']] = generate_kline_chart(s['_hist'], s['代码'], s['名称'])
        except Exception as e:
            logger.error(f"K线图失败 {s['代码']}: {e}")

    lines = [
        f"📈 A股选股日报 · {date_str} {weekday}",
        "="*50,
        f"扫描主板+创业板，符合全部条件共 {total} 只，推荐Top5：", ""
    ]
    for i, s in enumerate(top5):
        ff    = s.get('主力净流入3日', 0)
        north = s.get('北向变化', None)
        lines += [
            f"{medals[i]} {s['名称']}（{s['代码']}）  {s.get('行业','未知')}",
            f"   价格：{s['最新价']}元  今日：{s.get('今日涨跌幅',0):+.2f}%  近3日：{s.get('近3日涨幅',0):+.2f}%",
            f"   主力净流入3日：{ff:+.0f}万  北向：{f'{north:+.4f}%' if north else '暂无'}",
            f"   量能放大：+{s.get('成交量放大幅度',0):.0f}%  金叉：{s['金叉日期']}",
            f"   【推荐理由】{'；'.join(_reasons(s))}", ""
        ]
    lines += ["="*50,
              "⚠️ 风险提示：以上为技术面自动筛选，不构成投资建议。",
              f"GitHub Actions 自动发送 · {now.strftime('%H:%M')}"]
    plain_text = "\n".join(lines)

    colors  = ['#e8f5e9','#e3f2fd','#fff3e0','#f3e5f5','#fce4ec']
    borders = ['#43a047','#1e88e5','#fb8c00','#8e24aa','#e53935']
    cards   = ""
    for i, s in enumerate(top5):
        ff        = s.get('主力净流入3日', 0)
        north     = s.get('北向变化', None)
        today_c   = '#c62828' if s.get('今日涨跌幅',0) < 0 else '#2e7d32'
        ff_color  = '#c62828' if ff < 0 else '#2e7d32'
        ff_arrow  = '▼' if ff < 0 else '▲'
        nc_str    = f"{north:+.4f}%" if north is not None else "暂无"
        nc_color  = '#c62828' if (north or 0) < 0 else '#2e7d32'
        has_chart = s['代码'] in charts

        cards += f"""
        <div style="background:{colors[i]};border-left:5px solid {borders[i]};
                    padding:16px 20px;margin:14px 0;border-radius:8px;">
          <h3 style="margin:0 0 10px;color:#222;font-size:1.05em;">
            {medals[i]}&nbsp;{s['名称']}&nbsp;
            <span style="color:#666;font-weight:normal;font-size:0.85em;">
              ({s['代码']}) · {s.get('行业','未知')}
            </span>
          </h3>
          <table style="font-size:0.88em;color:#444;border-collapse:collapse;width:100%;">
            <tr>
              <td style="padding:3px 10px 3px 0;">💰 当前价格</td>
              <td style="padding:3px 16px 3px 0;"><b>{s['最新价']} 元</b></td>
              <td style="padding:3px 8px 3px 0;">📊 今日涨跌</td>
              <td style="color:{today_c};"><b>{s.get('今日涨跌幅',0):+.2f}%</b></td>
            </tr>
            <tr>
              <td style="padding:3px 10px 3px 0;">💵 主力净流入3日</td>
              <td style="padding:3px 16px 3px 0;color:{ff_color};">
                <b>{ff_arrow} {abs(ff):.0f} 万元</b></td>
              <td style="padding:3px 8px 3px 0;">🧭 北向3日变化</td>
              <td style="color:{nc_color};"><b>{nc_str}</b></td>
            </tr>
            <tr>
              <td style="padding:3px 10px 3px 0;">📦 量能放大</td>
              <td style="padding:3px 16px 3px 0;color:#2e7d32;">
                <b>+{s.get('成交量放大幅度',0):.0f}%</b></td>
              <td style="padding:3px 8px 3px 0;">🔔 金叉日期</td>
              <td><b>{s['金叉日期']}</b></td>
            </tr>
          </table>
          <div style="margin-top:10px;padding:8px 10px;background:rgba(255,255,255,0.65);
                      border-radius:4px;font-size:0.85em;color:#333;line-height:1.7;">
            💡 <b>推荐理由：</b>{'；'.join(_reasons(s, html=True))}
          </div>
          {'<div style="margin-top:12px;"><img src="cid:chart_'+s["代码"]+'" style="max-width:100%;border-radius:6px;border:1px solid #ddd;"></div>' if has_chart else ''}
        </div>"""

    html_text = f"""
    <html><body style="font-family:'PingFang SC',Arial,sans-serif;max-width:680px;
                        margin:0 auto;padding:24px 20px;color:#222;">
      <h2 style="color:#1565c0;border-bottom:3px solid #1565c0;padding-bottom:10px;">
        📈 A股选股日报 · {date_str} {weekday}
      </h2>
      <p style="color:#555;font-size:0.9em;">
        扫描主板+创业板（盈利·无涨停），符合全部条件 <b>{total}</b> 只 · 推荐Top5
      </p>
      {cards}
      <div style="background:#fff8e1;padding:12px 16px;border-radius:6px;
                  font-size:0.82em;color:#6d4c41;margin-top:18px;line-height:1.7;">
        ⚠️ <b>风险提示：</b>以上为技术指标自动筛选，不构成投资建议。股市有风险，请谨慎决策。<br>
        <span style="color:#aaa;">GitHub Actions 自动发送 · {now.strftime('%H:%M')}</span>
      </div>
    </body></html>"""

    return plain_text, html_text, charts

# ============================================================
# 邮件发送
# ============================================================

def send_email(plain_text, html_text, charts, subject):
    sender      = os.environ.get('EMAIL_SENDER', '')
    password    = os.environ.get('EMAIL_PASSWORD', '')
    receiver    = os.environ.get('EMAIL_RECEIVER', '')
    smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port   = int(os.environ.get('SMTP_PORT', '465'))

    if not all([sender, password, receiver]):
        logger.warning("邮件环境变量未配置，打印到控制台")
        print("\n" + "="*50 + "\n" + plain_text + "\n" + "="*50)
        return

    msg = MIMEMultipart('mixed')
    msg['Subject'] = subject
    msg['From']    = sender
    msg['To']      = receiver
    related = MIMEMultipart('related')
    alt     = MIMEMultipart('alternative')
    alt.attach(MIMEText(plain_text, 'plain', 'utf-8'))
    alt.attach(MIMEText(html_text,  'html',  'utf-8'))
    related.attach(alt)
    for code, img_bytes in charts.items():
        img = MIMEImage(img_bytes, 'png')
        img.add_header('Content-ID', f'<chart_{code}>')
        img.add_header('Content-Disposition', 'inline', filename=f'kline_{code}.png')
        related.attach(img)
    msg.attach(related)

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as srv:
            srv.login(sender, password)
            srv.sendmail(sender, receiver, msg.as_string())
        logger.info(f"✅ 邮件发送成功，含 {len(charts)} 张K线图")
    except Exception as e:
        logger.error(f"❌ 邮件发送失败: {e}")
        print(plain_text)

# ============================================================
# 主入口
# ============================================================

def main():
    logger.info("="*50)
    logger.info("🚀 A股选股机器人 v4 启动")
    logger.info("="*50)

    now      = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    today    = now.strftime('%Y-%m-%d')
    subject  = f"📈 A股选股日报 · {date_str}"

    stocks     = get_all_a_stocks()
    candidates = scan_stocks(stocks)
    candidates = enrich_and_filter(candidates)

    if not candidates:
        msg = f"{date_str} 今日未发现符合全部条件的股票，市场可能处于调整阶段。"
        send_email(msg, msg, {}, subject + " · 今日无信号")
        return

    top5 = score_and_recommend(candidates)
    save_to_supabase(top5, today)

    plain_text, html_text, charts = build_report_and_charts(top5, len(candidates))
    send_email(plain_text, html_text, charts, subject)
    logger.info("✅ 运行完成")

if __name__ == '__main__':
    main()
