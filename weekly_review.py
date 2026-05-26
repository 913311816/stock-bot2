#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股周回测报告
=============
每周日运行，从 Supabase 读取上周5天的推荐记录，
模拟「推荐日次日开盘买入，持有5个交易日收盘卖出」的收益，
汇总成表格发送邮件。
"""

import os, sys, logging, smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import tushare as ts
import pandas as pd
import numpy as np
from supabase import create_client

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

_pro = None
_supabase = None

def get_pro():
    global _pro
    if _pro is None:
        token = os.environ.get('TUSHARE_TOKEN', '')
        if not token:
            raise RuntimeError("缺少 TUSHARE_TOKEN")
        ts.set_token(token)
        _pro = ts.pro_api()
    return _pro

def get_supabase():
    global _supabase
    if _supabase is None:
        url = os.environ.get('SUPABASE_URL', '')
        key = os.environ.get('SUPABASE_KEY', '')
        if not url or not key:
            raise RuntimeError("缺少 SUPABASE_URL / SUPABASE_KEY")
        _supabase = create_client(url, key)
    return _supabase

# ============================================================
# 获取上周推荐记录
# ============================================================

def get_last_week_recommendations() -> pd.DataFrame:
    """从 Supabase 读取上周一到上周五的推荐记录"""
    now  = datetime.now()
    # 上周一
    last_monday = now - timedelta(days=now.weekday() + 7)
    # 上周五
    last_friday = last_monday + timedelta(days=4)

    start = last_monday.strftime('%Y-%m-%d')
    end   = last_friday.strftime('%Y-%m-%d')

    logger.info(f"查询推荐记录：{start} ~ {end}")
    sb = get_supabase()
    resp = (sb.table('stock_recommendations')
              .select('*')
              .gte('report_date', start)
              .lte('report_date', end)
              .order('report_date')
              .order('rank_in_report')
              .execute())

    if not resp.data:
        logger.warning("上周无推荐记录")
        return pd.DataFrame()

    df = pd.DataFrame(resp.data)
    logger.info(f"共获取 {len(df)} 条推荐记录")
    return df

# ============================================================
# 获取实际价格计算收益
# ============================================================

def get_price_on_date(ts_code: str, target_date: str,
                      direction: str = 'next', field: str = 'open') -> float:
    """
    获取 target_date 之后（direction='next'）或之前最近一个交易日的价格
    field: 'open'=开盘价，'close'=收盘价
    """
    pro = get_pro()
    if direction == 'next':
        start = target_date
        end   = (datetime.strptime(target_date, '%Y-%m-%d')
                 + timedelta(days=10)).strftime('%Y%m%d')
    else:
        start = (datetime.strptime(target_date, '%Y-%m-%d')
                 - timedelta(days=10)).strftime('%Y%m%d')
        end   = target_date

    start_fmt = start.replace('-', '')
    end_fmt   = end.replace('-', '')

    df = pro.daily(ts_code=ts_code, start_date=start_fmt, end_date=end_fmt,
                   fields=f'trade_date,{field}')
    if df is None or df.empty:
        return None
    df = df.sort_values('trade_date')
    val = pd.to_numeric(df.iloc[0 if direction == 'next' else -1][field], errors='coerce')
    return float(val) if not np.isnan(float(val)) else None

def calc_returns(recs: pd.DataFrame) -> pd.DataFrame:
    """
    对每条推荐记录计算：
      买入价  = 推荐日次日开盘价
      卖出价  = 买入后第5个交易日收盘价
      收益率  = (卖出价 - 买入价) / 买入价 × 100%
    """
    results = []
    for _, row in recs.iterrows():
        ts_code     = row['ts_code']
        report_date = row['report_date']  # 格式 YYYY-MM-DD

        # 次日开盘买入
        next_day = (datetime.strptime(report_date, '%Y-%m-%d')
                    + timedelta(days=1)).strftime('%Y-%m-%d')
        buy_price = get_price_on_date(ts_code, next_day, direction='next', field='open')

        # 持有5个交易日后收盘卖出
        sell_day = (datetime.strptime(next_day, '%Y-%m-%d')
                    + timedelta(days=9)).strftime('%Y-%m-%d')   # 留缓冲找交易日
        sell_price = get_price_on_date(ts_code, sell_day, direction='next', field='close')

        if buy_price and sell_price and buy_price > 0:
            ret = (sell_price - buy_price) / buy_price * 100
        else:
            ret = None

        results.append({
            '推荐日期':   report_date,
            '排名':       row.get('rank_in_report', '-'),
            '代码':       row.get('code', ''),
            '名称':       row.get('name', ''),
            '行业':       row.get('industry', ''),
            '推荐时收盘': row.get('close_price', '-'),
            '买入价(次日开盘)': round(buy_price, 2)  if buy_price  else '未开盘',
            '卖出价(5日后收盘)': round(sell_price, 2) if sell_price else '未到期',
            '收益率%':    round(ret, 2) if ret is not None else '计算中',
            '主力净流入(万)': row.get('fund_flow_3d', '-'),
        })

        import time; time.sleep(0.3)

    return pd.DataFrame(results)

# ============================================================
# 报告生成
# ============================================================

def build_weekly_report(df: pd.DataFrame, week_start: str, week_end: str) -> tuple:
    now = datetime.now()

    # 统计有效收益
    valid = df[df['收益率%'].apply(lambda x: isinstance(x, (int, float)))]
    avg_ret   = valid['收益率%'].mean() if not valid.empty else None
    win_rate  = (valid['收益率%'] > 0).sum() / len(valid) * 100 if not valid.empty else None
    best      = valid.loc[valid['收益率%'].idxmax()] if not valid.empty else None
    worst     = valid.loc[valid['收益率%'].idxmin()] if not valid.empty else None

    # ---- 纯文本 ----
    lines = [
        f"📊 A股选股周回测报告 · {week_start} ~ {week_end}",
        "="*52,
        f"本周共推荐 {len(df)} 只次（{df['推荐日期'].nunique()} 个交易日 × 每日5只）",
        f"模拟策略：推荐日次日开盘买入，持有5个交易日收盘卖出", ""
    ]

    if avg_ret is not None:
        win_emoji = '🟢' if avg_ret > 0 else '🔴'
        lines += [
            "【本周汇总】",
            f"  平均收益率：{win_emoji} {avg_ret:+.2f}%",
            f"  胜率（正收益占比）：{win_rate:.0f}%",
        ]
        if best is not None:
            lines.append(f"  最佳：{best['名称']}（{best['代码']}）{best['收益率%']:+.2f}%")
        if worst is not None:
            lines.append(f"  最差：{worst['名称']}（{worst['代码']}）{worst['收益率%']:+.2f}%")
        lines.append("")

    lines.append("【逐日明细】")
    for date, group in df.groupby('推荐日期'):
        lines.append(f"\n  📅 {date}")
        for _, r in group.iterrows():
            ret = r['收益率%']
            arrow = ('▲' if isinstance(ret, float) and ret > 0
                     else '▼' if isinstance(ret, float) and ret < 0 else '－')
            ret_str = f"{ret:+.2f}%" if isinstance(ret, float) else str(ret)
            lines.append(f"    {r['排名']}. {r['名称']}({r['代码']})  "
                         f"买:{r['买入价(次日开盘)']} → 卖:{r['卖出价(5日后收盘)']}  "
                         f"{arrow} {ret_str}")

    lines += [
        "", "="*52,
        "⚠️ 本回测为理想化模拟（次日开盘价买入），实际交易存在滑点和冲击成本。",
        f"GitHub Actions 自动发送 · {now.strftime('%Y-%m-%d %H:%M')}"
    ]
    plain_text = "\n".join(lines)

    # ---- HTML ----
    summary_html = ""
    if avg_ret is not None:
        avg_color = '#2e7d32' if avg_ret > 0 else '#c62828'
        summary_html = f"""
        <div style="background:#f5f5f5;border-radius:8px;padding:14px 18px;margin-bottom:16px;">
          <h3 style="margin:0 0 10px;color:#444;">📊 本周汇总</h3>
          <table style="font-size:0.95em;width:100%;border-collapse:collapse;">
            <tr>
              <td style="padding:4px 16px 4px 0;color:#666;">平均收益率</td>
              <td style="color:{avg_color};font-size:1.3em;font-weight:bold;">{avg_ret:+.2f}%</td>
              <td style="padding:4px 16px 4px 0;color:#666;">胜率</td>
              <td style="font-size:1.1em;font-weight:bold;">{win_rate:.0f}%</td>
            </tr>
            {'<tr><td style="color:#666;">最佳</td><td style="color:#2e7d32;"><b>' + str(best["名称"]) + ' ' + str(best["收益率%"]) + '%</b></td>' if best is not None else ''}
            {'<td style="color:#666;">最差</td><td style="color:#c62828;"><b>' + str(worst["名称"]) + ' ' + str(worst["收益率%"]) + '%</b></td></tr>' if worst is not None else ''}
          </table>
        </div>"""

    # 按日期分组生成表格
    tables_html = ""
    for date, group in df.groupby('推荐日期'):
        rows_html = ""
        for _, r in group.iterrows():
            ret    = r['收益率%']
            if isinstance(ret, float):
                ret_color = '#2e7d32' if ret > 0 else '#c62828'
                ret_str   = f"{ret:+.2f}%"
            else:
                ret_color = '#888'
                ret_str   = str(ret)
            rows_html += f"""
            <tr style="border-bottom:1px solid #eee;">
              <td style="padding:6px 8px;">{r['排名']}</td>
              <td style="padding:6px 8px;"><b>{r['名称']}</b><br>
                <span style="color:#888;font-size:0.82em;">{r['代码']} · {r['行业']}</span></td>
              <td style="padding:6px 8px;text-align:right;">{r['推荐时收盘']}</td>
              <td style="padding:6px 8px;text-align:right;">{r['买入价(次日开盘)']}</td>
              <td style="padding:6px 8px;text-align:right;">{r['卖出价(5日后收盘)']}</td>
              <td style="padding:6px 8px;text-align:right;color:{ret_color};font-weight:bold;">{ret_str}</td>
            </tr>"""
        tables_html += f"""
        <h4 style="margin:20px 0 6px;color:#555;">📅 {date}</h4>
        <table style="width:100%;border-collapse:collapse;font-size:0.88em;
                      background:white;border-radius:6px;overflow:hidden;
                      box-shadow:0 1px 4px rgba(0,0,0,0.08);">
          <thead style="background:#1565c0;color:white;">
            <tr>
              <th style="padding:8px;text-align:left;">排名</th>
              <th style="padding:8px;text-align:left;">股票</th>
              <th style="padding:8px;text-align:right;">推荐收盘</th>
              <th style="padding:8px;text-align:right;">买入价</th>
              <th style="padding:8px;text-align:right;">卖出价</th>
              <th style="padding:8px;text-align:right;">收益率</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>"""

    html_text = f"""
    <html><body style="font-family:'PingFang SC',Arial,sans-serif;max-width:700px;
                        margin:0 auto;padding:24px 20px;color:#222;">
      <h2 style="color:#1565c0;border-bottom:3px solid #1565c0;padding-bottom:10px;">
        📊 A股选股周回测报告
      </h2>
      <p style="color:#555;font-size:0.9em;">
        回测周期：<b>{week_start} ~ {week_end}</b> ·
        策略：推荐日次日开盘买入，持有5个交易日收盘卖出
      </p>
      {summary_html}
      {tables_html}
      <div style="background:#fff8e1;padding:12px 16px;border-radius:6px;
                  font-size:0.82em;color:#6d4c41;margin-top:20px;line-height:1.7;">
        ⚠️ 本回测为理想化模拟，实际交易存在滑点、冲击成本及停牌等情况，仅供参考。<br>
        <span style="color:#aaa;">GitHub Actions 自动发送 · {now.strftime('%Y-%m-%d %H:%M')}</span>
      </div>
    </body></html>"""

    return plain_text, html_text

# ============================================================
# 邮件发送
# ============================================================

def send_email(plain_text, html_text, subject):
    sender      = os.environ.get('EMAIL_SENDER', '')
    password    = os.environ.get('EMAIL_PASSWORD', '')
    receiver    = os.environ.get('EMAIL_RECEIVER', '')
    smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port   = int(os.environ.get('SMTP_PORT', '465'))

    if not all([sender, password, receiver]):
        print("\n" + plain_text)
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
        logger.info(f"✅ 周报邮件已发送至 {receiver}")
    except Exception as e:
        logger.error(f"❌ 邮件发送失败: {e}")
        print(plain_text)

# ============================================================
# 主入口
# ============================================================

def main():
    logger.info("="*50)
    logger.info("📊 A股选股周回测报告启动")
    logger.info("="*50)

    now         = datetime.now()
    last_monday = now - timedelta(days=now.weekday() + 7)
    last_friday = last_monday + timedelta(days=4)
    week_start  = last_monday.strftime('%Y-%m-%d')
    week_end    = last_friday.strftime('%Y-%m-%d')
    subject     = f"📊 A股选股周回测报告 · {week_start} ~ {week_end}"

    recs = get_last_week_recommendations()
    if recs.empty:
        msg = f"上周（{week_start} ~ {week_end}）暂无推荐记录，无法生成回测报告。"
        send_email(msg, msg, subject)
        return

    result_df   = calc_returns(recs)
    plain_text, html_text = build_weekly_report(result_df, week_start, week_end)
    send_email(plain_text, html_text, subject)
    logger.info("✅ 周回测报告运行完成")

if __name__ == '__main__':
    main()
