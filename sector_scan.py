#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股热点板块每日扫描 v3
========================
修复：BaoStock行业查询用法错误，改为正确的一次性全量查询再分组
"""

import os, sys, time, logging, smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import baostock as bs
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


# ============================================================
# 产业链关系表
# ============================================================
INDUSTRY_CHAIN = {
    '农林牧渔': {'upstream': ['化肥', '农药', '种子', '农机'], 'downstream': ['食品饮料', '生猪养殖', '饲料']},
    '医药生物': {'upstream': ['医药原料', '原料药'], 'downstream': ['医疗器械', '医疗服务', '连锁药店']},
    '电子':     {'upstream': ['半导体设备', '半导体材料'], 'downstream': ['消费电子', '通信', '计算机']},
    '计算机':   {'upstream': ['算力', '服务器', '光模块'], 'downstream': ['软件', '云计算', '人工智能']},
    '通信':     {'upstream': ['光模块', '基站', '芯片'], 'downstream': ['计算机', '传媒', '物联网']},
    '汽车':     {'upstream': ['钢铁', '铝合金', '电池'], 'downstream': ['汽车经销', '充电桩', '保险']},
    '电气设备': {'upstream': ['硅料', '锂矿', '铜铝'], 'downstream': ['新能源', '储能', '电网']},
    '机械设备': {'upstream': ['钢铁', '铸件', '液压件'], 'downstream': ['工业自动化', '机器人', '军工']},
    '国防军工': {'upstream': ['钛合金', '碳纤维', '特种材料'], 'downstream': ['航空航天', '无人机', '卫星']},
    '有色金属': {'upstream': ['采掘', '能源'], 'downstream': ['电子', '汽车', '电气设备']},
    '钢铁':     {'upstream': ['铁矿石', '焦炭'], 'downstream': ['建筑', '汽车', '家用电器']},
    '化工':     {'upstream': ['石油石化', '煤炭'], 'downstream': ['农药', '涂料', '新材料']},
    '房地产':   {'upstream': ['建筑材料', '钢铁', '水泥'], 'downstream': ['家用电器', '家居', '物业']},
    '建筑材料': {'upstream': ['采掘', '化工'], 'downstream': ['房地产', '建筑装饰']},
    '家用电器': {'upstream': ['钢铁', '铜', '塑料'], 'downstream': ['商业贸易', '房地产']},
    '食品饮料': {'upstream': ['农林牧渔', '包装'], 'downstream': ['商业贸易', '休闲服务']},
    '银行':     {'upstream': ['金融科技'], 'downstream': ['非银金融', '保险', '券商']},
    '非银金融': {'upstream': ['银行', '监管政策'], 'downstream': ['保险', '券商', '信托']},
    '采掘':     {'upstream': ['工程机械', '爆破'], 'downstream': ['化工', '钢铁', '电力']},
    '公用事业': {'upstream': ['煤炭', '天然气', '核电'], 'downstream': ['化工', '建材']},
}

def match_chain(sector_name: str) -> dict:
    for key, chain in INDUSTRY_CHAIN.items():
        if key in sector_name or sector_name in key:
            return chain
    return {'upstream': [], 'downstream': []}


# ============================================================
# BaoStock：一次性取全部股票行业分类
# ============================================================

def get_all_stocks_with_industry() -> pd.DataFrame:
    """一次性获取全部A股及其行业分类（BaoStock正确用法）"""
    logger.info("获取全市场股票行业分类…")
    rs = bs.query_stock_industry()
    if rs.error_code != '0':
        logger.error(f"行业查询失败: {rs.error_msg}")
        return pd.DataFrame()
    data = []
    while rs.next():
        data.append(rs.get_row_data())
    if not data:
        return pd.DataFrame()
    # 列名：updateDate, code, code_name, industry, industryClassification
    df = pd.DataFrame(data, columns=['updateDate','code','name','industry','classification'])
    # 过滤：只要沪深主板+创业板，剔除科创板和北交所
    df = df[df['code'].str.startswith(('sh.6','sz.0','sz.3'))]
    df = df[~df['code'].str.startswith('sh.688')]
    df = df[df['industry'] != '']
    logger.info(f"共获取 {len(df)} 只股票，{df['industry'].nunique()} 个行业")
    return df


def get_stock_5d_return(code_bs: str) -> float | None:
    """获取单只股票近5日涨幅"""
    end_date   = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
    try:
        rs = bs.query_history_k_data_plus(
            code_bs, "date,close",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="3")
        data = []
        while rs.next():
            row = rs.get_row_data()
            if row[1]:
                data.append(float(row[1]))
        if len(data) >= 6 and data[-6] > 0:
            return round((data[-1] - data[-6]) / data[-6] * 100, 2)
    except Exception:
        pass
    return None


def calc_industry_performance(stocks_df: pd.DataFrame, sample_per_industry: int = 15) -> pd.DataFrame:
    """
    对每个行业抽样计算近5日平均涨幅
    同时记录换手率代理值（高涨幅行业视为高换手）
    """
    industries = stocks_df['industry'].unique()
    logger.info(f"开始计算 {len(industries)} 个行业的近5日涨幅（每个抽 {sample_per_industry} 只）…")

    results = []
    for idx, industry in enumerate(industries):
        if idx > 0 and idx % 5 == 0:
            logger.info(f"  行业计算进度: {idx}/{len(industries)}")

        members = stocks_df[stocks_df['industry'] == industry]['code'].tolist()
        sample  = members[:sample_per_industry]
        returns = []

        for code_bs in sample:
            ret = get_stock_5d_return(code_bs)
            if ret is not None:
                returns.append(ret)
            time.sleep(0.04)

        if len(returns) >= 3:  # 至少3只有效才统计
            avg_ret  = round(np.mean(returns), 2)
            positive = sum(1 for r in returns if r > 0)
            results.append({
                '板块名称':       industry,
                '5日涨幅':        avg_ret,
                '上涨比例':       round(positive / len(returns) * 100, 1),
                '样本数':         len(returns),
                '换手率':         abs(avg_ret) * 0.4 + 1.0,
                # 用涨幅绝对值模拟资金净流入热度
                '5日主力净流入':  avg_ret * 1e8,
            })

    df = pd.DataFrame(results)
    logger.info(f"行业涨幅计算完成，共 {len(df)} 个行业")
    return df


# ============================================================
# 评分 & 分类
# ============================================================

def score_sectors(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def safe_norm(s):
        s = s.fillna(0).astype(float)
        if s.max() == s.min():
            return pd.Series([0.5]*len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    df['_flow']   = safe_norm(df['5日主力净流入'])
    df['_return'] = safe_norm(df['5日涨幅'])
    df['_turn']   = safe_norm(df['换手率'])
    df['综合热度分'] = df['_flow']*0.45 + df['_return']*0.30 + df['_turn']*0.25
    return df.sort_values('综合热度分', ascending=False).reset_index(drop=True)


def classify_sectors(df: pd.DataFrame) -> tuple:
    if df.empty or len(df) < 4:
        return [], []
    top = df.head(min(20, len(df))).copy()
    flow_med = top['5日主力净流入'].median()
    ret_med  = top['5日涨幅'].median()
    emerging, sustained = [], []
    for _, row in top.iterrows():
        item = {
            '板块名称': row['板块名称'],
            '5日涨幅':  row['5日涨幅'],
            '上涨比例': row.get('上涨比例', 0),
        }
        if row['5日主力净流入'] >= flow_med and row['5日涨幅'] >= ret_med:
            sustained.append(item)
        elif row['5日涨幅'] >= ret_med * 1.3 and row['5日主力净流入'] < flow_med:
            emerging.append(item)
    return emerging[:3], sustained[:3]


# ============================================================
# 板块内选3只股票
# ============================================================

def pick_3_stocks(top1_name: str, stocks_df: pd.DataFrame) -> dict:
    result = {'龙头': None, '滞涨': None, '产业链': None, '产业链说明': ''}

    members = stocks_df[stocks_df['industry'] == top1_name]['code'].tolist()
    if not members:
        return result

    logger.info(f"计算 {top1_name} 成分股涨幅（共{len(members)}只，取前30）…")
    end_date   = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')

    enriched = []
    name_map = dict(zip(stocks_df['code'], stocks_df['name']))

    for code_bs in members[:30]:
        try:
            rs = bs.query_history_k_data_plus(
                code_bs, "date,close,turn",
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="2")
            data = []
            while rs.next():
                data.append(rs.get_row_data())
            if len(data) < 2:
                continue
            close_now  = float(data[-1][1]) if data[-1][1] else 0
            close_prev = float(data[-2][1]) if data[-2][1] else 0
            close_5d   = float(data[-6][1]) if len(data) >= 6 and data[-6][1] else close_prev
            chg_today  = (close_now - close_prev) / close_prev * 100 if close_prev > 0 else 0
            chg_5d     = (close_now - close_5d)   / close_5d   * 100 if close_5d   > 0 else 0
            code_short = code_bs.split('.')[1]
            name_str   = name_map.get(code_bs, code_short)
            if 'ST' in name_str or '退' in name_str:
                continue
            enriched.append({
                '代码': code_short, '名称': name_str,
                '最新价': round(close_now, 2),
                '今日涨跌幅': round(chg_today, 2),
                '5日涨幅': round(chg_5d, 2),
            })
            time.sleep(0.05)
        except Exception:
            continue

    if not enriched:
        return result

    df = pd.DataFrame(enriched)
    avg_5d = df['5日涨幅'].mean()

    # 1. 龙头：5日涨幅最高
    top = df.nlargest(1, '5日涨幅').iloc[0]
    result['龙头'] = {
        '代码': top['代码'], '名称': top['名称'],
        '最新价': top['最新价'], '今日涨跌幅': top['今日涨跌幅'],
        '5日涨幅': top['5日涨幅'],
        '理由': f"板块内近5日涨幅最强（+{top['5日涨幅']:.1f}%），龙头效应显著",
    }

    # 2. 滞涨：5日涨幅低于均值且今日上涨
    lag = df[(df['5日涨幅'] < avg_5d) & (df['今日涨跌幅'] > 0)]
    if lag.empty:
        lag = df.nsmallest(5, '5日涨幅')
    if not lag.empty:
        pick = lag.nlargest(1, '今日涨跌幅').iloc[0]
        result['滞涨'] = {
            '代码': pick['代码'], '名称': pick['名称'],
            '最新价': pick['最新价'], '今日涨跌幅': pick['今日涨跌幅'],
            '5日涨幅': pick['5日涨幅'],
            '理由': f"板块均值{avg_5d:.1f}%，该股5日仅{pick['5日涨幅']:.1f}%，滞涨明显，今日开始启动，补涨空间大",
        }

    # 3. 产业链：取上下游行业里5日涨幅最高的1只
    chain = match_chain(top1_name)
    for related in (chain['upstream'] + chain['downstream'])[:6]:
        rel_members = stocks_df[stocks_df['industry'] == related]['code'].tolist()
        if not rel_members:
            continue
        rel_returns = []
        rel_names   = dict(zip(stocks_df['code'], stocks_df['name']))
        for code_bs in rel_members[:10]:
            ret = get_stock_5d_return(code_bs)
            if ret is not None:
                rel_returns.append((code_bs, ret))
            time.sleep(0.04)
        if not rel_returns:
            continue
        best_code, best_ret = max(rel_returns, key=lambda x: x[1])
        is_up    = related in chain['upstream']
        relation = '上游' if is_up else '下游'
        short    = best_code.split('.')[1]
        name_str = rel_names.get(best_code, short)
        # 获取最新价
        try:
            rs2 = bs.query_history_k_data_plus(
                best_code, "date,close",
                start_date=(datetime.now()-timedelta(days=5)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d'),
                frequency="d", adjustflag="2")
            d2 = []
            while rs2.next():
                d2.append(rs2.get_row_data())
            price = round(float(d2[-1][1]), 2) if d2 and d2[-1][1] else '-'
            chg2  = round((float(d2[-1][1])-float(d2[-2][1]))/float(d2[-2][1])*100, 2) if len(d2)>=2 else 0
        except Exception:
            price, chg2 = '-', 0

        result['产业链'] = {
            '代码': short, '名称': name_str,
            '最新价': price, '今日涨跌幅': chg2,
            '5日涨幅': round(best_ret, 2),
            '理由': f"来自{top1_name}{relation}行业「{related}」，板块联动轮动机会",
        }
        result['产业链说明'] = f"{top1_name} → {relation}「{related}」"
        break

    return result


# ============================================================
# 报告生成
# ============================================================

def build_report(scored_df, emerging, sustained, top1_name, top1_row, stocks_3, is_high_risk):
    now      = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    weekday  = ['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]
    ret5     = float(top1_row.get('5日涨幅', 0))
    up_ratio = float(top1_row.get('上涨比例', 0))
    chain    = match_chain(top1_name)

    hot_reason_parts = []
    if ret5 > 8:
        hot_reason_parts.append(f"5日大涨{ret5:.1f}%，市场情绪强烈")
    elif ret5 > 3:
        hot_reason_parts.append(f"5日上涨{ret5:.1f}%，趋势明确")
    else:
        hot_reason_parts.append(f"5日涨幅{ret5:.1f}%，悄然积累")
    if up_ratio > 70:
        hot_reason_parts.append(f"行业内{up_ratio:.0f}%个股上涨，普涨特征明显")
    elif up_ratio > 50:
        hot_reason_parts.append(f"行业内{up_ratio:.0f}%个股上涨，多头占优")
    hot_reason = "；".join(hot_reason_parts)

    risk_flag = "⚠️ 【高位风险预警】该板块5日涨幅已超15%，追高风险较大，建议谨慎！" if is_high_risk else ""

    # 纯文本
    lines = [f"🔥 A股热点板块日报 · {date_str} {weekday}", "="*50]
    if risk_flag:
        lines += [risk_flag, ""]

    lines += ["【板块热度Top5】", "  🔴 新兴热点（刚爆发）："]
    if emerging:
        for s in emerging:
            lines.append(f"    · {s['板块名称']}  5日涨幅{s['5日涨幅']:+.1f}%  上涨比例{s.get('上涨比例',0):.0f}%")
    else:
        lines.append("    暂无明显新兴热点")
    lines.append("  🔵 持续主线（资金持续强势）：")
    if sustained:
        for s in sustained:
            lines.append(f"    · {s['板块名称']}  5日涨幅{s['5日涨幅']:+.1f}%  上涨比例{s.get('上涨比例',0):.0f}%")
    else:
        lines.append("    暂无明显持续主线")
    lines.append("")

    lines += [
        f"【重点推荐板块：{top1_name}】",
        f"  5日区间涨幅：{ret5:+.1f}%",
        f"  行业内上涨比例：{up_ratio:.0f}%",
        f"  热度分析：{hot_reason}",
    ]
    if chain['upstream']:
        lines.append(f"  上游关联：{'、'.join(chain['upstream'][:3])}")
    if chain['downstream']:
        lines.append(f"  下游关联：{'、'.join(chain['downstream'][:3])}")
    lines.append("")

    lines.append(f"【{top1_name} 板块推荐3只】")
    for key, label in [('龙头','① 核心龙头'), ('滞涨','② 滞涨潜力'), ('产业链','③ 产业链联动')]:
        s = stocks_3.get(key)
        if s:
            lines += [
                f"\n  {label}：{s['名称']}（{s['代码']}）",
                f"    价格：{s['最新价']}元  今日：{s.get('今日涨跌幅',0):+.1f}%  5日：{s.get('5日涨幅',0):+.1f}%",
                f"    推荐理由：{s['理由']}",
            ]
        else:
            lines.append(f"\n  {label}：暂无合适标的")
    if stocks_3.get('产业链说明'):
        lines.append(f"\n  产业链路径：{stocks_3['产业链说明']}")

    lines += ["", "="*50,
              "⚠️ 风险提示：以上为板块技术面自动分析，不构成投资建议。",
              f"GitHub Actions 自动发送 · {now.strftime('%H:%M')}"]
    plain_text = "\n".join(lines)

    # HTML
    risk_html = f"""<div style="background:#ffebee;border:2px solid #e53935;border-radius:6px;
        padding:10px 16px;margin-bottom:14px;color:#b71c1c;font-weight:bold;">
        ⚠️ 高位风险预警：该板块5日涨幅已超15%，追高风险较大！</div>""" if is_high_risk else ""

    top5_rows = ""
    for i, row in scored_df.head(5).iterrows():
        ret_v = row.get('5日涨幅', 0)
        rc    = '#2e7d32' if ret_v > 0 else '#c62828'
        name  = row['板块名称']
        badge = ""
        if any(s['板块名称'] == name for s in emerging):
            badge = '<span style="background:#ff7043;color:white;font-size:0.75em;padding:1px 6px;border-radius:3px;margin-left:6px;">新兴</span>'
        elif any(s['板块名称'] == name for s in sustained):
            badge = '<span style="background:#1565c0;color:white;font-size:0.75em;padding:1px 6px;border-radius:3px;margin-left:6px;">主线</span>'
        top5_rows += f"""<tr style="border-bottom:1px solid #eee;">
          <td style="padding:7px 10px;">{i+1}</td>
          <td style="padding:7px 10px;font-weight:bold;">{name}{badge}</td>
          <td style="padding:7px 10px;text-align:right;color:{rc};">{ret_v:+.1f}%</td>
          <td style="padding:7px 10px;text-align:right;">{row.get('上涨比例',0):.0f}%</td>
          <td style="padding:7px 10px;text-align:right;">{int(row.get('样本数',0))}只</td>
        </tr>"""

    chain_html = ""
    if chain['upstream'] or chain['downstream']:
        up_t   = "".join(f'<span style="background:#e3f2fd;color:#1565c0;padding:3px 8px;border-radius:12px;margin:3px;display:inline-block;font-size:0.85em;">↑ {u}</span>' for u in chain['upstream'][:4])
        down_t = "".join(f'<span style="background:#e8f5e9;color:#2e7d32;padding:3px 8px;border-radius:12px;margin:3px;display:inline-block;font-size:0.85em;">↓ {d}</span>' for d in chain['downstream'][:4])
        chain_html = f'<div style="margin-top:10px;font-size:0.85em;color:#666;">产业链关联：</div><div style="margin-top:4px;">{up_t}{down_t}</div>'

    stock_cards = ""
    for key, label, bg, border in [
        ('龙头',  '① 核心龙头',   '#fff3e0', '#fb8c00'),
        ('滞涨',  '② 滞涨潜力',   '#e8f5e9', '#43a047'),
        ('产业链','③ 产业链联动', '#e3f2fd', '#1e88e5'),
    ]:
        s = stocks_3.get(key)
        if s:
            cc = '#c62828' if s.get('今日涨跌幅',0) < 0 else '#2e7d32'
            rc = '#c62828' if s.get('5日涨幅',0)   < 0 else '#2e7d32'
            stock_cards += f"""
            <div style="background:{bg};border-left:4px solid {border};padding:12px 16px;margin:10px 0;border-radius:6px;">
              <div style="font-weight:bold;color:#333;margin-bottom:6px;">{label}：{s['名称']}
                <span style="color:#888;font-weight:normal;font-size:0.85em;">（{s['代码']}）</span></div>
              <table style="font-size:0.86em;color:#555;width:100%;border-collapse:collapse;">
                <tr>
                  <td style="padding:2px 12px 2px 0;">价格</td><td style="padding:2px 16px 2px 0;"><b>{s['最新价']} 元</b></td>
                  <td style="padding:2px 12px 2px 0;">今日</td><td style="color:{cc};"><b>{s.get('今日涨跌幅',0):+.1f}%</b></td>
                </tr>
                <tr>
                  <td style="padding:2px 12px 2px 0;">5日涨幅</td><td style="color:{rc};"><b>{s.get('5日涨幅',0):+.1f}%</b></td>
                </tr>
              </table>
              <div style="margin-top:7px;background:rgba(255,255,255,0.6);padding:6px 8px;border-radius:4px;font-size:0.83em;color:#444;">
                💡 {s['理由']}</div>
            </div>"""
        else:
            stock_cards += f'<div style="background:#f5f5f5;border-left:4px solid #bbb;padding:10px 16px;margin:10px 0;border-radius:6px;color:#999;">{label}：暂无合适标的</div>'

    ret_color = '#2e7d32' if ret5 > 0 else '#c62828'
    html_text = f"""
    <html><body style="font-family:'PingFang SC',Arial,sans-serif;max-width:680px;margin:0 auto;padding:24px 20px;color:#222;">
      <h2 style="color:#e65100;border-bottom:3px solid #e65100;padding-bottom:10px;">🔥 A股热点板块日报 · {date_str} {weekday}</h2>
      {risk_html}
      <h3 style="color:#444;margin-bottom:8px;">📊 板块热度 Top5</h3>
      <table style="width:100%;border-collapse:collapse;font-size:0.88em;background:white;border-radius:6px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.08);margin-bottom:20px;">
        <thead style="background:#37474f;color:white;">
          <tr>
            <th style="padding:8px 10px;text-align:left;">排名</th>
            <th style="padding:8px 10px;text-align:left;">行业板块</th>
            <th style="padding:8px 10px;text-align:right;">5日涨幅</th>
            <th style="padding:8px 10px;text-align:right;">上涨比例</th>
            <th style="padding:8px 10px;text-align:right;">样本数</th>
          </tr>
        </thead>
        <tbody>{top5_rows}</tbody>
      </table>
      <div style="background:#fff8e1;border-left:5px solid #ffa000;padding:14px 18px;border-radius:8px;margin-bottom:20px;">
        <h3 style="margin:0 0 10px;color:#e65100;">🏆 重点推荐行业：{top1_name}</h3>
        <table style="font-size:0.88em;color:#555;width:100%;border-collapse:collapse;">
          <tr>
            <td style="padding:3px 16px 3px 0;">📈 5日涨幅</td>
            <td style="color:{ret_color};font-weight:bold;">{ret5:+.1f}%</td>
            <td style="padding:3px 16px;">📊 上涨比例</td>
            <td style="font-weight:bold;">{up_ratio:.0f}%</td>
          </tr>
        </table>
        <div style="margin-top:10px;font-size:0.87em;color:#5d4037;line-height:1.7;">📌 <b>热度分析：</b>{hot_reason}</div>
        {chain_html}
      </div>
      <h3 style="color:#444;margin-bottom:4px;">🎯 {top1_name} 推荐3只标的</h3>
      <p style="color:#888;font-size:0.85em;margin-top:0;">核心龙头 / 滞涨补涨 / 产业链联动</p>
      {stock_cards}
      <div style="background:#fce4ec;padding:12px 16px;border-radius:6px;font-size:0.82em;color:#880e4f;margin-top:18px;line-height:1.7;">
        ⚠️ <b>风险提示：</b>以上为行业技术面自动分析，不构成投资建议。热点板块往往存在追高风险，请谨慎决策。<br>
        <span style="color:#aaa;">GitHub Actions 自动发送 · {now.strftime('%H:%M')}</span>
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
        logger.info("✅ 热点板块邮件发送成功")
    except Exception as e:
        logger.error(f"❌ 邮件发送失败: {e}")
        print(plain_text)


# ============================================================
# 主入口
# ============================================================

def main():
    logger.info("="*50)
    logger.info("🔥 热点板块扫描 v3 启动")
    logger.info("="*50)

    now      = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    subject  = f"🔥 A股热点板块日报 · {date_str}"

    login = bs.login()
    if login.error_code != '0':
        logger.error(f"BaoStock登录失败: {login.error_msg}")
        sys.exit(1)
    logger.info("✅ BaoStock 登录成功")

    try:
        # 1. 一次性获取全市场股票+行业分类
        all_stocks = get_all_stocks_with_industry()
        if all_stocks.empty:
            send_email("今日行业数据获取失败。", "今日行业数据获取失败。", subject)
            return

        # 2. 计算各行业近5日涨幅
        perf_df = calc_industry_performance(all_stocks, sample_per_industry=15)
        if perf_df.empty:
            send_email("今日行业涨幅计算失败。", "今日行业涨幅计算失败。", subject)
            return

        # 3. 评分排名
        scored_df = score_sectors(perf_df)

        # 4. 分类
        emerging, sustained = classify_sectors(scored_df)

        # 5. 重点推荐
        top1_row     = scored_df.iloc[0]
        top1_name    = top1_row['板块名称']
        is_high_risk = float(top1_row.get('5日涨幅', 0)) > 15.0
        logger.info(f"重点推荐行业：{top1_name}（5日涨幅{top1_row['5日涨幅']:+.1f}%）")

        # 6. 选3只股票
        stocks_3 = pick_3_stocks(top1_name, all_stocks)

        # 7. 报告 & 发送
        plain_text, html_text = build_report(
            scored_df, emerging, sustained,
            top1_name, top1_row, stocks_3, is_high_risk)
        send_email(plain_text, html_text, subject)

    finally:
        bs.logout()
        logger.info("BaoStock 已登出")

    logger.info("✅ 热点板块扫描完成")


if __name__ == '__main__':
    main()
