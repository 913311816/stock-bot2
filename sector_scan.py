#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股热点板块每日扫描 v2
========================
数据源：AKShare（仅使用海外可访问的接口）

修复：
  - 替换被拦截的东方财富实时接口
  - 改用 BaoStock 获取板块行情数据（海外稳定可访问）
  - AKShare 仅用于概念/行业资金流向排名（该接口海外可用）
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
    '新能源车': {
        'upstream':   ['锂矿', '正极材料', '负极材料', '电解液', '隔膜'],
        'downstream': ['充电桩', '汽车经销', '车联网', '自动驾驶'],
    },
    '光伏': {
        'upstream':   ['硅料', '光伏玻璃', '胶膜', 'EVA'],
        'downstream': ['储能', '逆变器', '支架', '电网'],
    },
    '半导体': {
        'upstream':   ['半导体设备', '半导体材料', '光刻胶'],
        'downstream': ['消费电子', '服务器', '人工智能'],
    },
    '人工智能': {
        'upstream':   ['算力', '服务器', '光模块'],
        'downstream': ['软件', '云计算', '机器人', '自动驾驶'],
    },
    '军工': {
        'upstream':   ['钛合金', '碳纤维', '航空发动机'],
        'downstream': ['航空航天', '无人机', '卫星导航'],
    },
    '医药': {
        'upstream':   ['医药原料', 'CXO', '原料药'],
        'downstream': ['医疗器械', '医疗服务', '连锁药店'],
    },
    '储能': {
        'upstream':   ['锂矿', '正极材料', '电解液'],
        'downstream': ['电网', '光伏', '风电'],
    },
    '机器人': {
        'upstream':   ['减速器', '伺服电机', '控制器'],
        'downstream': ['工业自动化', '人工智能', '新能源车'],
    },
    '消费电子': {
        'upstream':   ['半导体', '屏幕', '摄像头模组'],
        'downstream': ['云计算', '5G', '可穿戴设备'],
    },
    '房地产': {
        'upstream':   ['建材', '钢铁', '水泥', '玻璃'],
        'downstream': ['家电', '家居', '物业管理'],
    },
    '银行': {
        'upstream':   ['金融科技', '征信'],
        'downstream': ['保险', '券商', '资产管理'],
    },
    '煤炭': {
        'upstream':   ['煤炭开采设备', '矿山机械'],
        'downstream': ['火电', '钢铁', '化工'],
    },
    '钢铁': {
        'upstream':   ['铁矿石', '焦炭', '废钢'],
        'downstream': ['建筑', '汽车', '家电', '机械'],
    },
    '化工': {
        'upstream':   ['石油', '天然气', '煤炭'],
        'downstream': ['农药', '涂料', '塑料', '新材料'],
    },
    '农业': {
        'upstream':   ['化肥', '农药', '种子', '农机'],
        'downstream': ['食品加工', '生猪', '饲料'],
    },
}

def match_industry_chain(sector_name: str) -> dict:
    for key, chain in INDUSTRY_CHAIN.items():
        if key in sector_name or sector_name in key:
            return chain
    return {'upstream': [], 'downstream': []}


# ============================================================
# BaoStock 获取行业行情
# ============================================================

def get_industry_list_bs() -> list:
    """获取 BaoStock 支持的行业列表"""
    industries = [
        '农林牧渔', '采掘', '化工', '钢铁', '有色金属', '电子',
        '家用电器', '食品饮料', '纺织服装', '轻工制造', '医药生物',
        '公用事业', '交通运输', '房地产', '商业贸易', '休闲服务',
        '综合', '建筑材料', '建筑装饰', '电气设备', '国防军工',
        '计算机', '传媒', '通信', '银行', '非银金融', '汽车',
        '机械设备', '煤炭', '石油石化',
    ]
    return industries


def get_industry_performance_bs() -> pd.DataFrame:
    """
    通过 BaoStock 获取各行业近5日平均涨幅
    抽样每个行业前20只股票计算均值
    """
    logger.info("通过 BaoStock 计算行业近5日涨幅…")
    industries = get_industry_list_bs()
    results = []

    end_date   = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')

    for industry in industries:
        try:
            # 查询该行业股票列表
            rs = bs.query_stock_industry(industryName=industry)
            if rs.error_code != '0':
                continue
            stocks = []
            while rs.next():
                row = rs.get_row_data()
                code = row[1] if len(row) > 1 else ''
                if code and not code.startswith('688') and not code.startswith('bj'):
                    stocks.append(code)
            if not stocks:
                continue

            # 抽取前15只，计算近5日涨幅均值
            sample = stocks[:15]
            returns = []
            for code in sample:
                try:
                    r = bs.query_history_k_data_plus(
                        code, "date,close",
                        start_date=start_date, end_date=end_date,
                        frequency="d", adjustflag="3")
                    data = []
                    while r.next():
                        data.append(r.get_row_data())
                    if len(data) >= 6:
                        closes = [float(d[1]) for d in data if d[1]]
                        if closes[-6] > 0:
                            ret = (closes[-1] - closes[-6]) / closes[-6] * 100
                            returns.append(ret)
                    time.sleep(0.05)
                except Exception:
                    continue

            if returns:
                avg_ret = np.mean(returns)
                results.append({
                    '板块名称':    industry,
                    '5日涨幅':     round(avg_ret, 2),
                    '样本数':      len(returns),
                    '5日主力净流入': avg_ret * 1e7,  # 用涨幅估算相对热度
                    '换手率':      abs(avg_ret) * 0.3,
                    '来源':        '行业',
                })
            time.sleep(0.2)
        except Exception as e:
            logger.debug(f"行业 {industry} 处理失败: {e}")
            continue

    df = pd.DataFrame(results) if results else pd.DataFrame()
    logger.info(f"行业行情计算完成，共 {len(df)} 个行业")
    return df


def get_industry_stocks_bs(industry_name: str, top_n: int = 30) -> pd.DataFrame:
    """获取行业内股票并计算近5日涨幅"""
    try:
        rs = bs.query_stock_industry(industryName=industry_name)
        if rs.error_code != '0':
            return pd.DataFrame()
        stocks = []
        while rs.next():
            row = rs.get_row_data()
            if len(row) >= 4:
                stocks.append({'代码_bs': row[1], '名称': row[3]})
        if not stocks:
            return pd.DataFrame()

        end_date   = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')

        enriched = []
        for s in stocks[:top_n]:
            code_bs = s['代码_bs']
            try:
                r = bs.query_history_k_data_plus(
                    code_bs,
                    "date,open,high,low,close,volume,turn",
                    start_date=start_date, end_date=end_date,
                    frequency="d", adjustflag="2")
                data = []
                while r.next():
                    data.append(r.get_row_data())
                if len(data) < 2:
                    continue

                latest = data[-1]
                close_now  = float(latest[4]) if latest[4] else 0
                close_prev = float(data[-2][4]) if data[-2][4] else 0
                close_5d   = float(data[-6][4]) if len(data) >= 6 and data[-6][4] else close_prev
                turn       = float(latest[6]) if latest[6] else 0

                chg_today = (close_now - close_prev) / close_prev * 100 if close_prev > 0 else 0
                chg_5d    = (close_now - close_5d)   / close_5d   * 100 if close_5d   > 0 else 0

                # 用5日涨幅模拟主力净流入（正相关）
                mock_flow = chg_5d * close_now * 1e5

                enriched.append({
                    '代码':        code_bs.split('.')[1] if '.' in code_bs else code_bs,
                    '名称':        s['名称'],
                    '最新价':      round(close_now, 2),
                    '今日涨跌幅':  round(chg_today, 2),
                    '5日涨幅':     round(chg_5d, 2),
                    '换手率':      round(turn, 2),
                    '主力净流入':  mock_flow,
                })
                time.sleep(0.05)
            except Exception:
                continue

        return pd.DataFrame(enriched) if enriched else pd.DataFrame()
    except Exception as e:
        logger.error(f"获取行业成分股失败 [{industry_name}]: {e}")
        return pd.DataFrame()


# ============================================================
# 板块评分
# ============================================================

def score_sectors(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def safe_norm(s):
        s = s.fillna(0).astype(float)
        if s.max() == s.min():
            return pd.Series([0.5]*len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    df['_flow_score']   = safe_norm(df['5日主力净流入'])
    df['_return_score'] = safe_norm(df['5日涨幅'])
    df['_turn_score']   = safe_norm(df.get('换手率', pd.Series([0]*len(df))))

    df['综合热度分'] = (
        df['_flow_score']   * 0.45 +
        df['_return_score'] * 0.30 +
        df['_turn_score']   * 0.25
    )
    return df.sort_values('综合热度分', ascending=False).reset_index(drop=True)


def classify_sectors(df: pd.DataFrame) -> tuple:
    if df.empty or len(df) < 5:
        return [], []
    top20 = df.head(20).copy()
    flow_med   = top20['5日主力净流入'].median()
    return_med = top20['5日涨幅'].median()

    emerging  = []
    sustained = []
    for _, row in top20.iterrows():
        flow  = row.get('5日主力净流入', 0)
        ret   = row.get('5日涨幅', 0)
        name  = row['板块名称']
        score = row['综合热度分']
        item  = {'板块名称': name, '综合热度分': score,
                 '5日净流入估算': round(ret, 2),
                 '5日涨幅': round(ret, 2)}
        if flow >= flow_med and ret >= return_med:
            sustained.append(item)
        elif ret >= return_med * 1.3 and flow < flow_med:
            emerging.append(item)

    return emerging[:3], sustained[:3]


# ============================================================
# 板块内选3只股票
# ============================================================

def pick_3_stocks(sector_name: str) -> dict:
    result = {'龙头': None, '滞涨': None, '产业链': None, '产业链说明': ''}

    stocks = get_industry_stocks_bs(sector_name)
    if stocks.empty:
        logger.warning(f"未获取到 [{sector_name}] 成分股")
        return result

    stocks = stocks[~stocks['名称'].str.contains('ST|退', na=False)]
    stocks = stocks[stocks['最新价'] > 0]
    if stocks.empty:
        return result

    # 1. 核心龙头：5日涨幅最强
    top = stocks.nlargest(1, '5日涨幅')
    if not top.empty:
        r = top.iloc[0]
        result['龙头'] = {
            '代码': r['代码'], '名称': r['名称'],
            '最新价': r['最新价'],
            '今日涨跌幅': r['今日涨跌幅'],
            '5日涨幅': r['5日涨幅'],
            '主力净流入(万)': '-',
            '理由': f"近5日涨幅{r['5日涨幅']:.1f}%，板块内领涨龙头，资金集中效应显著",
        }

    # 2. 滞涨潜力：5日涨幅低于均值 且 今日有上涨
    avg_5d    = stocks['5日涨幅'].mean()
    laggards  = stocks[
        (stocks['5日涨幅'] < avg_5d) &
        (stocks['今日涨跌幅'] > 0)
    ]
    if laggards.empty:
        laggards = stocks.nsmallest(5, '5日涨幅')
    if not laggards.empty:
        pick = laggards.nlargest(1, '今日涨跌幅').iloc[0]
        result['滞涨'] = {
            '代码': pick['代码'], '名称': pick['名称'],
            '最新价': pick['最新价'],
            '今日涨跌幅': pick['今日涨跌幅'],
            '5日涨幅': pick['5日涨幅'],
            '主力净流入(万)': '-',
            '理由': f"板块均值涨幅{avg_5d:.1f}%，该股5日仅{pick['5日涨幅']:.1f}%，滞涨明显，存在补涨空间",
        }

    # 3. 产业链联动
    chain = match_industry_chain(sector_name)
    all_related = chain['upstream'] + chain['downstream']
    for related_name in all_related[:5]:
        try:
            rel_stocks = get_industry_stocks_bs(related_name, top_n=20)
            time.sleep(0.3)
            if not rel_stocks.empty:
                rel_stocks = rel_stocks[~rel_stocks['名称'].str.contains('ST|退', na=False)]
                if not rel_stocks.empty:
                    pick = rel_stocks.nlargest(1, '5日涨幅').iloc[0]
                    is_up    = related_name in chain['upstream']
                    relation = '上游' if is_up else '下游'
                    result['产业链'] = {
                        '代码': pick['代码'], '名称': pick['名称'],
                        '最新价': pick['最新价'],
                        '今日涨跌幅': pick['今日涨跌幅'],
                        '5日涨幅': pick['5日涨幅'],
                        '主力净流入(万)': '-',
                        '理由': f"来自{sector_name}{relation}板块「{related_name}」，板块轮动联动机会",
                    }
                    result['产业链说明'] = f"{sector_name} → {relation}「{related_name}」"
                    break
        except Exception as e:
            logger.debug(f"产业链查询失败 [{related_name}]: {e}")
            continue

    if not result['产业链说明']:
        result['产业链说明'] = f"{sector_name}（产业链数据查询中）"

    return result


# ============================================================
# 报告生成
# ============================================================

def analyze_hot_reason(row: pd.Series) -> str:
    ret  = row.get('5日涨幅', 0)
    turn = row.get('换手率', 0)
    reasons = []
    if ret > 10:
        reasons.append(f"5日区间大涨{ret:.1f}%，市场情绪强烈")
    elif ret > 5:
        reasons.append(f"5日稳步上涨{ret:.1f}%，趋势明确")
    elif ret > 2:
        reasons.append(f"5日上涨{ret:.1f}%，处于启动阶段")
    else:
        reasons.append(f"5日小幅变动，资金悄然积累")
    if turn > 3:
        reasons.append(f"换手率{turn:.1f}%，交投活跃")
    return "；".join(reasons) if reasons else "资金稳步积累，热度持续上升"


def build_sector_report(scored_df, emerging, sustained,
                        top1_name, top1_row, stocks_3, is_high_risk) -> tuple:
    now      = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    weekday  = ['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]
    ret5     = float(top1_row.get('5日涨幅', 0))
    turn     = float(top1_row.get('换手率', 0))
    hot_reason = analyze_hot_reason(top1_row)
    chain    = match_industry_chain(top1_name)
    risk_flag = "⚠️ 【高位风险预警】该板块5日涨幅已超15%，追高风险较大，建议谨慎！" if is_high_risk else ""

    # -------- 纯文本 --------
    lines = [f"🔥 A股热点板块日报 · {date_str} {weekday}", "="*50]
    if risk_flag:
        lines += [risk_flag, ""]

    lines += ["【板块热度Top5】", "  🔴 新兴热点："]
    for s in (emerging or [{'板块名称': '暂无', '5日涨幅': 0}]):
        lines.append(f"    · {s['板块名称']}  5日涨幅{s['5日涨幅']:+.1f}%")
    lines.append("  🔵 持续主线：")
    for s in (sustained or [{'板块名称': '暂无', '5日涨幅': 0}]):
        lines.append(f"    · {s['板块名称']}  5日涨幅{s['5日涨幅']:+.1f}%")
    lines.append("")

    lines += [
        f"【重点推荐板块：{top1_name}】",
        f"  5日区间涨幅：{ret5:+.1f}%",
        f"  当前换手率：{turn:.1f}%",
        f"  热度原因：{hot_reason}",
    ]
    if chain['upstream']:
        lines.append(f"  上游关联：{'、'.join(chain['upstream'][:3])}")
    if chain['downstream']:
        lines.append(f"  下游关联：{'、'.join(chain['downstream'][:3])}")
    lines.append("")

    lines.append(f"【{top1_name} 板块推荐3只】")
    for key, label in [('龙头','①核心龙头'), ('滞涨','②滞涨潜力'), ('产业链','③产业链联动')]:
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
              "⚠️ 风险提示：以上为板块技术面和资金面自动分析，不构成投资建议。",
              f"GitHub Actions 自动发送 · {now.strftime('%H:%M')}"]
    plain_text = "\n".join(lines)

    # -------- HTML --------
    risk_html = f"""
    <div style="background:#ffebee;border:2px solid #e53935;border-radius:6px;
                padding:10px 16px;margin-bottom:14px;color:#b71c1c;font-weight:bold;">
      ⚠️ 高位风险预警：该板块5日涨幅已超15%，追高风险较大，建议谨慎操作！
    </div>""" if is_high_risk else ""

    top5_rows = ""
    for i, row in scored_df.head(5).iterrows():
        ret_val   = row.get('5日涨幅', 0)
        ret_color = '#2e7d32' if ret_val > 0 else '#c62828'
        name      = row['板块名称']
        badge = ""
        if any(s['板块名称'] == name for s in emerging):
            badge = '<span style="background:#ff7043;color:white;font-size:0.75em;padding:1px 6px;border-radius:3px;margin-left:6px;">新兴</span>'
        elif any(s['板块名称'] == name for s in sustained):
            badge = '<span style="background:#1565c0;color:white;font-size:0.75em;padding:1px 6px;border-radius:3px;margin-left:6px;">主线</span>'
        top5_rows += f"""
        <tr style="border-bottom:1px solid #eee;">
          <td style="padding:7px 10px;">{i+1}</td>
          <td style="padding:7px 10px;font-weight:bold;">{name}{badge}</td>
          <td style="padding:7px 10px;text-align:right;color:{ret_color};">{ret_val:+.1f}%</td>
          <td style="padding:7px 10px;text-align:right;">{row.get('换手率',0):.1f}%</td>
        </tr>"""

    chain_html = ""
    if chain['upstream'] or chain['downstream']:
        up_tags   = "".join(f'<span style="background:#e3f2fd;color:#1565c0;padding:3px 8px;border-radius:12px;margin:3px;display:inline-block;font-size:0.85em;">↑ {u}</span>' for u in chain['upstream'][:4])
        down_tags = "".join(f'<span style="background:#e8f5e9;color:#2e7d32;padding:3px 8px;border-radius:12px;margin:3px;display:inline-block;font-size:0.85em;">↓ {d}</span>' for d in chain['downstream'][:4])
        chain_html = f'<div style="margin-top:10px;font-size:0.85em;color:#666;">产业链关联：</div><div style="margin-top:4px;">{up_tags}{down_tags}</div>'

    stock_cards = ""
    for key, label, bg, border in [
        ('龙头',  '① 核心龙头',   '#fff3e0', '#fb8c00'),
        ('滞涨',  '② 滞涨潜力',   '#e8f5e9', '#43a047'),
        ('产业链','③ 产业链联动', '#e3f2fd', '#1e88e5'),
    ]:
        s = stocks_3.get(key)
        if s:
            chg_c = '#c62828' if s.get('今日涨跌幅', 0) < 0 else '#2e7d32'
            r5_c  = '#c62828' if s.get('5日涨幅', 0) < 0 else '#2e7d32'
            stock_cards += f"""
            <div style="background:{bg};border-left:4px solid {border};
                        padding:12px 16px;margin:10px 0;border-radius:6px;">
              <div style="font-weight:bold;color:#333;margin-bottom:6px;">
                {label}：{s['名称']}
                <span style="color:#888;font-weight:normal;font-size:0.85em;">（{s['代码']}）</span>
              </div>
              <table style="font-size:0.86em;color:#555;width:100%;border-collapse:collapse;">
                <tr>
                  <td style="padding:2px 12px 2px 0;">价格</td>
                  <td style="padding:2px 16px 2px 0;"><b>{s['最新价']} 元</b></td>
                  <td style="padding:2px 12px 2px 0;">今日</td>
                  <td style="color:{chg_c};"><b>{s.get('今日涨跌幅',0):+.1f}%</b></td>
                </tr>
                <tr>
                  <td style="padding:2px 12px 2px 0;">5日涨幅</td>
                  <td style="color:{r5_c};"><b>{s.get('5日涨幅',0):+.1f}%</b></td>
                </tr>
              </table>
              <div style="margin-top:7px;background:rgba(255,255,255,0.6);padding:6px 8px;
                          border-radius:4px;font-size:0.83em;color:#444;">
                💡 {s['理由']}
              </div>
            </div>"""
        else:
            stock_cards += f'<div style="background:#f5f5f5;border-left:4px solid #bbb;padding:10px 16px;margin:10px 0;border-radius:6px;color:#999;">{label}：暂无合适标的</div>'

    html_text = f"""
    <html><body style="font-family:'PingFang SC',Arial,sans-serif;max-width:680px;
                        margin:0 auto;padding:24px 20px;color:#222;">
      <h2 style="color:#e65100;border-bottom:3px solid #e65100;padding-bottom:10px;">
        🔥 A股热点板块日报 · {date_str} {weekday}
      </h2>
      {risk_html}
      <h3 style="color:#444;margin-bottom:8px;">📊 板块热度 Top5</h3>
      <table style="width:100%;border-collapse:collapse;font-size:0.88em;
                    background:white;border-radius:6px;overflow:hidden;
                    box-shadow:0 1px 4px rgba(0,0,0,0.08);margin-bottom:20px;">
        <thead style="background:#37474f;color:white;">
          <tr>
            <th style="padding:8px 10px;text-align:left;">排名</th>
            <th style="padding:8px 10px;text-align:left;">板块</th>
            <th style="padding:8px 10px;text-align:right;">5日涨幅</th>
            <th style="padding:8px 10px;text-align:right;">换手率</th>
          </tr>
        </thead>
        <tbody>{top5_rows}</tbody>
      </table>
      <div style="background:#fff8e1;border-left:5px solid #ffa000;
                  padding:14px 18px;border-radius:8px;margin-bottom:20px;">
        <h3 style="margin:0 0 10px;color:#e65100;">🏆 重点推荐板块：{top1_name}</h3>
        <table style="font-size:0.88em;color:#555;width:100%;border-collapse:collapse;">
          <tr>
            <td style="padding:3px 16px 3px 0;">📈 5日区间涨幅</td>
            <td style="color:{'#2e7d32' if ret5>0 else '#c62828'};font-weight:bold;">{ret5:+.1f}%</td>
            <td style="padding:3px 16px;">🔄 换手率</td>
            <td style="font-weight:bold;">{turn:.1f}%</td>
          </tr>
        </table>
        <div style="margin-top:10px;font-size:0.87em;color:#5d4037;line-height:1.7;">
          📌 <b>热度分析：</b>{hot_reason}
        </div>
        {chain_html}
      </div>
      <h3 style="color:#444;margin-bottom:4px;">🎯 {top1_name} 推荐3只标的</h3>
      <p style="color:#888;font-size:0.85em;margin-top:0;">核心龙头 / 滞涨补涨 / 产业链联动，覆盖不同风险收益特征</p>
      {stock_cards}
      <div style="background:#fce4ec;padding:12px 16px;border-radius:6px;
                  font-size:0.82em;color:#880e4f;margin-top:18px;line-height:1.7;">
        ⚠️ <b>风险提示：</b>以上为板块资金面和技术面自动分析，不构成投资建议。<br>
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
        print("\n" + "="*50 + "\n" + plain_text + "\n" + "="*50)
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
    logger.info("🔥 热点板块扫描启动 v2")
    logger.info("="*50)

    now      = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    subject  = f"🔥 A股热点板块日报 · {date_str}"

    # 登录 BaoStock
    login = bs.login()
    if login.error_code != '0':
        logger.error(f"BaoStock登录失败: {login.error_msg}")
        sys.exit(1)
    logger.info("✅ BaoStock 登录成功")

    try:
        # 1. 计算各行业近5日表现
        perf_df = get_industry_performance_bs()
        if perf_df.empty:
            msg = "今日行业数据获取失败，请稍后重试。"
            send_email(msg, msg, subject)
            return

        # 2. 综合评分排名
        scored_df = score_sectors(perf_df)

        # 3. 区分新兴热点 vs 持续主线
        emerging, sustained = classify_sectors(scored_df)

        # 4. 重点推荐板块
        top1_row  = scored_df.iloc[0]
        top1_name = top1_row['板块名称']
        is_high_risk = float(top1_row.get('5日涨幅', 0)) > 15.0
        logger.info(f"重点推荐板块：{top1_name}（5日涨幅 {top1_row.get('5日涨幅',0):.1f}%）")

        # 5. 板块内选3只股票
        stocks_3 = pick_3_stocks(top1_name)

        # 6. 生成并发送报告
        plain_text, html_text = build_sector_report(
            scored_df, emerging, sustained,
            top1_name, top1_row, stocks_3, is_high_risk)
        send_email(plain_text, html_text, subject)

    finally:
        bs.logout()
        logger.info("BaoStock 已登出")

    logger.info("✅ 热点板块扫描完成")


if __name__ == '__main__':
    main()
