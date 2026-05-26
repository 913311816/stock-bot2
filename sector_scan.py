#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股热点板块每日扫描 · AKShare版
=================================
数据源：AKShare（东方财富板块接口，海外服务器可访问）

输出内容：
  第一部分：板块热度Top5（区分「新兴热点」和「持续主线」）
  第二部分：重点推荐1个板块及热度原因分析
  第三部分：该板块推荐3只股票
            - 核心龙头（板块内最强）
            - 滞涨潜力（还没跟涨的）
            - 产业链联动（上下游关联）
  第四部分：高位风险预警（5日涨幅>15%标注警示）
"""

import os, sys, time, logging, smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import akshare as ak
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


# ============================================================
# 产业链关系表（上下游映射）
# ============================================================
# 格式：{ 板块关键词: { 'upstream': [...], 'downstream': [...] } }
INDUSTRY_CHAIN = {
    '新能源车': {
        'upstream':   ['锂矿', '正极材料', '负极材料', '电解液', '隔膜', '钴'],
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
        'upstream':   ['算力', '服务器', '光模块', 'GPU'],
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
    """模糊匹配产业链关系"""
    for key, chain in INDUSTRY_CHAIN.items():
        if key in sector_name or sector_name in key:
            return chain
    return {'upstream': [], 'downstream': []}


# ============================================================
# AKShare 数据获取
# ============================================================

def get_industry_fund_flow() -> pd.DataFrame:
    """获取行业板块5日资金流向排名"""
    logger.info("获取行业板块5日资金流向…")
    try:
        df = ak.stock_board_industry_fund_flow_rank(symbol="5日")
        df = df.rename(columns={
            '名称': '板块名称',
            '5日主力净流入-净额': '5日主力净流入',
            '5日主力净流入-净占比': '5日净占比',
        })
        # 兼容不同版本列名
        if '5日主力净流入' not in df.columns:
            for col in df.columns:
                if '净额' in col or '净流入' in col:
                    df = df.rename(columns={col: '5日主力净流入'})
                    break
        df['5日主力净流入'] = pd.to_numeric(
            df.get('5日主力净流入', 0), errors='coerce').fillna(0)
        return df
    except Exception as e:
        logger.error(f"行业资金流向获取失败: {e}")
        return pd.DataFrame()


def get_industry_spot() -> pd.DataFrame:
    """获取行业板块实时行情（含5日涨幅、换手率等）"""
    logger.info("获取行业板块实时行情…")
    try:
        df = ak.stock_board_industry_spot_em()
        df = df.rename(columns={
            '板块名称': '板块名称',
            '5日涨跌': '5日涨幅',
            '换手率': '换手率',
            '总市值': '总市值',
            '上涨家数': '上涨家数',
            '下跌家数': '下跌家数',
        })
        for col in ['5日涨幅', '换手率']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        logger.error(f"行业实时行情获取失败: {e}")
        return pd.DataFrame()


def get_concept_fund_flow() -> pd.DataFrame:
    """获取概念板块5日资金流向（补充行业板块）"""
    logger.info("获取概念板块5日资金流向…")
    try:
        df = ak.stock_board_concept_fund_flow_rank(symbol="5日")
        df = df.rename(columns={'名称': '板块名称'})
        for col in df.columns:
            if '净额' in col or '净流入' in col:
                df = df.rename(columns={col: '5日主力净流入'})
                break
        df['5日主力净流入'] = pd.to_numeric(
            df.get('5日主力净流入', 0), errors='coerce').fillna(0)
        df['来源'] = '概念'
        return df[['板块名称', '5日主力净流入', '来源']].head(30)
    except Exception as e:
        logger.error(f"概念资金流向获取失败: {e}")
        return pd.DataFrame()


def get_sector_stocks(sector_name: str, is_concept: bool = False) -> pd.DataFrame:
    """获取板块内所有成分股"""
    try:
        if is_concept:
            df = ak.stock_board_concept_cons_em(symbol=sector_name)
        else:
            df = ak.stock_board_industry_cons_em(symbol=sector_name)
        if df is None or df.empty:
            return pd.DataFrame()
        # 统一列名
        df = df.rename(columns={
            '代码': '代码', '名称': '名称',
            '最新价': '最新价', '涨跌幅': '今日涨跌幅',
            '5日涨跌': '5日涨幅', '换手率': '换手率',
            '成交额': '成交额', '总市值': '总市值',
            '主力净流入': '主力净流入',
        })
        for col in ['今日涨跌幅', '5日涨幅', '换手率', '主力净流入', '总市值']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        logger.error(f"获取板块成分股失败 [{sector_name}]: {e}")
        return pd.DataFrame()


# ============================================================
# 板块评分逻辑
# ============================================================

def score_sectors(fund_df: pd.DataFrame, spot_df: pd.DataFrame) -> pd.DataFrame:
    """
    综合评分（行业板块）：
      5日主力净流入  45%
      5日涨幅        30%
      换手率         25%
    """
    if fund_df.empty:
        return pd.DataFrame()

    df = fund_df.copy()
    df['来源'] = df.get('来源', '行业')

    # 合并实时行情
    if not spot_df.empty and '板块名称' in spot_df.columns:
        merge_cols = ['板块名称']
        for col in ['5日涨幅', '换手率', '上涨家数', '下跌家数']:
            if col in spot_df.columns:
                merge_cols.append(col)
        df = df.merge(spot_df[merge_cols], on='板块名称', how='left')

    for col in ['5日涨幅', '换手率']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0)

    def safe_norm(s):
        s = s.fillna(0).astype(float)
        if s.max() == s.min():
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    df['_flow_score']  = safe_norm(df['5日主力净流入'])
    df['_return_score'] = safe_norm(df['5日涨幅'])
    df['_turn_score']  = safe_norm(df['换手率'])

    df['综合热度分'] = (
        df['_flow_score']   * 0.45 +
        df['_return_score'] * 0.30 +
        df['_turn_score']   * 0.25
    )
    df = df.sort_values('综合热度分', ascending=False).reset_index(drop=True)
    return df


def classify_sectors(df: pd.DataFrame) -> tuple:
    """
    区分「新兴热点」和「持续主线」：
      - 新兴热点：5日涨幅高 但 资金流入排名相对靠后（刚爆发）
      - 持续主线：5日资金净流入持续居前（连续强势）
    """
    if df.empty:
        return [], []

    top20 = df.head(20).copy()
    flow_median  = top20['5日主力净流入'].median()
    return_median = top20.get('5日涨幅', pd.Series([0]*len(top20))).median()

    emerging = []   # 新兴热点：涨幅高 but 流入排名相对靠后
    sustained = []  # 持续主线：流入持续居前

    for _, row in top20.iterrows():
        flow   = row.get('5日主力净流入', 0)
        ret    = row.get('5日涨幅', 0)
        name   = row['板块名称']
        score  = row['综合热度分']

        if flow >= flow_median and ret >= return_median:
            sustained.append({'板块名称': name, '综合热度分': score,
                               '5日净流入(亿)': round(flow / 1e8, 2),
                               '5日涨幅': round(ret, 2)})
        elif ret >= return_median * 1.3 and flow < flow_median:
            emerging.append({'板块名称': name, '综合热度分': score,
                              '5日净流入(亿)': round(flow / 1e8, 2),
                              '5日涨幅': round(ret, 2)})

    return emerging[:3], sustained[:3]


# ============================================================
# 板块内股票推荐
# ============================================================

def pick_3_stocks(sector_name: str, is_concept: bool = False) -> dict:
    """
    从板块内挑选3只代表股：
      1. 核心龙头：主力净流入最高 且 今日涨幅前列
      2. 滞涨潜力：5日涨幅低于板块均值 但 今日有净流入
      3. 产业链联动：匹配上下游板块，取各上下游板块资金流最高的1只
    """
    result = {'龙头': None, '滞涨': None, '产业链': None, '产业链说明': ''}

    stocks = get_sector_stocks(sector_name, is_concept)
    if stocks.empty:
        logger.warning(f"未能获取 [{sector_name}] 成分股")
        return result

    # 过滤ST和停牌
    stocks = stocks[~stocks['名称'].str.contains('ST|退', na=False)]
    stocks = stocks[stocks.get('最新价', pd.Series([1]*len(stocks))) > 0]

    if stocks.empty:
        return result

    # 1. 核心龙头：主力净流入最高
    if '主力净流入' in stocks.columns:
        top_flow = stocks.nlargest(1, '主力净流入')
    else:
        top_flow = stocks.nlargest(1, '总市值') if '总市值' in stocks.columns else stocks.head(1)

    if not top_flow.empty:
        r = top_flow.iloc[0]
        result['龙头'] = {
            '代码': r.get('代码', ''), '名称': r.get('名称', ''),
            '最新价': r.get('最新价', '-'),
            '今日涨跌幅': r.get('今日涨跌幅', 0),
            '5日涨幅': r.get('5日涨幅', 0),
            '主力净流入(万)': round(r.get('主力净流入', 0) / 1e4, 0) if '主力净流入' in stocks.columns else '-',
            '理由': '板块内主力资金流入最强的核心标的，龙头效应显著',
        }

    # 2. 滞涨潜力：5日涨幅低于均值 且 主力净流入为正
    if '5日涨幅' in stocks.columns:
        avg_5d = stocks['5日涨幅'].mean()
        laggards = stocks[
            (stocks['5日涨幅'] < avg_5d) &
            (stocks.get('主力净流入', pd.Series([0]*len(stocks))) > 0)
        ]
        if laggards.empty:
            # 退而求其次：5日涨幅最低的股
            laggards = stocks.nsmallest(3, '5日涨幅')

        if not laggards.empty:
            # 从滞涨股里选今日涨幅最好的（开始启动迹象）
            col = '今日涨跌幅' if '今日涨跌幅' in laggards.columns else laggards.columns[0]
            pick = laggards.nlargest(1, col).iloc[0]
            result['滞涨'] = {
                '代码': pick.get('代码', ''), '名称': pick.get('名称', ''),
                '最新价': pick.get('最新价', '-'),
                '今日涨跌幅': pick.get('今日涨跌幅', 0),
                '5日涨幅': pick.get('5日涨幅', 0),
                '主力净流入(万)': round(pick.get('主力净流入', 0) / 1e4, 0) if '主力净流入' in stocks.columns else '-',
                '理由': f"板块平均5日涨幅{avg_5d:.1f}%，该股仅{pick.get('5日涨幅',0):.1f}%，属于板块内滞涨股，存在补涨潜力",
            }

    # 3. 产业链联动：查找上下游板块最强股
    chain = match_industry_chain(sector_name)
    all_related = chain['upstream'] + chain['downstream']

    if all_related:
        # 尝试每个上下游板块，找到第一个能获取到成分股的
        for related_name in all_related[:6]:
            try:
                rel_stocks = get_sector_stocks(related_name, is_concept=False)
                time.sleep(0.5)
                if rel_stocks.empty:
                    rel_stocks = get_sector_stocks(related_name, is_concept=True)
                if not rel_stocks.empty:
                    rel_stocks = rel_stocks[~rel_stocks['名称'].str.contains('ST|退', na=False)]
                    if '主力净流入' in rel_stocks.columns:
                        pick = rel_stocks.nlargest(1, '主力净流入').iloc[0]
                    elif '总市值' in rel_stocks.columns:
                        pick = rel_stocks.nlargest(1, '总市值').iloc[0]
                    else:
                        pick = rel_stocks.iloc[0]

                    is_up = related_name in chain['upstream']
                    relation = '上游' if is_up else '下游'
                    result['产业链'] = {
                        '代码': pick.get('代码', ''), '名称': pick.get('名称', ''),
                        '最新价': pick.get('最新价', '-'),
                        '今日涨跌幅': pick.get('今日涨跌幅', 0),
                        '5日涨幅': pick.get('5日涨幅', 0),
                        '主力净流入(万)': round(pick.get('主力净流入', 0) / 1e4, 0) if '主力净流入' in rel_stocks.columns else '-',
                        '理由': f"来自{sector_name}的{relation}板块「{related_name}」，板块联动效应下存在跟涨机会",
                    }
                    result['产业链说明'] = f"{sector_name} → {relation}「{related_name}」"
                    break
            except Exception as e:
                logger.debug(f"产业链查询失败 [{related_name}]: {e}")
                continue
    else:
        result['产业链说明'] = f"{sector_name}（暂未收录产业链数据）"

    return result


# ============================================================
# 报告生成
# ============================================================

def analyze_hot_reason(sector_row: pd.Series) -> str:
    """根据数据推断板块热度原因"""
    flow  = sector_row.get('5日主力净流入', 0)
    ret   = sector_row.get('5日涨幅', 0)
    turn  = sector_row.get('换手率', 0)

    reasons = []
    if flow > 5e8:
        reasons.append(f"5日主力净流入高达{flow/1e8:.1f}亿元，机构资金大规模进场")
    elif flow > 1e8:
        reasons.append(f"5日主力净流入{flow/1e8:.1f}亿元，资金关注度持续提升")
    else:
        reasons.append(f"资金开始流入（5日净额{flow/1e8:.2f}亿元），处于建仓阶段")

    if ret > 10:
        reasons.append(f"5日区间涨幅达{ret:.1f}%，市场情绪高涨")
    elif ret > 5:
        reasons.append(f"5日稳步上涨{ret:.1f}%，趋势明确")
    elif ret > 0:
        reasons.append(f"5日小幅上涨{ret:.1f}%，处于启动初期")

    if turn > 5:
        reasons.append(f"换手率{turn:.1f}%，市场交投非常活跃")
    elif turn > 2:
        reasons.append(f"换手率{turn:.1f}%，交投相对活跃")

    return "；".join(reasons) if reasons else "资金流入稳定，热度持续积累"


def build_sector_report(
    top5_df: pd.DataFrame,
    emerging: list,
    sustained: list,
    top1_name: str,
    top1_row: pd.Series,
    stocks_3: dict,
    is_high_risk: bool,
) -> tuple:
    now      = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    weekday  = ['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]

    flow5  = top1_row.get('5日主力净流入', 0)
    ret5   = top1_row.get('5日涨幅', 0)
    turn   = top1_row.get('换手率', 0)
    hot_reason = analyze_hot_reason(top1_row)
    chain  = match_industry_chain(top1_name)

    risk_flag = "⚠️ 【高位风险预警】该板块5日涨幅已超15%，追高风险较大，建议谨慎！" if is_high_risk else ""

    # -------- 纯文本 --------
    lines = [
        f"🔥 A股热点板块日报 · {date_str} {weekday}",
        "="*50,
    ]

    if risk_flag:
        lines += [risk_flag, ""]

    # Part1: Top5热度
    lines += ["【板块热度Top5】", ""]
    lines.append("  🔴 新兴热点（近期刚爆发）：")
    if emerging:
        for s in emerging:
            lines.append(f"    · {s['板块名称']}  5日涨幅{s['5日涨幅']:+.1f}%  净流入{s['5日净流入(亿)']}亿")
    else:
        lines.append("    暂无明显新兴热点")

    lines.append("  🔵 持续主线（资金持续流入）：")
    if sustained:
        for s in sustained:
            lines.append(f"    · {s['板块名称']}  5日涨幅{s['5日涨幅']:+.1f}%  净流入{s['5日净流入(亿)']}亿")
    else:
        lines.append("    暂无明显持续主线")
    lines.append("")

    # Part2: 重点推荐板块
    lines += [
        f"【重点推荐板块：{top1_name}】",
        f"  5日主力净流入：{flow5/1e8:.2f}亿元",
        f"  5日区间涨幅：{ret5:+.1f}%",
        f"  当前换手率：{turn:.1f}%",
        f"  热度原因：{hot_reason}",
    ]
    if chain['upstream']:
        lines.append(f"  上游关联：{'、'.join(chain['upstream'][:3])}")
    if chain['downstream']:
        lines.append(f"  下游关联：{'、'.join(chain['downstream'][:3])}")
    lines.append("")

    # Part3: 3只推荐股票
    lines.append(f"【{top1_name} 板块推荐3只】")
    labels = {'龙头': '①核心龙头', '滞涨': '②滞涨潜力', '产业链': '③产业链联动'}
    for key, label in labels.items():
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

    lines += [
        "", "="*50,
        "⚠️ 风险提示：以上为板块技术面和资金面自动分析，不构成投资建议。",
        f"GitHub Actions 自动发送 · {now.strftime('%H:%M')}"
    ]
    plain_text = "\n".join(lines)

    # -------- HTML --------
    risk_html = ""
    if is_high_risk:
        risk_html = """
        <div style="background:#ffebee;border:2px solid #e53935;border-radius:6px;
                    padding:10px 16px;margin-bottom:14px;color:#b71c1c;font-weight:bold;">
          ⚠️ 高位风险预警：该板块5日涨幅已超15%，追高风险较大，建议谨慎操作！
        </div>"""

    # Top5表格
    top5_rows = ""
    for i, row in top5_df.head(5).iterrows():
        flow_val = row.get('5日主力净流入', 0)
        ret_val  = row.get('5日涨幅', 0)
        ret_color = '#2e7d32' if ret_val > 0 else '#c62828'
        flow_color = '#2e7d32' if flow_val > 0 else '#c62828'
        badge = ""
        name  = row['板块名称']
        if any(s['板块名称'] == name for s in emerging):
            badge = '<span style="background:#ff7043;color:white;font-size:0.75em;padding:1px 5px;border-radius:3px;margin-left:4px;">新兴</span>'
        elif any(s['板块名称'] == name for s in sustained):
            badge = '<span style="background:#1565c0;color:white;font-size:0.75em;padding:1px 5px;border-radius:3px;margin-left:4px;">主线</span>'
        top5_rows += f"""
        <tr style="border-bottom:1px solid #eee;">
          <td style="padding:7px 10px;">{i+1}</td>
          <td style="padding:7px 10px;font-weight:bold;">{name}{badge}</td>
          <td style="padding:7px 10px;text-align:right;color:{flow_color};">
            {'+' if flow_val>0 else ''}{flow_val/1e8:.2f}亿</td>
          <td style="padding:7px 10px;text-align:right;color:{ret_color};">
            {ret_val:+.1f}%</td>
          <td style="padding:7px 10px;text-align:right;">{row.get('换手率',0):.1f}%</td>
        </tr>"""

    # 产业链 HTML
    chain_html = ""
    if chain['upstream'] or chain['downstream']:
        up_tags   = "".join(f'<span style="background:#e3f2fd;color:#1565c0;padding:3px 8px;border-radius:12px;margin:3px;display:inline-block;font-size:0.85em;">↑ {u}</span>' for u in chain['upstream'][:4])
        down_tags = "".join(f'<span style="background:#e8f5e9;color:#2e7d32;padding:3px 8px;border-radius:12px;margin:3px;display:inline-block;font-size:0.85em;">↓ {d}</span>' for d in chain['downstream'][:4])
        chain_html = f"""
        <div style="margin-top:10px;">
          <div style="font-size:0.85em;color:#666;margin-bottom:4px;">产业链关联：</div>
          {up_tags}{down_tags}
        </div>"""

    # 3只股票卡片
    stock_cards = ""
    card_configs = [
        ('龙头',  '①核心龙头',   '#fff3e0', '#fb8c00'),
        ('滞涨',  '②滞涨潜力',   '#e8f5e9', '#43a047'),
        ('产业链','③产业链联动', '#e3f2fd', '#1e88e5'),
    ]
    for key, label, bg, border in card_configs:
        s = stocks_3.get(key)
        if s:
            chg_color = '#c62828' if s.get('今日涨跌幅', 0) < 0 else '#2e7d32'
            r5_color  = '#c62828' if s.get('5日涨幅', 0) < 0 else '#2e7d32'
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
                  <td style="color:{chg_color};"><b>{s.get('今日涨跌幅',0):+.1f}%</b></td>
                </tr>
                <tr>
                  <td style="padding:2px 12px 2px 0;">5日涨幅</td>
                  <td style="color:{r5_color};padding:2px 16px 2px 0;"><b>{s.get('5日涨幅',0):+.1f}%</b></td>
                  <td style="padding:2px 12px 2px 0;">主力净流入</td>
                  <td><b>{s.get('主力净流入(万)','-')} 万</b></td>
                </tr>
              </table>
              <div style="margin-top:7px;background:rgba(255,255,255,0.6);padding:6px 8px;
                          border-radius:4px;font-size:0.83em;color:#444;">
                💡 {s['理由']}
              </div>
            </div>"""
        else:
            stock_cards += f"""
            <div style="background:#f5f5f5;border-left:4px solid #bbb;
                        padding:10px 16px;margin:10px 0;border-radius:6px;color:#999;">
              {label}：暂无合适标的
            </div>"""

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
            <th style="padding:8px 10px;text-align:right;">5日净流入</th>
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
            <td style="padding:3px 16px 3px 0;">💰 5日主力净流入</td>
            <td style="color:{'#2e7d32' if flow5>0 else '#c62828'};font-weight:bold;">
              {flow5/1e8:.2f} 亿元</td>
            <td style="padding:3px 16px 3px 16px;">📈 5日区间涨幅</td>
            <td style="color:{'#2e7d32' if ret5>0 else '#c62828'};font-weight:bold;">
              {ret5:+.1f}%</td>
          </tr>
          <tr>
            <td style="padding:3px 16px 3px 0;">🔄 当前换手率</td>
            <td style="font-weight:bold;">{turn:.1f}%</td>
          </tr>
        </table>
        <div style="margin-top:10px;font-size:0.87em;color:#5d4037;line-height:1.7;">
          📌 <b>热度分析：</b>{hot_reason}
        </div>
        {chain_html}
      </div>

      <h3 style="color:#444;margin-bottom:4px;">🎯 {top1_name} 推荐3只标的</h3>
      <p style="color:#888;font-size:0.85em;margin-top:0;">
        分别代表：核心龙头 / 滞涨补涨 / 产业链联动，覆盖不同风险收益特征
      </p>
      {stock_cards}

      <div style="background:#fce4ec;padding:12px 16px;border-radius:6px;
                  font-size:0.82em;color:#880e4f;margin-top:18px;line-height:1.7;">
        ⚠️ <b>风险提示：</b>以上为板块资金面和技术面自动分析，不构成投资建议。
        热点板块往往存在追高风险，请结合个股基本面审慎决策。<br>
        <span style="color:#aaa;">GitHub Actions 自动发送 · {now.strftime('%H:%M')}</span>
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
        logger.warning("邮件环境变量未配置，打印到控制台")
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
        logger.info(f"✅ 热点板块邮件发送成功")
    except Exception as e:
        logger.error(f"❌ 邮件发送失败: {e}")
        print(plain_text)


# ============================================================
# 主入口
# ============================================================

def main():
    logger.info("="*50)
    logger.info("🔥 热点板块扫描启动")
    logger.info("="*50)

    now      = datetime.now()
    date_str = now.strftime('%Y年%m月%d日')
    subject  = f"🔥 A股热点板块日报 · {date_str}"

    # 1. 获取数据
    fund_df  = get_industry_fund_flow()
    time.sleep(1)
    spot_df  = get_industry_spot()
    time.sleep(1)

    if fund_df.empty:
        logger.error("板块资金数据获取失败，退出")
        send_email("今日板块数据获取失败，请检查网络。", "今日板块数据获取失败。", subject)
        return

    # 2. 综合评分
    scored_df = score_sectors(fund_df, spot_df)
    if scored_df.empty:
        logger.error("板块评分失败")
        return

    # 3. 分类：新兴热点 vs 持续主线
    emerging, sustained = classify_sectors(scored_df)

    # 4. 确定重点推荐板块（综合得分第一）
    top1_row  = scored_df.iloc[0]
    top1_name = top1_row['板块名称']
    logger.info(f"重点推荐板块：{top1_name}")

    # 5. 高位风险判断
    ret5 = float(top1_row.get('5日涨幅', 0))
    is_high_risk = ret5 > 15.0

    # 6. 板块内选3只股票
    time.sleep(1)
    stocks_3 = pick_3_stocks(top1_name, is_concept=False)

    # 7. 生成报告
    plain_text, html_text = build_sector_report(
        scored_df, emerging, sustained,
        top1_name, top1_row, stocks_3, is_high_risk
    )

    # 8. 发送邮件
    send_email(plain_text, html_text, subject)
    logger.info("✅ 热点板块扫描完成")


if __name__ == '__main__':
    main()
