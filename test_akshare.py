#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AKShare 概念板块接口可用性测试
================================
测试以下接口是否能在 GitHub 服务器（海外）访问：
  1. 概念板块今日资金流向排名
  2. 概念板块5日资金流向排名
  3. 概念板块成分股
结果打印到日志，不发邮件。
"""

import sys, time, logging
import akshare as ak

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def test_concept_flow_today():
    logger.info("=" * 40)
    logger.info("测试1：概念板块今日资金流向排名")
    try:
        df = ak.stock_board_concept_fund_flow_rank(symbol="今日")
        if df is not None and not df.empty:
            logger.info(f"✅ 成功！返回 {len(df)} 行，列名：{list(df.columns)}")
            logger.info(f"   前3行：\n{df.head(3).to_string()}")
            return True
        else:
            logger.error("❌ 返回空数据")
            return False
    except Exception as e:
        logger.error(f"❌ 失败：{e}")
        return False


def test_concept_flow_5d():
    logger.info("=" * 40)
    logger.info("测试2：概念板块5日资金流向排名")
    try:
        df = ak.stock_board_concept_fund_flow_rank(symbol="5日")
        if df is not None and not df.empty:
            logger.info(f"✅ 成功！返回 {len(df)} 行，列名：{list(df.columns)}")
            logger.info(f"   前3行：\n{df.head(3).to_string()}")
            return True
        else:
            logger.error("❌ 返回空数据")
            return False
    except Exception as e:
        logger.error(f"❌ 失败：{e}")
        return False


def test_concept_spot():
    logger.info("=" * 40)
    logger.info("测试3：概念板块实时行情")
    try:
        df = ak.stock_board_concept_spot_em()
        if df is not None and not df.empty:
            logger.info(f"✅ 成功！返回 {len(df)} 行，列名：{list(df.columns)}")
            logger.info(f"   前3行：\n{df.head(3).to_string()}")
            return True
        else:
            logger.error("❌ 返回空数据")
            return False
    except Exception as e:
        logger.error(f"❌ 失败：{e}")
        return False


def test_concept_stocks():
    logger.info("=" * 40)
    logger.info("测试4：获取某概念板块成分股（用'人工智能'测试）")
    try:
        df = ak.stock_board_concept_cons_em(symbol="人工智能")
        if df is not None and not df.empty:
            logger.info(f"✅ 成功！返回 {len(df)} 行，列名：{list(df.columns)}")
            logger.info(f"   前3行：\n{df.head(3).to_string()}")
            return True
        else:
            logger.error("❌ 返回空数据")
            return False
    except Exception as e:
        logger.error(f"❌ 失败：{e}")
        return False


def main():
    logger.info("🔍 开始测试 AKShare 概念板块接口可用性")
    logger.info(f"AKShare 版本：{ak.__version__}")

    results = {}
    results['今日资金流向'] = test_concept_flow_today()
    time.sleep(2)
    results['5日资金流向']  = test_concept_flow_5d()
    time.sleep(2)
    results['实时行情']     = test_concept_spot()
    time.sleep(2)
    results['成分股查询']   = test_concept_stocks()

    logger.info("=" * 40)
    logger.info("📋 测试结果汇总：")
    all_pass = True
    for name, ok in results.items():
        status = "✅ 可用" if ok else "❌ 不可用"
        logger.info(f"  {name}：{status}")
        if not ok:
            all_pass = False

    if all_pass:
        logger.info("🎉 全部接口可用，可以升级为概念板块版本！")
    else:
        logger.info("⚠️  部分接口不可用，需要评估替代方案")


if __name__ == '__main__':
    main()
