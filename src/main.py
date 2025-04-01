#!/usr/bin/env python
# -*- coding: utf-8 -*-
# autopep8: off

"""
中法数字化转型文化差异分析项目主入口文件
运行整个分析流程，包括数据收集、预处理、分析、可视化和报告生成
"""

# 确保能正确导入项目模块

import os
import time
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块

from src.data_collection.collector import DataCollector
from src.preprocessing.preprocessor import DataPreprocessor
from src.analysis.analyzer import DataAnalyzer
from src.visualization.visualizer import DataVisualizer
from src.reporting.reporter import ReportGenerator
from src.utils.logger import setup_logger

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='中法数字化转型文化差异分析')
    parser.add_argument('--include-collection',
                        action='store_true', help='数据收集')
    parser.add_argument('--include-preprocessing',
                        action='store_true', help='数据预处理')
    parser.add_argument('--include-analysis',
                        action='store_true', help='数据分析')
    parser.add_argument('--include-visualization',
                        action='store_true', help='数据可视化')
    parser.add_argument('--include-report',
                        action='store_true', help='生成报告')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志级别')
    return parser.parse_args()


def setup_directories():
    """创建必要的目录结构"""
    directories = [
        'data/raw',
        'data/processed',
        'data/results',
        'logs',
        'reports/html',
        'reports/pdf',
        'reports/figures'
    ]
    for directory in directories:
        os.makedirs(os.path.join(project_root, directory), exist_ok=True)


def main():
    """主函数，运行完整分析流程"""
    start_time = time.time()
    args = parse_arguments()

    # 设置日志
    setup_directories()
    logger = setup_logger(args.log_level)

    logger.info("="*80)
    logger.info("中法数字化转型文化差异分析项目启动")
    logger.info("="*80)

    # 各模块
    collector: DataCollector = None
    preprocessor: DataPreprocessor = None
    analyzer: DataAnalyzer = None
    visualizer: DataVisualizer = None
    report_generator: ReportGenerator = None

    analysis_results = None
    visualization_results = None

    # 主流程
    # 1. 数据收集
    if args.include_collection:
        logger.info("开始数据收集...")
        collector = DataCollector()
        collector.collect_all()
        logger.info("数据收集完成")
    else:
        logger.info("跳过数据收集步骤")

    # 2. 数据预处理
    if args.include_preprocessing:
        logger.info("开始数据预处理...")
        preprocessor = DataPreprocessor()
        preprocessor.preprocess_all()
        logger.info("数据预处理完成")
    else:
        logger.info("跳过数据预处理步骤")

    # 3. 数据分析
    if args.include_analysis:
        logger.info("开始数据分析...")
        analyzer = DataAnalyzer()
        analysis_results = analyzer.analyze_all()
        logger.info("数据分析完成")
    else:
        logger.info("跳过数据分析")

    # 4. 数据可视化
    if args.include_visualization:
        logger.info("开始数据可视化...")
        visualizer = DataVisualizer()
        visualization_results = visualizer.visualize_all()
        logger.info("数据可视化完成")
    else:
        logger.info("跳过数据可视化")

    # 5. 报告生成
    if args.include_report:
        logger.info("开始生成分析报告...")
        # 从保存的结果加载数据
        analyzer = analyzer or DataAnalyzer()
        analysis_results = analysis_results or analyzer.load_results()

        visualizer = visualizer or DataVisualizer()
        visualization_results = visualization_results or visualizer.load_visualization_results()

        report_generator = ReportGenerator(
            analysis_results, visualization_results)
        report_generator.generate_report()
        logger.info("分析报告生成完成")
    else:
        logger.info("跳过分析报告生成")

    # 计算总运行时间
    elapsed_time = time.time() - start_time
    logger.info(f"分析流程完成，总用时: {elapsed_time:.2f} 秒")
    logger.info("="*80)


if __name__ == "__main__":
    main()
