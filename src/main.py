#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中法数字化转型文化差异分析项目主入口文件
运行整个分析流程，包括数据收集、预处理、分析、可视化和报告生成
"""
# 确保能正确导入项目模块
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from src.reporting.reporter import ReportGenerator
from src.visualization.visualizer import DataVisualizer
from src.analysis.analyzer import DataAnalyzer
from src.preprocessing.preprocessor import DataPreprocessor
from src.data_collection.collector import DataCollector
import os

import time
import argparse

# 导入项目模块


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='中法数字化转型文化差异分析')
    parser.add_argument('--include-collection',
                        action='store_true', help='包含数据收集步骤')
    parser.add_argument('--include-preprocessing',
                        action='store_true', help='包含数据预处理步骤')
    parser.add_argument('--include-analysis',
                        action='store_true', help='包含数据分析步骤')
    parser.add_argument('--include-visualization',
                        action='store_true', help='包含数据可视化步骤')
    parser.add_argument('--only-report', action='store_true', help='只生成报告')
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
    analyzer: DataAnalyzer = DataAnalyzer()
    visualizer: DataVisualizer = None
    report_generator: ReportGenerator = None

    analysis_results = None

    # 主流程
    if args.only_report:
        logger.info("仅生成报告模式，跳过数据处理步骤")
    else:
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
            analysis_results = analyzer.analyze_all()
            logger.info("数据分析完成")
        else:
            logger.info("跳过数据分析步骤")
            # 从保存的结果加载分析数据
            analysis_results = analyzer.load_results()

        # 4. 数据可视化
        if args.include_visualization:
            logger.info("开始数据可视化...")
            visualizer = DataVisualizer()
            if analysis_results is not None:
                visualization_results = visualizer.visualize_all(
                    analysis_results)
                logger.info("数据可视化完成")
            else:
                logger.info("跳过数据可视化步骤")
                # 从保存的结果加载可视化数据
                visualization_results = visualizer.load_visualization_results()

    # 5. 报告生成
    logger.info("开始生成分析报告...")
    if args.only_report:
        # 从保存的结果加载数据
        if analysis_results is None:
            analysis_results = analyzer.load_results()
        if visualization_results is None:
            visualization_results = visualizer.load_visualization_results()

    report_generator = ReportGenerator(analysis_results, visualization_results)
    report_generator.generate_report()
    logger.info("分析报告生成完成")

    # 计算总运行时间
    elapsed_time = time.time() - start_time
    logger.info(f"分析流程完成，总用时: {elapsed_time:.2f} 秒")
    logger.info("="*80)


if __name__ == "__main__":
    main()
