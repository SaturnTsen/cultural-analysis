#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件
包含项目所需的所有配置参数
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeds")
# 日志目录
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "analysis.log")

# 创建必要的目录
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# API
BASE_URL = "https://api.siliconflow.cn/v1/embeddings"
PAYLOAD_MODEL = "Pro/BAAI/bge-m3"
API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# Hofstede文化维度配置
# 参考: https://www.hofstede-insights.com/country-comparison/china,france/
HOFSTEDE_DIMENSIONS = {
    "china": {
        "power_distance": 80,
        "individualism": 20,
        "masculinity": 66,
        "uncertainty_avoidance": 30,
        "long_term_orientation": 87,
        "indulgence": 24,
    },
    "france": {
        "power_distance": 68,
        "individualism": 71,
        "masculinity": 43,
        "uncertainty_avoidance": 86,
        "long_term_orientation": 63,
        "indulgence": 48,
    },
}

# 分析配置
ANALYSIS_CONFIG = {
    # TF-IDF分析配置
    "tfidf": {
        "max_features": 200,
        "ngram_range": (1, 2),
        "min_df": 2,
        "languages": ["fr", "zh", "en"],  # 支持多语言分析
    },

    # 情感分析配置
    "sentiment": {
        "model": "transformer",  # 使用transformer模型进行多语言情感分析
        "transformer_model": {
            "zh": "bert-base-chinese",
            "fr": "camembert-base",
            "en": "bert-base-uncased"
        },
        "threshold": {
            "positive": 0.6,
            "negative": 0.4,
        },
    },

    # 主题建模配置
    "topic_modeling": {
        "num_topics": 5,
        "max_iter": 100,
        "alpha": "auto",
        "random_state": 42,
        "languages": ["fr", "zh", "en"],  # 支持多语言主题建模
    },
}

# 可视化配置
VISUALIZATION_CONFIG = {
    "wordcloud": {
        "width": 800,
        "height": 400,
        "background_color": "white",
        "max_words": 100,
        "languages": ["fr", "zh", "en"],  # 支持多语言词云
    },
    "colors": {
        "china": "#E30613",  # 中国红
        "france": "#002395",  # 法国蓝
    },
    "sentiment_colors": {
        "positive": "#4CAF50",  # 绿色
        "neutral": "#9E9E9E",   # 灰色
        "negative": "#F44336",  # 红色
    },
    "font_family": ['PingFang HK'],
    "font_path": "C:/Users/ming/AppData/Local/Microsoft/Windows/Fonts/PingFang.ttc",
    "dpi": 300,
    "figsize": (10, 6),
}
