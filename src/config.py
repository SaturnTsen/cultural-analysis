#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件
包含项目所需的所有配置参数
"""

import os
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List, Dict
# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# 报告目录
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
HTML_REPORTS_DIR = os.path.join(REPORTS_DIR, "html")
PDF_REPORTS_DIR = os.path.join(REPORTS_DIR, "pdf")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# 模板目录
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")

# 日志目录
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "analysis.log")

# 创建必要的目录
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR,
                  HTML_REPORTS_DIR, PDF_REPORTS_DIR, FIGURES_DIR,
                  LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# 数据源配置


class DataSource(BaseModel):
    industrial: List[str]
    financial: List[str]
    healthcare: List[str]
    education: List[str]
    public_services: List[str]


class DataSourcesList(BaseModel):
    # 直接使用字典结构
    china: DataSource
    france: DataSource


# 原始数据
raw_data_sources = {
    "china": {
        "industrial": [
            # Made in China 2025
            "https://www.isdp.eu/wp-content/uploads/2018/06/Made-in-China-Backgrounder.pdf",
            "https://app.xinhuanet.com/news/article.html?articleId=d3b74efeaf4ba927ea1f65e334dbfd56",  # 工业互联网创新发展报告
        ],
        "financial": [
            # FinTech Development Plan
            "https://www.china-briefing.com/news/a-close-reading-china-fintech-development-plan-for-2022-2025",
            # Five Fintech Trends
            "https://www.smefinanceforum.org/post/five-major-fintech-trends-where-china-is-leading-the-world",
        ],
        "healthcare": [
            "https://www.oliverwyman.com/our-expertise/perspectives/health/2020/apr/covid-19-makes-digital-care-the-norm-in-china.html",  # Digital Care in China
        ],
        "education": [
            "https://edu.sh.gov.cn/xxgk2_zhzw_ghjh_01/20201015/v2-0015-gw_3022018002.html",  # 教育信息化2.0行动计划
            "https://internationaleducation.gov.au/international-network/china/PolicyUpdates-China/Pages/China's-education-arrangements-during-COVID-19-pandemic-period.aspx",  # 停课不停学
        ],
        "public_services": [
            # Internet Plus Government Services (2016)
            "http://english.www.gov.cn/policies/latest_releases/2016/09/29/content_281475454498314.htm",
            # Digital Government Plan (2021)
            "https://govinsider.asia/intl-en/article/china-sets-out-four-step-plan-for-digital-government",
            # 数字中国战略 (2018)
            "https://www.gov.cn/xinwen/2018-04/22/content_5284959.htm",
        ],
    },
    "france": {
        "industrial": [
            # Plan Industrie du Futur
            "https://www.entreprises.gouv.fr/files/files/Publications/2019/Rapports-activite/dge-rapport-d-activite-2018.pdf",
            # Air Liquide Case Study
            "https://www.airliquide.com/fr/histoires/innovation/transformation-numerique-une-aventure-humaine-avant-tout",
        ],
        "financial": [
            "https://acpr.banque-france.fr/fr/publications-et-statistiques/publications/ndeg-131-la-transformation-numerique-dans-le-secteur-bancaire-francais",  # ACPR Report
            "https://www.banque-france.fr/fr/stabilite-financiere/mandat-stabilite-financiere/accompagner-transformation-numerique-secteur-financier/enjeux",  # Banque de France
        ],
        "healthcare": [
            "https://esante.gouv.fr",  # Stratégie nationale du numérique en santé
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC8007943",  # Télémédecine et COVID-19
        ],
        "education": [
            # Plan numérique pour l'éducation
            "https://www.cedefop.europa.eu/en/news/france-digital-plan-education-500-schools-and-colleges-be-connected-internet",
            # Bilan du Plan numérique
            "https://www.lagazettedescommunes.com/595402/numerique-educatif-les-promesses-non-tenues-de-letat",
        ],
        "public_services": [
            # Cour des Comptes Report
            "https://www.ccomptes.fr/fr/publications/le-pilotage-de-la-transformation-numerique-de-letat-par-la-direction",
            "https://www.vie-publique.fr/eclairage/18925-e-administration-la-transformation-numerique-de-letat",  # France Relance
        ],
    },
}

# 创建验证后的数据源配置
DATA_SOURCES = DataSourcesList(**raw_data_sources)

# 文件路径配置
FILE_PATHS = {
    "china": {
        "industrial": os.path.join(RAW_DATA_DIR, "china_industrial.txt"),
        "financial": os.path.join(RAW_DATA_DIR, "china_financial.txt"),
        "healthcare": os.path.join(RAW_DATA_DIR, "china_healthcare.txt"),
        "education": os.path.join(RAW_DATA_DIR, "china_education.txt"),
        "public_services": os.path.join(RAW_DATA_DIR, "china_public_services.txt"),
    },
    "france": {
        "industrial": os.path.join(RAW_DATA_DIR, "france_industrial.txt"),
        "financial": os.path.join(RAW_DATA_DIR, "france_financial.txt"),
        "healthcare": os.path.join(RAW_DATA_DIR, "france_healthcare.txt"),
        "education": os.path.join(RAW_DATA_DIR, "france_education.txt"),
        "public_services": os.path.join(RAW_DATA_DIR, "france_public_services.txt"),
    },
}

# 语言配置
LANGUAGE_CONFIG = {
    "china": "zh",
    "france": "fr",
    "en": "en",
}

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

# 文化维度解释
HOFSTEDE_DESCRIPTIONS = {
    "power_distance": "权力距离：社会中权力的不平等分配的接受程度",
    "individualism": "个人主义：个人对团体的独立程度",
    "masculinity": "男性气质：成就、英雄主义、果断和物质成功的偏好",
    "uncertainty_avoidance": "不确定性规避：社会对不确定性和模糊性的不适程度",
    "long_term_orientation": "长期导向：注重长期规划和传统的程度",
    "indulgence": "放纵度：控制自身欲望和冲动的程度",
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

# 报告配置
REPORT_CONFIG = {
    "title": "中法数字化转型文化差异分析",
    "subtitle": "基于自然语言处理的跨文化比较研究 (2015-2025)",
    "author": "您的姓名",
    "institution": "您的学校/机构",
    "template_file": os.path.join(TEMPLATES_DIR, "report_template.html"),
    "css_file": os.path.join(TEMPLATES_DIR, "report_style.css"),
    "output_html": os.path.join(HTML_REPORTS_DIR, "report.html"),
    "output_pdf": os.path.join(PDF_REPORTS_DIR, "report.pdf"),
    "sectors": ["industrial", "financial", "healthcare", "education", "public_services"],
    "languages": ["fr", "zh", "en"],
    "time_period": {
        "start": "2015",
        "end": "2025"
    }
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
}

# HTTP请求配置
HTTP_CONFIG = {
    "timeout": 30,
    "retries": 3,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
}
