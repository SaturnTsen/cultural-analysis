"""
可视化模块：将分析结果以图形方式展示
"""

from src.config import RESULTS_DIR, VISUALIZATION_CONFIG
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib
from typing import *

matplotlib.use('Agg')  # 设置后端为Agg，不显示图形窗口


logger = get_logger("visualization.visualizer")


class DataVisualizer:
    """数据可视化类，负责生成各种可视化图表"""

    def __init__(self, results_file_path=None):
        """初始化可视化器"""
        self.results_dir = RESULTS_DIR

        # 如果传入了分析结果，则使用传入的结果
        results_file_path = results_file_path or os.path.join(
            self.results_dir, "analysis_results.json")

        self.analysis_results = self._load_analysis_results(
            analysis_json_path=results_file_path)

        if not self.analysis_results:
            logger.error("无法获取分析结果，无法生成可视化")
            return {}

        # 绘图参数
        plt.style.use('seaborn-v0_8-whitegrid')

        # 确保可视化目录存在
        self.viz_dir = os.path.join(self.results_dir, "visualizations")
        ensure_dir(self.viz_dir)

        # 设置字体
        plt.rcParams['font.family'] = VISUALIZATION_CONFIG['font_family']

        if os.path.exists(VISUALIZATION_CONFIG['font_path']):
            self.font_path = VISUALIZATION_CONFIG['font_path']
        else:
            logger.info(f"字体{VISUALIZATION_CONFIG['font_path']} 不存在，回退至默认字体")

        # 设置颜色方案
        self.colors = {
            'france': VISUALIZATION_CONFIG['colors']['france'],  # 法国蓝
            'china': VISUALIZATION_CONFIG['colors']['china'],    # 中国红
            # 绿色
            'positive': VISUALIZATION_CONFIG['sentiment_colors']['positive'],
            # 灰色
            'neutral': VISUALIZATION_CONFIG['sentiment_colors']['neutral'],
            # 红色
            'negative': VISUALIZATION_CONFIG['sentiment_colors']['negative']
        }

    def visualize_all(self):
        """生成所有可视化"""
        logger.info("开始生成所有可视化")

        # 生成词云
        word_cloud_results = self.generate_word_clouds()

        # 生成情感分析图表
        sentiment_results = self.generate_sentiment_charts()

        # 生成主题模型可视化
        topic_results = self.generate_topic_visualizations()

        # 生成文化维度雷达图
        culture_results = self.generate_cultural_radar_charts()

        # 生成中法对比图
        comparison_results = self.generate_country_comparison_charts()

        # 组合结果
        visualization_results = {
            'word_clouds': word_cloud_results,
            'sentiment_charts': sentiment_results,
            'topic_visualizations': topic_results,
            'cultural_radar_charts': culture_results,
            'country_comparison_charts': comparison_results,
            'visualization_date': datetime.now().isoformat()
        }

        # 保存可视化结果
        self._save_visualization_results(visualization_results)

        logger.info("所有可视化生成完成")

        return visualization_results

    def _load_analysis_results(self, analysis_json_path=None):
        """加载分析结果"""
        if analysis_json_path is None:
            results_file = os.path.join(
                self.results_dir, "analysis_results.json")
        else:
            results_file = analysis_json_path

        if not os.path.exists(results_file):
            logger.warning(f"分析结果文件不存在: {results_file}")
            return None

        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info("成功加载分析结果")
            return results
        except Exception as e:
            logger.error(f"加载分析结果时出错: {str(e)}")
            return None

    def _save_visualization_results(self, results):
        """保存可视化结果"""
        results_file = os.path.join(
            self.results_dir, "visualization_results.json")

        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"可视化结果已保存至: {results_file}")
        except Exception as e:
            logger.error(f"保存可视化结果时出错: {str(e)}")

    def load_visualization_results(self):
        """从文件加载可视化结果"""
        logger.info("从文件加载可视化结果")

        results_file = os.path.join(
            self.results_dir, "visualization_results.json")

        if not os.path.exists(results_file):
            logger.warning(f"可视化结果文件不存在: {results_file}")
            # 返回空结果
            return {
                'word_clouds': {},
                'sentiment_charts': {},
                'topic_visualizations': {},
                'cultural_radar_charts': {},
                'country_comparison_charts': {},
                'visualization_date': datetime.now().isoformat()
            }

        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"成功加载可视化结果")
            return results
        except Exception as e:
            logger.error(f"加载可视化结果时出错: {str(e)}")
            # 返回空结果
            return {
                'word_clouds': {},
                'sentiment_charts': {},
                'topic_visualizations': {},
                'cultural_radar_charts': {},
                'country_comparison_charts': {},
                'visualization_date': datetime.now().isoformat()
            }

    def generate_word_clouds(self):
        """生成词云可视化"""
        logger.info("生成词云可视化")

        results = {}

        if 'tf_idf' not in self.analysis_results:
            logger.warning("找不到TF-IDF分析结果，无法生成词云")
            return results

        tf_idf_results = self.analysis_results['tf_idf']

        for country in tf_idf_results.keys():
            country_dir = os.path.join(self.viz_dir, country)
            if not os.path.exists(country_dir):
                os.makedirs(country_dir)

            results[country] = {}

            for sector, sector_data in tf_idf_results[country].items():
                logger.info(f"生成{country}的{sector}领域词云")

                if 'top_words' not in sector_data or not sector_data['top_words']:
                    logger.warning(f"{country}的{sector}领域没有词频数据")
                    continue

                # 创建词-权重字典
                word_weights = {item['word']: item['score']
                                for item in sector_data['top_words']}

                # 设置词云颜色
                color = self.colors[country]

                # 创建词云对象
                wc = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='Blues' if country == 'france' else 'Reds',
                    max_words=100,
                    contour_width=1,
                    contour_color='steelblue' if country == 'france' else 'firebrick',
                    font_path=self.font_path)

                # 生成词云
                wc.generate_from_frequencies(word_weights)

                # 创建图形
                plt.figure(figsize=(10, 6))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                plt.title(
                    f"{country.capitalize()} - {sector.capitalize()} Sector", fontsize=16)
                plt.tight_layout()

                # 保存图像
                output_path = os.path.join(
                    country_dir, f"wordcloud_{sector}.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close('all')  # 确保关闭所有图形

                results[country][sector] = output_path

        logger.info("词云可视化生成完成")

        return results

    def generate_sentiment_charts(self):
        """生成情感分析图表"""
        logger.info("生成情感分析图表")

        results = {}

        if 'sentiment' not in self.analysis_results:
            logger.warning("找不到情感分析结果，无法生成图表")
            return results

        sentiment_results = self.analysis_results['sentiment']

        # 准备数据
        data = []
        for country in sentiment_results.keys():
            for sector, sector_data in sentiment_results[country].items():
                if 'sentiment_ratios' not in sector_data:
                    continue

                ratios = sector_data['sentiment_ratios']

                data.append({
                    'country': country.capitalize(),
                    'sector': sector.capitalize(),
                    'sentiment': 'Positive',
                    'ratio': ratios.get('positive', 0)
                })
                data.append({
                    'country': country.capitalize(),
                    'sector': sector.capitalize(),
                    'sentiment': 'Neutral',
                    'ratio': ratios.get('neutral', 0)
                })
                data.append({
                    'country': country.capitalize(),
                    'sector': sector.capitalize(),
                    'sentiment': 'Negative',
                    'ratio': ratios.get('negative', 0)
                })

        if not data:
            logger.warning("没有情感分析数据")
            return results

        df = pd.DataFrame(data)

        # 1. 创建按国家和行业的情感条形图
        plt.figure(figsize=(12, 8))

        # 设置颜色
        colors = [self.colors['positive'],
                  self.colors['neutral'], self.colors['negative']]

        # 创建图表
        ax = sns.barplot(
            x='sector',
            y='ratio',
            hue='sentiment',
            data=df,
            palette=colors
        )

        # 添加标题和标签
        plt.title('Sentiment Analysis by Sector and Country', fontsize=16)
        plt.xlabel('Sector', fontsize=12)
        plt.ylabel('Ratio', fontsize=12)
        plt.legend(title='Sentiment')

        # 添加数据标签
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=8)

        # 调整布局
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 保存图像
        output_path = os.path.join(self.viz_dir, "sentiment_by_sector.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close('all')  # 确保关闭所有图形

        results['by_sector'] = output_path

        # 2. 创建国家间情感对比图
        country_sentiment = df.groupby(['country', 'sentiment'])[
            'ratio'].mean().reset_index()

        plt.figure(figsize=(10, 6))

        # 创建图表
        sns.barplot(
            x='country',
            y='ratio',
            hue='sentiment',
            data=country_sentiment,
            palette=colors
        )

        # 添加标题和标签
        plt.title('Sentiment Comparison: France vs China', fontsize=16)
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Average Ratio', fontsize=12)
        plt.legend(title='Sentiment')

        # 添加数据标签
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=10)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        output_path = os.path.join(
            self.viz_dir, "sentiment_country_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close('all')  # 确保关闭所有图形

        results['country_comparison'] = output_path

        logger.info("情感分析图表生成完成")

        return results

    def generate_topic_visualizations(self):
        """生成主题模型可视化"""
        logger.info("生成主题模型可视化")

        results = {}

        if 'topics' not in self.analysis_results:
            logger.warning("找不到主题分析结果，无法生成可视化")
            return results

        topic_results = self.analysis_results['topics']

        for country in topic_results.keys():
            country_dir = os.path.join(self.viz_dir, country)
            if not os.path.exists(country_dir):
                os.makedirs(country_dir)

            results[country] = {}

            for sector, sector_data in topic_results[country].items():
                logger.info(f"生成{country}的{sector}领域主题可视化")

                if 'topics' not in sector_data or not sector_data['topics']:
                    logger.warning(f"{country}的{sector}领域没有主题数据")
                    continue

                topics = sector_data['topics']

                # 创建多个子图
                n_topics = len(topics)
                fig, axes = plt.subplots(1, n_topics, figsize=(n_topics*4, 5))

                # 如果只有一个主题，将axes转换为列表
                if n_topics == 1:
                    axes = [axes]

                # 设置颜色映射
                cmap = cm.get_cmap('Blues' if country == 'france' else 'Reds')
                norm = Normalize(vmin=0, vmax=max(
                    [max(topic['weights']) for topic in topics]))

                # 绘制每个主题的词条形图
                for i, topic in enumerate(topics):
                    words = topic['words']
                    weights = topic['weights']

                    # 对权重和词组合排序
                    word_weight_pairs = list(zip(words, weights))
                    word_weight_pairs.sort(key=lambda x: x[1], reverse=True)
                    words = [pair[0] for pair in word_weight_pairs]
                    weights = [pair[1] for pair in word_weight_pairs]

                    # 反转顺序，使最重要的词在顶部
                    words = words[:10]
                    weights = weights[:10]
                    words.reverse()
                    weights.reverse()

                    # 绘制水平条形图
                    colors = [cmap(norm(weight)) for weight in weights]
                    axes[i].barh(words, weights, color=colors)

                    # 设置标题和标签
                    axes[i].set_title(f"Topic {i+1}", fontsize=14)
                    axes[i].set_xlabel('Weight', fontsize=12)

                    # 添加格线
                    axes[i].grid(axis='x', linestyle='--', alpha=0.7)

                # 调整布局
                plt.suptitle(
                    f"{country.capitalize()} - {sector.capitalize()} Sector: Topic Model", fontsize=16)
                plt.tight_layout()

                # 保存图像
                output_path = os.path.join(country_dir, f"topics_{sector}.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close('all')  # 确保关闭所有图形

                results[country][sector] = output_path

        logger.info("主题模型可视化生成完成")

        return results

    def generate_cultural_radar_charts(self):
        """生成文化维度雷达图"""
        logger.info("生成文化维度雷达图")

        results = {}

        if 'cultural_dimensions' not in self.analysis_results:
            logger.warning("找不到文化维度分析结果，无法生成雷达图")
            return results

        cultural_results = self.analysis_results['cultural_dimensions']

        # 准备数据
        countries = list(cultural_results.keys())

        if not countries:
            logger.warning("没有国家数据")
            return results

        # 创建雷达图数据
        fig = go.Figure()

        for country in countries:
            # 获取维度分数和维度名称
            dimensions = cultural_results[country].keys()
            scores = [cultural_results[country][dim]['score']
                      for dim in dimensions]

            # 闭合雷达图
            scores.append(scores[0])
            dimensions_closed = list(dimensions) + [list(dimensions)[0]]

            # 添加轨迹
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=dimensions_closed,
                fill='toself',
                name=country.capitalize(),
                line_color=self.colors[country]
            ))

        # 更新布局
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Cultural Dimensions: France vs China",
            showlegend=True
        )

        # 保存图像
        output_path = os.path.join(
            self.viz_dir, "cultural_dimensions_radar.html")
        fig.write_html(output_path)

        # 也保存为图像
        # img_path = os.path.join(self.viz_dir, "cultural_dimensions_radar.png")
        # fig.write_image(img_path, width=800, height=600, scale=2)

        results['radar_chart'] = output_path
        # results['radar_image'] = img_path

        logger.info("文化维度雷达图生成完成")

        return results

    def generate_country_comparison_charts(self):
        """生成中法对比图表"""
        logger.info("生成中法对比图表")

        results = {}

        # 1. 情感对比
        if 'sentiment' in self.analysis_results:
            sentiment_results = self.analysis_results['sentiment']

            # 准备数据
            sentiment_data = []

            for country in sentiment_results.keys():
                # 计算各行业平均情感比例
                positive_ratios = []
                neutral_ratios = []
                negative_ratios = []

                for sector, sector_data in sentiment_results[country].items():
                    if 'sentiment_ratios' not in sector_data:
                        continue

                    ratios = sector_data['sentiment_ratios']
                    positive_ratios.append(ratios.get('positive', 0))
                    neutral_ratios.append(ratios.get('neutral', 0))
                    negative_ratios.append(ratios.get('negative', 0))

                if positive_ratios:
                    sentiment_data.append({
                        'country': country.capitalize(),
                        'positive': np.mean(positive_ratios),
                        'neutral': np.mean(neutral_ratios),
                        'negative': np.mean(negative_ratios)
                    })

            if sentiment_data:
                # 创建对比条形图
                fig = make_subplots(rows=1, cols=1)

                x = [d['country'] for d in sentiment_data]

                # 添加积极情感条
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=[d['positive'] for d in sentiment_data],
                        name='Positive',
                        marker_color=self.colors['positive']
                    )
                )

                # 添加中性情感条
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=[d['neutral'] for d in sentiment_data],
                        name='Neutral',
                        marker_color=self.colors['neutral']
                    )
                )

                # 添加消极情感条
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=[d['negative'] for d in sentiment_data],
                        name='Negative',
                        marker_color=self.colors['negative']
                    )
                )

                # 更新布局
                fig.update_layout(
                    title="Sentiment Comparison: France vs China",
                    xaxis_title="Country",
                    yaxis_title="Average Sentiment Ratio",
                    barmode='group',
                    legend_title="Sentiment"
                )

                # 保存图表
                output_path = os.path.join(
                    self.viz_dir, "france_china_sentiment_comparison.html")
                fig.write_html(output_path)

                # 也保存为图像
                img_path = os.path.join(
                    self.viz_dir, "france_china_sentiment_comparison.png")
                # fig.write_image(img_path, width=800, height=600, scale=2)

                results['sentiment_comparison'] = {
                    'html': output_path,
                    # 'image': img_path
                }

        # 2. 词频对比
        # 这部分需要更复杂的处理，可以选择最常见的一些词进行对比

        logger.info("中法对比图表生成完成")

        return results
