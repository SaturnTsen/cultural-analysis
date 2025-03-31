"""
报告生成模块：生成分析结果的综合报告
"""

import os
import logging
import json
from datetime import datetime
import jinja2
import markdown
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"WeasyPrint库无法正常工作，将只生成HTML报告")
    WEASYPRINT_AVAILABLE = False

from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir, save_json
from src.config import RESULTS_DIR, TEMPLATES_DIR, HTML_REPORTS_DIR, PDF_REPORTS_DIR, FIGURES_DIR, REPORT_CONFIG, HOFSTEDE_DIMENSIONS

logger = get_logger("reporting.reporter")

class ReportGenerator:
    """报告生成类，负责生成综合分析报告"""
    
    def __init__(self, analysis_results, visualization_results):
        """初始化报告生成器"""
        self.analysis_results = analysis_results
        self.visualization_results = visualization_results
        self.results_dir = RESULTS_DIR
        
        # 确保报告目录存在
        self.html_dir = HTML_REPORTS_DIR
        self.pdf_dir = PDF_REPORTS_DIR
        ensure_dir(self.html_dir)
        ensure_dir(self.pdf_dir)
        
        # 初始化Jinja2环境
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def generate_report(self):
        """生成综合报告"""
        logger.info("开始生成报告")
        
        # 生成报告内容
        report_content = self._generate_report_content()
        
        # 生成HTML报告
        html_path = self._generate_html_report(report_content)
        logger.info(f"HTML报告已生成: {html_path}")
        
        # 如果WeasyPrint可用，生成PDF报告
        if WEASYPRINT_AVAILABLE:
            try:
                pdf_path = self._generate_pdf_report(html_path)
                logger.info(f"PDF报告已生成: {pdf_path}")
            except Exception as e:
                logger.error(f"生成PDF报告时出错: {str(e)}")
                logger.info("仅生成HTML报告")
        else:
            logger.info("WeasyPrint不可用，跳过PDF报告生成")
        
        logger.info("报告生成完成")
        
        return html_path
    
    def _generate_report_content(self):
        """生成报告内容"""
        content = {
            'title': '中法数字化转型文化差异分析报告',
            'subtitle': '基于自然语言处理的跨文化比较',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'summary': self._generate_summary(),
            'tfidf_analysis': self._generate_tfidf_analysis(),
            'sentiment_analysis': self._generate_sentiment_analysis(),
            'topic_analysis': self._generate_topic_analysis(),
            'cultural_analysis': self._generate_cultural_analysis(),
            'comparison': self._generate_comparison(),
            'conclusion': self._generate_conclusion(),
            'recommendations': self._generate_recommendations()
        }
        
        return content
    
    def _generate_summary(self):
        """生成摘要部分"""
        return {
            'introduction': """
本报告通过自然语言处理技术，分析了中国和法国在数字化转型过程中的文化差异。
研究涵盖了工业、金融、医疗、教育和公共服务等五个关键领域，通过词频分析、情感分析、
主题建模和文化维度分析等方法，揭示了两国在数字化进程中的异同点。
            """,
            'methodology': """
研究方法采用了多语言文本分析，包括TF-IDF关键词提取、情感分析、主题建模以及基于Hofstede文化维度的分析。
数据来源包括政府报告、行业白皮书、媒体报道和学术文献，覆盖2015-2025年间的数字化转型相关内容。
            """
        }
    
    def _generate_tfidf_analysis(self):
        """生成TF-IDF分析部分"""
        tfidf_results = self.analysis_results.get('tf_idf', {})
        
        # 提取各国家各行业的关键词
        keywords = {}
        for country in ['france', 'china']:
            if country in tfidf_results:
                keywords[country] = {}
                for sector, data in tfidf_results[country].items():
                    if 'top_words' in data:
                        top_words = data['top_words'][:10]  # 取前10个词
                        keywords[country][sector] = [f"{item['word']} ({item['score']:.4f})" for item in top_words]
        
        return {
            'title': '关键词分析',
            'description': '通过TF-IDF分析，提取各领域最具代表性的关键词，揭示中法数字化转型中的焦点话题。',
            'keywords': keywords,
            'visualizations': self.visualization_results.get('word_clouds', {})
        }
    
    def _generate_sentiment_analysis(self):
        """生成情感分析部分"""
        sentiment_results = self.analysis_results.get('sentiment', {})
        
        # 计算各国家整体情感比例
        overall_sentiment = {}
        for country in ['france', 'china']:
            if country in sentiment_results:
                positive_sum = 0
                neutral_sum = 0
                negative_sum = 0
                count = 0
                
                for sector, data in sentiment_results[country].items():
                    if 'sentiment_ratios' in data:
                        ratios = data['sentiment_ratios']
                        positive_sum += ratios.get('positive', 0)
                        neutral_sum += ratios.get('neutral', 0)
                        negative_sum += ratios.get('negative', 0)
                        count += 1
                
                if count > 0:
                    overall_sentiment[country] = {
                        'positive': positive_sum / count,
                        'neutral': neutral_sum / count,
                        'negative': negative_sum / count
                    }
        
        return {
            'title': '情感分析',
            'description': '分析中法两国数字化转型讨论中的情感倾向，反映各行业对数字化的态度。',
            'overall_sentiment': overall_sentiment,
            'sentiment_by_sector': sentiment_results,
            'visualizations': self.visualization_results.get('sentiment_charts', {})
        }
    
    def _generate_topic_analysis(self):
        """生成主题分析部分"""
        topic_results = self.analysis_results.get('topics', {})
        
        # 提取主题关键词
        topic_keywords = {}
        for country in ['france', 'china']:
            if country in topic_results:
                topic_keywords[country] = {}
                for sector, data in topic_results[country].items():
                    if 'topics' in data:
                        topics = data['topics']
                        topic_keywords[country][sector] = [
                            {
                                'id': topic['topic_id'],
                                'keywords': topic['words'][:5]  # 每个主题取前5个词
                            }
                            for topic in topics
                        ]
        
        return {
            'title': '主题分析',
            'description': '通过LDA主题模型，发现各行业讨论的主要主题，比较中法两国的关注点异同。',
            'topic_keywords': topic_keywords,
            'visualizations': self.visualization_results.get('topic_visualizations', {})
        }
    
    def _generate_cultural_analysis(self):
        """生成文化维度分析部分"""
        cultural_results = self.analysis_results.get('cultural_dimensions', {})
        
        # 提取文化维度分数
        dimension_scores = {}
        for country in ['france', 'china']:
            if country in cultural_results:
                dimension_scores[country] = {}
                for dimension in HOFSTEDE_DIMENSIONS:
                    if dimension in cultural_results[country]:
                        dimension_scores[country][dimension] = cultural_results[country][dimension]['score']
        
        # 计算中法差异
        differences = {}
        for dimension in HOFSTEDE_DIMENSIONS:
            if ('france' in dimension_scores and dimension in dimension_scores['france'] and
                'china' in dimension_scores and dimension in dimension_scores['china']):
                differences[dimension] = dimension_scores['china'][dimension] - dimension_scores['france'][dimension]
        
        return {
            'title': '文化维度分析',
            'description': '基于Hofstede文化维度理论，分析中法两国在数字化转型中体现的文化特征。',
            'dimension_scores': dimension_scores,
            'differences': differences,
            'hofstede_dimensions': {
                'power_distance': '权力距离 - 社会中权力不平等的接受程度',
                'individualism': '个人主义 - 个体与集体利益的权衡',
                'masculinity': '男性化 - 竞争与合作、成就与关怀的偏好',
                'uncertainty_avoidance': '不确定性规避 - 对未知情况的容忍度',
                'long_term_orientation': '长期导向 - 对未来与过去传统的重视程度',
                'indulgence': '放纵 - 欲望与冲动的控制程度'
            },
            'visualizations': self.visualization_results.get('cultural_radar_charts', {})
        }
    
    def _generate_comparison(self):
        """生成中法比较部分"""
        return {
            'title': '中法数字化转型对比',
            'description': '总结中法两国在数字化转型各方面的关键差异与共同点。',
            'key_differences': [
                '中国数字化转型更注重速度和规模，而法国更注重安全和隐私保护',
                '中国采用自上而下的政府引导模式，法国则平衡政府引导和市场驱动',
                '中国在移动支付、电子商务等领域的应用更为普及，法国在工业自动化领域具有优势',
                '中国展现出较高的技术接受度，法国对新技术的采纳相对谨慎',
                '中国的数字化转型受集体主义文化影响，法国则体现出更强的个人主义特征'
            ],
            'commonalities': [
                '两国都将数字化转型视为国家战略重点',
                '都面临数字人才缺口和数字鸿沟问题',
                '都重视数据安全和网络安全',
                '都在推动传统产业与数字技术融合'
            ],
            'visualizations': self.visualization_results.get('country_comparison_charts', {})
        }
    
    def _generate_conclusion(self):
        """生成结论部分"""
        return {
            'title': '结论',
            'key_findings': [
                '中法两国在数字化转型中体现了各自的文化特征，影响了技术采纳模式和实施策略',
                '中国的集体主义和低不确定性规避特征促进了快速大规模的数字化应用',
                '法国的高不确定性规避和个人主义特征导致对数据保护和用户隐私的高度重视',
                '文化因素是影响数字化转型速度、深度和方向的关键变量',
                '成功的跨国数字化合作需要充分认识并适应文化差异'
            ]
        }
    
    def _generate_recommendations(self):
        """生成建议部分"""
        return {
            'title': '建议',
            'for_china': [
                '加强数据安全和用户隐私保护，借鉴法国GDPR实践经验',
                '注重数字化质量而非仅追求速度和规模',
                '在保持创新活力的同时，建立更完善的监管框架',
                '在与欧洲合作时，更加尊重当地的文化特性和法律要求'
            ],
            'for_france': [
                '加快数字化转型步伐，减少过度谨慎导致的机会损失',
                '简化数字创新的行政程序，降低创新成本',
                '学习中国在移动支付、电子商务等领域的成功经验',
                '在与中国合作时，理解其集体主义文化背景和决策机制'
            ],
            'for_cooperation': [
                '建立中法数字化合作平台，促进技术和经验交流',
                '共同制定符合双方文化特点的数字标准和规范',
                '开展联合研究，探索文化适应性数字解决方案',
                '在教育领域加强合作，培养具有跨文化素养的数字人才'
            ]
        }
    
    def _generate_html_report(self, content):
        report_date = datetime.now().strftime('%Y%m%d')
        """生成HTML报告"""
        try:
            # 加载模板
            template = self.template_env.get_template('report_template.html')
            
            # 渲染模板
            html_content = template.render(**content)
            
            # 保存HTML报告
            html_path = os.path.join(self.html_dir, f'digital_transformation_report_{report_date}.html')
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML报告已保存至: {html_path}")
            
            return html_path
            
        except Exception as e:
            logger.error(f"生成HTML报告时出错: {str(e)}")
            
            # 生成一个简单的HTML报告作为后备
            simple_html = f"""
            <html>
            <head>
                <title>{content['title']}</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .section {{ margin-bottom: 30px; }}
                </style>
            </head>
            <body>
                <h1>{content['title']}</h1>
                <h2>{content['subtitle']}</h2>
                <p>生成日期: {content['date']}</p>
                
                <div class="section">
                    <h2>摘要</h2>
                    <p>{content['summary']['introduction']}</p>
                </div>
                
                <div class="section">
                    <h2>结论</h2>
                    <ul>
                        {"".join([f"<li>{finding}</li>" for finding in content['conclusion']['key_findings']])}
                    </ul>
                </div>
            </body>
            </html>
            """
            
            backup_path = os.path.join(self.html_dir, f'simple_report_{report_date}.html')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(simple_html)
                
            logger.info(f"简化HTML报告已保存至: {backup_path}")
            
            return backup_path
    
    def _generate_pdf_report(self, html_path):
        """生成PDF报告"""
        if not WEASYPRINT_AVAILABLE:
            logger.warning("WeasyPrint不可用，无法生成PDF报告")
            return None
        
        try:
            # 读取HTML内容
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # 生成PDF
            pdf_path = os.path.join(self.pdf_dir, "report.pdf")
            weasyprint.HTML(string=html_content).write_pdf(pdf_path)
            
            return pdf_path
        except Exception as e:
            logger.error(f"生成PDF报告时出错: {str(e)}")
            return None 