"""
数据分析模块：分析预处理后的文本数据
"""

import os
import json
import numpy as np
from glob import glob
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

from src.utils.logger import get_logger
from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, HOFSTEDE_DIMENSIONS, ANALYSIS_CONFIG

logger = get_logger("analysis.analyzer")


class DataAnalyzer:
    """数据分析类，实现各种文本分析方法"""

    def __init__(self):
        """初始化数据分析器"""
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.results_dir = RESULTS_DIR

        # 确保结果目录存在
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # 加载文本语料库
        self.corpus = self._load_corpus()

        # 初始化情感分析模型
        self.sentiment_models = {}

    def analyze_all(self):
        """运行所有分析"""
        logger.info("开始所有数据分析")

        # 词频分析
        tf_idf_results = self.analyze_tf_idf()

        # 情感分析
        sentiment_results = self.analyze_sentiment()

        # 主题分析
        topic_results = self.analyze_topics()

        # 文化维度分析
        culture_results = self.analyze_cultural_dimensions()

        # 组合结果
        analysis_results = {
            'tf_idf': tf_idf_results,
            'sentiment': sentiment_results,
            'topics': topic_results,
            'cultural_dimensions': culture_results,
            'analysis_date': datetime.now().isoformat()
        }

        # 保存综合结果
        results_file = os.path.join(self.results_dir, "analysis_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)

        logger.info("所有数据分析完成")

        return analysis_results

    def _load_corpus(self):
        """加载预处理后的文本语料库"""
        logger.info("加载预处理后的文本语料库")

        corpus = {
            'france': {'processed_texts': {}, 'original_texts': {}, 'sentences': {}},
            'china': {'processed_texts': {}, 'original_texts': {}, 'sentences': {}}
        }

        # 加载预处理后的文本
        for country in ['france', 'china']:
            country_dir = os.path.join(self.processed_data_dir, country)
            if not os.path.exists(country_dir):
                logger.warning(f"找不到{country}的数据目录: {country_dir}")
                continue

            # 获取所有行业目录
            sector_dirs = glob(os.path.join(country_dir, "*"))

            for sector_dir in sector_dirs:
                sector = os.path.basename(sector_dir)
                logger.info(f"加载{country}的{sector}领域数据")

                # 为每个行业初始化数据存储
                if sector not in corpus[country]['processed_texts']:
                    corpus[country]['processed_texts'][sector] = []
                    corpus[country]['original_texts'][sector] = []
                    corpus[country]['sentences'][sector] = []

                # 查找所有metadata文件
                metadata_files = glob(os.path.join(
                    sector_dir, "metadata_*.json"))

                for metadata_file in metadata_files:
                    try:
                        # 读取metadata文件
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)

                        # 获取对应的处理后的文本文件和句子文件
                        original_file = metadata.get('file_path')
                        processed_file = metadata.get('processed_file')
                        sentences_file = metadata.get('sentences_file')

                        if not processed_file or not sentences_file:
                            logger.warning(
                                f"metadata文件{metadata_file}缺少必要的文件路径信息")
                            continue

                        # 加载原始文本
                        if os.path.exists(original_file):
                            with open(original_file, 'r', encoding='utf-8') as f:
                                original_text = f.read()
                                corpus[country]['original_texts'][sector].append(
                                    original_text)
                        else:
                            logger.warning(f"找不到原始文本文件: {original_file}")

                        # 加载分词后的文本
                        if os.path.exists(processed_file):
                            with open(processed_file, 'r', encoding='utf-8') as f:
                                processed_text = f.read()
                                corpus[country]['processed_texts'][sector].append(
                                    processed_text)
                        else:
                            logger.warning(f"找不到处理后的文本文件: {processed_file}")

                        # 加载分句后的文本
                        if os.path.exists(sentences_file):
                            with open(sentences_file, 'r', encoding='utf-8') as f:
                                sentences = f.read().split('\n')
                                corpus[country]['sentences'][sector].append(
                                    sentences)
                        else:
                            logger.warning(
                                f"找不到分句后的文本文件: {sentences_file}")

                    except Exception as e:
                        logger.error(
                            f"处理metadata文件{metadata_file}时出错: {str(e)}")
                        continue

        logger.info(f"语料库加载完成")
        return corpus

    def analyze_tf_idf(self):
        """TF-IDF分析，找出每个国家和行业的关键词"""
        logger.info("开始TF-IDF分析")

        results = {}

        for country in self.corpus.keys():
            results[country] = {}

            # 组合该国家所有行业的文本
            all_texts = []
            sector_indices = {}
            current_index = 0

            for sector, texts in self.corpus[country]['processed_texts'].items():
                sector_indices[sector] = (
                    current_index, current_index + len(texts))
                all_texts.extend(texts)
                current_index += len(texts)

            # 如果没有文本，跳过
            if not all_texts:
                logger.warning(f"{country}没有文本数据进行TF-IDF分析")
                continue

            # 创建TF-IDF向量化器
            vectorizer = TfidfVectorizer(max_features=1000)

            # 转换文本
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # 获取特征名称，即Tf-idf矩阵的每一列对应的单词
            feature_names = vectorizer.get_feature_names_out()

            # 对每个行业计算平均TF-IDF值
            for sector, (start_idx, end_idx) in sector_indices.items():
                if start_idx == end_idx:  # 空行业
                    continue

                # 提取该行业的TF-IDF子矩阵
                sector_tfidf = tfidf_matrix[start_idx:end_idx]

                # 计算平均TF-IDF
                avg_tfidf = sector_tfidf.mean(axis=0).A1

                # 创建词-分数对，当前国家当前行业里每个词的tf-idf分数
                word_scores = list(zip(feature_names, avg_tfidf))

                # 排序
                word_scores.sort(key=lambda x: x[1], reverse=True)

                # 取前50个词
                top_words = word_scores[:50]

                # 存储结果
                results[country][sector] = {
                    'top_words': [{'word': word, 'score': float(score)} for word, score in top_words],
                    'word_count': len(all_texts)
                }

        # 保存结果
        results_file = os.path.join(self.results_dir, "tf_idf_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("TF-IDF分析完成")

        return results

    def analyze_sentiment(self):
        """情感分析，评估文本的情感倾向"""
        logger.info("开始情感分析")

        results = {}

        for country in self.corpus.keys():
            results[country] = {}

            for sector, sentences_list in self.corpus[country]['sentences'].items():
                logger.info(f"分析{country}的{sector}领域情感")

                if not sentences_list:  # 空行业
                    continue

                # 扁平化句子列表
                # TODO 去掉短句
                all_sentences = [
                    sentence for doc_sentences in sentences_list for sentence in doc_sentences if sentence.strip()]

                if not all_sentences:  # 没有句子
                    continue

                # 获取样本（最多500句）
                sample_size = min(500, len(all_sentences))
                sample_sentences = list(np.random.choice(
                    all_sentences, sample_size, replace=False))

                # 分析情感
                sentiments = self._analyze_sentences_sentiment(
                    sample_sentences, country)

                # 统计结果
                sentiment_counts = Counter(sentiments)
                total = sum(sentiment_counts.values())

                # 计算比例
                sentiment_ratios = {
                    'positive': sentiment_counts.get('positive', 0) / total,
                    'neutral': sentiment_counts.get('neutral', 0) / total,
                    'negative': sentiment_counts.get('negative', 0) / total
                }

                # 存储结果
                results[country][sector] = {
                    'sentiment_counts': {k: v for k, v in sentiment_counts.items()},
                    'sentiment_ratios': sentiment_ratios,
                    'sample_size': sample_size,
                    'total_sentences': len(all_sentences)
                }

        # 保存结果
        results_file = os.path.join(self.results_dir, "sentiment_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("情感分析完成")

        return results

    def _analyze_sentences_sentiment(self, sentences, country):
        """分析句子列表的情感"""
        results = []

        # 获取适合该国家的语言模型
        # TODO 那么语料库应该中国相关的都变成汉语，法国相关的都变成法语
        if country == 'france':
            lang = 'fr'
        elif country == 'china':
            lang = 'zh'
        else:
            lang = 'en'  # 默认英语

        # 如果该语言的模型尚未加载，则加载
        if lang not in self.sentiment_models:
            try:
                # 使用transformers的pipeline
                model_name = ANALYSIS_CONFIG['sentiment']['transformer_model'][lang]
                logger.info(f"加载{lang}情感分析模型: {model_name}")

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name)
                self.sentiment_models[lang] = pipeline(
                    'sentiment-analysis', model=model, tokenizer=tokenizer)
            except Exception as e:
                logger.error(f"加载{lang}情感分析模型时出错: {str(e)}")

                # 出错时使用英文模型代替
                if 'en' not in self.sentiment_models:
                    tokenizer = AutoTokenizer.from_pretrained(
                        ANALYSIS_CONFIG['sentiment']['transformer_model']['en'])
                    model = AutoModelForSequenceClassification.from_pretrained(
                        ANALYSIS_CONFIG['sentiment']['transformer_model']['en'])
                    self.sentiment_models['en'] = pipeline(
                        'sentiment-analysis', model=model, tokenizer=tokenizer)
                self.sentiment_models[lang] = self.sentiment_models['en']

        # 批处理分析
        batch_size = 16
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]

            try:
                # 使用模型分析情感
                outputs = self.sentiment_models[lang](batch)

                # 解析结果
                for output in outputs:
                    if isinstance(output, list):
                        output = output[0]  # 某些模型返回列表

                    label = output['label']
                    score = output['score']

                    # 根据阈值和标签确定情感
                    if 'positive' in label.lower() or score > ANALYSIS_CONFIG['sentiment']['threshold']['positive']:
                        results.append('positive')
                    elif 'negative' in label.lower() or score < ANALYSIS_CONFIG['sentiment']['threshold']['negative']:
                        results.append('negative')
                    else:
                        results.append('neutral')

            except Exception as e:
                logger.error(f"情感分析批处理时出错: {str(e)}")
                # 出错时标记为中性
                results.extend(['neutral'] * len(batch))

        return results

    def analyze_topics(self):
        """主题分析，使用LDA提取文本主题"""
        logger.info("开始主题分析")

        results = {}
        num_topics = 5  # 主题数量

        for country in self.corpus.keys():
            results[country] = {}

            for sector, texts in self.corpus[country]['processed_texts'].items():
                logger.info(f"分析{country}的{sector}领域主题")

                if not texts:  # 空行业
                    continue

                # 创建词袋向量化器
                vectorizer = CountVectorizer(max_features=1000)

                # 转换文本
                try:
                    dtm = vectorizer.fit_transform(texts)
                    feature_names = vectorizer.get_feature_names_out()

                    # 训练LDA模型
                    lda = LatentDirichletAllocation(
                        n_components=num_topics, random_state=42)
                    lda.fit(dtm)

                    # 提取主题词
                    topics = []
                    for topic_idx, topic in enumerate(lda.components_):
                        # 获取排名前10的词
                        top_words_idx = topic.argsort()[:-11:-1]
                        top_words = [feature_names[i] for i in top_words_idx]

                        topics.append({
                            'topic_id': topic_idx,
                            'words': top_words,
                            'weights': [float(topic[i]) for i in top_words_idx]
                        })

                    # 存储结果
                    results[country][sector] = {
                        'topics': topics,
                        'document_count': len(texts)
                    }

                except Exception as e:
                    logger.error(f"分析{country}的{sector}领域主题时出错: {str(e)}")
                    results[country][sector] = {
                        'topics': [],
                        'document_count': len(texts),
                        'error': str(e)
                    }

        # 保存结果
        results_file = os.path.join(self.results_dir, "topic_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("主题分析完成")

        return results

    def analyze_cultural_dimensions(self):
        """基于Hofstede文化维度理论进行分析"""
        # 把： 文化维度分数 （结构性指标）和 文本关键词计数 （局部样本特征）
        # 强行混在一起加权平均 → 理论上是不成立的
        logger.info("开始文化维度分析")

        results = {}

        for country in self.corpus.keys():
            results[country] = {}

            # 合并所有行业的文本进行整体分析
            all_texts = []
            for sector, texts in self.corpus[country]['processed_texts'].items():
                all_texts.extend(texts)

            if not all_texts:
                logger.warning(f"{country}没有文本数据进行文化维度分析")
                continue

            # 将所有文本合并为一个大文本进行分析
            combined_text = " ".join(all_texts)

            # 对每个文化维度进行分析
            for dimension in HOFSTEDE_DIMENSIONS[country].keys():
                # 定义与该维度相关的关键词（这需要根据实际情况定制）
                # 这里简单演示，实际应用时需要更复杂的分析
                if dimension == 'power_distance':
                    keywords_high = ['hierarchy', 'authority',
                                     'power', 'order', 'obedience']
                    keywords_low = ['equality', 'flat',
                                    'democratic', 'participation', 'consensus']
                elif dimension == 'individualism':
                    keywords_high = ['individual', 'personal',
                                     'unique', 'self', 'independent']
                    keywords_low = ['collective', 'group',
                                    'team', 'cooperation', 'harmony']
                # ... 其他维度的关键词

                # 计算关键词频率
                high_score = sum(combined_text.count(word)
                                 for word in keywords_high)
                low_score = sum(combined_text.count(word)
                                for word in keywords_low)

                # 如果有足够的关键词匹配
                if high_score + low_score > 10:
                    # 计算偏好比例
                    ratio = high_score / \
                        (high_score + low_score) if high_score + \
                        low_score > 0 else 0.5

                    # 基于Hofstede的标准国家分数进行调整
                    base_score = HOFSTEDE_DIMENSIONS[country][dimension]

                    # 文本分析分数影响最终分数的权重
                    weight = 0.2

                    # 计算最终分数，介于0-100之间
                    dimension_score = base_score * \
                        (1 - weight) + 100 * ratio * weight
                else:
                    # 如果没有关键词匹配，使用标准分数
                    dimension_score = HOFSTEDE_DIMENSIONS[country][dimension]

                # 存储结果
                results[country][dimension] = {
                    'score': dimension_score,
                    'base_score': HOFSTEDE_DIMENSIONS[country][dimension],
                    'text_analysis_keywords': {
                        'high': {word: combined_text.count(word) for word in keywords_high},
                        'low': {word: combined_text.count(word) for word in keywords_low}
                    }
                }

        # 保存结果
        results_file = os.path.join(
            self.results_dir, "cultural_dimensions_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("文化维度分析完成")

        return results

    def load_results(self):
        """从保存的文件加载分析结果"""
        logger.info("从文件加载分析结果")

        results_file = os.path.join(self.results_dir, "analysis_results.json")

        if not os.path.exists(results_file):
            logger.warning(f"分析结果文件不存在: {results_file}")
            # 返回空结果
            return {
                'tf_idf': {},
                'sentiment': {},
                'topics': {},
                'cultural_dimensions': {},
                'analysis_date': datetime.now().isoformat()
            }

        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"成功加载分析结果")
            return results
        except Exception as e:
            logger.error(f"加载分析结果时出错: {str(e)}")
            # 返回空结果
            return {
                'tf_idf': {},
                'sentiment': {},
                'topics': {},
                'cultural_dimensions': {},
                'analysis_date': datetime.now().isoformat()
            }
