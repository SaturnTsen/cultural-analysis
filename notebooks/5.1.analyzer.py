from config import PROCESSED_DATA_DIR, RESULTS_DIR, HOFSTEDE_DIMENSIONS, ANALYSIS_CONFIG, EMBEDDINGS_DIR
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime
from typing import *
import asyncio
from glob import glob
from tqdm import tqdm
import numpy as np
import requests
import json
import logging
import httpx
import sys
import os
"""
数据分析模块：分析预处理后的文本数据
"""

base_url = "https://api.siliconflow.cn/v1/embeddings"
payload_model = "Pro/BAAI/bge-m3"
logger = logging.getLogger('Data Analyzer')


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
        self.model = payload_model
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url
        if not self.api_key:
            raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
        self.embedding_dir = EMBEDDINGS_DIR

    def analyze_all(self):
        """运行所有分析"""
        logger.info("开始所有数据分析")
        # 词频分析
        tf_idf_results = self.analyze_tf_idf()
        # 主题分析
        topic_results = self.analyze_topics()
        # 文化维度分析
        culture_results = self.analyze_cultural_dimensions()
        # 组合结果
        analysis_results = {
            'tf_idf': tf_idf_results,
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
                    sector_dir, "processed_metadata_*.json"))

                for metadata_file in tqdm(metadata_files, desc=f"加载{country}的{sector}领域数据"):
                    try:
                        # 读取metadata文件
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)

                        # 构造对应的 processed_sentences 和 processed_words 文件路径
                        base_name = os.path.basename(metadata_file).replace(
                            "processed_metadata_", "").replace(".json", "")
                        directory = os.path.dirname(metadata_file)
                        sentences_file = os.path.join(
                            directory, f"processed_sentences_{base_name}.txt")
                        words_file = os.path.join(
                            directory, f"processed_words_{base_name}.txt")
                        if not sentences_file or not words_file:
                            logger.warning(
                                f"metadata文件{metadata_file}缺少必要的文件路径信息")
                            continue

                        # 加载分词后的文本
                        if os.path.exists(words_file):
                            with open(words_file, 'r', encoding='utf-8') as f:
                                processed_text = f.read()
                                corpus[country]['processed_texts'][sector].append(
                                    processed_text)
                        else:
                            logger.warning(f"找不到处理后的文本文件: {words_file}")
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
            for sector, texts in tqdm(self.corpus[country]['processed_texts'].items(), desc=f"TF-IDF"):
                logger.info(f"正在处理 {country} 的 {sector} 领域")
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
        logger.info("TF-IDF分析完成")

        return results

    def analyze_topics(self):
        """主题分析，使用LDA提取文本主题"""
        logger.info("开始主题分析")
        results = {}
        num_topics = 3  # 主题数量
        num_topics_overeall = 5  # 整体主题数量
        for country in self.corpus.keys():
            results[country] = {}
            for sector, texts in tqdm(self.corpus[country]['processed_texts'].items(), desc=f"主题分析"):
                logger.info(f"分析{country}的{sector}领域主题")
                if not texts:  # 空行业
                    raise ValueError(f"{country}的{sector}领域没有文本数据进行主题分析")
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

            all_texts = []
            for sector_texts in self.corpus[country]['processed_texts'].values():
                all_texts.extend(sector_texts)
                if all_texts:
                    vectorizer = CountVectorizer(max_features=1000)
                try:
                    dtm = vectorizer.fit_transform(all_texts)
                    feature_names = vectorizer.get_feature_names_out()
                    lda = LatentDirichletAllocation(
                        n_components=num_topics_overeall, random_state=42)
                    lda.fit(dtm)
                    topics = []
                    for topic_idx, topic in enumerate(lda.components_):
                        top_words_idx = topic.argsort()[:-11:-1]
                        top_words = [feature_names[i] for i in top_words_idx]
                        topics.append({
                            'topic_id': topic_idx,
                            'words': top_words,
                            'weights': [float(topic[i]) for i in top_words_idx]
                        })
                    results[country]['overall'] = {
                        'topics': topics,
                        'document_count': len(all_texts)
                    }
                except Exception as e:
                    logger.error(f"分析{country}的整体主题时出错: {str(e)}")
                    results[country]['overall'] = {
                        'topics': [],
                        'document_count': len(all_texts),
                        'error': str(e)
                    }

        # 保存结果
        logger.info("主题分析完成")

        return results

    def analyze_cultural_dimensions(self):
        """基于 Hofstede 文化维度分析（并行跑中法两国）"""
        # 为六个维度指定正-负 anchor 词
        HOFSTEDE_DESCRIPTIONS_ANCHORS = {
            #   维度名 : [正向 anchor（+） , 负向 anchor（–）]
            # 权力距离
            "power_distance": ["authority", "equality"],
            # 个人主义
            "individualism": ["individual", "collective"],
            # 成就 / 关系
            "Motivation_towards_achievement": ["achievement", "harmony"],
            # 不确定性规避
            "uncertainty_avoidance": ["certainty", "uncertainty"],
            # 长期导向
            "long_term_orientation": ["future", "tradition"],
            # 纵欲 / 克制
            "indulgence": ["indulgence", "restraint"]
        }

        logger.info("开始文化维度分析")
        anchor_vectors = asyncio.run(
            self._compute_anchor_vectors(HOFSTEDE_DESCRIPTIONS_ANCHORS))

        # -------- 收集句子 ----------
        all_sentences = {'china': [], 'france': []}
        for country in self.corpus:
            for sent_lists in self.corpus[country]['sentences'].values():
                for s_list in sent_lists:
                    all_sentences[country].extend(
                        [s for s in s_list if s.strip()])

        results = {'china': {}, 'france': {}}

        async def process_country(country: str):
            """嵌入 + 缓存 + 投影，返回 (country, projections_dict)"""
            sentences = all_sentences[country]
            cache_file = os.path.join(
                self.embedding_dir, f"{country}_sentence_embs.npy")

            # ---------- 读 / 计算嵌入 ----------
            if os.path.exists(cache_file):
                logger.info(f"[{country}] 载入缓存 {cache_file}")
                sentence_embs = np.load(cache_file, mmap_mode='r')
            else:
                if not sentences:
                    logger.warning(f"[{country}] 无句子可分析，跳过")
                    return country, {}
                logger.info(f"[{country}] 句子 {len(sentences)}，开始嵌入")
                # ↓ 调低并发，避免两国叠加超限
                embs = await self._embed_sentences(sentences, max_concurrency=75)
                sentence_embs = np.asarray(embs, dtype=np.float32)
                np.save(cache_file, sentence_embs)
                logger.info(f"[{country}] 嵌入完成并缓存至 {cache_file}")

            # ---------- 投影 ----------
            proj_dict = {}
            for dim, anchor_vec in anchor_vectors.items():
                projections = sentence_embs @ anchor_vec
                proj_dict[dim] = {
                    'description': dim,
                    'avg_value': float(projections.mean()),
                    'sentence_count': len(sentences)
                }
            return country, proj_dict

        # -------- 并发跑两个国家 --------
        async def _run_two_countries():
            """在事件循环内并发执行两个 process_country 协程"""
            return await asyncio.gather(*(process_country(c) for c in ('china', 'france')))

        projections = asyncio.run(_run_two_countries())
        for country, proj in projections:
            results[country] = proj

        logger.info("文化维度分析完成")
        return results

    async def _get_embedding(self, text: str, timeout=60.0) -> List[float]:
        # returns a list of 1024 floats
        # Log the first 50 characters of the text
        logger.info(f"Fetching embedding for text: {text[:50]}...")
        payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float"
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()['data'][0]['embedding']

        # 计算所有 anchor 向量（并行）
    async def _compute_anchor_vectors(self, description_text_anchors):
        tasks = []
        for pos, neg in description_text_anchors.values():
            tasks.append(self._get_embedding(pos))
            tasks.append(self._get_embedding(neg))
        emb_list = await asyncio.gather(*tasks)           # 一次性并发
        it = iter(emb_list)
        anchor_vecs = {}
        for dim, (pos, neg) in description_text_anchors.items():
            pos_vec = np.array(next(it))
            neg_vec = np.array(next(it))
            vec = pos_vec - neg_vec                       # v = v_pos – v_neg
            anchor_vecs[dim] = vec / (np.linalg.norm(vec) + 1e-8)  # 单位化
        return anchor_vecs

    async def _embed_sentences(self, sentences, max_concurrency=100,
                               batch_size=256, retries=10, base_backoff=3.0):
        """
        并发获取句子嵌入：
        - tqdm 进度条
        - 每条句子在 RequestError/Timeout 时自动重试 (指数退避)
        """
        sem = asyncio.Semaphore(max_concurrency)
        pbar = tqdm(total=len(sentences), desc="Embedding sentences", ncols=80)

        async def _embed_once(text):
            async with sem:
                return await self._get_embedding(text)

        async def _embed_with_retry(text):
            for attempt in range(retries):
                try:
                    return await _embed_once(text)
                except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                    if attempt == retries - 1:
                        logger.error(f"获取嵌入失败（已重试 {retries} 次）: {exc}")
                        # 仍抛出，让上层决定如何处理（可改为返回 None）
                        raise
                    backoff = base_backoff * 2 ** attempt
                    logger.warning(f"请求失败，第 {attempt+1} 次重试，等待 {backoff:.1f}s")
                    await asyncio.sleep(backoff)

        results, idx = [], 0
        while idx < len(sentences):
            batch = sentences[idx: idx + batch_size]
            batch_embs = await asyncio.gather(*[_embed_with_retry(t) for t in batch])
            results.extend(batch_embs)
            idx += len(batch)
            pbar.update(len(batch))

        pbar.close()
        return results


if __name__ == "__main__":
    #     logging.basicConfig(
    #         level=logging.INFO,
    #         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #         handlers=[
    #             logging.StreamHandler(sys.stdout),
    #             logging.FileHandler(os.path.join(RESULTS_DIR, 'data_analyzer.log'))
    #         ])
    analyzer = DataAnalyzer()
    analyzer.analyze_all()
