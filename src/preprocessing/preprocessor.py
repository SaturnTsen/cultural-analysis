"""
数据预处理模块：处理收集的原始数据
"""

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LANGUAGE_CONFIG
from src.utils.helpers import ensure_dir, save_json, get_file_extension
from src.utils.logger import get_logger
from pathlib import Path
import os
import re
import json
import logging
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
import langdetect
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import jieba
from typing import *

logger = logging.getLogger(__name__)


# 确保NLTK相关资源下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理类，负责清洗和准备数据进行分析"""

    def __init__(self):
        """初始化数据预处理器"""
        self.raw_data_dir: str = RAW_DATA_DIR
        self.processed_data_dir: str = PROCESSED_DATA_DIR

        # 确保目录存在
        ensure_dir(self.processed_data_dir)

        # 加载停用词
        self.stopwords = {}
        for lang_code in LANGUAGE_CONFIG.values():
            try:
                self.stopwords[lang_code] = set(stopwords.words(
                    self._get_nltk_language_name(lang_code)))
            except:
                logger.warning(f"无法加载{lang_code}的停用词，将使用空集")
                self.stopwords[lang_code] = set()

    def preprocess_all(self):
        """预处理所有数据"""
        logger.info("开始预处理所有数据")

        # 获取所有国家目录
        country_dirs = glob(os.path.join(self.raw_data_dir, "*"))

        for country_dir in country_dirs:
            country = os.path.basename(country_dir)
            logger.info(f"预处理{country}数据")

            # 获取该国家的所有行业目录
            sector_dirs = glob(os.path.join(country_dir, "*"))

            for sector_dir in sector_dirs:
                sector = os.path.basename(sector_dir)
                logger.info(f"预处理{country}的{sector}领域数据")

                # 创建对应的处理后数据目录
                processed_sector_dir = os.path.join(
                    self.processed_data_dir, country, sector)
                if not os.path.exists(processed_sector_dir):
                    os.makedirs(processed_sector_dir)

                # 处理所有metadata文件
                metadata_files = glob(os.path.join(
                    sector_dir, "*_metadata.json"))
                for metadata_file in metadata_files:
                    try:
                        # 读取metadata
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)

                        # 获取对应的txt文件路径
                        txt_file = metadata.get('file_path')
                        if not txt_file or not os.path.exists(txt_file):
                            logger.warning(f"未找到对应的txt文件: {txt_file}")
                            continue

                        assert txt_file.endswith(
                            '.txt'), f"txt文件格式错误: {txt_file}"

                        # 处理文本文件
                        self._process_text_file(
                            country, sector, txt_file, processed_sector_dir, metadata)

                    except Exception as e:
                        logger.error(
                            f"处理metadata文件{metadata_file}时出错: {str(e)}")

        # 生成总体元数据统计
        self._generate_metadata_summary()

        logger.info("所有数据预处理完成")

    def _process_text_file(self, country: Literal['france', 'china'], sector: str, text_file: Path, output_dir: Path, metadata: Dict):
        """处理单个文本文件"""
        filename = os.path.basename(text_file)
        logger.info(f"处理文本文件: {filename}")

        try:
            # 读取文本
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # 检测语言
            lang = self._detect_language(text)

            # 标准化处理
            processed_text = self._normalize_text(text, lang)

            # 分词
            tokens = self._tokenize_text(processed_text, lang)

            # 移除停用词
            filtered_tokens = self._remove_stopwords(tokens, lang)

            # 保存处理后的文本
            processed_file = os.path.join(output_dir, f"processed_{filename}")
            with open(processed_file, 'w', encoding='utf-8') as f:
                f.write(' '.join(filtered_tokens))

            # 保存分句结果，用于后续情感分析
            sentences = self._split_sentences(text, lang)
            sentences_file = os.path.join(output_dir, f"sentences_{filename}")
            with open(sentences_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(sentences))

            # 更新元数据
            metadata.update({
                'id': filename.replace('.txt', ''),
                'processed_file': processed_file,  # 分词加过滤停用词后的文本
                'sentences_file': sentences_file,  # 分句后的文本
                'token_count': len(tokens),  # 分词后的token数量
                'filtered_token_count': len(filtered_tokens),  # 过滤停用词后的token数量
                'sentence_count': len(sentences),  # 分句后的句子数量
                'processing_date': datetime.now().isoformat()  # 处理时间
            })

            # 保存更新后的元数据
            metadata_file = os.path.join(
                output_dir, f"metadata_{filename.replace('.txt', '.json')}")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"文本文件处理完成: {filename}")

        except Exception as e:
            logger.error(f"处理文本文件{filename}时出错: {str(e)}")

    def _detect_language(self, text):
        """检测文本语言，只支持中文、英文和法文"""
        try:
            # 首先检查是否包含中文字符
            if re.search(r'[\u4e00-\u9fa5]', text):
                return 'zh'

            # 检查是否包含法语特殊字符
            if re.search(r'[éèêëàâçîïôûùüÿ]', text):
                return 'fr'

            # 使用langdetect进行检测
            lang = langdetect.detect(text)

            # 严格检查语言是否在我们支持的范围内
            if lang in ['fr']:
                return 'fr'
            elif lang in ['en']:
                return 'en'
            else:
                raise ValueError(f"检测到不支持的语言: {lang}，只支持中文(zh)、英文(en)和法文(fr)")

        except Exception as e:
            raise ValueError(f"语言检测失败: {str(e)}，只支持中文(zh)、英文(en)和法文(fr)")

    def _normalize_text(self, text, lang: Literal['zh', 'en', 'fr']):
        """标准化文本（移除特殊字符、统一大小写等）"""
        assert lang in ['zh', 'en', 'fr'], f"不支持的语言: {lang}"

        # 1. 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)

        # 2. 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text).strip()

        # 3. 对于英文和法文，转换为小写
        if lang in ['en', 'fr']:
            text = text.lower()
            # 移除英文和法文的标点符号
            text = re.sub(r'[^\w\s]', ' ', text)
            # 移除数字
            text = re.sub(r'\d+', ' ', text)

        # 中文处理
        elif lang == 'zh':
            # 移除中文标点符号
            text = re.sub(r'[^\u4e00-\u9fa5\s]', ' ', text)
            # 移除数字
            text = re.sub(r'\d+', ' ', text)

        # 4. 再次清理多余的空白字符
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _tokenize_text(self, text: str, lang: Literal['zh', 'en', 'fr']) -> List[str]:
        """分词"""
        if lang in ['en', 'fr']:
            return word_tokenize(text, language='french' if lang == 'fr' else 'english')
        elif lang == 'zh':
            return list(jieba.cut(text))
        return []

    def _remove_stopwords(self, tokens: List[str], lang: Literal['zh', 'en', 'fr']) -> List[str]:
        """移除停用词"""
        assert lang in ['zh', 'en', 'fr'], f"不支持的语言: {lang}"

        return [token for token in tokens if token not in self.stopwords[lang] and len(token) > 1]

    def _split_sentences(self, text: str, lang: Literal['zh', 'en', 'fr']) -> List[str]:
        """分句"""
        assert lang in ['zh', 'en', 'fr'], f"不支持的语言: {lang}"

        if lang in ['en', 'fr']:
            # 使用NLTK分句
            tokens = sent_tokenize(
                text, language='french' if lang == 'fr' else 'english')

            # 去掉纯标点符号和数字的句子
            tokens = [token for token in tokens if re.search(
                r'[^\w\s]', token)]

            # 去掉罗马数字、标点符号和空格的组合
            tokens = [token for token in tokens if not re.match(
                r'^[\sIVX\.,;:]+$', token.strip())]

            # 清理句子中的标点符号
            punctuation = '()[]{}"\'\'`~*_-+=,.!?;'
            cleaned_sentences = []
            for sentence in tokens:
                for char in punctuation:
                    sentence = sentence.replace(char, '')
                cleaned_sentences.append(sentence.strip())

            # 过滤空句子
            return [s for s in cleaned_sentences if s]

        elif lang == 'zh':
            # 按段落处理中文文本
            sentences = []
            for paragraph in text.split('\n'):
                if paragraph.strip():
                    # 使用句号、感叹号、问号和分号分句
                    sentences.extend(re.split(r'[。！？；]', paragraph))
            # 过滤空句子
            return [s.strip() for s in sentences if s.strip()]

    def _generate_metadata_summary(self):
        """生成元数据汇总"""
        # 查找所有元数据文件
        metadata_files = glob(os.path.join(
            self.processed_data_dir, "**", "metadata_*.json"), recursive=True)

        summary_data = []

        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # 解析路径以获取国家和行业
                parts = metadata_file.split(os.sep)
                country_idx = parts.index(
                    os.path.basename(self.processed_data_dir)) + 1
                country = parts[country_idx] if country_idx < len(
                    parts) else "unknown"
                sector = parts[country_idx + 1] if country_idx + \
                    1 < len(parts) else "unknown"

                summary_entry = {
                    'id': metadata.get('id', ''),
                    'country': country,
                    'sector': sector,
                    'language': metadata.get('language', 'unknown'),
                    'token_count': metadata.get('token_count', 0),
                    'filtered_token_count': metadata.get('filtered_token_count', 0),
                    'sentence_count': metadata.get('sentence_count', 0),
                    'processing_date': metadata.get('processing_date', ''),
                    'metadata_file': metadata_file
                }

                summary_data.append(summary_entry)

            except Exception as e:
                logger.error(f"处理元数据文件{metadata_file}时出错: {str(e)}")

        # 创建数据表
        df = pd.DataFrame(summary_data)

        # 保存汇总
        summary_file = os.path.join(
            self.processed_data_dir, "corpus_summary.csv")
        df.to_csv(summary_file, index=False, encoding='utf-8')

        # 生成统计信息
        stats = {
            'total_documents': len(df),
            'documents_by_country': df['country'].value_counts().to_dict(),
            'documents_by_sector': df['sector'].value_counts().to_dict(),
            'documents_by_language': df['language'].value_counts().to_dict(),
            'total_tokens': int(df['token_count'].sum()),
            'total_filtered_tokens': int(df['filtered_token_count'].sum()),
            'total_sentences': int(df['sentence_count'].sum()),
            'average_tokens_per_document': float(df['token_count'].mean()),
            'average_sentences_per_document': float(df['sentence_count'].mean()),
            'generation_date': datetime.now().isoformat()
        }

        # 保存统计信息
        stats_file = os.path.join(self.processed_data_dir, "corpus_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"元数据汇总生成完成，总共处理了{len(df)}个文档")

    def _get_nltk_language_name(self, lang_code):
        """根据语言代码获取NLTK语言名称"""
        # NLTK语言代码映射
        nltk_language_map = {
            'zh': 'chinese',
            'fr': 'french',
            'en': 'english',
            'de': 'german',
            'es': 'spanish',
            'it': 'italian',
            'nl': 'dutch',
            'pt': 'portuguese',
            'ru': 'russian'
        }

        if lang_code in nltk_language_map:
            return nltk_language_map[lang_code]

        logger.warning(f"无法映射语言代码 {lang_code} 到NLTK语言名称，将使用英语")
        return 'english'  # 默认返回英语
