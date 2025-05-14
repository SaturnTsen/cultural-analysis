import os
import importlib.util
import sys

config_path = Path(__file__).parent.parent / 'notebooks' / "config.py"
module_name = "config"

spec = importlib.util.spec_from_file_location(module_name, config_path)
config = importlib.util.module_from_spec(spec)
sys.modules[module_name] = config
spec.loader.exec_module(config)

RAW_DATA_DIR = config.RAW_DATA_DIR
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR

from pathlib import Path
import re
import json
import logging
import pandas as pd
from glob import glob
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import wordnet, stopwords
import jieba
from typing import *
from tqdm import tqdm
import spacy

# 确保NLTK相关资源下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger('Data Preprocessor')

LANGUAGE_CONFIG = {
    "china": "zh",
    "france": "fr",
    "en": "en",
}


class DataPreprocessor:
    """数据预处理类，负责清洗和准备数据进行分析"""

    def __init__(self):
        """初始化数据预处理器"""
        self.raw_data_dir: str = RAW_DATA_DIR
        self.processed_data_dir: str = PROCESSED_DATA_DIR
        self._lemmatizer = WordNetLemmatizer()
        self._nlp_fr = spacy.load("fr_core_news_sm")
        # 确保目录存在
        os.makedirs(self.processed_data_dir, exist_ok=True)

        # 加载停用词
        self.stopwords = {}
        for lang_code in LANGUAGE_CONFIG.values():
            try:
                self.stopwords[lang_code] = \
                    set(stopwords.words(self._get_nltk_language_name(lang_code))). \
                    union(spacy.blank(lang_code).Defaults.stop_words)
            except:
                raise ValueError(
                    f"无法加载{lang_code}的停用词，请检查NLTK数据是否完整")

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

                for txt_file in tqdm(os.listdir(sector_dir), desc=f"处理{sector}领域数据"):
                    if not txt_file.endswith('.txt') or not os.path.exists(os.path.join(sector_dir, txt_file)):
                        continue
                    # 处理文本文件
                    text_path = os.path.join(sector_dir, txt_file)
                    json_file = txt_file.replace('.txt', '.json')
                    json_path = os.path.join(sector_dir, json_file)
                    if not os.path.isfile(json_path):
                        continue

                    # 读取对应的 metadata
                    with open(json_path, 'r', encoding='utf-8') as f:
                        # print(f"正在处理元数据文件: {json_path}")
                        metadata = json.load(f)
                    self._process_text_file(
                        country, sector, text_path, processed_sector_dir, metadata)

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
            lang = metadata['语言']
            # 标准化处理
            processed_text = self._normalize_text(text, lang)
            # 分词
            tokens = self._tokenize_text(processed_text, lang)
            # 词形还原
            lemmatized_tokens = self._lemmatize_tokens(tokens, lang)
            # 移除停用词
            filtered_tokens = self._remove_stopwords(lemmatized_tokens, lang)

            # 保存处理后的文本
            processed_file = os.path.join(
                output_dir, f"processed_words_{filename}")
            with open(processed_file, 'w', encoding='utf-8') as f:
                f.write(' '.join(filtered_tokens))

            # 保存分句结果，用于后续情感分析
            sentences = self._split_sentences(text, lang)
            sentences_file = os.path.join(
                output_dir, f"processed_sentences_{filename}")
            with open(sentences_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(sentences))

            # 更新元数据
            metadata.update({
                'token_count': len(tokens),  # 分词后的token数量
                'filtered_token_count': len(filtered_tokens),  # 过滤停用词后的token数量
                'sentence_count': len(sentences),  # 分句后的句子数量
            })

            # 保存更新后的元数据
            metadata_file = os.path.join(
                output_dir, f"processed_metadata_{filename.replace('.txt', '.json')}")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"文本文件处理完成: {filename}")

        except Exception as e:
            logger.error(f"处理文本文件{filename}时出错: {str(e)}")

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

    def _lemmatize_tokens(self, tokens: Literal['zh', 'en', 'fr'], lang: str) -> List[str]:
        if lang.lower() == 'en':
            return [self._lemmatizer.lemmatize(token) for token in tokens]
        elif lang.lower() == 'fr':
            doc = self._nlp_fr(" ".join(tokens))
            return [token.lemma_ for token in doc]

        else:
            # 默认原样返回
            return tokens

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
        """按行分句（每行视为一个句子），忽略空行"""
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _generate_metadata_summary(self):
        """生成元数据汇总和按国家的详细统计"""
        # 查找所有元数据文件
        metadata_files = glob(os.path.join(
            self.processed_data_dir, "**", "processed_metadata_*.json"), recursive=True)

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

        # 保存汇总表
        summary_file = os.path.join(
            self.processed_data_dir, "corpus_summary.csv")
        df.to_csv(summary_file, index=False, encoding='utf-8')

        # 总体统计
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
        }

        # 按国家统计详细token信息
        token_stats_by_country = df.groupby('country').agg({
            'token_count': 'sum',
            'filtered_token_count': 'sum',
            'sentence_count': 'sum',
            'id': 'count'
        }).rename(columns={
            'token_count': 'total_tokens',
            'filtered_token_count': 'total_filtered_tokens',
            'sentence_count': 'total_sentences',
            'id': 'document_count'
        }).to_dict(orient='index')

        # 合并入总stats
        stats['token_stats_by_country'] = token_stats_by_country

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
            'en': 'english'
        }

        return nltk_language_map[lang_code]


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_all()
