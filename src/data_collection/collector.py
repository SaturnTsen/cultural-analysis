"""
数据收集模块：从各种来源收集中法数字化转型相关数据
"""

import os
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse
from datetime import datetime
import time
import random
import json
from pathlib import Path
import PyPDF2
from docx import Document  # 使用python-docx库

from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir, save_json, detect_language
from src.config import DATA_SOURCES, RAW_DATA_DIR, DataSourcesList

logger = logging.getLogger(__name__)

class DataCollector:
    """数据收集类，负责从不同来源收集数据"""
    
    def __init__(self):
        """初始化数据收集器"""
        self.sources = DATA_SOURCES.model_dump()
        self.raw_data_dir : str  = RAW_DATA_DIR
        
        # 确保目录存在
        for country in self.sources.keys():
            country_dir = os.path.join(self.raw_data_dir, country)
            if not os.path.exists(country_dir):
                os.makedirs(country_dir)
                
            for sector in self.sources[country]:
                sector_dir = os.path.join(country_dir, sector)
                if not os.path.exists(sector_dir):
                    os.makedirs(sector_dir)
    
    def _extract_text_from_pdf(self, pdf_path):
        """从PDF文件中提取文本"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"使用PyPDF2提取PDF文本时出错: {str(e)}")
        return text
    
    def _extract_text_from_docx(self, docx_path):
        """从DOCX文件中提取文本"""
        text = ""
        try:
            doc = Document(docx_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            logger.error(f"使用python-docx提取DOCX文本时出错: {str(e)}")
        return text
    
    def _extract_text_from_file(self, file_path):
        """根据文件类型提取文本"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self._extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return self._extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"读取文本文件时出错: {str(e)}")
                    return ""
        else:
            logger.warning(f"不支持的文件类型: {file_ext}，且textract不可用")
            return ""
    
    def collect_all(self):
        """收集所有数据源的数据"""
        logger.info("开始收集所有数据")
        
        for country, sectors in self.sources.items():
            logger.info(f"收集{country}数据")
            
            for sector, urls in sectors.items():
                logger.info(f"收集{country}的{sector}领域数据")
                
                for url in urls:
                    try:
                        self._collect_from_url(country, sector, url)
                        # 添加随机延迟，避免请求过快
                        time.sleep(random.uniform(1, 3))
                    except Exception as e:
                        logger.error(f"从{url}收集数据时出错: {str(e)}")
        
        logger.info("所有数据收集完成")
    
    def _collect_from_url(self, country, sector, url):
        """从特定URL收集数据"""
        logger.info(f"从{url}收集数据")
        
        # 解析URL域名，用于文件命名
        domain = urlparse(url).netloc
        filename_base = f"{domain}_{datetime.now().strftime('%Y%m%d')}"
        
        # 发送请求
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            logger.warning(f"获取{url}失败，状态码: {response.status_code}")
            return
        
        # 检测内容类型
        content_type = response.headers.get('Content-Type', '').lower()
        
        # 根据内容类型保存数据
        if 'application/pdf' in content_type:
            # 保存PDF文件
            output_path = os.path.join(self.raw_data_dir, country, sector, f"{filename_base}.pdf")
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"PDF文件已保存至: {output_path}")
            
            # 提取文本
            text = self._extract_text_from_pdf(output_path)
            if text:
                text_path = os.path.join(self.raw_data_dir, country, sector, f"{filename_base}_extracted.txt")
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                logger.info(f"PDF提取的文本已保存至: {text_path}")
        
        elif 'text/html' in content_type:
            # 保存HTML内容和提取文本
            html_path = os.path.join(self.raw_data_dir, country, sector, f"{filename_base}.html")
            with open(html_path, 'wb') as f:
                f.write(response.content)
            
            # 使用BeautifulSoup提取文本
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 移除脚本和样式元素
            for script in soup(["script", "style"]):
                script.extract()
            
            # 获取文本
            text = soup.get_text(separator='\n', strip=True)
            
            # 保存提取的文本
            text_path = os.path.join(self.raw_data_dir, country, sector, f"{filename_base}.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
                
            logger.info(f"HTML已保存至: {html_path}, 文本已提取至: {text_path}")
            
        elif 'application/json' in content_type:
            # 保存JSON内容
            json_path = os.path.join(self.raw_data_dir, country, sector, f"{filename_base}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logger.info(f"JSON已保存至: {json_path}")
            
        else:
            # 保存原始内容
            raw_path = os.path.join(self.raw_data_dir, country, sector, f"{filename_base}.txt")
            with open(raw_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"原始内容已保存至: {raw_path}")
        
        # 保存元数据
        metadata = {
            'url': url,
            'country': country,
            'sector': sector,
            'collection_date': datetime.now().isoformat(),
            'content_type': content_type,
            'status_code': response.status_code,
            'file_path': raw_path if 'raw_path' in locals() else (
                text_path if 'text_path' in locals() else (
                    json_path if 'json_path' in locals() else (
                        html_path if 'html_path' in locals() else output_path
                    )
                )
            )
        }
        
        metadata_path = os.path.join(self.raw_data_dir, country, sector, f"{filename_base}_metadata.json")
        pd.Series(metadata).to_json(metadata_path, orient='index', force_ascii=False)
        logger.info(f"元数据已保存至: {metadata_path}")
        
    def collect_from_specific_source(self, country, sector, url):
        """从特定来源收集数据"""
        try:
            return self._collect_from_url(country, sector, url)
        except Exception as e:
            logger.error(f"从{url}收集数据时出错: {str(e)}")
            return None 