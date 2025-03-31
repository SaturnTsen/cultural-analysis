#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实用工具函数模块
包含项目中可能会重复使用的通用函数
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from functools import wraps
from src.utils.logger import get_logger
from src.config import HTTP_CONFIG

logger = get_logger("utils.helpers")

def timer_decorator(func):
    """
    函数执行计时装饰器
    
    参数:
        func: 被装饰的函数
    
    返回:
        包装后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"函数 {func.__name__} 执行时间: {end_time - start_time:.4f} 秒")
        return result
    return wrapper

def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.debug(f"创建目录: {directory}")

def save_json(data, filepath):
    """
    将数据保存为JSON文件
    
    参数:
        data: 要保存的数据
        filepath: 保存路径
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.debug(f"数据已保存至 {filepath}")

def load_json(filepath):
    """
    从JSON文件加载数据
    
    参数:
        filepath: JSON文件路径
    
    返回:
        加载的数据
    """
    if not os.path.exists(filepath):
        logger.warning(f"文件不存在: {filepath}")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.debug(f"从 {filepath} 加载数据")
    return data

def save_dataframe(df, filepath, format='csv'):
    """
    保存DataFrame到文件
    
    参数:
        df: pandas DataFrame
        filepath: 保存路径
        format: 文件格式 ('csv' 或 'excel')
    """
    ensure_dir(os.path.dirname(filepath))
    
    if format.lower() == 'csv':
        df.to_csv(filepath, encoding='utf-8', index=False)
    elif format.lower() == 'excel':
        df.to_excel(filepath, index=False)
    else:
        raise ValueError(f"不支持的文件格式: {format}")
    
    logger.debug(f"DataFrame已保存至 {filepath}")

def load_dataframe(filepath, format=None):
    """
    从文件加载DataFrame
    
    参数:
        filepath: 文件路径
        format: 文件格式，如果为None，则从文件扩展名推断
    
    返回:
        pandas DataFrame
    """
    if not os.path.exists(filepath):
        logger.warning(f"文件不存在: {filepath}")
        return None
    
    if format is None:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.csv':
            format = 'csv'
        elif ext in ['.xls', '.xlsx']:
            format = 'excel'
        else:
            raise ValueError(f"无法从扩展名推断文件格式: {ext}")
    
    if format.lower() == 'csv':
        df = pd.read_csv(filepath, encoding='utf-8')
    elif format.lower() == 'excel':
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {format}")
    
    logger.debug(f"从 {filepath} 加载DataFrame")
    return df

def safe_request(url, method='get', **kwargs):
    """
    安全的HTTP请求函数，包含重试和错误处理
    
    参数:
        url: 请求URL
        method: HTTP方法，默认为'get'
        **kwargs: 传递给requests的额外参数
    
    返回:
        requests.Response对象或None（如果所有尝试均失败）
    """
    headers = kwargs.get('headers', {})
    if 'User-Agent' not in headers:
        headers['User-Agent'] = HTTP_CONFIG.get('user_agent')
    kwargs['headers'] = headers
    
    timeout = kwargs.pop('timeout', HTTP_CONFIG.get('timeout', 30))
    retries = kwargs.pop('retries', HTTP_CONFIG.get('retries', 3))
    
    for attempt in range(retries):
        try:
            response = getattr(requests, method.lower())(url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"请求失败 (尝试 {attempt+1}/{retries}): {url} - {str(e)}")
            if attempt + 1 == retries:
                logger.error(f"达到最大重试次数，请求失败: {url}")
                return None
            time.sleep(2 ** attempt)  # 指数退避
    
    return None

def detect_language(text, min_length=20):
    """
    检测文本语言
    
    参数:
        text: 要检测的文本
        min_length: 最小文本长度，小于此长度将不进行检测
    
    返回:
        检测到的语言代码
    """
    if len(text) < min_length:
        logger.warning(f"文本过短 ({len(text)} 字符)，无法可靠检测语言")
        return None
    
    try:
        from langdetect import detect
        return detect(text)
    except ImportError:
        logger.warning("langdetect未安装，无法检测语言")
        return None
    except Exception as e:
        logger.error(f"语言检测失败: {str(e)}")
        return None

def get_file_extension(filename):
    """
    获取文件扩展名
    
    参数:
        filename: 文件名或路径
    
    返回:
        扩展名（不包含点）
    """
    return os.path.splitext(filename)[1][1:].lower()

def get_country_and_sector(filename):
    """
    从文件名解析国家和行业
    
    参数:
        filename: 文件名，格式应为"{country}_{sector}.xxx"
    
    返回:
        (country, sector) 元组
    """
    basename = os.path.basename(filename)
    name_parts = os.path.splitext(basename)[0].split('_', 1)
    
    if len(name_parts) != 2:
        logger.warning(f"无法从文件名解析国家和行业: {basename}")
        return None, None
    
    country, sector = name_parts
    return country, sector 