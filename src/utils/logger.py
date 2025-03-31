#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志工具模块
设置和配置项目的日志系统
"""

import os
import logging
from datetime import datetime
from src.config import LOGS_DIR, LOGGING_CONFIG

def setup_logger(level=None):
    """
    设置日志记录器
    
    参数:
        level (str, optional): 日志级别，默认为配置文件中的设置
    
    返回:
        logging.Logger: 配置好的日志记录器
    """
    # 使用命令行参数指定的级别，或者配置文件中的默认设置
    if level is None:
        level = LOGGING_CONFIG.get("level", "INFO")
    
    # 将字符串日志级别转换为对应的常量
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"无效的日志级别: {level}")
    
    # 确保日志目录存在
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # 创建带有日期时间的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"analysis_{timestamp}.log")
    
    # 配置根日志记录器
    logging.basicConfig(
        level=numeric_level,
        format=LOGGING_CONFIG.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        datefmt=LOGGING_CONFIG.get("datefmt", "%Y-%m-%d %H:%M:%S"),
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    # 获取项目的根日志记录器
    logger = logging.getLogger("china_france_analysis")
    logger.info(f"日志已设置，级别: {level}，日志文件: {log_file}")
    
    return logger

def get_logger(name=None):
    """
    获取指定名称的日志记录器
    
    参数:
        name (str, optional): 日志记录器名称，默认为项目根日志记录器
    
    返回:
        logging.Logger: 日志记录器
    """
    if name:
        return logging.getLogger(f"china_france_analysis.{name}")
    return logging.getLogger("china_france_analysis") 