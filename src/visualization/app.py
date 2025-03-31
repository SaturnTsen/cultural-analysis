"""
数据可视化应用：展示数据收集和预处理的结果
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import os

# 设置页面配置
st.set_page_config(
    page_title="中法数字化转型数据分析",
    page_icon="📊",
    layout="wide"
)

# 设置标题
st.title("中法数字化转型数据分析仪表板")

# 获取数据目录
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 加载语料库统计信息
@st.cache_data
def load_corpus_stats():
    stats_file = PROCESSED_DATA_DIR / "corpus_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# 加载语料库汇总
@st.cache_data
def load_corpus_summary():
    summary_file = PROCESSED_DATA_DIR / "corpus_summary.csv"
    if summary_file.exists():
        return pd.read_csv(summary_file)
    return None

# 加载metadata文件
@st.cache_data
def load_metadata(metadata_path):
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return None

# 加载txt文件
@st.cache_data
def load_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return None

# 加载数据
stats = load_corpus_stats()
df = load_corpus_summary()

if stats is None or df is None:
    st.error("未找到数据文件，请先运行数据收集和预处理程序。")
else:
    # 创建三列布局
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("总文档数", stats['total_documents'])
    
    with col2:
        st.metric("总词数", stats['total_tokens'])
    
    with col3:
        st.metric("总句子数", stats['total_sentences'])
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        # 按国家分布的文档数
        country_counts = pd.DataFrame(
            list(stats['documents_by_country'].items()),
            columns=['国家', '文档数']
        )
        fig_country = px.bar(
            country_counts,
            x='国家',
            y='文档数',
            title='按国家分布的文档数'
        )
        st.plotly_chart(fig_country, use_container_width=True)
    
    with col2:
        # 按领域分布的文档数
        sector_counts = pd.DataFrame(
            list(stats['documents_by_sector'].items()),
            columns=['领域', '文档数']
        )
        fig_sector = px.bar(
            sector_counts,
            x='领域',
            y='文档数',
            title='按领域分布的文档数'
        )
        st.plotly_chart(fig_sector, use_container_width=True)
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        # 按语言分布的文档数
        language_counts = pd.DataFrame(
            list(stats['documents_by_language'].items()),
            columns=['语言', '文档数']
        )
        fig_lang = px.pie(
            language_counts,
            values='文档数',
            names='语言',
            title='按语言分布的文档数'
        )
        st.plotly_chart(fig_lang, use_container_width=True)
    
    with col2:
        # 平均词数和句子数
        fig_avg = go.Figure()
        fig_avg.add_trace(go.Bar(
            name='平均词数',
            x=['平均'],
            y=[stats['average_tokens_per_document']]
        ))
        fig_avg.add_trace(go.Bar(
            name='平均句子数',
            x=['平均'],
            y=[stats['average_sentences_per_document']]
        ))
        fig_avg.update_layout(title='文档平均词数和句子数')
        st.plotly_chart(fig_avg, use_container_width=True)
    
    # 详细数据分析
    st.header("详细数据分析")
    
    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["按国家分析", "按领域分析", "按语言分析"])
    
    with tab1:
        # 按国家分析
        country_analysis = df.groupby('country').agg({
            'token_count': ['count', 'sum', 'mean'],
            'filtered_token_count': ['sum', 'mean'],
            'sentence_count': ['sum', 'mean']
        }).round(2)
        country_analysis.columns = ['文档数', '总词数', '平均词数', '总过滤后词数', '平均过滤后词数', '总句子数', '平均句子数']
        st.dataframe(country_analysis)
        
        # 按国家-领域分布
        country_sector = pd.crosstab(df['country'], df['sector'])
        fig_country_sector = px.bar(
            country_sector,
            title='各国各领域文档分布'
        )
        st.plotly_chart(fig_country_sector, use_container_width=True)
    
    with tab2:
        # 按领域分析
        sector_analysis = df.groupby('sector').agg({
            'token_count': ['count', 'sum', 'mean'],
            'filtered_token_count': ['sum', 'mean'],
            'sentence_count': ['sum', 'mean']
        }).round(2)
        sector_analysis.columns = ['文档数', '总词数', '平均词数', '总过滤后词数', '平均过滤后词数', '总句子数', '平均句子数']
        st.dataframe(sector_analysis)
        
        # 按领域-语言分布
        sector_lang = pd.crosstab(df['sector'], df['language'])
        fig_sector_lang = px.bar(
            sector_lang,
            title='各领域语言分布'
        )
        st.plotly_chart(fig_sector_lang, use_container_width=True)
    
    with tab3:
        # 按语言分析
        language_analysis = df.groupby('language').agg({
            'token_count': ['count', 'sum', 'mean'],
            'filtered_token_count': ['sum', 'mean'],
            'sentence_count': ['sum', 'mean']
        }).round(2)
        language_analysis.columns = ['文档数', '总词数', '平均词数', '总过滤后词数', '平均过滤后词数', '总句子数', '平均句子数']
        st.dataframe(language_analysis)
        
        # 按语言-国家分布
        lang_country = pd.crosstab(df['language'], df['country'])
        fig_lang_country = px.bar(
            lang_country,
            title='各语言国家分布'
        )
        st.plotly_chart(fig_lang_country, use_container_width=True)
    
    # 文档详情
    st.header("文档详情")
    
    # 创建文档选择器
    selected_doc = st.selectbox(
        "选择要查看的文档",
        df['metadata_path'].tolist(),
        format_func=lambda x: os.path.basename(x)
    )
    
    if selected_doc:
        # 获取对应的txt文件路径
        txt_path = selected_doc.replace('metadata.json', 'processed.txt')
        
        # 加载metadata和txt内容
        metadata = load_metadata(selected_doc)
        txt_content = load_txt(txt_path)
        
        if metadata:
            st.subheader("文档元信息")
            st.json(metadata)
        
        if txt_content:
            st.subheader("文档内容")
            st.text_area("", txt_content, height=400)
    
    # 添加时间信息
    st.sidebar.subheader("数据信息")
    st.sidebar.write(f"数据生成时间：{stats['generation_date']}")
    
    # 添加数据来源信息
    st.sidebar.subheader("数据来源")
    st.sidebar.write(f"原始数据目录：{DATA_DIR}")
    st.sidebar.write(f"处理后数据目录：{PROCESSED_DATA_DIR}")
    
    # 添加数据统计信息
    st.sidebar.subheader("数据统计")
    st.sidebar.write(f"总过滤后词数：{stats['total_filtered_tokens']}")
    st.sidebar.write(f"平均过滤后词数：{stats['total_filtered_tokens'] / stats['total_documents']:.2f}") 