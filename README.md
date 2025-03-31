# 中法数字化转型文化差异分析

本项目通过自然语言处理技术，分析中国和法国在数字化转型过程中的文化差异。研究涵盖工业、金融、医疗、教育和公共服务等五个关键领域，通过词频分析、情感分析、主题建模和基于Hofstede文化维度理论的分析，揭示两国在数字化进程中的异同点。

## 项目架构

项目采用模块化设计，包含以下主要模块：

1. **数据采集模块**（`src/data_collection`）：从各种来源收集中法数字化转型相关数据
2. **数据预处理模块**（`src/preprocessing`）：清洗和准备数据进行分析
3. **数据分析模块**（`src/analysis`）：实现词频分析、情感分析、主题建模和文化维度分析
4. **可视化模块**（`src/visualization`）：生成分析结果的可视化图表
5. **报告生成模块**（`src/reporting`）：生成分析报告

## 安装与配置

### 环境要求
- Python 3.8+
- 见requirements.txt文件

### 安装步骤

1. 克隆此仓库
   ```bash
   git clone https://github.com/yourusername/france-china-digital-transformation.git
   cd france-china-digital-transformation
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 下载NLTK资源（如有需要）
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## 使用方法

### 运行完整分析流程

```bash
python src/main.py
```

### 只运行特定模块

```bash
# 只进行数据收集
python -c "from src.data_collection.collector import DataCollector; DataCollector().collect_all()"

# 只进行数据预处理
python -c "from src.preprocessing.preprocessor import DataPreprocessor; DataPreprocessor().preprocess_all()"

# 只进行数据分析
python -c "from src.analysis.analyzer import DataAnalyzer; DataAnalyzer().analyze_all()"
```

## 项目结构

```
.
├── data                    # 数据目录（自动创建）
│   ├── raw                 # 原始数据
│   ├── processed           # 预处理后的数据
│   └── results             # 分析结果
├── logs                    # 日志文件（自动创建）
├── src                     # 源代码
│   ├── data_collection     # 数据收集模块
│   ├── preprocessing       # 数据预处理模块
│   ├── analysis            # 数据分析模块
│   ├── visualization       # 可视化模块
│   ├── reporting           # 报告生成模块
│   ├── utils               # 工具函数
│   ├── config.py           # 配置文件
│   └── main.py             # 主入口文件
├── templates               # 报告模板
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明
```

## 配置文件

项目配置位于`src/config.py`，包含以下主要配置：

- 数据源配置
- 语言配置
- Hofstede文化维度配置
- 数据存储路径
- 模型参数
- 情感分析阈值
- 日志配置

## 输出结果

分析结果将保存在`data/results`目录下，包括：

- TF-IDF分析结果
- 情感分析结果
- 主题建模结果
- 文化维度分析结果
- 可视化图表
- 综合分析报告（HTML和PDF格式）

## 许可证

[MIT License](LICENSE)

## 作者

[您的姓名]

## 致谢

本项目受益于以下研究和资源：
- Hofstede的文化维度理论
- 中法两国政府发布的数字化转型政策文件
- 各行业数字化转型相关报告和研究 