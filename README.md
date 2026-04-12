# RAG-Papers-Search 📄🔍

这是一个为了练习 RAG (Retrieval-Augmented Generation) 流程而编写的实验性项目。主要目的是打通从 PDF 解析、向量存储到大模型检索生成的全链路。

> **项目声明**：本项目主要用于自学练习。

- **Embedding 模型**: BAAI/bge-m3
- **向量数据库**: Milvus
- **大语言模型**: DeepSeek
- **数据源**: ACL 2025 论文集

---

## 🛠️ 环境配置

建议先使用 Anaconda 创建一个干净的虚拟环境，再通过 `pip` 安装依赖。

### 第一步：创建并激活环境

打开终端（Anaconda Prompt），执行以下命令：

```bash
# 创建环境
conda create -n rag_env python=3.10 -y

# 激活环境
conda activate rag_env
```

### 第二步：安装项目依赖

在激活环境的状态下执行：

```bash
pip install -r requirements.txt
```

---

## 🚀 快速开始

### 修改配置

在 `config.py` 中配置你的私钥和数据库地址：

- **Milvus**: 修改 `index_connection_uri`， 本项目用的默认值
- **DeepSeek**: 配置你的 `DEEPSEEK_API_KEY` 环境变量

### 数据初始化

一键从网上获取到25年的acl论文数据集

```bash
python download_papers.py
```

### 启动Docker，milvus，自行安装

### 启动检索

```bash
python main.py