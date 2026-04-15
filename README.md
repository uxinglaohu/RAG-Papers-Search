# RAG-Papers-Search 📄🔍

这是一个为了练习 RAG (Retrieval-Augmented Generation) 并且突然想整一个论文检索的工具而编写的项目。主要目的是打通从 PDF 解析、向量存储到大模型检索生成的全链路。（需要先会用docker的milvus部署，因为本人也是第一次用，对这个milvus使用还略有生疏，所以导致项目容错能力比较一般。）

> **项目声明**：对比了claude，自用起来检索功能测试感觉比较准确的，但是本人是小白且刚接触RAG且第一次发github，且工程能力偏弱，有bug的话，后续我会处理。
>
> **项目声明2**：下载论文需要一定时间， 第一次将论文变成向量存数据库那块，我多用大模型去预识别了一下论文标题，这里花费的时间比较多，因为api限流，其次存向量数据库那块也花费了一些时间，然后会生成缓存vector_index文件，第二次询问就不会花费太多时间了。
>
> **项目声明3**：没有开发动态添加的功能，如果你往data/papers里面新添了论文，只能把缓存vector_index文件和milvus存储的向量库删掉，然后重新跑。。（再花费3个小时以及微量的调用模型api的💰，这个就是我提到的项目容错能力一般）。
>
> **项目声明4**：这个项目基本满足了我的需求了，所以有的query相关的优化还没有开发，但是不影响使用
>
> 
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

从网上获取到25年的acl论文数据集，下载位置data/papers

```bash
python download_papers.py
```

### 启动Docker，milvus，自行安装

### 启动检索

```bash
python main.py