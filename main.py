import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from config import RAGConfig,DEFAULT_CONFIG
from pathlib import Path
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.index_construction import IndexConstructionModule
from rag_modules.generation_integration import GenerationIntegrationModule
from rag_modules.retrieval_optimization import RetrievalOptimizationModule
from langchain_openai import ChatOpenAI
import os
import pickle
import json
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class PaperRAG:
    def __init__(self, config: RAGConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=os.environ[self.config.llm_key_name],
            base_url="https://api.deepseek.com",
            top_p=self.config.top_p,
        )
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")
        if not os.getenv(self.config.llm_key_name):
            raise ValueError(f"请设置 环境变量{self.config.llm_key_name}")

    def initialize_system(self):
        print("正在初始化论文检索系统")
        print("正在初始化数据准备模块.....")
        llm = self.llm
        self.data_module = DataPreparationModule(self.config.data_path, llm=llm)

        print("初始化索引构建模块......")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_connection_uri=self.config.index_connection_uri,
            collection_name=self.config.collection_name,
            db_name = self.config.db_name,
        )

        print("初始化生成集成模块......")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            api_key_name = self.config.llm_key_name,
            max_tokens=self.config.max_tokens,
            top_p = self.config.top_p,
            llm = llm,
        )

        print("系统全部初始化已经完成!")

    def build_knowledge_base(self):
        vectorstore = self.index_module.load_index()
        collection_name = self.index_module.collection_name
        count = vectorstore.col.num_entities
        if count == 0:
            logger.warning(f"没有名为 {collection_name} 的 Collection，需要创建并存储向量")
            self.data_module.load_documents()
            chunks = self.data_module.chunk_documents()
            logger.info("正在构建向量索引")
            self.index_module.build_vector_index(chunks)
            with open(self.config.cache_path, "wb") as f:
                pickle.dump(chunks, f)
            logger.info(f"✅ chunks 已缓存到本地 {self.config.cache_path}")
        else:
            logger.info(f"✅ 集合 {collection_name} 已存在，跳过向量入库步骤,直接读取数据...")
            with open(self.config.cache_path, "rb") as f:
                chunks = pickle.load(f)
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        print(f"\n📊 知识库统计:")
        print(f"   文本块数: {len(chunks)}")
        print("知识库构建已完成！😃")

    def _extract_filters_from_query(self, query: str) -> dict:
        prompt = f"""请从以下用户问题中提取过滤条件，以JSON格式返回。
    如果没有特定过滤条件，返回空字典 {{}}

    可提取的字段：
    - title: 论文标题（用户提到了具体论文名）

    示例：
    用户问题："RetroLLM这篇论文的创新点是什么？"
    返回：{{"title": "RetroLLM"}}

    用户问题："有哪些关于情感分析的论文？"
    返回：{{}}

    用户问题："{query}"
    请只返回JSON，不要其他内容。"""

        response = self.llm.invoke(prompt)
        try:
            filters = json.loads(response.content.strip())
        except:
            filters = {}
        print(f"query抽取出的过滤特征为: {filters}")
        return filters
    def chat(self, query: str):
        # 1. 查询路由
        route_type = self.generation_module.query_router(query)
        print(f"查询类型:{route_type}")

        # 2. 智能查询重写（论文检索路由版）
        if route_type == 'keyword':
            # 关键词检索 → 原样搜索（最适合论文检索）
            rewritten_query = query
            print(f"🔍 关键词检索，保持原样: {query}")

        elif route_type == 'title':
            # 标题检索 → 精确搜索，不改动
            rewritten_query = query
            print(f"📄 标题精确检索，保持原样: {query}")

        else:
            print("🤖 学术问题，执行智能查询优化...")
            rewritten_query = query

        # 3. 检索相关子块
        filters = self._extract_filters_from_query(rewritten_query)
        relevant_chunks = self.retrieval_module.filtered_hybrid_search(rewritten_query, filters,
                                                                         top_k=self.config.top_k)

        print(f"找到 {len(relevant_chunks)} 个相关文档块")

        if not relevant_chunks:
            return "抱歉，没有找到相关的论文内容。请尝试其他关键词。"
        if route_type == 'keyword':
            print("✅ 关键词检索，返回相关论文标题")
            return self.generation_module.generate_keyword_answer(query, relevant_chunks)
        elif route_type == 'title':
            print("✅ 标题检索，精确匹配论文标题")
            return self.generation_module.generate_title_answer(query, relevant_chunks)
        else:
            print("✅ 通用问法，返回相关论文标题")
            return self.generation_module.generate_detailed_answer(query, relevant_chunks)

    def run_interactive(self):
        """运行交互式问答"""
        self.initialize_system()
        self.build_knowledge_base()
        print("=" * 60)

        print("🍽️  论文检索RAG系统 - 交互式问答  🍽️")

        print("\n交互式回答 （输入'退出'结束）：")

        while True:
            try:
                user_input = input("💭：")
                if user_input.lower() == "退出":
                    break
                answer = self.chat(user_input)
                print(f"🤖 {answer}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")

        print("\n感谢使用论文检索系统！")

def main():
    rag_system = PaperRAG()

    rag_system.run_interactive()
if __name__ == "__main__":
    main()
