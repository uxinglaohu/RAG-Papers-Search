import logging
import os
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
logger = logging.getLogger(__name__)
class GenerationIntegrationModule:
    def __init__(self, model_name: str, api_key_name: str, temperature: float, max_tokens: int, top_p: float, llm: MoonshotChat):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = os.environ[api_key_name]
        self.llm = llm
        self.top_p = top_p
        # self.setup_llm()
        self.show_prompt = True
    # def setup_llm(self):
    #     """初始化大语言模型"""
    #     logger.info(f"正在初始化LLM: {self.model_name}")
    #
    #     self.llm = MoonshotChat(
    #         model=self.model_name,
    #         temperature=self.temperature,
    #         max_tokens=self.max_tokens,
    #         moonshot_api_key=self.api_key,
    #         top_p=self.top_p,
    #     )
    #
    #     logger.info("LLM初始化完成")

    def query_router(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_template("""
        你是一个论文检索意图分类器，请根据用户问题判断检索类型，只返回结果。

        分类规则：
        1. 'keyword' - 用户输入【关键词、研究领域、主题】，用于检索相关论文
           例如：大语言模型检索增强生成、图像分割研究、深度学习推荐

        2. 'title' - 用户输入【完整/部分论文标题】，精确查找某篇论文
           例如：Attention Is All You Need、BERT模型、Transformer机器翻译

        3. 'general' - 用户对论文内容提问、学术咨询、不属于检索需求
   例如：这篇论文的创新点、什么是检索增强生成、实验结果如何

        请只返回分类结果：keyword、title 或 general

        用户问题: {query}
        分类结果:""")

        # 构建链
        chain = prompt | self.llm | StrOutputParser()

        # 执行
        result = chain.invoke({"query": query}).strip().lower()

        for label in ['keyword', 'title', 'general']:
            if label in result:
                return label
        return 'general'
    def query_rewrite(self, query: str) -> str:
        """
        论文检索 - 智能查询重写
        """
        prompt = PromptTemplate(
            template="""
       你是一个专业的学术查询助手。请分析用户的查询，判断是否需要重写以提高论文检索效果。
    
       原始查询: {query}
    
       分析规则：
       1. **具体明确的学术查询**（直接返回原查询）：
          - 包含明确研究方向：如"大语言模型检索增强生成研究"
          - 明确论文主题：如"基于Transformer的图像分割"
          - 具体学术问题：如"知识图谱在推荐系统中的应用"
    
       2. **模糊、简短的查询**（需要重写）：
          - 过于简短：如"大模型"、"深度学习"
          - 模糊宽泛：如"人工智能研究"、"图像处理"
          - 不完整表达：如"推荐系统"、"NLP"
    
       重写原则：
       - 保持原意不变
       - 变成适合论文检索的标准学术表达
       - 保持简洁、专业
    
       请输出最终查询（不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query}).strip()

        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response

    def _build_context(self, docs: list[Document], max_length: int = 2000)-> str:
        if not docs:
            return "暂无相关论文信息。"

        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs, 1):
            # 论文元数据信息（改成论文字段）
            metadata_info = f"【论文片段 {i}】"
            if 'title' in doc.metadata:
                metadata_info += f" 标题: {doc.metadata['title']}"
            if 'source' in doc.metadata:
                metadata_info += f" | 来源: {doc.metadata['source']}"
            if 'page' in doc.metadata:
                metadata_info += f" | 页码: {doc.metadata['page']}"

            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"

            # 长度限制
            if current_length + len(doc_text) > max_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)
        return "\n" + "=" * 50 + "\n".join(context_parts)

    def generate_keyword_answer(self, query: str, docs: list[Document]) -> str:
        context_parts = []
        for doc in docs:
            title = doc.metadata.get("title", "未知标题")
            context_parts.append(f"论文标题: {title}\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        prompt = ChatPromptTemplate.from_template("""
        你是一个学术论文检索助手。以下内容均来自 ACL 2025 论文库。
        用户想查找某个研究方向的相关论文。

        检索到的可能相关内容：
        {context}

        用户查询：{query}

        要求：
        1. 列出所有与该研究方向相关的论文标题
        2. 按相关度从高到低排序
        3. 每篇论文用一句话说明与查询的关联性
        4. 如果没有找到，直接回复"未找到相关论文"

        回答：""")
        if self.show_prompt:
            print("RAG检索的相关内容:", context)
            print("用户询问:", query)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "query": query})

    def generate_title_answer(self, query: str, docs: list[Document]) -> str:
        context_parts = []
        for doc in docs:
            title = doc.metadata.get("title", "未知标题")
            context_parts.append(f"论文标题: {title}\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        prompt = ChatPromptTemplate.from_template("""
        你是一个学术论文检索助手。以下内容均来自 ACL 2025 论文库。
        用户记得一篇论文的部分名称，想找到它的完整标题。

        检索到的内容：
        {context}

        用户查询：{query}

        要求：
        1. 从检索内容中找出与用户描述最匹配的论文完整标题
        2. 如果有多篇相似的，按匹配度排列
        3. 如果没有找到，直接回复"未找到相关论文"

        回答：""")
        if self.show_prompt:
            print("RAG检索的相关内容:", context)
            print("用户询问:", query)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "query": query})

    def generate_detailed_answer(self, query: str, docs: list[Document]) -> str:
        context_parts = []
        for doc in docs:
            title = doc.metadata.get("title", "未知标题")
            context_parts.append(f"论文标题: {title}\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        prompt = ChatPromptTemplate.from_template("""
        你是一个学术论文检索助手。以下内容均来自 ACL 2025 论文库。

        检索到的可能相关内容：
        {context}

        用户查询：{query}

        要求：
        1. 从检索内容中找出与用户查询相关的论文标题
        2. 每篇论文用一句话说明与查询的关联性
        3. 如果没有找到，直接回复"未找到相关论文"

        回答：""")
        if self.show_prompt:
            print("RAG检索的相关内容:", context)
            print("用户询问:", query)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "query": query})