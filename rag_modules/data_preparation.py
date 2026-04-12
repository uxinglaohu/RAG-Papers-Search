from langchain_classic import text_splitter
from langchain_core.documents import Document
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
from langchain_community.chat_models.moonshot import MoonshotChat
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm
import re
logger = logging.getLogger(__name__)
class DataPreparationModule:
    def __init__(self, data_path: str, llm: MoonshotChat):
        self.data_path = data_path
        self.documents: list[Document] = []
        self.chunks: list[Document] = []
        self.parent_child_map: dict[str, str] = {}
        self.llm = llm

    def _extract_title(self, first_page_text: str) -> str:
        prompt = ChatPromptTemplate.from_template("""
        从以下论文首页内容中提取论文标题，只返回标题文本，不要其他任何内容。

        内容：
        {text}

        论文标题：""")

        chain = prompt | self.llm | StrOutputParser()
        import time
        time.sleep(2)
        try:
            return chain.invoke({"text": first_page_text[:1000]}).strip()
        except Exception as e:
            logger.error(f"标题提取失败: {e}")
            return "未知标题"


    def load_documents(self) -> list[Document]:
        logger.info(f"正在从 {self.data_path} 加载文档...")

        documents = []
        root_path = Path(self.data_path).resolve()
        pdf_files = list(root_path.glob("*.pdf"))
        for path in tqdm(pdf_files, desc="📄 加载PDF文档", total=len(pdf_files), colour="blue"):
            relative_path = path.resolve().relative_to(root_path).as_posix()
            #    MD5识别字节，需要encode，然后变成hex 十六进制的哈希表示
            parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()

                first_page_text = docs[0].page_content if docs else ""
                title = self._extract_title(first_page_text)
                print(f"📄 {path.name} -> 标题: {title}")
                for doc in docs:
                    clean_meta = {
                        "source": str(path),
                        "page": doc.metadata.get("page", 0),
                        "title": title,
                        "total_pages": doc.metadata.get("total_pages", 0),
                        "parent_id": parent_id
                    }
                    doc.metadata = clean_meta
                    # print(doc)
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"读取文件 {path} 失败: {e}")
        self.documents = documents
        # 元数据增强 pass
        return self.documents


    def chunk_documents(self) -> list[Document]:
        logger.info("正在进行分块操作...")
        if not self.documents:
            raise ValueError("请先加载文档")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 600,
            chunk_overlap = 100,
            separators=[
                "\n\n",  # 1. 优先保住段落
                "\n",  # 2. 其次保住行
                ". ",  # 3. 英文句号（加空格是为了避开 3.14 这种小数点）
                "。",  # 4. 中文句号
                "! ",  # 5. 英文感叹号
                "！",  # 6. 中文感叹号
                "? ",  # 7. 英文问号
                "？",  # 8. 中文问号
                " ",  # 9. 单词间的空格
                ""  # 10. 最后的保底
            ],
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(self.documents)
        chunks = [c for c in chunks if c.page_content and c.page_content.strip() != ""]
        for chunk in chunks:
            chunk.page_content = re.sub(r'[\ud800-\udfff]+', '[UNK]', chunk.page_content)
            chunk.page_content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]+', '[UNK]', chunk.page_content)
            chunk.page_content = re.sub(r'(\[UNK\]\s*)+', '[UNK] ', chunk.page_content)
            chunk.page_content = re.sub(r'\s+', ' ', chunk.page_content).strip()
        print(len(self.documents))
        print("原始文档数:", len(self.documents))
        print("分块后有效块数:", len(chunks))
        self.chunks = chunks
        return chunks


