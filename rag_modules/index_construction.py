from langchain_huggingface import HuggingFaceEmbeddings
import logging
import torch
from langchain_milvus import Milvus
from langchain_core.documents import Document
from pymilvus import connections, Collection, utility, DataType, FieldSchema, CollectionSchema
logger = logging.getLogger(__name__)
class IndexConstructionModule:
    def __init__(self, model_name: str, collection_name: str, index_connection_uri: str, db_name: str):
        self.model_name = model_name
        self.collection_name = collection_name
        self.db_name = db_name
        self.index_connection_uri = index_connection_uri
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()

    def setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"✅ 嵌入模型初始化完成 | 当前运行设备: {device}")

    def load_index(self):
        if not self.embeddings:
            self.setup_embeddings()

        try:
            connections.connect(
                alias="default",
                uri="http://localhost:19530",
                db_name=self.db_name,
            )
            print(f"当前所有集合: {utility.list_collections()}")
            if not utility.has_collection(self.collection_name):
                print(f"未发现 Collection: {self.collection_name}，正在创建...")
                fields = [
                    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=65535, auto_id=False),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
                    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=65535),  # 新增
                    FieldSchema(name="page", dtype=DataType.INT64),
                    FieldSchema(name="total_pages", dtype=DataType.INT64),
                    FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="start_index", dtype=DataType.INT64),
                ]
                schema = CollectionSchema(fields, description="论文知识库")
                collection = Collection(name=self.collection_name, schema=schema)
                collection.create_index(
                    field_name="vector",
                    index_params={
                        "index_type": "IVF_FLAT",
                        "metric_type": "COSINE",
                        "params": {"nlist": 128},
                    }
                )

            self.vectorstore = Milvus(
                embedding_function=self.embeddings,
                connection_args={
                    "uri": "http://localhost:19530",
                    "db_name": self.db_name,
                },
                collection_name=self.collection_name,
            )
            return self.vectorstore

        except Exception as e:
            logger.error(f"❌ 初始化 Milvus 失败: {str(e)}")
            raise

    def build_vector_index(self, chunks: list[Document]):
        """
        将分块后的文档存入 Milvus 向量数据库
        """
        if not chunks:
            raise ValueError("❌ chunks 列表不能为空，请检查前置加载步骤")
        ids = [f"{c.metadata['parent_id']}_{i}" for i, c in enumerate(chunks)]

        try:
            # 2. 写入数据库
            self.vectorstore.add_documents(chunks, ids=ids)
            logger.info(f"✅ 向量索引构建完成，共插入 {len(chunks)} 条数据")
        except Exception as e:
            logger.error(f"❌ 向量写入失败: {str(e)}")
            raise

        return self.vectorstore
