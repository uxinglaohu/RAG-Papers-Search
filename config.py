from dataclasses import dataclass
@dataclass
class RAGConfig:
    data_path: str = "./data/papers"

    embedding_model: str = "BAAI/bge-m3"
    index_connection_uri: str = "http://localhost:19530"
    db_name: str = "papers"
    collection_name: str = "papers_index"
    llm_model: str = "deepseek-chat"
    llm_key_name = "DEEPSEEK_API_KEY"

    cache_path: str = "./vector_index/chunks_cache.pkl"
    temperature: float = 0.5
    top_p: float = 0.5
    max_tokens: int = 5000

    top_k: int = 50
    def __post_init__(self):
        pass

DEFAULT_CONFIG = RAGConfig()