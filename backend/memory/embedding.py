"""
LocalEmbedding — BGE 模型本地加载，text → 512-dim 归一化向量
零外部 API 依赖，首次使用自动下载模型 (~100MB)，后续从缓存加载。
"""
import numpy as np


class LocalEmbedding:
    """BGE-small-zh 嵌入模型封装，返回归一化向量供 ChromaDB 使用。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转为向量列表，L2 归一化。返回 list[list[float]] 兼容 ChromaDB。"""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @property
    def dim(self) -> int:
        """向量维度"""
        return self._model.get_embedding_dimension()
