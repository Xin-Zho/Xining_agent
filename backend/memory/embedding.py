"""
LocalEmbedding — BGE 模型本地加载，text → 512-dim 归一化向量
零外部 API 依赖，首次使用自动下载模型 (~100MB)，后续从缓存加载。
"""
import os

# 中国大陆服务器无法直接访问 huggingface.co，使用镜像
_HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = _HF_MIRROR


class LocalEmbedding:
    """BGE-small-zh 嵌入模型封装，返回归一化向量供 ChromaDB 使用。
    模型惰性加载：只在首次调用 embed() 或 dim 时才下载/加载模型。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self._model_name = model_name
        self._model = None  # 惰性加载

    def _ensure_model(self):
        """懒加载：首次使用时才实例化 SentenceTransformer"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转为向量列表，L2 归一化。返回 list[list[float]] 兼容 ChromaDB。"""
        if isinstance(texts, str):
            texts = [texts]
        self._ensure_model()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @property
    def dim(self) -> int:
        """向量维度"""
        self._ensure_model()
        return self._model.get_embedding_dimension()
