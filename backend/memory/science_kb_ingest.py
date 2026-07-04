"""
科学知识库摄入管道

隔离设计：
  science_kb   — 自动爬取的公开权威源（IUPAC/NIST/Wikipedia/OpenStax）
  user_uploads — 用户上传的讲义、教材、论文

特性：
  - 公式感知分块：$...$ / $$...$$ 边界不切开
  - 来源类型自动检测 + 引用格式化
  - 嵌入模型可选切换（通过 EMBEDDING_MODEL 环境变量）
  - 去重：同 URL 重复摄入自动跳过
"""
import os
import re
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────

CHROMA_PATH = os.environ.get(
    "CHROMA_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma")
)

COLLECTIONS = {
    "science_kb": "公开权威科学知识库",
    "user_uploads": "用户上传文献",
}

CHUNK_SIZE = 400       # characters
CHUNK_OVERLAP = 80

SOURCE_LABELS = {
    "iupac": "IUPAC Gold Book",
    "nist": "NIST Chemistry WebBook",
    "nist-materials": "NIST Materials Data",
    "materials-project": "Materials Project",
    "wikipedia": "Wikipedia",
    "crc": "CRC Handbook",
    "openstax": "OpenStax",
}

CONFIDENCE_TIERS = {
    "iupac": "peer-reviewed",
    "nist": "peer-reviewed",
    "nist-materials": "peer-reviewed",
    "materials-project": "peer-reviewed",
    "crc": "peer-reviewed",
    "openstax": "textbook",
    "wikipedia": "wikipedia",
    "user": "user-provided",
}


# ── Helpers ───────────────────────────────────────────────────────────

def _source_type_from_url(url: str) -> str:
    url_l = url.lower()
    if "goldbook.iupac" in url_l or "iupac" in url_l: return "iupac"
    if "materialsdata.nist" in url_l or ("nist.gov" in url_l and "materials" in url_l): return "nist-materials"
    if "webbook.nist" in url_l or "nist.gov" in url_l: return "nist"
    if "materialsproject.org" in url_l or "materials-project" in url_l: return "materials-project"
    if "wikipedia.org" in url_l: return "wikipedia"
    if "crc" in url_l or "hbcp" in url_l: return "crc"
    if "openstax" in url_l: return "openstax"
    return "other"


def _format_citation(source_type: str, source_url: str) -> str:
    label = SOURCE_LABELS.get(source_type, "Reference")
    return f"{label}, <{source_url}>"


def _extract_formulas(text: str) -> list[str]:
    """Extract LaTeX formulas from text."""
    formulas = []
    for m in re.findall(r'\$\$([^$]+)\$\$', text):
        formulas.append(m.strip())
    for m in re.findall(r'\$([^$]+)\$', text):
        formulas.append(m.strip())
    return formulas


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


# ── Formula-aware chunking ────────────────────────────────────────────

def _formula_aware_split(text: str, chunk_size: int = CHUNK_SIZE,
                         overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks, keeping LaTeX formulas intact."""
    chunks = []
    paragraphs = re.split(r'\n{2,}', text)
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(_validate_formulas(current))
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sub = ""
                for sent in sentences:
                    if len(sub) + len(sent) <= chunk_size:
                        sub = (sub + " " + sent).strip() if sub else sent
                    else:
                        if sub:
                            chunks.append(_validate_formulas(sub))
                        sub = sent
                current = sub
            else:
                current = para
    if current:
        chunks.append(_validate_formulas(current))
    return [c for c in chunks if len(c) > 20]


def _validate_formulas(chunk: str) -> str:
    """Ensure $$ and $ are paired. If broken, close them."""
    doubles = chunk.count("$$")
    if doubles % 2 != 0:
        chunk += "$$"
    singles = len(re.findall(r'(?<!\$)\$(?!\$)', chunk))
    if singles % 2 != 0:
        chunk += "$"
    return chunk


# ── Embedding ─────────────────────────────────────────────────────────

def _get_embedding_function():
    from .embedding import LocalEmbedding
    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    return LocalEmbedding(model_name=model_name)


# ── ChromaDB ──────────────────────────────────────────────────────────

def _get_collection(collection_name: str):
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    ef = _get_embedding_function()
    try:
        return client.get_collection(collection_name, embedding_function=ef)
    except Exception:
        return client.create_collection(
            collection_name,
            embedding_function=ef,
            metadata={"description": COLLECTIONS.get(collection_name, "")},
        )


# ── Public API ────────────────────────────────────────────────────────

def ingest_science_document(content: str, source_url: str,
                            title: str = "", collection: str = "science_kb") -> dict:
    """Ingest a document into a knowledge collection.

    Args:
        content: Document text content
        source_url: Source URL (used for dedup and citation)
        title: Document title
        collection: Target collection ("science_kb" or "user_uploads")

    Returns:
        {"status": "ok", "chunks": N, "collection": name, "source": url}
    """
    try:
        source_type = _source_type_from_url(source_url)
        citation = _format_citation(source_type, source_url)
        chunks = _formula_aware_split(content)

        coll = _get_collection(collection)
        url_id = _url_hash(source_url)

        # Dedup: check if chunks from this URL already exist
        existing = coll.get(where={"source_url": source_url})
        if existing and existing["ids"]:
            return {"status": "ok", "chunks": 0, "collection": collection,
                    "source": source_url, "note": "Already ingested (dedup by URL)"}

        ids, documents, metadatas = [], [], []
        ts = datetime.now(timezone.utc).isoformat()
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_type}_{url_id}_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source_url": source_url,
                "source_type": source_type,
                "confidence_tier": CONFIDENCE_TIERS.get(source_type, "other"),
                "citation": citation,
                "title": title,
                "formulas": json.dumps(_extract_formulas(chunk), ensure_ascii=False),
                "chunk_index": i,
                "ingested_at": ts,
            })

        if ids:
            coll.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info(f"Ingested {len(ids)} chunks from {source_url} → {collection}")
        return {"status": "ok", "chunks": len(ids), "collection": collection,
                "source": source_url, "source_type": source_type}

    except Exception as e:
        logger.error(f"Ingestion failed for {source_url}: {e}")
        return {"status": "error", "error": str(e)}


def search_knowledge(query: str, collection: str = "all",
                     top_k: int = 5) -> dict:
    """Search knowledge collections.

    Args:
        query: Search query
        collection: "all" (default), "science_kb", or "user_uploads"
        top_k: Number of results
    """
    items = []
    search_collections = (
        list(COLLECTIONS.keys()) if collection == "all" else [collection]
    )

    for col_name in search_collections:
        try:
            coll = _get_collection(col_name)
            results = coll.query(query_texts=[query], n_results=top_k)
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else None

                # Score: combine distance + confidence
                tier = meta.get("confidence_tier", "other")
                tier_weight = {"peer-reviewed": 1.0, "textbook": 0.9,
                               "wikipedia": 0.7, "user-provided": 0.8,
                               "other": 0.5}
                weight = tier_weight.get(tier, 0.5)
                raw_score = 1.0 / (1.0 + (dist or 1.0))
                combined = raw_score * weight

                items.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i][:500],
                    "collection": col_name,
                    "citation": meta.get("citation", ""),
                    "source_url": meta.get("source_url", ""),
                    "confidence_tier": tier,
                    "distance": dist,
                    "score": round(combined, 4),
                })
        except Exception as e:
            logger.warning(f"Search failed for collection {col_name}: {e}")

    items.sort(key=lambda x: x["score"], reverse=True)
    return {"status": "ok", "query": query, "collection": collection,
            "results": items[:top_k], "count": len(items[:top_k])}


def get_collection_count(collection: str = "all") -> int:
    """Return number of chunks in a collection."""
    total = 0
    cols = list(COLLECTIONS.keys()) if collection == "all" else [collection]
    for name in cols:
        try:
            coll = _get_collection(name)
            total += coll.count()
        except Exception:
            pass
    return total


def delete_collection(collection: str) -> dict:
    """Delete an entire collection."""
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        client.delete_collection(collection)
        return {"status": "ok", "deleted": collection}
    except Exception as e:
        return {"status": "error", "error": str(e)}
