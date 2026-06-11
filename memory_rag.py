"""
memory_rag.py — 基于 sentence-transformers + FAISS 的 RAG 记忆检索引擎
语义向量检索，替代简单关键词匹配
"""
import os
import json
import numpy as np

# 延迟导入，未安装时降级为关键词模式
_model = None
_faiss = None
_index = None
_fact_ids = []  # FAISS 索引位置 → fact_id 映射
RAG_READY = False

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".memory")
os.makedirs(MEMORY_DIR, exist_ok=True)
FAISS_INDEX_FILE = os.path.join(MEMORY_DIR, "faiss_index.pkl")

# 使用轻量中文嵌入模型（国内镜像）
MODEL_NAME = "shibing624/text2vec-base-chinese"
HF_MIRROR = "https://hf-mirror.com"  # 国内镜像


def init_rag():
    """初始化 RAG 引擎，失败则返回 False"""
    global _model, _faiss, RAG_READY
    if RAG_READY:
        return True
    try:
        import os
        os.environ["HF_ENDPOINT"] = HF_MIRROR  # 设置 HuggingFace 镜像

        from sentence_transformers import SentenceTransformer
        import faiss as _faiss_mod

        _faiss = _faiss_mod
        print("[RAG] 加载嵌入模型（首次需下载约400MB）...")
        _model = SentenceTransformer(MODEL_NAME)
        RAG_READY = True
        print("[RAG] 引擎就绪")
        return True
    except ImportError as e:
        print(f"[RAG] 依赖未安装，降级为关键词模式: {e}")
        return False
    except Exception as e:
        print(f"[RAG] 初始化失败: {e}")
        return False


def encode_texts(texts: list) -> np.ndarray:
    """将文本列表编码为向量矩阵"""
    global _model
    if _model is None:
        return None
    embeddings = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)


def build_index(facts: list):
    """从 facts 列表构建 FAISS 索引"""
    global _index, _fact_ids

    if not facts:
        _index = None
        _fact_ids = []
        _save_index()
        return

    texts = [f["content"] for f in facts]
    _fact_ids = [f["id"] for f in facts]

    embeddings = encode_texts(texts)
    if embeddings is None:
        return

    dim = embeddings.shape[1]
    _index = _faiss.IndexFlatIP(dim)  # 内积（已归一化 = 余弦相似度）
    _index.add(embeddings)
    _save_index()


def add_to_index(fact: dict):
    """增量添加一条 fact 到索引"""
    global _index, _fact_ids

    embedding = encode_texts([fact["content"]])
    if embedding is None:
        return

    dim = embedding.shape[1]
    if _index is None:
        _index = _faiss.IndexFlatIP(dim)

    _index.add(embedding)
    _fact_ids.append(fact["id"])
    _save_index()


def remove_from_index(fact_id: int):
    """从索引中移除一条 fact（FAISS 不支持删除，需要重建）"""
    global _index, _fact_ids
    if fact_id in _fact_ids:
        _fact_ids.remove(fact_id)


def rebuild_index(facts: list):
    """完全重建索引"""
    build_index(facts)


def search_by_embedding(query: str, facts: list, top_k: int = 10) -> list:
    """用向量相似度搜索最相关的 facts"""
    global _index, _fact_ids

    if _index is None or _model is None or not facts:
        return []

    query_embedding = encode_texts([query])
    if query_embedding is None:
        return []

    # 确保索引和 facts 同步
    if _index.ntotal != len(_fact_ids) or _index.ntotal == 0:
        rebuild_index(facts)
        if _index is None or _index.ntotal == 0:
            return []

    # 搜索
    k = min(top_k, _index.ntotal)
    scores, indices = _index.search(query_embedding, k)

    results = []
    fact_map = {f["id"]: f for f in facts}

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_fact_ids):
            continue
        fact_id = _fact_ids[idx]
        fact = fact_map.get(fact_id)
        if fact:
            results.append({**fact, "_score": float(score)})

    return results


def _save_index():
    """保存索引到磁盘（用 pickle 绕过 FAISS 的 Unicode 路径问题）"""
    global _index, _fact_ids
    try:
        import pickle
        os.makedirs(MEMORY_DIR, exist_ok=True)
        vectors = None
        if _index is not None and _index.ntotal > 0:
            vectors = _faiss.rev_swig_ptr(_index.get_xb(), _index.ntotal * _index.d).reshape(_index.ntotal, _index.d).copy()
        data = {
            "vectors": vectors,
            "ntotal": _index.ntotal if _index else 0,
            "d": _index.d if _index else 0,
            "fact_ids": _fact_ids
        }
        with open(FAISS_INDEX_FILE, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"[RAG] 保存索引失败: {e}")


def load_index():
    """从磁盘加载索引"""
    global _index, _fact_ids
    try:
        import pickle
        if os.path.exists(FAISS_INDEX_FILE):
            with open(FAISS_INDEX_FILE, "rb") as f:
                data = pickle.load(f)
            vectors = data.get("vectors")
            _fact_ids = data.get("fact_ids", [])
            if vectors is not None and len(vectors) > 0:
                dim = vectors.shape[1]
                _index = _faiss.IndexFlatIP(dim)
                _index.add(vectors)
            print(f"[RAG] 加载索引成功: {_index.ntotal if _index else 0} 条向量")
            return True
    except Exception as e:
        print(f"[RAG] 加载索引失败: {e}")
    return False


def hybrid_search(query: str, facts: list, top_k: int = 10) -> list:
    """
    混合检索：向量语义 + 关键词匹配，融合排序
    - 向量检索找到语义相似的记忆
    - 关键词匹配找到精确包含的记忆
    - 两者去重后按综合分数排序
    """
    import re

    results_map = {}  # fact_id → {fact, score, source}

    # 1. 向量语义检索
    if RAG_READY and _index is not None:
        vec_results = search_by_embedding(query, facts, top_k=top_k * 2)
        for r in vec_results:
            fid = r["id"]
            results_map[fid] = {
                "fact": r,
                "vec_score": r.get("_score", 0),
                "kw_score": 0
            }

    # 2. 关键词匹配（作为补充）
    query_lower = query.lower()
    keywords = set(re.findall(r'[一-鿿]+|[a-zA-Z]{2,}', query_lower))
    # 去掉太短的词
    keywords = {kw for kw in keywords if len(kw) >= 2}

    for fact in facts:
        fid = fact["id"]
        content_lower = fact["content"].lower()
        kw_score = 0
        for kw in keywords:
            if kw in content_lower:
                kw_score += 2

        if fid in results_map:
            results_map[fid]["kw_score"] = kw_score
        elif kw_score > 0:
            results_map[fid] = {
                "fact": fact,
                "vec_score": 0,
                "kw_score": kw_score
            }

    # 3. 融合排序
    scored = []
    for fid, data in results_map.items():
        # 综合分数：向量 * 0.6 + 关键词 * 0.4（归一化后）
        vec = data.get("vec_score", 0)
        kw = min(data.get("kw_score", 0), 10) / 10  # 归一化到 0-1
        combined = vec * 0.6 + kw * 0.4
        fact = data["fact"]
        scored.append({**fact, "_score": combined, "_vec": vec, "_kw": data.get("kw_score", 0)})

    scored.sort(key=lambda x: -x["_score"])
    return scored[:top_k]
