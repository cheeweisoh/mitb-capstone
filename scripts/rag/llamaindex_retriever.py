import csv
import re
from pathlib import Path
from typing import Any

import faiss
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore

from rag_prompts import content_terms, rerank_for_adherence

DEFAULT_CHUNKS_PATH = Path("dataset/rag/rag_chunks.csv")
DEFAULT_INDEX_DIR = Path("dataset/rag/llamaindex_faiss")
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_FUSION_CANDIDATES = 30
DEFAULT_RRF_K = 60

BM25_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def bm25_tokenizer(text: str) -> list[str]:
    return [token.casefold() for token in BM25_TOKEN_RE.findall(text)]


def load_chunk_rows(chunks_path: Path = DEFAULT_CHUNKS_PATH) -> list[dict[str, str]]:
    with chunks_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No chunks found in {chunks_path}")

    required_columns = {
        "chunk_id",
        "source_file",
        "start_page",
        "end_page",
        "heading",
        "word_count",
        "text",
    }
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{chunks_path} is missing required columns: {missing}")

    return rows


def load_nodes(chunks_path: Path = DEFAULT_CHUNKS_PATH) -> list[TextNode]:
    """Read rag_chunks.csv and build one TextNode per row, preserving the full
    existing metadata schema. text_template is cleared to '{content}' so only
    node.text is embedded, matching the legacy embed_texts() behavior exactly
    (metadata is not prepended into the embedding input)."""
    rows = load_chunk_rows(chunks_path)
    nodes = []
    for row in rows:
        metadata = {
            "source_file": row["source_file"],
            "start_page": int(row["start_page"]),
            "end_page": int(row["end_page"]),
            "heading": row["heading"],
            "chunk_index": row.get("chunk_index"),
            "word_count": int(row["word_count"]),
        }
        node = TextNode(id_=row["chunk_id"], text=row["text"], metadata=metadata)
        node.text_template = "{content}"
        node.excluded_embed_metadata_keys = list(metadata.keys())
        node.excluded_llm_metadata_keys = list(metadata.keys())
        nodes.append(node)
    return nodes


def build_or_load_index(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    device: str | None = None,
    rebuild: bool = False,
) -> VectorStoreIndex:
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name, device=device, normalize=True)

    if not rebuild and index_dir.exists() and (index_dir / "docstore.json").exists():
        vector_store = FaissVectorStore.from_persist_dir(str(index_dir))
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(index_dir))
        return load_index_from_storage(storage_context, embed_model=embed_model)

    nodes = load_nodes(chunks_path)
    embed_dim = len(embed_model.get_text_embedding(nodes[0].text))
    faiss_index = faiss.IndexFlatIP(embed_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

    index_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(index_dir))
    return index


def build_bm25_retriever(nodes: list[TextNode], top_k: int) -> BM25Retriever:
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k, tokenizer=bm25_tokenizer)


def node_to_result_dict(node_with_score: NodeWithScore) -> dict[str, Any]:
    node = node_with_score.node
    result = dict(node.metadata)
    result["chunk_id"] = node.node_id
    result["text"] = node.text
    result["score"] = float(node_with_score.score) if node_with_score.score is not None else 0.0
    return result


def dense_search(index: VectorStoreIndex, query: str, top_k: int) -> list[dict[str, Any]]:
    retriever = index.as_retriever(similarity_top_k=top_k)
    return [node_to_result_dict(nws) for nws in retriever.retrieve(query)]


def bm25_search(bm25: BM25Retriever, query: str, top_k: int) -> list[dict[str, Any]]:
    results = bm25.retrieve(QueryBundle(query_str=query))
    return [node_to_result_dict(nws) for nws in results[:top_k]]


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """Ported near-verbatim from the legacy retrieve_chunks.py."""
    fused: dict[str, dict[str, Any]] = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]
        fused[chunk_id] = dict(result)
        fused[chunk_id]["dense_rank"] = rank
        fused[chunk_id]["dense_score"] = result["score"]
        fused[chunk_id]["fusion_score"] = 1 / (rrf_k + rank)

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["chunk_id"]
        if chunk_id not in fused:
            fused[chunk_id] = dict(result)
            fused[chunk_id]["fusion_score"] = 0.0
        fused[chunk_id]["bm25_rank"] = rank
        fused[chunk_id]["bm25_score"] = result["score"]
        fused[chunk_id]["fusion_score"] += 1 / (rrf_k + rank)

    ranked = sorted(fused.values(), key=lambda result: result["fusion_score"], reverse=True)
    for result in ranked:
        result.setdefault("dense_rank", None)
        result.setdefault("dense_score", None)
        result.setdefault("bm25_rank", None)
        result.setdefault("bm25_score", None)
        result["score"] = result["fusion_score"]
    return ranked[:top_k]


class AdherenceRerankPostprocessor(BaseNodePostprocessor):
    """Re-sorts nodes by stopword-filtered content-term overlap between the
    query and (heading + text), stable sort. Ported from the legacy
    rag_generation.py rerank_for_adherence(), operating on NodeWithScore here."""

    @classmethod
    def class_name(cls) -> str:
        return "AdherenceRerankPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            return nodes
        query_terms = content_terms(query_bundle.query_str)

        def overlap(node_with_score: NodeWithScore) -> int:
            text = f"{node_with_score.node.metadata.get('heading', '')} {node_with_score.node.text}"
            return len(query_terms & content_terms(text))

        return [
            node
            for _, node in sorted(
                enumerate(nodes),
                key=lambda item: (-overlap(item[1]), item[0]),
            )
        ]


def rerank_result_dicts(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rerank_for_adherence(query, results)


class LlamaIndexRagRetriever:
    """Drop-in replacement for the legacy RagRetriever (rag_generation.py).
    Same constructor/retrieve() shape so callers change only their import."""

    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX_DIR,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        device: str | None = None,
        hybrid: bool = False,
        fusion_candidates: int = DEFAULT_FUSION_CANDIDATES,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.index = build_or_load_index(
            chunks_path=chunks_path,
            index_dir=index_dir,
            embed_model_name=embed_model_name,
            device=device,
        )
        self.hybrid = hybrid
        self.fusion_candidates = fusion_candidates
        self.rrf_k = rrf_k
        self.bm25 = None
        if hybrid:
            nodes = load_nodes(chunks_path)
            self.bm25 = build_bm25_retriever(nodes, top_k=fusion_candidates)

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        candidate_k = max(top_k * 3, self.fusion_candidates if self.hybrid else top_k)
        results = dense_search(self.index, query, candidate_k)

        if self.bm25 is not None:
            bm25_results = bm25_search(self.bm25, query, candidate_k)
            results = reciprocal_rank_fusion(
                dense_results=results,
                bm25_results=bm25_results,
                top_k=top_k,
                rrf_k=self.rrf_k,
            )
            return rerank_result_dicts(query, results)[:top_k]

        return rerank_result_dicts(query, results)[:top_k]
