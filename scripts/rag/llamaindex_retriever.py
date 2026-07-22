import csv
from pathlib import Path
from typing import Any

import faiss
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

DEFAULT_CHUNKS_PATH = Path("dataset/rag/rag_chunks.csv")
DEFAULT_INDEX_DIR = Path("dataset/rag/llamaindex_faiss")
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Cosine-similarity floor below which a retrieved chunk is dropped rather than
# forced into context just to fill top_k. 0.40 chosen from bucketed composite
# eval scores on the guidelines RAG run: below 0.40 avg composite ~4.20,
# at/above ~4.39-4.56 (Mann-Whitney p=0.0084, joined result/eval on question).
DEFAULT_MIN_SCORE = 0.40
# Cross-encoder used to re-rank the dense candidate pool. The dense retriever
# (a bi-encoder) frequently ranks a chunk highest on vocabulary/topic overlap
# alone -- e.g. a generic "Recommendation 6" from an unrelated guideline
# scoring above the real answer -- because it embeds query and chunk
# independently. A cross-encoder scores query and chunk jointly, so it catches
# that mismatch; see llamaindex_retriever manual smoke test in the RAG
# improvement work for a concrete example of this reordering.
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Dense candidates fetched (as a multiple of top_k) before reranking, so the
# cross-encoder has a wider pool to promote a genuinely on-topic chunk from,
# not just whatever the bi-encoder already ranked in the top_k.
DEFAULT_RERANK_FETCH_MULTIPLIER = 4


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


def node_to_result_dict(node_with_score: NodeWithScore) -> dict[str, Any]:
    node = node_with_score.node
    result = dict(node.metadata)
    result["chunk_id"] = node.node_id
    result["text"] = node.text
    result["score"] = float(node_with_score.score) if node_with_score.score is not None else 0.0
    return result


def dense_search_nodes(index: VectorStoreIndex, query: str, top_k: int) -> list[NodeWithScore]:
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)


def dense_search(index: VectorStoreIndex, query: str, top_k: int) -> list[dict[str, Any]]:
    return [node_to_result_dict(nws) for nws in dense_search_nodes(index, query, top_k)]


class LlamaIndexRagRetriever:
    """Drop-in replacement for the legacy RagRetriever (rag_generation.py).
    Same constructor/retrieve() shape so callers change only their import."""

    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX_DIR,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        device: str | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
        rerank_model: str | None = DEFAULT_RERANK_MODEL,
        rerank_fetch_multiplier: int = DEFAULT_RERANK_FETCH_MULTIPLIER,
    ) -> None:
        self.index = build_or_load_index(
            chunks_path=chunks_path,
            index_dir=index_dir,
            embed_model_name=embed_model_name,
            device=device,
        )
        self.min_score = min_score
        self.rerank_fetch_multiplier = rerank_fetch_multiplier
        # top_n is set per-call in retrieve() since top_k can vary by caller;
        # rerank_model=None disables reranking (plain dense top-k + min_score,
        # the pre-rerank behavior).
        self.reranker = (
            SentenceTransformerRerank(model=rerank_model, top_n=1, keep_retrieval_score=True, device=device)
            if rerank_model
            else None
        )

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self.reranker is None:
            results = dense_search(self.index, query, top_k)
            return [result for result in results if result["score"] >= self.min_score]

        fetch_k = max(top_k * self.rerank_fetch_multiplier, top_k)
        candidates = dense_search_nodes(self.index, query, fetch_k)
        if not candidates:
            return []

        self.reranker.top_n = top_k
        reranked = self.reranker.postprocess_nodes(candidates, query_str=query)

        results = []
        for node_with_score in reranked:
            result = node_to_result_dict(node_with_score)
            # keep_retrieval_score stashed the pre-rerank cosine score in
            # metadata (as "retrieval_score") before overwriting node.score
            # with the cross-encoder's own score. Restore the cosine score as
            # `score` so min_score / verify_score_floor keep comparing against
            # the calibrated 0-1 cosine scale; the cross-encoder's uncalibrated
            # logit score is kept separately for visibility/debugging.
            result["rerank_score"] = result["score"]
            result["score"] = float(result.pop("retrieval_score", result["rerank_score"]))
            results.append(result)

        return [result for result in results if result["score"] >= self.min_score]
