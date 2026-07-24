import csv
from pathlib import Path
from typing import Any

import faiss
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
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
# Hybrid retrieval: a dense bi-encoder alone can miss a chunk that shares an
# exact term with the query (drug names, acronyms, numeric codes) but isn't
# semantically close in embedding space. BM25 catches those on lexical
# overlap; its candidates are merged into the same fetch pool the
# cross-encoder reranks, so BM25 only ever *adds* candidates for the
# reranker to judge -- it never bypasses that judgment.
DEFAULT_HYBRID = True
# When source_file_filter narrows retrieval to specific document(s) (the
# router's job -- see topic_router.py), the generic top_k*multiplier fetch
# pulls candidates from the *whole* corpus first and only then filters down,
# so a large document (e.g. Stroke Rehabilitation Guidelines, ~180 of ~640
# corpus chunks) only ever contributes a small, roughly proportional slice of
# that fetch -- most of its own chunks never even reach the reranker. Once
# retrieval is already narrowed to a document, fetch enough candidates to
# cover that document's whole chunk count instead, capped here so a filtered
# fetch never gets unreasonably expensive to rerank.
DEFAULT_MAX_FILTERED_FETCH_K = 200
# Even once a large document's full chunk set reaches the reranker, the
# default 6-layer cross-encoder still struggles to discriminate among ~100
# similarly-worded "Recommendation N" chunks from the same guideline -- it
# was tuned for distinguishing topically-different chunks, not near-duplicate
# ones from a single document. A deeper cross-encoder has more resolving
# power for that; only worth paying for once a filtered document is actually
# large enough for this to matter (Stroke Rehab's ~185 chunks vs. the next
# largest guideline at 35), so it's loaded lazily and used only above the
# threshold below.
DEFAULT_LARGE_DOC_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_LARGE_DOC_CHUNK_THRESHOLD = 60


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
        hybrid: bool = DEFAULT_HYBRID,
        max_filtered_fetch_k: int = DEFAULT_MAX_FILTERED_FETCH_K,
        large_doc_rerank_model: str | None = DEFAULT_LARGE_DOC_RERANK_MODEL,
        large_doc_chunk_threshold: int = DEFAULT_LARGE_DOC_CHUNK_THRESHOLD,
    ) -> None:
        self.index = build_or_load_index(
            chunks_path=chunks_path,
            index_dir=index_dir,
            embed_model_name=embed_model_name,
            device=device,
        )
        self.min_score = min_score
        self.rerank_fetch_multiplier = rerank_fetch_multiplier
        self.max_filtered_fetch_k = max_filtered_fetch_k
        self.device = device
        self.doc_chunk_counts: dict[str, int] = {}
        for row in load_chunk_rows(chunks_path):
            self.doc_chunk_counts[row["source_file"]] = self.doc_chunk_counts.get(row["source_file"], 0) + 1
        # top_n is set per-call in retrieve() since top_k can vary by caller;
        # rerank_model=None disables reranking (plain dense top-k + min_score,
        # the pre-rerank behavior).
        self.reranker = (
            SentenceTransformerRerank(model=rerank_model, top_n=1, keep_retrieval_score=True, device=device)
            if rerank_model
            else None
        )
        # Lazily built the first time a filtered document actually crosses
        # large_doc_chunk_threshold -- most runs/documents never need it, so
        # this avoids loading a second cross-encoder model up front.
        self.large_doc_rerank_model = large_doc_rerank_model
        self.large_doc_chunk_threshold = large_doc_chunk_threshold
        self._large_doc_reranker: SentenceTransformerRerank | None = None
        # similarity_top_k set per-call in _fetch_candidates; BM25's own index
        # build (tokenize+stem all nodes) is local/cheap, no embedding calls.
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=load_nodes(chunks_path), similarity_top_k=1) if hybrid else None

    def _reranker_for(self, source_file_filter: list[str] | None) -> SentenceTransformerRerank | None:
        if not source_file_filter or self.reranker is None or self.large_doc_rerank_model is None:
            return self.reranker
        filtered_chunk_count = sum(self.doc_chunk_counts.get(sf, 0) for sf in source_file_filter)
        if filtered_chunk_count <= self.large_doc_chunk_threshold:
            return self.reranker
        if self._large_doc_reranker is None:
            self._large_doc_reranker = SentenceTransformerRerank(
                model=self.large_doc_rerank_model, top_n=1, keep_retrieval_score=True, device=self.device
            )
        return self._large_doc_reranker

    def _fetch_candidates(self, query: str, fetch_k: int) -> list[NodeWithScore]:
        dense = {nws.node.node_id: nws for nws in dense_search_nodes(self.index, query, fetch_k)}
        if self.bm25_retriever is None:
            return list(dense.values())

        self.bm25_retriever.similarity_top_k = fetch_k
        for nws in self.bm25_retriever.retrieve(query):
            node_id = nws.node.node_id
            if node_id in dense:
                continue  # keep the dense entry -- its score is the calibrated cosine scale
            # BM25-only find: no comparable cosine score exists for it. Treat
            # it as borderline-passing the min_score floor and let the
            # cross-encoder reranker be the real arbiter of relevance, same
            # as any chunk that just barely clears the floor on cosine alone.
            nws.score = self.min_score
            dense[node_id] = nws
        return list(dense.values())

    def retrieve(self, query: str, top_k: int, source_file_filter: list[str] | None = None) -> list[dict[str, Any]]:
        fetch_k = max(top_k * self.rerank_fetch_multiplier, top_k) if self.reranker else top_k
        if source_file_filter:
            filtered_chunk_count = sum(self.doc_chunk_counts.get(sf, 0) for sf in source_file_filter)
            fetch_k = min(max(fetch_k, filtered_chunk_count), self.max_filtered_fetch_k)
        candidates = self._fetch_candidates(query, fetch_k)
        if source_file_filter:
            candidates = [nws for nws in candidates if nws.node.metadata.get("source_file") in source_file_filter]
        if not candidates:
            return []

        reranker = self._reranker_for(source_file_filter)
        if reranker is None:
            candidates = sorted(candidates, key=lambda nws: nws.score or 0.0, reverse=True)[:top_k]
            results = [node_to_result_dict(nws) for nws in candidates]
            return [result for result in results if result["score"] >= self.min_score]

        reranker.top_n = top_k
        reranked = reranker.postprocess_nodes(candidates, query_str=query)

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
