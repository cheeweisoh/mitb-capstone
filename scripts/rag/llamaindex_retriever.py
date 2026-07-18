import csv
from pathlib import Path
from typing import Any

import faiss
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

DEFAULT_CHUNKS_PATH = Path("dataset/rag/rag_chunks.csv")
DEFAULT_INDEX_DIR = Path("dataset/rag/llamaindex_faiss")
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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


def dense_search(index: VectorStoreIndex, query: str, top_k: int) -> list[dict[str, Any]]:
    retriever = index.as_retriever(similarity_top_k=top_k)
    return [node_to_result_dict(nws) for nws in retriever.retrieve(query)]


class LlamaIndexRagRetriever:
    """Drop-in replacement for the legacy RagRetriever (rag_generation.py).
    Same constructor/retrieve() shape so callers change only their import."""

    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX_DIR,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        device: str | None = None,
    ) -> None:
        self.index = build_or_load_index(
            chunks_path=chunks_path,
            index_dir=index_dir,
            embed_model_name=embed_model_name,
            device=device,
        )

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        return dense_search(self.index, query, top_k)
