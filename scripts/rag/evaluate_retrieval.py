import argparse
import csv
import json
from pathlib import Path
from typing import Any

from llamaindex_retriever import (DEFAULT_CHUNKS_PATH,
                                  DEFAULT_FUSION_CANDIDATES,
                                  DEFAULT_INDEX_DIR, DEFAULT_EMBED_MODEL,
                                  DEFAULT_RRF_K, bm25_search,
                                  build_bm25_retriever, build_or_load_index,
                                  dense_search, load_nodes,
                                  reciprocal_rank_fusion)
from tqdm import tqdm

DEFAULT_INPUT_PATH = Path("dataset/guidelines/guidelines_qa_pairs.csv")
DEFAULT_OUTPUT_PATH = Path("dataset/rag/retrieval_eval.csv")


def load_qa_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No QA rows found in {path}")

    required_columns = {"source_file", "page_num", "question", "answer"}
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")

    if limit is not None:
        return rows[:limit]
    return rows


def page_hit(expected_page: str, result: dict[str, Any]) -> bool:
    if not expected_page.isdigit():
        return False
    page_num = int(expected_page)
    return int(result["start_page"]) <= page_num <= int(result["end_page"])


def source_hit(expected_source: str, result: dict[str, Any]) -> bool:
    return result["source_file"] == expected_source


def result_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "score": result["score"],
        "chunk_id": result["chunk_id"],
        "source_file": result["source_file"],
        "start_page": result["start_page"],
        "end_page": result["end_page"],
        "heading": result["heading"],
        "word_count": result["word_count"],
        "text": result["text"],
    }
    if "fusion_score" in result:
        summary["fusion_score"] = result["fusion_score"]
        summary["dense_rank"] = result["dense_rank"]
        summary["dense_score"] = result["dense_score"]
        summary["bm25_rank"] = result["bm25_rank"]
        summary["bm25_score"] = result["bm25_score"]
    return summary


def context_preview(results: list[dict[str, Any]], max_chars: int) -> str:
    blocks = []
    for rank, result in enumerate(results, start=1):
        text = result["text"]
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        blocks.append(
            "\n".join(
                [
                    f"[{rank}] score={result['score']:.4f}",
                    (f"{result['source_file']} " f"p{result['start_page']}-{result['end_page']} | {result['heading']}"),
                    f"chunk_id: {result['chunk_id']}",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks)


def evaluate_rows(
    qa_rows: list[dict[str, str]],
    index: Any,
    bm25: Any,
    top_k: int,
    preview_chars: int,
    hybrid: bool,
    fusion_candidates: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    candidate_k = max(top_k, fusion_candidates if hybrid else top_k)

    eval_rows = []
    for qa_row in tqdm(qa_rows, desc="Evaluating retrieval"):
        question = qa_row["question"]
        results = dense_search(index, question, candidate_k)

        if bm25 is not None:
            bm25_results = bm25_search(bm25, question, candidate_k)
            results = reciprocal_rank_fusion(
                dense_results=results,
                bm25_results=bm25_results,
                top_k=top_k,
                rrf_k=rrf_k,
            )
        else:
            results = results[:top_k]

        source_hits = [source_hit(qa_row["source_file"], result) for result in results]
        page_hits = [source_hit(qa_row["source_file"], result) and page_hit(qa_row["page_num"], result) for result in results]
        first_source_rank = next((i + 1 for i, hit in enumerate(source_hits) if hit), "")
        first_page_rank = next((i + 1 for i, hit in enumerate(page_hits) if hit), "")
        top_result = results[0] if results else {}

        eval_rows.append(
            {
                "question": question,
                "expected_answer": qa_row["answer"],
                "expected_source_file": qa_row["source_file"],
                "expected_page_num": qa_row["page_num"],
                "source_hit_at_k": bool(first_source_rank),
                "source_hit_rank": first_source_rank,
                "page_hit_at_k": bool(first_page_rank),
                "page_hit_rank": first_page_rank,
                "top_1_score": top_result.get("score", ""),
                "top_1_chunk_id": top_result.get("chunk_id", ""),
                "top_1_source_file": top_result.get("source_file", ""),
                "top_1_start_page": top_result.get("start_page", ""),
                "top_1_end_page": top_result.get("end_page", ""),
                "top_1_heading": top_result.get("heading", ""),
                "top_1_text": top_result.get("text", ""),
                "top_k_results_json": json.dumps(
                    [result_summary(result) for result in results],
                    ensure_ascii=False,
                ),
                "top_k_context": context_preview(results, max_chars=preview_chars),
            }
        )

    return eval_rows


def write_eval_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question",
        "expected_answer",
        "expected_source_file",
        "expected_page_num",
        "source_hit_at_k",
        "source_hit_rank",
        "page_hit_at_k",
        "page_hit_rank",
        "top_1_score",
        "top_1_chunk_id",
        "top_1_source_file",
        "top_1_start_page",
        "top_1_end_page",
        "top_1_heading",
        "top_1_text",
        "top_k_results_json",
        "top_k_context",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], top_k: int, output_path: Path) -> None:
    total = len(rows)
    source_hits = sum(row["source_hit_at_k"] for row in rows)
    page_hits = sum(row["page_hit_at_k"] for row in rows)
    print(f"Evaluated {total} QA rows")
    print(f"Source hit@{top_k}: {source_hits}/{total} ({source_hits / total:.1%})")
    print(f"Page hit@{top_k}: {page_hits}/{total} ({page_hits / total:.1%})")
    print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG chunk retrieval against guideline QA rows.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Guideline QA CSV path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output evaluation CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help=f"Chunk CSV path. Default: {DEFAULT_CHUNKS_PATH}",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"LlamaIndex FAISS storage directory. Default: {DEFAULT_INDEX_DIR}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBED_MODEL,
        help=f"HuggingFace embedding model name. Default: {DEFAULT_EMBED_MODEL}",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional embedding device, such as 'cpu', 'mps', or 'cuda'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question. Default: 5",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Limit evaluation to the first N QA rows.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=700,
        help="Maximum text characters per retrieved chunk in top_k_context. Default: 700",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the FAISS index before evaluation.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Fuse dense FAISS and BM25 rankings with reciprocal rank fusion before evaluation.",
    )
    parser.add_argument(
        "--fusion-candidates",
        type=int,
        default=DEFAULT_FUSION_CANDIDATES,
        help=f"Number of dense and BM25 candidates to fuse. Default: {DEFAULT_FUSION_CANDIDATES}",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=f"Reciprocal rank fusion smoothing constant. Default: {DEFAULT_RRF_K}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    if args.rows is not None and args.rows < 1:
        raise SystemExit("--rows must be at least 1")
    if args.preview_chars < 100:
        raise SystemExit("--preview-chars must be at least 100")
    if args.fusion_candidates < 1:
        raise SystemExit("--fusion-candidates must be at least 1")
    if args.rrf_k < 1:
        raise SystemExit("--rrf-k must be at least 1")

    qa_rows = load_qa_rows(args.input, limit=args.rows)
    index = build_or_load_index(
        chunks_path=args.chunks,
        index_dir=args.index,
        embed_model_name=args.model,
        device=args.device,
        rebuild=args.rebuild,
    )
    bm25 = None
    if args.hybrid:
        nodes = load_nodes(args.chunks)
        bm25 = build_bm25_retriever(nodes, top_k=args.fusion_candidates)

    eval_rows = evaluate_rows(
        qa_rows=qa_rows,
        index=index,
        bm25=bm25,
        top_k=args.top_k,
        preview_chars=args.preview_chars,
        hybrid=args.hybrid,
        fusion_candidates=args.fusion_candidates,
        rrf_k=args.rrf_k,
    )
    write_eval_rows(eval_rows, args.output)
    print_summary(eval_rows, top_k=args.top_k, output_path=args.output)


if __name__ == "__main__":
    main()
