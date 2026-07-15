import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
OUTPUT_PATH = "dataset.csv"


def has_physician_flair(comment: dict) -> bool:
    """Check if this comment or any reply contains physician flair."""
    if comment.get("author_flair") == "Physician":
        return True
    for reply in comment.get("replies", []):
        if has_physician_flair(reply):
            return True
    return False


def extract_all_comments(comment: dict, root_comment_id: str, order: int = 0) -> list[dict]:
    """Extract this comment and all replies into a flat list with order."""
    result = [
        {
            "comment": comment,
            "root_comment_id": root_comment_id,
            "order": order,
        }
    ]
    for i, reply in enumerate(comment.get("replies", []), start=1):
        result.extend(extract_all_comments(reply, root_comment_id, order + i))
    return result


def process_file(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        post = json.load(f)

    thread_id = post.get("id")
    thread_author = post.get("author")
    title = (post.get("title") or "").strip()
    selftext = (post.get("selftext") or "").strip()
    query = f"{title}\n\n{selftext}".strip() if selftext else title

    rows = []
    for root_comment in post.get("comments", []):
        if root_comment.get("author") == "AutoModerator":
            continue

        # Only include chains with physician flair
        if not has_physician_flair(root_comment):
            continue

        # Extract all comments from this chain
        all_comments = extract_all_comments(root_comment, root_comment.get("id"))

        for comment_data in all_comments:
            comment = comment_data["comment"]
            root_id = comment_data["root_comment_id"]
            order = comment_data["order"]

            rows.append(
                {
                    "thread_id": thread_id,
                    "thread_author": thread_author,
                    "root_comment_id": root_id,
                    "comment_id": comment.get("id"),
                    "comment_order": order,
                    "title": title,
                    "content": selftext,
                    "query": query,
                    "comment_body": comment.get("body"),
                    "comment_author": comment.get("author"),
                    "comment_score": comment.get("score"),
                    "comment_flair": comment.get("author_flair"),
                }
            )

    return rows


def main():
    json_files = sorted(DATA_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {DATA_DIR}")

    all_rows = []
    for path in json_files:
        all_rows.extend(process_file(path))

    df = pd.DataFrame(
        all_rows,
        columns=[
            "thread_id",
            "thread_author",
            "root_comment_id",
            "comment_id",
            "comment_order",
            "title",
            "content",
            "query",
            "comment_body",
            "comment_author",
            "comment_score",
            "comment_flair",
        ],
    )
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
