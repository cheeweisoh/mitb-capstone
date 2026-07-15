import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

SUBREDDIT = "AskDocs"
POST_LIMIT = 5000
FETCH_CATEGORIES = ["best", "hot", "new", "top", "rising"]
OUTPUT_DIR = Path("data")

API_BASE = "https://www.reddit.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AskDocs_scraper/1.0)"}


def api_get(path: str, params: dict = None) -> dict:
    resp = requests.get(f"{API_BASE}{path}.json", params=params, headers=HEADERS, timeout=15)

    remaining = float(resp.headers.get("X-Ratelimit-Remaining", 1))
    reset_secs = float(resp.headers.get("X-Ratelimit-Reset", 0))
    if remaining < 2:
        time.sleep(reset_secs)
    resp.raise_for_status()

    return resp.json()


def fetch_posts(category: str, limit: int, existing_ids: set[str]) -> list[dict]:
    if category not in ("hot", "new", "top", "rising", "best"):
        raise ValueError(f"Invalid category '{category}'. Choose from: best, hot, new, top, rising")

    posts = []
    after = None

    while len(posts) < limit:
        params = {"limit": 100}
        if after:
            params["after"] = after

        data = api_get(f"/r/{SUBREDDIT}/{category}", params)
        children = data["data"]["children"]
        if not children:
            break

        for c in children:
            if c["data"]["id"] not in existing_ids and c["data"]["link_flair_text"] == "Physician Responded":
                posts.append(c["data"])
                if len(posts) >= limit:
                    break

        after = data["data"].get("after")
        if not after:
            break

    return posts


def parse_comment_tree(comments: list) -> list[dict]:
    result = []

    for item in comments:
        if item["kind"] != "t1":  # t1 = comment
            continue

        d = item["data"]
        replies = []
        if d.get("replies") and isinstance(d["replies"], dict):
            replies = parse_comment_tree(d["replies"]["data"]["children"])

        result.append(
            {
                "id": d["id"],
                "author": d.get("author", "[deleted]"),
                "author_flair": d.get("author_flair_text"),
                "body": d.get("body", ""),
                "score": d.get("score", 0),
                "created_utc": datetime.fromtimestamp(d["created_utc"], tz=timezone.utc).isoformat(),
                "is_submitter": d.get("is_submitter", False),
                "replies": replies,
            }
        )

    return result


def fetch_comments(post_id: str) -> list[dict]:
    data = api_get(f"/r/{SUBREDDIT}/comments/{post_id}", {"limit": 500})
    comment_listing = data[1]["data"]["children"]

    return parse_comment_tree(comment_listing)


def serialize_post(post: dict, comments: list[dict]) -> dict:
    return {
        "id": post["id"],
        "title": post["title"],
        "author": post.get("author", "[deleted]"),
        "author_flair": post.get("author_flair_text"),
        "selftext": post.get("selftext", ""),
        "score": post.get("score", 0),
        "upvote_ratio": post.get("upvote_ratio", 0),
        "url": post.get("url", ""),
        "permalink": f"https://www.reddit.com{post['permalink']}",
        "created_utc": datetime.fromtimestamp(post["created_utc"], tz=timezone.utc).isoformat(),
        "num_comments": post.get("num_comments", 0),
        "flair": post.get("link_flair_text"),
        "comments": comments,
    }


def save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_existing_ids(directory: Path) -> set[str]:
    return {p.stem for p in directory.rglob("*.json") if p.stem.isalnum()}


def main():
    existing_ids = load_existing_ids(OUTPUT_DIR)
    all_posts = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for category in FETCH_CATEGORIES:
        print(f"Fetching {POST_LIMIT} new '{category}' posts from r/{SUBREDDIT} (skipping already-saved)...")
        posts = fetch_posts(category, POST_LIMIT, existing_ids)
        print(f"Retrieved {len(posts)} new posts. Now fetching comments...\n")

        for i, post in enumerate(posts, start=1):
            print(f"  [{i}/{len(posts)}] {post['title'][:70]!r}")
            try:
                comments = fetch_comments(post["id"])
                post_data = serialize_post(post, comments)
                all_posts.append(post_data)

                save_json(post_data, OUTPUT_DIR / f"{post['id']}.json")
            except Exception as exc:
                print(f"    ⚠ Skipped {post['id']}: {exc}")

    # combined_path = OUTPUT_DIR / f"askdocs_{FETCH_CATEGORY}_{timestamp}.json"
    # save_json(all_posts, combined_path)

    total_comments = sum(p["num_comments"] for p in all_posts)
    print(f"\n✓ Done! Saved {len(all_posts)} posts (~{total_comments} comments)")
    # print(f"  Combined file : {combined_path}")
    # print(f"  Per-post files: {OUTPUT_DIR / timestamp}/")


if __name__ == "__main__":
    main()
