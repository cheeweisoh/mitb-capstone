import json
import os
import time
from itertools import groupby as itertools_groupby
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

PHYSICIAN_FLAIRS = [
    "Physician",
    "Physician - Psychiatry",
    "Physician - Psychiatry | Moderator",
    "Medical Student",
    "Physician - Emergency Medicine",
    "Physician - Vascular Surgery",
    "Speech Language Pathologist",
    "Physician | Moderator | Top Contributor",
    "Registered Nurse",
    "Physician | Heme/Onc",
    "Physician - Dermatologist | Top Contributor",
    "Physician - Family Medicine",
    "Cardiology Acute Care Practitioner",
    "Physician - Neurology",
    "Pharmacist",
    "Registered Dietician ",
    "Imaging Technologist, MRI",
    "Physician - Pathology",
    "Physician - Pediatrics",
    "CRNA",
    "Physician | Top Contributor",
    "Dentist",
    "Physician Assistant",
    "Clinical Pharmacist",
    "RN",
    "Radiology/Lab Technician",
    "Physician - Ob/Gyn",
    "Licensed Mental Health Counselor",
    "Physician | FM &amp; PHPM",
    "Paramedic",
]

SYSTEM_PROMPT = """
You are a medical knowledge extraction system. Your job is to convert raw Reddit posts and physician responses into clean, high-quality question-and-answer pairs for a medical QA dataset. \

Rules: \
- QUESTION: Use the COMPLETE thread content (title and body). Extract the patient's main question/concern from the thread, while making it sound like a question that might be posed to an LLM. Keep the length to 2-3 sentences. Also preserve ALL important medical details including: \
    - Specific symptoms, conditions, or medications mentioned \
    - Relevant patient characteristics (age, gender, medical history if mentioned) \
    - Timeline information (duration, frequency) \
    - Specific concerns or uncertainties \
- ANSWER: Synthesise ALL of the physician's comments into a single coherent well-structured answer. Remove redundancy and present the medical advice cleary. Do not add any information that was not in the original physician comments. Keep the length to 2-3 sentences. Preserve ALL important medical details including: \
    - Specific recommendations and treatment suggestions \
    - Important warnings, contraindications, or precautions \
    - Diagnostic considerations or differential diagnoses mentioned \
    - Specific medication names, dosages, or procedures if mentioned \
    - Reasoning behind the advice when provided \
    - Avoid using sentences that are too technical or jargon-heavy. Make it clear and concise as though explaining to a patient, while preserving the medical information. \
    
Here are some examples of what kind of questions we extract (and their corresponding answers):

"Question": "I woke up this morning feeling the whole room is spinning when i was sitting down. I went to the bathroom walking unsteadily, as i tried to focus i feel nauseous. I try to vomit but it wont come out.. After taking panadol and sleep for few hours, i still feel the same.. By the way, if i lay down or sit down, my head do not spin, only when i want to move around then i feel the whole world is spinning.. And it is normal stomach discomfort at the same time? Earlier after i relieved myself, the spinning lessen so i am not sure whether its connected or coincidences.",
"Answer": "The most likely cause for your symptoms is benign paroxysmal positional vertigo (BPPV), a type of peripheral vertigo. In this condition, the most common symptom is dizziness or giddiness, which is made worse with movements. Accompanying nausea and vomiting are common. The condition is due to problem in the ear, and improves in a few days on own. Betahistine tablets would help relieve your symptoms. Doing vestibular rehabilitation or adaptation exercises would prevent the recurrence of these symptoms. An ENT evaluation would also help."

"Question": "My baby has been pooing 5-6 times a day for a week. In the last few days it has increased to 7 and they are very watery with green stringy bits in them. He does not seem unwell i.e no temperature and still eating. He now has a very bad nappy rash from the pooing",
"Answer": "It seems your kid is having viral diarrhea. Once it starts it will take 5-7 days to completely get better. Unless the kids having low urine output or very dull or excessively sleepy or blood in motion or green bilious vomiting...you need not worry. There is no need to use antibiotics unless there is blood in the motion. Antibiotics might worsen if unnecessarily used causing antibiotic associated diarrhea. I suggest you use zinc supplements.

Output ONLY a JSON object with exactly two keys: "question" and "answer". No preamble, no explanation, no markdown fences.
"""

USER_TEMPLATE = """
## Original Post

Title: {post_title}

{post_body}

## Physician Response(s)

{physician_comments}

Generate the QA pair now.\
"""


def build_chains(df):
    chains = []

    for thread_id, thread_df in raw_comments.groupby("thread_id"):
        physician_rows = thread_df[thread_df["is_physician"]]
        if physician_rows.empty:
            continue

        post_title = thread_df["title"].iloc[0]
        post_title = post_title if isinstance(post_title, str) else ""
        post_body = thread_df["content"].iloc[0]
        post_body = post_body if isinstance(post_body, str) else ""
        if post_body.strip() == post_title.strip():
            post_body = ""

        author_comments = [row["comment_body"].strip() for _, row in thread_df.iterrows() if row["is_author"]]
        n_author_turns = len(author_comments) + 1

        post_body = [
            post_body,
        ] + author_comments

        for root_id, chain_df in physician_rows.groupby("root_comment_id"):
            chain_df_sorted = chain_df.sort_values("comment_order")
            chain_score = chain_df_sorted[chain_df_sorted["is_physician"]]["comment_score"].sum()

            physician_comments = [row["comment_body"].strip() for _, row in chain_df_sorted.iterrows() if row["is_physician"]]

            if not physician_comments:
                continue

            chains.append(
                {
                    "thread_id": thread_id,
                    "root_comment_id": root_id,
                    "chain_score": chain_score,
                    "n_author_turns": n_author_turns,
                    "n_physician_turns": len(physician_comments),
                    "physician_authors": chain_df_sorted["comment_author"].tolist(),
                    "physician_flairs": chain_df_sorted["comment_flair"].tolist(),
                    "post_title": post_title,
                    "post_body": post_body,
                    "physician_comments": physician_comments,
                }
            )

    return chains


def select_top_n(chains, top_n):
    chains_sorted = sorted(chains, key=lambda x: (x["thread_id"], -x["chain_score"]))
    selected = []

    for _, group in itertools_groupby(chains_sorted, key=lambda x: x["thread_id"]):
        for rank, chain in enumerate(group, start=1):
            if rank > top_n:
                break
            chain["chain_rank"] = rank
            selected.append(chain)

    return selected


def fix_unescaped_quotes(raw: str) -> str:
    import re

    def escape_inner_quotes(m):
        key = m.group(1)
        value = m.group(2)
        value = re.sub(r'(?<!\\)"', '\\"', value)
        return f'"{key}": "{value}"'

    fixed = re.sub(
        r'"(question|answer)":\s*"(.*?)"(?=\s*[,}])',
        escape_inner_quotes,
        raw,
        flags=re.DOTALL,
    )

    return fixed


def _create_openai_client() -> Any:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key) if api_key else OpenAI()


OPENAI_CLIENT = _create_openai_client()


def generate_qa(chain, retries):
    author_block = "\n\n".join(x for x in chain["post_body"])
    physician_block = "\n\n".join(x for x in chain["physician_comments"])

    user_message = USER_TEMPLATE.format(
        post_title=chain["post_title"],
        post_body=author_block,
        physician_comments=physician_block,
    )

    for attempt in range(retries):
        try:

            response = OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = json.loads(fix_unescaped_quotes(raw))

            if "question" not in parsed or "answer" not in parsed:
                raise ValueError("Missing 'question' or 'answer' key")
            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Parse Error Attemp {attempt + 1}/{retries}: {e}")
            if attempt == retries:
                return None
            time.sleep(1)

        except Exception as e:
            status_code = getattr(e, "status_code", None)
            response = getattr(e, "response", None)
            if status_code is None and response is not None:
                status_code = getattr(response, "status_code", None)

            if status_code == 429:
                print("Rate limited by OpenAI (429). Sleeping 60 seconds before retrying.")
                time.sleep(60)
                continue

            print(f"API Error Attempt {attempt + 1}/{retries}: {e}")
            if attempt == retries:
                return None
            time.sleep(2**attempt)


if __name__ == "__main__":
    raw_comments = pd.read_csv("dataset/raw_dataset.csv")
    raw_comments["is_physician"] = raw_comments["comment_flair"].isin(PHYSICIAN_FLAIRS)
    raw_comments["is_author"] = (raw_comments["thread_author"] == raw_comments["comment_author"]) & (raw_comments["thread_author"] != "[deleted]")

    output_path = "dataset/generated_qa_pairs.csv"
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        existing_thread_ids = set(existing_df.get("root_comment_id", pd.Series()).dropna().astype(str))
    else:
        existing_df = None
        existing_thread_ids = set()

    chains = build_chains(raw_comments)
    top_n_chains = select_top_n(chains, 5)
    pending_chains = [chain for chain in top_n_chains if str(chain["root_comment_id"]) not in existing_thread_ids]

    raw_limit = input("How many samples to generate? (press Enter for all): ").strip()
    if raw_limit:
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError("Sample count must be an integer or empty.") from exc
        if limit < 1:
            raise ValueError("Sample count must be at least 1.")
        pending_chains = pending_chains[:limit]

    qa_pairs = []
    n_failed = 0
    n_success = 0
    write_header = not os.path.exists(output_path)

    for chain in tqdm(pending_chains, desc="Generating QA Pairs"):
        qa = generate_qa(chain, 3)

        if qa is None:
            n_failed += 1
            continue

        record = {
            "thread_id": chain["thread_id"],
            "chain_rank": chain["chain_rank"],
            "root_comment_id": chain["root_comment_id"],
            "physician_score": chain["chain_score"],
            "n_physician_turns": chain["n_physician_turns"],
            "physician_authors": chain["physician_authors"],
            "physician_flairs": chain["physician_flairs"],
            "question": qa["question"],
            "answer": qa["answer"],
        }
        qa_pairs.append(record)
        n_success += 1

        pd.DataFrame([record]).to_csv(output_path, mode="a", header=write_header, index=False)
        write_header = False

    tqdm.write(
        f"{len(existing_thread_ids)} chains already present. Processed {n_success + n_failed} chains. Success: {n_success}, Failed: {n_failed}. {len(top_n_chains) - len(existing_thread_ids) - n_success - n_failed} remaining"
    )

