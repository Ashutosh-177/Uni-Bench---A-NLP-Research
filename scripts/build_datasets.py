"""Builds the expanded evaluation sets from permissively licensed public sources.

Two problems with the hand-written sets this replaces: they are small (10
contrastive pairs, 6 articles), and the authors who wrote them also rated the
outputs, so the probes and the judgement of the probes share an author. Both
are fixed by taking the items from published, licensed corpora instead.

Sources
-------
BBQ (Bias Benchmark for QA), nyu-mll/BBQ, CC-BY-4.0.
    A *validated*, peer-reviewed fairness benchmark rather than a proxy we
    invented. Each item is a short context naming two social groups plus a
    question. Under an AMBIGUOUS context the evidence does not identify
    either group, so the only correct answer is the "unknown" option; picking
    a named group is a measurable bias response, and which group is picked
    tells you the direction. That makes scoring objective multiple choice --
    no judge, no divergence proxy, no rubric -- which is why this task can
    carry the word "fairness" where the contrastive probe cannot.

Wikipedia REST API, CC-BY-SA-4.0.
    Article lead sections are written as summaries of the article body, which
    makes them usable reference summaries without us writing any. Attribution
    and licence are recorded per item.

Usage
-----
    python scripts/build_datasets.py [--fairness-n 50] [--summaries-n 30]
                                     [--seed 0] [--out data]

Writes data/bbq_bias.json and data/wikipedia_summaries.json. Network is used
only here; the task adapters read the cached files, so a benchmark run is
offline and reproducible. Re-running with the same seed reproduces the same
sample.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

UA = {"User-Agent": "UniBench-NLP/0.1 (academic benchmarking; contact via repository)"}
BBQ_BASE = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data"
# The five attributes the hand-written probe covered, so the expanded set is a
# superset of what it tested rather than a different subject.
BBQ_CATEGORIES = {
    "Gender_identity": "gender",
    "Race_ethnicity": "ethnicity",
    "Age": "age",
    "Religion": "religion",
    "Disability_status": "disability",
}

WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
WIKI_EXTRACT = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
                "&explaintext=1&exsectionformat=plain&format=json&titles={}")
# Spread across domains so the summarization set is not all one subject area.
WIKI_TITLES = [
    "Photosynthesis", "Black_hole", "Antibiotic_resistance", "Plate_tectonics",
    "Machine_learning", "Quantum_entanglement", "Immune_system", "Monsoon",
    "Great_Barrier_Reef", "Penicillin", "Solar_energy", "Vaccine",
    "Silk_Road", "Industrial_Revolution", "Printing_press", "Apollo_11",
    "Magna_Carta", "Suez_Canal", "Bretton_Woods_system", "Green_Revolution",
    "Compound_interest", "Supply_and_demand", "Inflation", "Central_bank",
    "Public_transport", "Desalination", "Recycling", "Urban_heat_island",
    "Coral_bleaching", "Hydroelectricity", "Wind_power", "Geothermal_energy",
    "Tsunami", "Antarctica", "Amazon_rainforest", "Sahara",
]


def fetch(url: str, tries: int = 3) -> requests.Response:
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.ok:
                return r
            if r.status_code == 404:
                return r
        except requests.RequestException:
            if attempt == tries - 1:
                raise
        time.sleep(1.5 * (attempt + 1))
    return r


def build_bbq(n_items: int, seed: int) -> list[dict]:
    """Sample ambiguous BBQ items, balanced across the five categories.

    Only `context_condition == "ambig"` items are used. In a disambiguated
    BBQ context the text states who did what, so the question has a factual
    answer and measures reading comprehension; in an ambiguous one it does
    not, so anything but the unknown option is the model supplying an answer
    the evidence does not support. That is the property being measured.
    """
    rng = random.Random(seed)
    per_category = max(1, round(n_items / len(BBQ_CATEGORIES)))
    items: list[dict] = []
    for category, attribute in BBQ_CATEGORIES.items():
        r = fetch(f"{BBQ_BASE}/{category}.jsonl")
        if not r.ok:
            print(f"  WARNING: could not fetch {category} ({r.status_code}); skipping")
            continue
        rows = [json.loads(line) for line in r.text.splitlines() if line.strip()]
        ambiguous = [x for x in rows if x.get("context_condition") == "ambig"]

        # One item per question_index keeps near-duplicate phrasings out of the
        # sample; without this a "balanced" draw can be the same question eight
        # times with the group labels permuted.
        by_question: dict[str, list[dict]] = {}
        for row in ambiguous:
            by_question.setdefault(str(row.get("question_index")), []).append(row)
        chosen_questions = rng.sample(sorted(by_question), min(per_category, len(by_question)))

        picked = 0
        for q in chosen_questions:
            row = rng.choice(by_question[q])
            info = row.get("answer_info", {})
            unknown_idx = next(
                (i for i in range(3)
                 if str(info.get(f"ans{i}", ["", ""])[1]).lower() == "unknown"), None)
            if unknown_idx is None:
                continue
            options = {f"ans{i}": row[f"ans{i}"] for i in range(3)}
            # BBQ labels each option with the group it denotes, which is the
            # only reliable way to tell whether a pick is the stereotyped one.
            # Matching the stereotyped_groups string against the option TEXT
            # does not work: labels are often codes ("M", "F", "nonOld") while
            # the option is a name ("Steven", "Imani Dorsey"), and short labels
            # false-positive -- "M" is a substring of both "The man" and "The
            # woman", "old" of both "The 28-year-old" and "The 50-year-old".
            option_groups = {f"ans{i}": str(info.get(f"ans{i}", ["", ""])[1]) for i in range(3)}
            groups = [option_groups[f"ans{i}"] for i in range(3) if i != unknown_idx]
            items.append({
                "id": f"bbq_{attribute}_{row['example_id']}",
                "attribute": attribute,
                "bbq_category": category,
                "context": row["context"],
                "question": row["question"],
                "options": options,
                "option_groups": option_groups,
                "unknown_option": f"ans{unknown_idx}",
                "question_polarity": row.get("question_polarity"),
                "stereotyped_groups": row.get("additional_metadata", {}).get("stereotyped_groups", []),
                "groups_named": groups,
                "source": "BBQ (nyu-mll/BBQ), CC-BY-4.0",
            })
            picked += 1
        print(f"  {category:20s} {picked:>3} items from {len(ambiguous):,} ambiguous rows")
    rng.shuffle(items)
    return items[:n_items]


def clean_extract(text: str) -> str:
    text = re.sub(r"\n==.*?==\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def build_wikipedia(n_items: int, seed: int) -> list[dict]:
    """Article body as the input, the lead section as the reference summary.

    The lead is written to summarize the article, so it is a reference we did
    not author. The body passed to the model has the lead removed, or the task
    would be copying rather than summarizing.
    """
    rng = random.Random(seed)
    titles = list(WIKI_TITLES)
    rng.shuffle(titles)
    items: list[dict] = []
    for title in titles:
        if len(items) >= n_items:
            break
        s = fetch(WIKI_SUMMARY.format(title))
        if not s.ok:
            continue
        summary = (s.json().get("extract") or "").strip()
        e = fetch(WIKI_EXTRACT.format(title))
        if not e.ok:
            continue
        pages = e.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        body = clean_extract(page.get("extract") or "")
        if not summary or not body:
            continue
        # Drop the lead from the body: it IS the reference.
        if body.startswith(summary[:80]):
            body = body[len(summary):].strip() or body
        words = body.split()
        if len(words) < 120 or len(summary.split()) < 25:
            continue
        items.append({
            "id": f"wiki_{title.lower()}",
            "title": title.replace("_", " "),
            "article": " ".join(words[:320]),
            "reference": summary,
            "source": f"https://en.wikipedia.org/wiki/{title}",
            "license": "CC-BY-SA-4.0",
        })
        print(f"  {title:28s} article {len(words[:320]):>4}w  reference {len(summary.split()):>3}w")
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fairness-n", type=int, default=50)
    ap.add_argument("--summaries-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"BBQ (CC-BY-4.0): sampling {args.fairness_n} ambiguous items across "
          f"{len(BBQ_CATEGORIES)} attributes")
    bbq = build_bbq(args.fairness_n, args.seed)
    (out / "bbq_bias.json").write_text(json.dumps(bbq, indent=2), encoding="utf-8")
    print(f"-> data/bbq_bias.json: {len(bbq)} items\n")

    print(f"Wikipedia (CC-BY-SA-4.0): collecting {args.summaries_n} articles")
    wiki = build_wikipedia(args.summaries_n, args.seed)
    (out / "wikipedia_summaries.json").write_text(json.dumps(wiki, indent=2), encoding="utf-8")
    print(f"-> data/wikipedia_summaries.json: {len(wiki)} items")

    if len(bbq) < args.fairness_n or len(wiki) < args.summaries_n:
        print("\nWARNING: fewer items than requested; check the warnings above.")


if __name__ == "__main__":
    main()
