"""ReClaim matching engine.

Pipeline: attribute extraction -> embeddings -> vector similarity retrieval
-> (optional) LLM verification -> hybrid confidence score.
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import date

import numpy as np
import pandas as pd

LLM_MODEL = "claude-haiku-4-5-20251001"

# Hybrid score weights (your "custom algorithm" - tune these and report the results)
W_SEM, W_ATTR, W_LLM = 0.40, 0.30, 0.30
LIKELY_T, POSSIBLE_T = 0.75, 0.50

# --------------------------------------------------------------------------
# 1. Attribute extraction (lexicon-based NER; optional spaCy for brands)
# --------------------------------------------------------------------------
COLOR_SYNONYMS = {
    "black": ["black"], "white": ["white", "cream"], "blue": ["blue", "navy"],
    "red": ["red", "maroon"], "green": ["green"], "grey": ["grey", "gray"],
    "brown": ["brown", "tan", "leather brown"], "silver": ["silver", "steel", "metallic"],
    "gold": ["gold", "golden"], "yellow": ["yellow"], "pink": ["pink"],
    "orange": ["orange"], "purple": ["purple"],
}

# Order matters: first match wins (e.g. "laptop charger" -> charger, not laptop)
ITEM_SYNONYMS = {
    "charger": ["charger", "adapter", "adaptor", "power adapter"],
    "earbuds": ["earbuds", "earphones", "headphones", "airpods", "earpods"],
    "wallet": ["wallet", "purse", "billfold"],
    "keys": ["keys", "key set", "bunch of keys"],
    "glasses": ["glasses", "spectacles", "eyeglasses", "specs"],
    "watch": ["watch", "wristwatch"],
    "phone": ["phone", "iphone", "smartphone", "mobile", "android"],
    "bottle": ["bottle", "flask", "tumbler"],
    "umbrella": ["umbrella"],
    "backpack": ["backpack", "rucksack", "knapsack", "school bag", "bag"],
    "laptop": ["laptop", "macbook", "notebook computer"],
    "id card": ["id card", "identity card", "student id"],
    "book": ["book", "notebook", "textbook"],
}

BRANDS = ["nike", "adidas", "puma", "apple", "samsung", "hp", "dell", "lenovo", "casio",
          "boat", "sony", "jbl", "wildcraft", "skybags", "american tourister", "titan",
          "fastrack", "realme", "oneplus", "redmi", "asus", "acer", "milton", "cello"]


BRAND_ALIASES = {"iphone": "apple", "macbook": "apple", "ipad": "apple", "airpods": "apple",
                 "galaxy": "samsung", "thinkpad": "lenovo", "inspiron": "dell"}


def _has(word: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(word)}s?\b", text) is not None


def extract_attributes(text: str) -> dict:
    t = text.lower()
    item = next((canon for canon, syns in ITEM_SYNONYMS.items()
                 if any(_has(s, t) for s in syns)), None)
    colors = {canon for canon, syns in COLOR_SYNONYMS.items()
              if any(_has(s, t) for s in syns)}
    brand = next((b for b in BRANDS if _has(b, t)), None) \
        or next((b for alias, b in BRAND_ALIASES.items() if _has(alias, t)), None)
    return {"type": item, "colors": colors, "brand": brand}


def enrich_text(text: str, attrs: dict) -> str:
    """Append canonical attributes so 'rucksack' and 'backpack' share tokens."""
    parts = [attrs["type"] or "", " ".join(sorted(attrs["colors"])), attrs["brand"] or ""]
    return " ".join(p for p in parts if p) + " " + text


# --------------------------------------------------------------------------
# 2. Embeddings (Sentence-BERT, with TF-IDF fallback so it always runs)
# --------------------------------------------------------------------------
class Embedder:
    def __init__(self, prefer_sbert: bool = True):
        self.model, self.kind = None, "tfidf"
        if prefer_sbert:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.kind = "sbert"
            except Exception:
                self.model = None

    def similarity(self, queries: list[str], docs: list[str]) -> np.ndarray:
        """Cosine similarity matrix, shape (len(queries), len(docs))."""
        if not queries or not docs:
            return np.zeros((len(queries), len(docs)))
        if self.kind == "sbert":
            q = self.model.encode(queries, normalize_embeddings=True)
            d = self.model.encode(docs, normalize_embeddings=True)
            return np.clip(q @ d.T, 0, 1)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(queries + docs)
        return cosine_similarity(vec.transform(queries), vec.transform(docs))


# --------------------------------------------------------------------------
# 3. Rule-based attribute score + hard vetoes (false-positive control)
# --------------------------------------------------------------------------
def _eq(a, b) -> float:
    if not a or not b:
        return 0.5  # unknown -> neutral
    return 1.0 if a == b else 0.0


def _colors(a: set, b: set) -> float:
    if not a or not b:
        return 0.5
    return len(a & b) / len(a | b)


def _parse_date(d) -> date:
    return d if isinstance(d, date) else pd.to_datetime(d).date()


def time_score(lost_date, found_date) -> float:
    days = (_parse_date(found_date) - _parse_date(lost_date)).days
    return 0.2 if days < 0 else math.exp(-days / 7)


def attribute_score(a: dict, b: dict, loc_sim: float, t_score: float) -> float:
    return (0.40 * _eq(a["type"], b["type"]) + 0.25 * _colors(a["colors"], b["colors"])
            + 0.15 * _eq(a["brand"], b["brand"]) + 0.10 * loc_sim + 0.10 * t_score)


def apply_vetoes(score: float, a: dict, b: dict) -> tuple[float, str | None]:
    if a["type"] and b["type"] and a["type"] != b["type"]:
        return min(score, 0.20), "Different item types"
    if a["brand"] and b["brand"] and a["brand"] != b["brand"]:
        return min(score, 0.40), "Different brands"
    if a["colors"] and b["colors"] and not (a["colors"] & b["colors"]):
        return min(score, 0.50), "Conflicting colors"
    return score, None


# --------------------------------------------------------------------------
# 4. LLM verification (optional - needs ANTHROPIC_API_KEY)
# --------------------------------------------------------------------------
def llm_verify(lost: dict, found: dict) -> dict | None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        prompt = (
            "Decide whether these two reports describe the SAME physical item.\n"
            f"LOST: {lost['description']} | location: {lost['location']} | date: {lost['date']}\n"
            f"FOUND: {found['description']} | location: {found['location']} | date: {found['date']}\n"
            "Different wording of the same thing is fine. Contradictory details "
            "(different brand, color or type) mean no match. "
            'Reply with JSON only: {"score": <0-100>, "reason": "<one short sentence>"}'
        )
        resp = anthropic.Anthropic().messages.create(
            model=LLM_MODEL, max_tokens=200,
            messages=[{"role": "user", "content": prompt}])
        data = json.loads(re.search(r"\{.*\}", resp.content[0].text, re.S).group(0))
        return {"score": max(0.0, min(1.0, float(data["score"]) / 100)),
                "reason": data.get("reason", "")}
    except Exception:
        return None


# --------------------------------------------------------------------------
# 5. End-to-end matching
# --------------------------------------------------------------------------
def label_for(score: float) -> str:
    return "Likely" if score >= LIKELY_T else "Possible" if score >= POSSIBLE_T else "Unlikely"


def find_matches(query: dict, pool: pd.DataFrame, embedder: Embedder,
                 use_llm: bool = False, top_k: int = 5, retrieve_k: int = 10) -> list[dict]:
    """query: dict with type/description/location/date. pool: opposite-type reports."""
    if pool.empty:
        return []
    q_attrs = extract_attributes(query["description"])
    pool = pool.reset_index(drop=True)
    p_attrs = [extract_attributes(d) for d in pool["description"]]

    q_text = enrich_text(query["description"], q_attrs)
    p_texts = [enrich_text(d, a) for d, a in zip(pool["description"], p_attrs)]
    sem = embedder.similarity([q_text], p_texts)[0]
    loc = embedder.similarity([query["location"]], pool["location"].tolist())[0]

    # Stage 1: vector retrieval -> shortlist
    order = np.argsort(-sem)[:retrieve_k]
    results = []
    for i in order:
        row = pool.iloc[i]
        lost, found = (query, row.to_dict()) if query["type"] == "lost" else (row.to_dict(), query)
        t = time_score(lost["date"], found["date"])
        attr = attribute_score(q_attrs, p_attrs[i], float(loc[i]), t)
        score = W_SEM * float(sem[i]) + W_ATTR * float(attr)
        weight = W_SEM + W_ATTR
        results.append({"row": row.to_dict(), "semantic": float(sem[i]), "attribute": float(attr),
                        "location": float(loc[i]), "time": float(t), "llm": None,
                        "llm_reason": "", "_score": score, "_w": weight,
                        "_lost": lost, "_found": found, "_qa": q_attrs, "_pa": p_attrs[i]})

    results.sort(key=lambda r: -r["_score"])
    results = results[:top_k]

    # Stage 2: LLM verification on the shortlist only
    for r in results:
        score, weight = r.pop("_score"), r.pop("_w")
        if use_llm:
            v = llm_verify(r["_lost"], r["_found"])
            if v:
                r["llm"], r["llm_reason"] = v["score"], v["reason"]
                score, weight = score + W_LLM * v["score"], weight + W_LLM
        final, veto = apply_vetoes(score / weight, r.pop("_qa"), r.pop("_pa"))
        r.pop("_lost"), r.pop("_found")
        r["confidence"], r["veto"], r["label"] = float(final), veto, label_for(float(final))

    return sorted(results, key=lambda r: -r["confidence"])