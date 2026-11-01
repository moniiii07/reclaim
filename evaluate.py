"""Precision@1 / Recall@3 on labelled data (data/sample_reports.csv, `match_id` = ground truth)."""
import sys

import pandas as pd

from matcher import Embedder, extract_attributes, find_matches, nlp_backend

path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_reports.csv"
df = pd.read_csv(path, dtype=str, keep_default_na=False)
emb = Embedder()
lost = df[(df.type == "lost") & (df.match_id != "")]
found = df[df.type == "found"]

hit1 = hit3 = 0
for _, row in lost.iterrows():
    res = find_matches(row.to_dict(), found, emb, use_llm=False, top_k=3)
    ids = [r["row"]["id"] for r in res]
    hit1 += bool(ids) and ids[0] == row.match_id
    hit3 += row.match_id in ids

n = len(lost)
print(f"NLP: {nlp_backend()} | embeddings: {emb.kind} | labelled lost reports: {n}")
if n:
    print(f"Precision@1: {hit1 / n:.2%}   Recall@3: {hit3 / n:.2%}")
# false-match check: lost reports with NO real match should not get a 'Likely' result
orphans = df[(df.type == "lost") & (df.match_id == "")]
fp = sum(any(r["label"] == "Likely" for r in find_matches(r.to_dict(), found, emb, top_k=3))
         for _, r in orphans.iterrows())
if len(orphans):
    print(f"False 'Likely' on {len(orphans)} unmatched lost reports: {fp}")