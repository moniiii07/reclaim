"""Evaluate retrieval + scoring against the labelled sample data.
Usage: python evaluate.py            (rule-based + embeddings)
       python evaluate.py --llm      (adds LLM verification, needs ANTHROPIC_API_KEY)
"""
import sys
import pandas as pd
from matcher import Embedder, find_matches, LIKELY_T

use_llm = "--llm" in sys.argv
df = pd.read_csv("data/sample_reports.csv", dtype=str, keep_default_na=False)
lost, found = df[df.type == "lost"], df[df.type == "found"]
emb = Embedder()
print(f"Embedder: {emb.kind} | LLM: {use_llm}\n")

hit1 = hit3 = n_pos = fp = n_neg = 0
for _, q in lost.iterrows():
    res = find_matches(q.to_dict(), found, emb, use_llm=use_llm, top_k=5)
    ids = [r["row"]["id"] for r in res]
    if q.match_id:
        n_pos += 1
        hit1 += ids[:1] == [q.match_id]
        hit3 += q.match_id in ids[:3]
        print(f"{q.id:>4} -> top1 {ids[0]:>4} ({res[0]['confidence']:.2f}, {res[0]['label']})  true={q.match_id}")
    else:
        n_neg += 1
        wrong = bool(res) and res[0]["confidence"] >= LIKELY_T
        fp += wrong
        print(f"{q.id:>4} -> no true match; top {ids[0]} ({res[0]['confidence']:.2f}, {res[0]['label']})")

print(f"\nPrecision@1 (top-1 accuracy): {hit1}/{n_pos} = {hit1/n_pos:.0%}")
print(f"Recall@3:                     {hit3}/{n_pos} = {hit3/n_pos:.0%}")
print(f"False 'Likely' on no-match queries: {fp}/{n_neg}")
