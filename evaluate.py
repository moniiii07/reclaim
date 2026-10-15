"""
ReClaim - evaluate matching quality on the synthetic test sets (data/eval_*.csv, data/heldout_*.csv).

Systems compared:
  baseline : the original prototype (TF-IDF cosine + exact item/location equality)
  proposed : this project's pipeline (embeddings + attribute extraction + verification)

Usage:  python evaluate.py [--no-llm] [--force-fallback]
Metrics: Hit@1, Hit@3, Precision@3, MRR, Recall@Possible, false-match rate (no-match queries).
"""
import argparse, csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matcher as M

def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ---------- baseline (original prototype logic) ----------
def b_sim(a, b):
    a, b = M.preprocess(a), M.preprocess(b)
    if not a or not b: return 0.0
    v = TfidfVectorizer().fit_transform([a, b])
    return float(cosine_similarity(v[0:1], v[1:2])[0][0])

def b_score(lost, found):
    d = b_sim(lost["description"], found["description"])
    i = float(bool(M.preprocess(lost["item"]) and M.preprocess(lost["item"]) == M.preprocess(found["item"])))
    l = float(bool(M.preprocess(lost["location"]) and M.preprocess(lost["location"]) == M.preprocess(found["location"])))
    return round(0.60*d + 0.25*i + 0.15*l, 3)

def baseline(lost, pool, top_k=5):
    res = [(f["id"], b_score(lost, f)) for f in pool]
    res = [(i, s) for i, s in res if s >= 0.50]
    return sorted(res, key=lambda x: -x[1])[:top_k]

def proposed(lost, pool, emb, use_llm):
    out = M.find_matches(lost, pool, embedder=emb, top_k=5, use_llm=use_llm)
    return [(r["report"]["id"], r["confidence"]) for r in out]

def metrics(name, runner, lost_rows, pool):
    hit1 = hit3 = rr = poss = 0; p3 = 0.0; n = 0; fm = fmn = 0
    for q in lost_rows:
        ranked = [i for i, _ in runner(q, pool)]
        if q["true_match"]:
            n += 1
            if ranked[:1] == [q["true_match"]]: hit1 += 1
            if q["true_match"] in ranked[:3]: hit3 += 1; p3 += 1/3
            if q["true_match"] in ranked: rr += 1/(ranked.index(q["true_match"]) + 1); poss += 1
        else:
            fmn += 1
            if ranked: fm += 1
    n_, fmn_ = max(n, 1), max(fmn, 1)   # avoid ZeroDivisionError on sets without matchable / no-match queries
    return {"system": name, "Hit@1": hit1/n_, "Hit@3": hit3/n_, "P@3": p3/n_, "MRR": rr/n_,
            "Recall(any label)": poss/n_, "False-match rate": fm/fmn_, "n_matchable": n, "n_nomatch": fmn}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--force-fallback", action="store_true")
    ap.add_argument("--set", default="eval", choices=["eval", "heldout"], help="eval or heldout")
    args = ap.parse_args()
    lost, pool = read(f"data/{args.set}_lost.csv"), read(f"data/{args.set}_found.csv")
    emb = M.Embedder(force_fallback=args.force_fallback)
    print("embedding backend:", emb.backend)
    rows = [metrics("baseline (TF-IDF + exact match)", lambda q, p: baseline(q, p), lost, pool),
            metrics(f"proposed [{emb.backend}, {'rules' if args.no_llm else 'LLM if key set'}]",
                    lambda q, p: proposed(q, p, emb, not args.no_llm), lost, pool)]
    keys = ["Hit@1", "Hit@3", "P@3", "MRR", "Recall(any label)", "False-match rate"]
    print(f"\n{'system':58s}" + "".join(f"{k:>18s}" for k in keys))
    for r in rows:
        print(f"{r['system']:58s}" + "".join(f"{r[k]:18.3f}" for k in keys))
    print(f"\nmatchable queries: {rows[0]['n_matchable']}, no-match queries: {rows[0]['n_nomatch']}")
