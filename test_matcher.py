import json, types
import matcher as M

EMB = M.Embedder(force_fallback=True)

def L(item, desc, loc): return {"id": "L", "type": "lost", "item": item, "description": desc, "location": loc}
def F(i, item, desc, loc): return {"id": str(i), "type": "found", "item": item, "description": desc, "location": loc}

def test_extract_attributes_paper_example():
    a = M.extract_attributes("water bottle", "black water bottle with a sticker", "near the library")
    assert a.item == "bottle" and "dark" in a.colours and "sticker" in a.marks and a.location_tokens == {"library"}

def test_dark_equals_black_and_label_equals_sticker():
    a = M.extract_attributes("", "black bottle with a sticker", "library")
    b = M.extract_attributes("", "dark coloured bottle with a label", "outside the library")
    score, conflicts, _ = M.attribute_score(a, b)
    assert not conflicts and score > 0.9

def test_paper_example_is_matched():
    found = [F(1, "bottle", "dark coloured bottle with a label", "outside the library"),
             F(2, "bottle", "red bottle with a sticker", "library"),
             F(3, "wallet", "black wallet", "canteen")]
    res = M.find_matches(L("water bottle", "black water bottle with a sticker", "near the library"),
                         found, embedder=EMB, use_llm=False)
    assert res and res[0]["report"]["id"] == "1"
    assert all(r["report"]["id"] != "3" for r in res)

def test_colour_conflict_is_rejected():
    found = [F(1, "bottle", "red bottle with a sticker", "library")]
    assert M.find_matches(L("bottle", "black bottle with a sticker", "library"), found, embedder=EMB, use_llm=False) == []

def test_brand_conflict_is_rejected():
    found = [F(1, "phone", "black oneplus phone", "main block")]
    assert M.find_matches(L("phone", "black samsung phone", "main block"), found, embedder=EMB, use_llm=False) == []

def test_unrelated_returns_nothing():
    found = [F(1, "umbrella", "green umbrella", "gate")]
    assert M.find_matches(L("laptop", "silver laptop", "lab"), found, embedder=EMB, use_llm=False) == []

def test_empty_pool():
    assert M.find_matches(L("a", "b", "c"), [], embedder=EMB, use_llm=False) == []

def test_llm_verdict_is_used_and_can_veto(monkeypatch):
    class Block: type = "text"; text = "```json\n" + json.dumps({"verdict": "mismatch", "confidence": 0.9, "reason": "different owners' marks"}) + "\n```"
    class Client:
        def __init__(self, **k): self.messages = types.SimpleNamespace(create=lambda **kw: types.SimpleNamespace(content=[Block()]))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setitem(__import__("sys").modules, "anthropic", types.SimpleNamespace(Anthropic=Client))
    found = [F(1, "bottle", "dark coloured bottle with a label", "outside the library")]
    res = M.find_matches(L("bottle", "black bottle with a sticker", "library"), found, embedder=EMB, use_llm=True)
    assert res == []          # LLM vetoed an otherwise strong candidate

def test_llm_failure_falls_back_to_rules(monkeypatch):
    class Client:
        def __init__(self, **k): raise RuntimeError("network down")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setitem(__import__("sys").modules, "anthropic", types.SimpleNamespace(Anthropic=Client))
    found = [F(1, "bottle", "dark coloured bottle with a label", "outside the library")]
    res = M.find_matches(L("bottle", "black bottle with a sticker", "library"), found, embedder=EMB, use_llm=True)
    assert res and res[0]["verification"]["source"] == "rules"

def test_storage_roundtrip(tmp_path):
    p = str(tmp_path / "r.csv")
    M.save_report({"id": "1", "type": "found", "item": "bottle", "description": "x", "location": "y", "contact": "c"}, p)
    assert M.load_reports(p)[0]["item"] == "bottle" and M.next_id(M.load_reports(p)) == "2"


# ---------------------------------------------------------------- additions for the ReClaim port
def test_next_id_handles_prefixed_ids():
    rows = [{"id": "L1"}, {"id": "F7"}, {"id": "junk"}]
    assert M.next_id(rows, "L") == "L8" and M.next_id([], "F") == "F1"


def test_mask_contact():
    assert M.mask_contact("alice@x.edu") == "al***@x.edu"
    assert M.mask_contact("98765 43210").endswith("10") and "9876" not in M.mask_contact("98765 43210")
    assert M.mask_contact("") == "****"


def test_notification_is_logged_once(tmp_path):
    p = str(tmp_path / "n.csv")
    lost = {"id": "L1", "contact": "owner@x.edu"}
    assert M.record_notification(lost, {"id": "F1"}, 0.9, path=p) is True
    assert M.record_notification(lost, {"id": "F1"}, 0.9, path=p) is False      # duplicate pair ignored
    assert M.record_notification(lost, {"id": "F2"}, 0.5, auto=False, path=p) is True
    rows = M.load_notifications(p)
    assert len(rows) == 2 and rows[0]["owner_contact"] == "owner@x.edu" and rows[1]["auto"] == "False"


def test_seed_demo_data_runs_once_and_drops_ground_truth(tmp_path):
    p = str(tmp_path / "reports.csv")
    n = M.seed_demo_data(p, source_dir="data")
    rows = M.load_reports(p)
    assert n == len(rows) > 0 and {r["type"] for r in rows} == {"lost", "found"}
    assert "true_match" not in rows[0]
    assert M.seed_demo_data(p, source_dir="data") == 0                            # never overwrites


def test_api_key_is_passed_per_call_not_via_environment(monkeypatch):
    seen = {}
    class Block: type = "text"; text = json.dumps({"verdict": "match", "confidence": 0.95, "reason": "same"})
    class Client:
        def __init__(self, **k): seen.update(k); self.messages = types.SimpleNamespace(create=lambda **kw: types.SimpleNamespace(content=[Block()]))
    monkeypatch.setitem(__import__("sys").modules, "anthropic", types.SimpleNamespace(Anthropic=Client))
    found = [F(1, "bottle", "dark coloured bottle with a label", "outside the library")]
    res = M.find_matches(L("bottle", "black bottle with a sticker", "library"), found,
                         embedder=EMB, use_llm=True, api_key="session-key")
    assert seen.get("api_key") == "session-key" and res[0]["verification"]["source"] == "llm"
