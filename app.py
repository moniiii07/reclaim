import os
import shutil
from datetime import date

import pandas as pd
import streamlit as st

from matcher import Embedder, find_matches, LIKELY_T, POSSIBLE_T

DATA = "data/reports.csv"
NOTIFY_LOG = "data/notifications.csv"
if not os.path.exists(DATA):
    shutil.copy("data/sample_reports.csv", DATA)


def mask_contact(c: str) -> str:
    """Show just enough of a contact to prove it exists, without revealing it."""
    c = c.strip()
    if "@" in c:
        name, _, domain = c.partition("@")
        return f"{name[:2]}{'*' * max(len(name) - 2, 2)}@{domain}"
    digits = "".join(ch for ch in c if ch.isdigit())
    return f"{'*' * max(len(digits) - 2, 2)}{digits[-2:]}" if digits else "****"


def notify_owner(lost_row: dict, found_row: dict, confidence: float, auto: bool):
    """Simulate sending a notification to the person who reported the item lost,
    using the contact THEY provided — the finder never sees or enters it."""
    os.makedirs("data", exist_ok=True)
    entry = pd.DataFrame([{
        "lost_id": lost_row["id"], "found_id": found_row["id"],
        "notified_contact": lost_row["contact"], "confidence": round(confidence, 2),
        "auto": auto, "sent_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }])
    header = not os.path.exists(NOTIFY_LOG)
    entry.to_csv(NOTIFY_LOG, mode="a", header=header, index=False)


st.set_page_config(page_title="ReClaim", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return Embedder()


def load() -> pd.DataFrame:
    return pd.read_csv(DATA, dtype=str, keep_default_na=False)


def add_report(rtype, desc, loc, d, contact) -> str:
    df = load()
    prefix = "L" if rtype == "lost" else "F"
    n = sum(df.id.str.startswith(prefix)) + 1
    rid = f"{prefix}{n}"
    while rid in set(df.id):
        n += 1
        rid = f"{prefix}{n}"
    row = {"id": rid, "type": rtype, "description": desc, "location": loc,
           "date": str(d), "contact": contact, "match_id": ""}
    pd.concat([df, pd.DataFrame([row])]).to_csv(DATA, index=False)
    return rid


st.session_state.setdefault("confirmed", set())
embedder = get_embedder()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🔎 ReClaim")
    st.caption("AI Lost & Found Matching System")
    key = st.text_input("Anthropic API key (optional)", type="password",
                        help="Enables LLM verification of top candidates.")
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
    use_llm = st.toggle("LLM verification", value=bool(os.getenv("ANTHROPIC_API_KEY")),
                        disabled=not os.getenv("ANTHROPIC_API_KEY"))
    top_k = st.slider("Matches to show", 1, 10, 5)
    st.caption(f"Embeddings: **{embedder.kind}**  |  LLM: **{'on' if use_llm else 'off'}**")
    st.caption(f"Likely ≥ {LIKELY_T:.0%} · Possible ≥ {POSSIBLE_T:.0%}")


def show_matches(query: dict, scope: str = "t"):
    """query['type'] is the CURRENT poster's perspective.
    - 'lost': they're browsing found items for their own lost item -> they may confirm
      a match themselves, which reveals the finder's contact (the finder posted it, so
      it's theirs to share).
    - 'found': they're a finder whose found report is matched against lost reports ->
      they never see the lost owner's contact. The system notifies that owner instead,
      using the contact THAT owner supplied when they filed the lost report."""
    df = load()
    perspective = query["type"]
    other = "found" if perspective == "lost" else "lost"
    pool = df[df.type == other]
    with st.spinner("Matching..."):
        results = find_matches(query, pool, embedder, use_llm=use_llm, top_k=top_k)
    icons = {"Likely": "🟢", "Possible": "🟡", "Unlikely": "🔴"}

    likely = [r for r in results if r["label"] == "Likely"]
    possible = [r for r in results if r["label"] == "Possible"]

    # ---- Clear yes/no verdict at the top ----
    if likely:
        best = max(likely, key=lambda r: r["confidence"])
        extra = f" + {len(likely) - 1} more likely match(es)" if len(likely) > 1 else ""
        st.success(f"✅ **Match found!** Best match: **{best['row']['id']}** "
                   f"({best['confidence']:.0%} confidence){extra}.")
    elif possible:
        st.warning(f"🟡 **No confident match, but {len(possible)} possible match(es) found.** "
                   "Please review them below.")
    elif perspective == "lost":
        st.error("❌ **No matching item found yet.** Your report is saved. "
                 "Check back later, as new found reports are matched against it.")
    else:
        st.error("❌ **No matching lost report found yet.** Your report is saved and "
                 "will be included when owners search for their lost items.")

    # Only list real candidates; hide "Unlikely" noise
    results = [r for r in results if r["label"] != "Unlikely"]
    if results:
        st.subheader(f"Potential matches ({len(results)})")

    if perspective == "found" and results:
        auto_notified = [r for r in results if r["label"] == "Likely"
                         and (query.get("id", "new"), r["row"]["id"]) not in st.session_state.confirmed]
        for r in auto_notified:
            notify_owner(r["row"], query, r["confidence"], auto=True)
            st.session_state.confirmed.add((query.get("id", "new"), r["row"]["id"]))
        if auto_notified:
            st.success(f"📩 Auto-notified {len(auto_notified)} likely owner(s) using the "
                       "contact they provided in their lost report. You won't see their "
                       "contact — they'll reach out to you if they confirm it's their item.")

    for r in results:
        row = r["row"]
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{row['id']}** · {row['description']}")
            c1.caption(f"📍 {row['location']}  ·  📅 {row['date']}")
            c2.metric(f"{icons[r['label']]} {r['label']}", f"{r['confidence']:.0%}")
            st.progress(float(min(max(float(r["confidence"]), 0.0), 1.0)))
            with st.expander("Why this score?"):
                st.write(f"Semantic similarity: **{r['semantic']:.0%}**  |  "
                         f"Attribute match: **{r['attribute']:.0%}**  |  "
                         f"Location: **{r['location']:.0%}**  |  Time: **{r['time']:.0%}**"
                         + (f"  |  LLM: **{r['llm']:.0%}**" if r["llm"] is not None else ""))
                if r["llm_reason"]:
                    st.info(f"LLM: {r['llm_reason']}")
                if r["veto"]:
                    st.warning(f"Score capped: {r['veto']}")

            ckey = (query.get("id", "new"), row["id"])
            if perspective == "lost":
                # row is a FOUND report; its contact belongs to the finder, safe to reveal on confirm.
                if ckey in st.session_state.confirmed:
                    st.success(f"Match confirmed. Finder's contact: {row['contact']}")
                elif st.button("✅ This is my item", key=f"c_{scope}_{ckey}"):
                    st.session_state.confirmed.add(ckey)
                    st.rerun()
                else:
                    st.caption("🔒 The finder's contact is revealed after you confirm the match.")
            else:
                # row is a LOST report; its contact belongs to the owner — never shown to the finder.
                if ckey in st.session_state.confirmed:
                    st.success(f"📩 Owner notified at {mask_contact(row['contact'])}. "
                               "They'll contact you if they confirm it's their item.")
                elif r["label"] != "Likely":
                    if st.button("🔔 Notify this person anyway", key=f"c_{scope}_{ckey}"):
                        notify_owner(row, query, r["confidence"], auto=False)
                        st.session_state.confirmed.add(ckey)
                        st.rerun()


tab1, tab2, tab3 = st.tabs(["➕ Submit a report", "🔍 Search existing report", "📋 All reports"])

with tab1:
    with st.form("report"):
        rtype = st.radio("I have...", ["lost", "found"], horizontal=True,
                         format_func=lambda x: "Lost an item" if x == "lost" else "Found an item")
        desc = st.text_area("Describe the item", placeholder="e.g. Navy rucksack with Nike logo, zipper damaged")
        c1, c2 = st.columns(2)
        loc = c1.text_input("Where?", placeholder="e.g. Reading Hall")
        d = c2.date_input("When?", value=date.today())
        contact = st.text_input(
            "Your contact (email / phone)",
            help="If lost: we notify YOU here when a match turns up. "
                 "If found: this is shown to the owner only after they confirm the match — "
                 "you never enter or see someone else's contact.")
        ok = st.form_submit_button("Submit and find matches", type="primary")
    if ok:
        if not (desc.strip() and loc.strip() and contact.strip()):
            st.error("Please fill in description, location and contact.")
        else:
            rid = add_report(rtype, desc.strip(), loc.strip(), d, contact.strip())
            st.success(f"Report {rid} saved.")
            st.session_state.q1 = {"id": rid, "type": rtype, "description": desc.strip(),
                                   "location": loc.strip(), "date": str(d)}
    if st.session_state.get("q1"):
        show_matches(st.session_state.q1, scope="t1")

with tab2:
    df = load()
    pick = st.selectbox("Choose a report", df.id.tolist(),
                        format_func=lambda i: f"{i} - {df.set_index('id').loc[i, 'description'][:70]}")
    if st.button("Find matches", key="find"):
        st.session_state.q2 = df.set_index("id").loc[pick].to_dict() | {"id": pick}
    if st.session_state.get("q2"):
        show_matches(st.session_state.q2, scope="t2")

with tab3:
    st.dataframe(load().drop(columns=["contact", "match_id"]), hide_index=True)
    if os.path.exists(NOTIFY_LOG):
        st.caption("Notifications sent to lost-item owners (contact numbers withheld here too):")
        log = pd.read_csv(NOTIFY_LOG)
        log["notified_contact"] = log["notified_contact"].apply(mask_contact)
        st.dataframe(log, hide_index=True)