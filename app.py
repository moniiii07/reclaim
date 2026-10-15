"""ReClaim - Streamlit front end for the AI Lost & Found Matching System.

All matching / storage / privacy logic lives in matcher.py (and is unit-tested there);
this file only handles layout and session state.

Privacy model
  * lost report  -> the owner sees candidate FOUND items; the finder's contact is revealed only after
                    the owner clicks "This is my item".
  * found report -> the finder never sees an owner's contact; likely owners are notified using the
                    contact THEY supplied, and the notification log shows masked contacts only.
"""
import os

import pandas as pd
import streamlit as st

import matcher as M

st.set_page_config(page_title="ReClaim - AI Lost & Found", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return M.Embedder()


def secret_to_env():
    """Let the app owner provide a shared LLM key / model name via Streamlit Cloud secrets."""
    try:
        for k in ("ANTHROPIC_API_KEY", "LLM_MODEL", "EMBED_BACKEND"):
            if k in st.secrets and not os.environ.get(k):
                os.environ[k] = st.secrets[k]
    except Exception:
        pass  # no secrets file locally - that's fine


secret_to_env()
M.seed_demo_data()          # first run only: demo reports from data/eval_*.csv

# Draw the page header BEFORE loading the (large) embedding model, so the app is never just blank.
st.title("🔎 AI Lost & Found Matching System")
st.caption("Semantic matching of lost and found reports. Results are *potential* matches - "
           "ownership must be confirmed by the owner or authorised staff.")
embedder = get_embedder()

ss = st.session_state
VIEW_VERSION = 2                     # bump when the shape of stored results changes
if ss.get("view") and ss["view"].get("v") != VIEW_VERSION:
    ss["view"] = None                # stale results from before an app update - discard, don't render
ss.setdefault("view", None)          # last submitted report + its match results
ss.setdefault("confirmed", set())    # (lost_id, found_id) pairs the owner confirmed -> contact revealed
ss.setdefault("notified", set())     # (lost_id, found_id) pairs the finder manually notified

ICON = {"Likely Match": "🟢", "Possible Match": "🟡"}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔎 ReClaim")
    st.caption("AI Lost & Found Matching System")
    typed_key = st.text_input(
        "Anthropic API key (optional)", type="password",
        help="Enables LLM verification of candidate matches. Kept only in your browser session.")
    # The key stays in this session and is passed to the matcher per call - it is NOT written to
    # os.environ, which would share it with every other visitor of the app.
    api_key = typed_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
    use_llm = st.toggle("LLM verification", value=bool(api_key), disabled=not api_key) and bool(api_key)
    st.caption(f"Embeddings: `{embedder.backend}`")
    st.caption(f"Verification: `{'LLM (' + M.LLM_MODEL + ')' if use_llm else 'rule-based fallback'}`")
    st.caption(f"Likely ≥ {M.LIKELY_T:.0%} · Possible ≥ {M.POSSIBLE_T:.0%}")
    count_slot = st.empty()

# ---------------------------------------------------------------- actions
def submit_report(kind: str, item: str, description: str, location: str, contact: str):
    """Save the report, run matching once, and keep the results in session state."""
    reports = M.load_reports()
    rtype = "lost" if kind == "I lost something" else "found"
    report = {"id": M.next_id(reports, "L" if rtype == "lost" else "F"), "type": rtype,
              "item": item, "description": description, "location": location, "contact": contact}
    M.save_report(report)
    kw = dict(embedder=embedder, use_llm=use_llm, api_key=api_key or None)

    with st.spinner("Searching for matches..."):
        if rtype == "lost":
            ss.view = {"v": VIEW_VERSION, "kind": "lost", "report": report, "matches": M.search_lost_item(report, **kw)}
        else:
            hits = []
            for lost in (r for r in reports if r.get("type") == "lost"):
                res = M.find_matches(lost, [report], **kw)
                if res:
                    hits.append({"lost": lost, "match": res[0]})
            hits.sort(key=lambda h: -h["match"]["confidence"])
            auto = sum(1 for h in hits if h["match"]["label"] == "Likely Match"
                       and M.record_notification(h["lost"], report, h["match"]["confidence"], auto=True))
            ss.view = {"v": VIEW_VERSION, "kind": "found", "report": report, "hits": hits, "auto": auto}


def breakdown(m: dict):
    v = m["verification"]
    with st.expander("Why this score?"):
        st.write(f"Semantic similarity: **{max(m['similarity'], 0):.0%}** | "
                 f"Attribute agreement: **{m['attribute_score']:.0%}**")
        st.info(f"Verifier ({v['source']}): **{v['verdict']}** - {v['reason']}")


def render_lost_view(view: dict):
    lost, matches = view["report"], view["matches"]
    likely = [m for m in matches if m["label"] == "Likely Match"]
    if likely:
        best = likely[0]
        st.success(f"✅ **Match found!** Best match: found item **#{best['report'].get('id', '?')}** "
                   f"({best['confidence']:.0%} confidence).")
    elif matches:
        st.warning(f"🟡 No confident match, but {len(matches)} possible match(es) - please review them below.")
    else:
        st.info("No potential matches yet. Your report is saved and will be checked against future found reports.")

    for m in matches:
        r = m["report"]
        ckey = (lost["id"], r["id"])
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**#{r.get('id', '?')}** · *{r.get('item') or 'item'}* - {r.get('description', '')}")
            c1.caption(f"📍 {r.get('location', '')}")
            c2.metric(f"{ICON[m['label']]} {m['label']}", f"{m['confidence']:.0%}")
            st.progress(float(min(max(m["confidence"], 0.0), 1.0)))
            breakdown(m)
            if ckey in ss.confirmed:
                st.success(f"Match confirmed. Finder's contact: {r.get('contact') or 'not provided'}")
            elif st.button("✅ This is my item", key=f"confirm_{lost['id']}_{r['id']}"):
                ss.confirmed.add(ckey)
                st.rerun()
            else:
                st.caption("🔒 The finder's contact is revealed after you confirm the match.")


def render_found_view(view: dict):
    found, hits = view["report"], view["hits"]
    if not hits:
        st.info("No matching lost report yet. Your report is saved and will be shown to owners who search later.")
    else:
        st.warning(f"This found item may match {len(hits)} lost report(s):")
    if view["auto"]:
        st.success(f"📩 Auto-notified {view['auto']} likely owner(s) using the contact they gave in their lost "
                   "report. You won't see their contact - they'll reach out if they confirm it's their item.")

    for h in hits:
        lost, m = h["lost"], h["match"]
        ckey = (lost["id"], found["id"])
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**Lost report #{lost.get('id', '?')}** · *{lost.get('item') or 'item'}* - {lost.get('description', '')}")
            c1.caption(f"📍 {lost.get('location', '')}")
            c2.metric(f"{ICON[m['label']]} {m['label']}", f"{m['confidence']:.0%}")
            st.progress(float(min(max(m["confidence"], 0.0), 1.0)))
            breakdown(m)
            if m["label"] == "Likely Match" or ckey in ss.notified:
                st.success(f"📩 Owner notified at {M.mask_contact(lost.get('contact'))}.")
            elif st.button("🔔 Notify this person anyway", key=f"notify_{lost['id']}_{found['id']}"):
                M.record_notification(lost, found, m["confidence"], auto=False)
                ss.notified.add(ckey)
                st.rerun()


# ---------------------------------------------------------------- tabs
tab_report, tab_browse = st.tabs(["➕ Submit a report", "📋 Browse reports"])

with tab_report:
    kind = st.radio("What are you reporting?", ["I lost something", "I found something"], horizontal=True)
    with st.form("report_form", clear_on_submit=False):
        item = st.text_input("Item (e.g. water bottle, wallet, phone)")
        description = st.text_area("Description (colour, brand, marks, contents...)")
        location = st.text_input("Location (where it was lost / found)")
        contact = st.text_input(
            "Your contact (email or phone)",
            help="Lost: we notify YOU here when a match turns up. Found: shown to the owner only after "
                 "they confirm the match. You never see anyone else's contact.")
        submitted = st.form_submit_button("Submit and find matches", type="primary")

    if submitted:
        if not (description.strip() and location.strip() and contact.strip()):
            st.error("Please fill in the description, location and contact.")
        else:
            submit_report(kind, item.strip(), description.strip(), location.strip(), contact.strip())
            st.success(f"Report #{ss.view['report']['id']} saved.")

    if ss.view:
        st.subheader("Potential matches")
        (render_lost_view if ss.view["kind"] == "lost" else render_found_view)(ss.view)

with tab_browse:
    data = M.load_reports()
    if not data:
        st.write("No reports yet.")
    else:
        st.dataframe(pd.DataFrame(data).drop(columns=["contact"], errors="ignore"), hide_index=True)  # contacts stay private
    log = M.load_notifications()
    if log:
        st.caption("Notifications sent to lost-item owners (contacts masked):")
        df_log = pd.DataFrame(log)
        df_log["owner_contact"] = df_log["owner_contact"].apply(M.mask_contact)
        st.dataframe(df_log, hide_index=True)

count_slot.caption(f"Stored reports: {len(M.load_reports())}")