import os
import shutil
from datetime import date

import pandas as pd
import streamlit as st

import notifier
from matcher import (LIKELY_T, POSSIBLE_T, Embedder, extract_attributes, find_matches,
                     get_index, llm_available, nlp_backend)

DATA = "data/reports.csv"
NOTIFY_LOG = "data/notifications.csv"

COLS = ["id", "type", "description", "location", "date", "contact", "match_id"]

os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA):
    if os.path.exists("data/sample_reports.csv"):
        shutil.copy("data/sample_reports.csv", DATA)
    else:
        pd.DataFrame(columns=COLS).to_csv(DATA, index=False)

st.set_page_config(page_title="ReClaim", page_icon="🔎", layout="wide")

# Streamlit Cloud: copy root-level Secrets into environment variables used by
# notifier.py. SMTP_USERNAME is the canonical name; SMTP_USER is retained only
# for older deployments that already used it.
try:
    for k in ("ANTHROPIC_API_KEY", "SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM"):
        if k in st.secrets and not os.getenv(k):
            os.environ[k] = str(st.secrets[k])
    if not os.getenv("SMTP_USERNAME") and os.getenv("SMTP_USER"):
        os.environ["SMTP_USERNAME"] = os.environ["SMTP_USER"]
except Exception:
    pass


def mask_contact(c: str) -> str:
    """Show just enough of a contact to prove it exists, without revealing it."""
    c = c.strip()
    if "@" in c:
        name, _, domain = c.partition("@")
        return f"{name[:2]}{'*' * max(len(name) - 2, 2)}@{domain}"
    digits = "".join(ch for ch in c if ch.isdigit())
    return f"{'*' * max(len(digits) - 2, 2)}{digits[-2:]}" if digits else "****"


def already_notified(lost_id: str, found_id: str) -> bool:
    """Suppress duplicates only after SMTP accepted the email.

    A simulated or failed notification must be retried after the SMTP secrets
    are corrected, so it must not block later deliveries.
    """
    if not os.path.exists(NOTIFY_LOG):
        return False
    try:
        log = pd.read_csv(NOTIFY_LOG, dtype=str, keep_default_na=False)
        required = {"lost_id", "found_id", "status"}
        if not required.issubset(log.columns):
            return False
        sent = log["status"].str.strip().str.lower().eq("sent")
        same_match = log["lost_id"].eq(str(lost_id)) & log["found_id"].eq(str(found_id))
        return bool((same_match & sent).any())
    except (OSError, pd.errors.EmptyDataError):
        return False


def notify_owner(lost_row: dict, found_row: dict, confidence: float, auto: bool) -> str:
    """E-mail the person who reported the item lost, using the contact THEY provided
    (the finder never sees or enters it). Returns the delivery status."""
    subject, body = notifier.build_message(found_row, confidence)
    status, detail = notifier.send_email(lost_row["contact"], subject, body)
    entry = pd.DataFrame([{
        "lost_id": lost_row["id"], "found_id": found_row["id"],
        "notified_contact": lost_row["contact"], "confidence": round(confidence, 2),
        "auto": auto, "status": status, "detail": detail,
        "sent_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }])
    entry.to_csv(NOTIFY_LOG, mode="a", header=not os.path.exists(NOTIFY_LOG), index=False)
    return status


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return Embedder()


def load() -> pd.DataFrame:
    """Read the reports CSV and make sure every expected column exists."""
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    # give rows without an id a unique one (L1, F2, ...)
    for i in df.index[df["id"].str.strip() == ""]:
        prefix = "L" if str(df.at[i, "type"]).lower().startswith("l") else "F"
        df.at[i, "id"] = f"{prefix}{i + 1}"
    return df


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
    pd.concat([df, pd.DataFrame([row])], ignore_index=True)[COLS].to_csv(DATA, index=False)
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
    use_llm = st.toggle("LLM verification", value=llm_available(), disabled=not llm_available())
    top_k = st.slider("Matches to show", 1, 10, 5)

    with st.expander("📧 E-mail notification (SMTP)"):
        st.caption("Needed to send real e-mails to lost-item owners. Without it, "
                   "notifications are only logged.")
        for env, label, pw in (("SMTP_HOST", "SMTP host (e.g. smtp.gmail.com)", False),
                               ("SMTP_PORT", "Port (587 or 465)", False),
                               ("SMTP_USERNAME", "SMTP user", False),
                               ("SMTP_PASSWORD", "SMTP password / app password", True),
                               ("SMTP_FROM", "From address (optional)", False)):
            val = st.text_input(label, value=os.getenv(env, ""), type="password" if pw else "default")
            if val:
                os.environ[env] = val
        st.caption(f"Status: **{'configured' if notifier.smtp_configured() else 'not configured'}**")

    st.caption(f"NLP: **{nlp_backend()}** | Embeddings: **{embedder.kind}**")
    st.caption(f"LLM: **{'on' if use_llm else 'off'}**")
    st.caption(f"Likely ≥ {LIKELY_T:.0%} · Possible ≥ {POSSIBLE_T:.0%}")

st.info("ℹ️ Scores indicate a **potential match only**. They do not confirm ownership or prove that two "
        "reports describe the same object. Final confirmation rests with the users or administrators.")

STATUS_TEXT = {"sent": "e-mailed", "simulated": "logged (e-mail not configured)",
               "no_email": "logged (no e-mail address on the lost report)", "failed": "logged (e-mail failed)"}


def show_matches(query: dict, scope: str = "t"):
    """query['type'] is the CURRENT poster's perspective.

    - 'lost': they're browsing found items for their own lost item -> they may confirm
      a match themselves, which reveals the finder's contact.
    - 'found': they're a finder whose report is matched against lost reports -> they never
      see the owner's contact. The system notifies that owner instead, using the e-mail
      THAT owner supplied in their lost report."""
    df = load()
    perspective = query["type"]
    other = "found" if perspective == "lost" else "lost"
    pool = df[df.type == other]

    with st.spinner("Matching..."):
        results = find_matches(query, pool, embedder, use_llm=use_llm, top_k=top_k)

    icons = {"Likely": "🟢", "Possible": "🟡", "Unlikely": "🔴"}
    likely = [r for r in results if r["label"] == "Likely"]
    possible = [r for r in results if r["label"] == "Possible"]

    # What the system understood (text processing + attribute extraction stages)
    attrs = extract_attributes(query["description"])
    with st.expander("🧠 What the system understood from your description"):
        st.write(f"**Item type:** {attrs['type'] or '—'} · **Colours:** {', '.join(sorted(attrs['colors'])) or '—'} · "
                 f"**Brand:** {attrs['brand'] or '—'}")
        if len(pool):
            idx = get_index(embedder, pool["description"].tolist())
            st.caption(f"Compared against {len(pool)} stored {other} report(s) using a "
                       f"{idx.backend.upper()} vector index.")

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

    results = [r for r in results if r["label"] != "Unlikely"]
    if results:
        st.subheader(f"Potential matches ({len(results)})")

    if perspective == "found" and results:
        qid = query.get("id", "new")
        statuses = []
        for r in results:
            if r["label"] != "Likely":
                continue
            ckey = (qid, r["row"]["id"])
            if ckey in st.session_state.confirmed or already_notified(r["row"]["id"], qid):
                continue
            statuses.append(notify_owner(r["row"], query, r["confidence"], auto=True))
            st.session_state.confirmed.add(ckey)
        if statuses:
            summary = ", ".join(f"{statuses.count(s)} {STATUS_TEXT[s]}" for s in dict.fromkeys(statuses))
            st.success(f"📩 Likely owner(s) notified using the contact they gave in their lost report: {summary}. "
                       "You won't see their contact - they'll reach out if they confirm it's their item.")

    for r in results:
        row = r["row"]
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{row['id']}** · {row['description']}")
            c1.caption(f"📍 {row['location']} · 📅 {row['date']}")
            c2.metric(f"{icons[r['label']]} {r['label']}", f"{r['confidence']:.0%}")
            st.progress(float(min(max(float(r["confidence"]), 0.0), 1.0)))
            with st.expander("Why this score?"):
                st.write(f"Semantic similarity: **{r['semantic']:.0%}** | "
                         f"Attribute match: **{r['attribute']:.0%}** | "
                         f"Location: **{r['location']:.0%}** | Time: **{r['time']:.0%}**"
                         + (f" | LLM: **{r['llm']:.0%}**" if r["llm"] is not None else ""))
                if r["llm_reason"]:
                    st.info(f"LLM: {r['llm_reason']}")
                if r["llm_similarities"]:
                    st.markdown("**Similarities:** " + "; ".join(r["llm_similarities"]))
                if r["llm_differences"]:
                    st.markdown("**Differences:** " + "; ".join(r["llm_differences"]))
                if r["veto"]:
                    st.warning(f"Score capped: {r['veto']}")

            ckey = (query.get("id", "new"), row["id"])
            if perspective == "lost":
                if ckey in st.session_state.confirmed:
                    st.success(f"Match confirmed. Finder's contact: {row['contact']}")
                elif st.button("✅ This is my item", key=f"c_{scope}_{ckey}"):
                    st.session_state.confirmed.add(ckey)
                    st.rerun()
                else:
                    st.caption("🔒 The finder's contact is revealed after you confirm the match.")
            else:
                if ckey in st.session_state.confirmed or already_notified(row["id"], ckey[0]):
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
        desc = st.text_area("Describe the item",
                            placeholder="e.g. Black water bottle with a sticker near the library")
        c1, c2 = st.columns(2)
        loc = c1.text_input("Where?", placeholder="e.g. Reading Hall")
        d = c2.date_input("When?", value=date.today())
        contact = st.text_input(
            "Your contact (e-mail preferred)",
            help="If lost: we e-mail YOU here when a match turns up. "
                 "If found: this is shown to the owner only after they confirm the match - "
                 "you never enter or see someone else's contact.")
        ok = st.form_submit_button("Submit and find matches", type="primary")
    if ok:
        if not (desc.strip() and loc.strip() and contact.strip()):
            st.error("Please fill in description, location and contact.")
        else:
            rid = add_report(rtype, desc.strip(), loc.strip(), d, contact.strip())
            st.success(f"Report {rid} saved.")
            if rtype == "lost" and not notifier.valid_email(contact):
                st.warning("This contact is not an e-mail address, so e-mail alerts can't be sent to you.")
            st.session_state.q1 = {"id": rid, "type": rtype, "description": desc.strip(),
                                   "location": loc.strip(), "date": str(d)}
    if st.session_state.get("q1"):
        show_matches(st.session_state.q1, scope="t1")

with tab2:
    df = load()
    if df.empty:
        st.info("No reports yet. Submit one in the first tab.")
    else:
        pick = st.selectbox("Choose a report", df.id.tolist(),
                            format_func=lambda i: f"{i} - {df.set_index('id').loc[i, 'description'][:70]}")
        if st.button("Find matches", key="find"):
            st.session_state.q2 = df.set_index("id").loc[pick].to_dict() | {"id": pick}
        if st.session_state.get("q2"):
            show_matches(st.session_state.q2, scope="t2")

with tab3:
    st.dataframe(load().drop(columns=["contact", "match_id"], errors="ignore"), hide_index=True)
    if os.path.exists(NOTIFY_LOG):
        st.caption("Notifications to lost-item owners (contacts masked):")
        log = pd.read_csv(NOTIFY_LOG)
        log["notified_contact"] = log["notified_contact"].apply(mask_contact)
        st.dataframe(log, hide_index=True)
