# ReClaim - AI Lost & Found Matching System

Pipeline: report -> attribute extraction (NER) -> Sentence-BERT embeddings -> cosine
similarity retrieval -> optional LLM verification -> hybrid confidence score -> Streamlit UI.

## Run (about 5 minutes)
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
python evaluate.py            # Precision@1 / Recall@3 on the sample data
```
Optional LLM verification: paste an Anthropic API key in the sidebar, or
`export ANTHROPIC_API_KEY=...` (Windows: `set ANTHROPIC_API_KEY=...`). Without a key the
system still works using embeddings + attribute scoring. If sentence-transformers cannot be
installed, it falls back to TF-IDF automatically.

## The scoring algorithm (your custom contribution)
    confidence = (0.40*semantic + 0.30*attribute + 0.30*LLM) / total weight used
    attribute  = 0.40*type + 0.25*color + 0.15*brand + 0.10*location + 0.10*time
    vetoes     = different type -> cap 0.20 | different brand -> cap 0.40 | conflicting color -> cap 0.50
Weights and thresholds live at the top of `matcher.py`. Tune them and report the effect.

## 60-minute plan
| Min | Task |
|-----|------|
| 0-10 | Install, run the app, run evaluate.py |
| 10-25 | Replace sample_reports.csv with ~30-50 of your own realistic pairs (include tricky look-alikes) |
| 25-40 | Tune weights/thresholds in matcher.py, re-run evaluate.py, note the numbers |
| 40-50 | Take screenshots: submit page, matches with breakdown, confirm-to-reveal-contact |
| 50-60 | Paste results into your report/PPT (Results slide + screenshots) |

## Files
- `matcher.py` - the engine (extraction, embeddings, scoring, LLM verify)
- `app.py` - Streamlit interface
- `evaluate.py` - metrics on labelled data (`match_id` column = ground truth)
- `data/sample_reports.csv` - 12 lost + 14 found sample reports
