# ReClaim - AI Lost & Found Matching System

Pipeline: report -> text preprocessing -> attribute extraction (spaCy) -> Sentence-BERT embeddings
-> FAISS vector similarity search -> optional LLM verification -> hybrid confidence score
-> Streamlit UI -> e-mail notification of the lost-item owner.

## Run
```
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
python evaluate.py            # Precision@1 / Recall@3 on data/sample_reports.csv
```

## Optional settings (environment variables, Streamlit secrets, or the sidebar)
- `ANTHROPIC_API_KEY` - enables LLM verification (similarities / differences / compatibility).
- `SMTP_HOST`, `SMTP_PORT` (587 or 465), `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` - send real e-mails.
  Gmail: use smtp.gmail.com, port 587 and an App Password. Without SMTP the notification is only logged.

Fallbacks keep the app running: spaCy missing -> regex lexicons; sentence-transformers missing -> TF-IDF;
faiss missing -> NumPy search.

## Scoring
```
confidence = (0.40*semantic + 0.30*attribute + 0.30*LLM) / total weight used
attribute  = 0.35*type + 0.20*color + 0.15*brand + 0.10*marks + 0.10*location + 0.10*time
vetoes     = different type -> cap 0.20 | different brand -> cap 0.40 | conflicting color -> cap 0.50
```
"dark" / "light" are treated as compatible with the matching real colours. Weights live at the top of `matcher.py`.

## Files
- `matcher.py` - engine (preprocessing, spaCy extraction, embeddings, FAISS index, scoring, LLM verify)
- `notifier.py` - SMTP e-mail to the lost-item owner
- `app.py` - Streamlit interface
- `evaluate.py` - metrics on labelled data (`match_id` column = ground truth)
- `data/sample_reports.csv` - your existing sample data (unchanged)

Scores are a potential match only; final confirmation rests with the users or administrators.