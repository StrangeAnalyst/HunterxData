# Course Project — Part 1: S&P 500 news embeddings & technical analysis

Two self-contained Jupyter notebooks covering both sections of the assignment.

| File | Section | Topic |
|---|---|---|
| `Section_A_news_embeddings_clustering.ipynb` | A | Text embeddings on S&P 500 news → K-Means → silhouette → PCA → cluster interpretation |
| `Section_B_technical_analysis_backtesting.ipynb` | B | MA50/MA200 Golden & Death Crosses → 14-day filter → volatility → visualisation → event-study backtest |

## How to run

```bash
pip install -r requirements.txt
jupyter lab
```

Then **Run All** on each notebook, top to bottom. Both notebooks start with a `%pip install` cell, so
they also run as-is on Google Colab with no prior setup.

Expected runtime on a laptop with a normal connection:

* **Section A** — ~10–20 min. The `.news` endpoint is queried once per ticker (~500 HTTP calls with a
  0.15 s pause), then `all-MiniLM-L6-v2` is downloaded (~80 MB, first run only) and the embeddings are
  computed on CPU.
* **Section B** — ~3–6 min, dominated by the batched price download (9 batches of 60 tickers).

Both notebooks need outbound access to `en.wikipedia.org`, `query*.finance.yahoo.com` and — for
Section A — `huggingface.co`.

## What each notebook produces

**Section A**
1. S&P 500 constituents from Wikipedia, with `BRK.B → BRK-B` style symbol translation for Yahoo.
2. `news_dict`: raw `.news` payload per ticker, fetched defensively (per-ticker `try/except`, rate
   limiting, support for both the modern `content{…}` schema and the legacy flat schema).
3. `df_news`: cleaned DataFrame with `TICKER, TITLE, SUMMARY, PUBLICATION_DATE, URL, PROVIDER`.
4. `EMBEDDED_TEXT` (the title) and `EMBEDDINGS` (384-d vectors from `all-MiniLM-L6-v2`).
5. `df_news_unique`: one article per ticker — the matrix that gets clustered.
6. Silhouette scores for k = 2…6 plotted against the elbow curve, plus the best k.
7. K-Means with k = 3, cluster labels written back to the DataFrame (`CLUSTER`, `CLUSTER_BEST_K`,
   per-article `SILHOUETTE`).
8. PCA scatter plots (plain and headline-annotated), TF-IDF characteristic terms per cluster,
   representative headlines, and a cluster × GICS-sector cross-tab.
9. Answers to all ten reflection questions.

**Section B**
1. Closing prices for every constituent, downloaded in batches with retries and a data-quality report.
   A ~420-day warm-up window is downloaded *in addition* to the required 2024-05-01 → 2025-05-01
   analysis window, so that MA200 is already defined on the first day of the analysis window.
2. `df_close`, `df_ma50`, `df_ma200` over the required window.
3. `detect_crosses()` — vectorised detection of every MA50/MA200 crossover, returned as a tidy
   one-row-per-event table.
4. `df_golden_cross_14d` / `df_death_cross_14d`, enriched with annualised 1-year volatility,
   volatility around the cross date, company name and GICS sector.
5. Ten annotated charts per signal type: price + MA50 + MA200, the last 14 days shaded, the cross
   marked, and both volatility measures in the title.
6. An event study of every cross of the year: forward returns at 5/10/21/63 sessions vs. the
   unconditional baseline, hit rates, and Welch t-tests.
7. Answers to all reflection questions.

## Two answers to complete after running

Section B asks for the *names* of the companies that crossed in the last 14 days. Those depend on the
day the notebook is run, so the notebook prints both lists programmatically and the corresponding
markdown cells are marked with 📋 — paste the printed tickers there after your run.

## Notes on methodology

* **Symbol translation.** Wikipedia writes class shares with a dot; Yahoo expects a dash. Without the
  mapping, those tickers return empty and silently disappear from the universe.
* **No look-ahead.** Prices are forward-filled only (never back-filled), moving averages use
  `min_periods` equal to the window, and the backtest measures returns *after* the signal date.
* **`k = 1` is skipped** in the silhouette loop: the score is mathematically undefined for a single
  cluster (`scikit-learn` raises `ValueError` for `n_labels = 1`), so the loop runs k = 2…6.
* **Known limits, stated in the notebooks:** one year of data in a single market regime, survivorship
  bias from using today's index membership, no transaction costs, and truncated forward windows for the
  most recent signals.
