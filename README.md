# Predicting Next‑Match Injury Risk in Professional Soccer (Public Data)

# utilized help from LLMs and Machine Learning Models 

> DS 340W • Applied Data Science • Parent‑paper data source: Transfermarkt injury histories; supplemental public sources (FBref, Understat, StatsBomb Open Data)

---

## TL;DR

This repo implements an end‑to‑end, **leakage‑safe** pipeline to predict **time‑loss injury risk in the next 7–10 days** for players in Europe’s top clubs using **public data**. It reproduces the data source described in the parent paper (Transfermarkt injury pages) and adds workload/context features from FBref and Understat. Baselines: regularized logistic; advanced: XGBoost/Random Forest; optional: Cox/discrete‑time survival. Models are **calibrated** and **interpreted** with SHAP.

---

## Data Sources (first‑hand access)

* **Injury labels (primary):** Transfermarkt player “Injury history” pages (diagnosis, from, until, days, games missed). Start page: [https://www.transfermarkt.com/](https://www.transfermarkt.com/)
  *We extract directly, season by season, for the specified clubs.*
* **Workload proxies (minutes, positions, starts):** FBref match/player logs. [https://fbref.com/en/](https://fbref.com/en/)
* **Match context (xG, shots, opponent strength):** Understat. [https://understat.com/](https://understat.com/)
* **Event data (optional leagues):** StatsBomb Open Data (free). [https://github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)
* **Mirrors for quick prototyping (optional):** Kaggle datasets that replicate Transfermarkt injuries. [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)

> **ToS & ethics:** Use respectful rate limiting, cache requests, and comply with each site’s Terms of Use. This project uses publicly viewable pages and does **not** attempt to access private data.

---

## Repo Structure

```
injury-risk-soccer/
├─ README.md                      # this file
├─ LICENSE                        # MIT (suggested)
├─ .gitignore                     # e.g., data/, .env, .Rproj.user, .venv
├─ config/
│  ├─ clubs.yml                   # list of clubs + Transfermarkt URL templates
│  ├─ params.yml                  # feature horizons, CV params, thresholds
│  └─ secrets-template.env        # example of env keys (no secrets committed)
├─ data/
│  ├─ raw/
│  │  ├─ transfermarkt/           # per-club raw injury HTML/CSV dumps
│  │  ├─ fbref/                   # raw match & player logs
│  │  └─ understat/               # raw xG/context
│  ├─ interim/                    # merged but not yet featured
│  ├─ processed/                  # feature tables used for modeling
│  └─ external/                   # optional Kaggle mirrors
├─ notebooks/
│  ├─ 00_eda_data_quality.ipynb
│  ├─ 01_feature_engineering.ipynb
│  ├─ 02_model_baselines_logit.ipynb
│  ├─ 03_model_xgboost_rf_shap.ipynb
│  ├─ 04_calibration_curves.ipynb
│  └─ 05_survival_time_to_injury.ipynb
├─ R/
│  ├─ scrape_transfermarkt.R      # builds injury table per club/season
│  ├─ fetch_fbref.R               # pulls minutes/starts/positions
│  ├─ merge_build_features.R      # creates rolling loads, congestion flags
│  └─ renv.lock                   # optional: pinned R packages
├─ src/
│  ├─ py/
│  │  ├─ build_dataset.py         # merges raw -> processed feature table
│  │  ├─ tscv.py                  # time‑series + group CV utilities
│  │  ├─ train_logit.py
│  │  ├─ train_xgb.py
│  │  ├─ calibrate.py             # isotonic/Platt, reliability diagrams
│  │  ├─ eval_metrics.py          # AUROC/AUPRC/F1/Brier/PRC plots
│  │  └─ shap_report.py           # global/local SHAP plots
│  └─ rutils/
│     └─ helpers.R                # small shared R utilities
├─ scripts/
│  ├─ 00_pull_injuries.sh         # orchestrates R scrape
│  ├─ 01_pull_fbref.sh
│  ├─ 02_features.sh
│  ├─ 03_train_baselines.sh
│  ├─ 04_train_xgb.sh
│  └─ 05_make_figures.sh
├─ reports/
│  ├─ figures/                    # SHAP, calibration, PR curves, incidence
│  └─ paper/
│     ├─ final_paper.docx (or .qmd/.tex)
│     └─ slides.pdf               # final 15‑min presentation
└─ tests/
   └─ test_data_checks.py         # schema, leakage, splits
```

---

## Quickstart

### Option A — Python

```bash
# create env
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib pyyaml joblib requests beautifulsoup4 tqdm fastparquet pyarrow

# build processed features (after raw data exists)
python src/py/build_dataset.py --config config/params.yml

# train & evaluate
python src/py/train_logit.py   --config config/params.yml
python src/py/train_xgb.py     --config config/params.yml
python src/py/calibrate.py     --config config/params.yml
python src/py/shap_report.py   --config config/params.yml
```

---

## Parent Paper (data source declared)

* **Hoenig, T. et al. (2022)** *Analysis of more than 20,000 injuries in European professional football by using a citizen science‑based approach* — Data sourced from **Transfermarkt** injury pages (clearly specified). Use this as methodological precedent for public media‑sourced injury labels.

> For DS 340W policy: This choice ensures **first‑hand access** to the same source the paper used, enabling full implementation.

---

## Feature Set (public‑data friendly)

* **Workload:** rolling minutes (1/3/5 matches), %90s completed, days since last match, congestion flag (≤3–4 rest days)
* **History:** prior time‑loss injuries, days since last injury, counts by diagnosis
* **Profile:** age, height (if available), position/role
* **Context:** home/away, opponent strength, xG for/against (Understat), team tempo proxies (shots/pressures)
* **Label:** injury occurs within next 10 days (binary) or weekly time‑to‑injury (survival)

---

## Modeling Plan

1. **Splits:** temporal train→val→test (earlier seasons → later), grouped by player to avoid leakage
2. **Baselines:** class‑weighted logistic (L1/L2), report AUROC, AUPRC, Brier, calibration curve
3. **Boosted trees:** XGBoost/LightGBM with time‑series CV and probability calibration
4. **Interpretability:** SHAP global/local; partial‑dependence for rest days & rolling minutes
5. **Decision value:** threshold selection via PR curves & decision curves (cost of rest vs. cost of injury)
6. **(Optional)** Cox / discrete‑time survival for time‑to‑injury

---

## Repro Steps (scripted)

```bash
bash scripts/00_pull_injuries.sh  # R scraper -> data/raw/transfermarkt
bash scripts/01_pull_fbref.sh     # R scraper -> data/raw/fbref
bash scripts/02_features.sh       # R -> processed features
bash scripts/03_train_baselines.sh
bash scripts/04_train_xgb.sh
bash scripts/05_make_figures.sh
```

---

## DS 340W Deliverables Map

* **Week 1**: Topic + parent paper + data‑access links (this README)
* **Check‑in #2**: Raw pulls complete; first merged table
* **Check‑in #3**: Baseline logistic + early calibration plot
* **Check‑in #4**: XGBoost + SHAP + decision curves
* **Midterm**: Proposal deck (design, features, leakage controls)
* **Final**: Paper + slides + repo (figures in `/reports/figures`)

---

## Contributing & Project Hygiene

* Open issues for **data quality** (missing dates/diagnoses), **feature drift**, and **calibration**.
* Commit only code/notebooks; **never commit raw data** (kept local by default). Use DVC or a cloud bucket if needed.
* Add unit tests for: split integrity (no future leakage), label windowing, schema checks.

---

## License

MIT for code. Data remain property of their respective owners; follow each source’s ToS.

---

## Appendix: R Scrape Skeleton

```r
# R/scrape_transfermarkt.R
library(worldfootballR); library(dplyr); library(purrr); library(lubridate); library(readr)
seasons <- 2022:2024
clubs <- list(
  "Arsenal" = "https://www.transfermarkt.com/arsenal-fc/startseite/verein/11/saison_id/%d",
  "Manchester City" = "https://www.transfermarkt.com/manchester-city/startseite/verein/281/saison_id/%d",
  "Liverpool" = "https://www.transfermarkt.com/liverpool-fc/startseite/verein/31/saison_id/%d",
  "Chelsea" = "https://www.transfermarkt.com/chelsea-fc/startseite/verein/631/saison_id/%d",
  "Manchester United" = "https://www.transfermarkt.com/manchester-united/startseite/verein/985/saison_id/%d",
  "Tottenham" = "https://www.transfermarkt.com/tottenham-hotspur/startseite/verein/148/saison_id/%d",
  "Bayern Munich" = "https://www.transfermarkt.com/fc-bayern-munich/startseite/verein/27/saison_id/%d",
  "Paris Saint Germain" = "https://www.transfermarkt.com/paris-saint-germain/startseite/verein/583/saison_id/%d",
  "Barcelona" = "https://www.transfermarkt.com/fc-barcelona/startseite/verein/131/saison_id/%d",
  "Real Madrid" = "https://www.transfermarkt.com/real-madrid/startseite/verein/418/saison_id/%d"
)

season_window <- function(y) c(as.Date(sprintf("%d-07-01", y)), as.Date(sprintf("%d-06-30", y+1)))

fetch_team_season <- function(club, url_fmt, y){
  p_urls <- tm_team_player_urls(team_url = sprintf(url_fmt, y))
  Sys.sleep(1)
  inj <- tm_player_injury_history(player_urls = p_urls)
  if (is.null(inj) || nrow(inj)==0) return(NULL)
  rng <- season_window(y)
  inj %>% mutate(from = suppressWarnings(lubridate::dmy(from)),
                 until = suppressWarnings(lubridate::dmy(until)),
                 season = sprintf("%d/%02d", y, (y+1) %% 100),
                 club = club) %>%
    filter(!is.na(from), from >= rng[1], from <= rng[2]) %>%
    transmute(player = player_name, club, season,
              diagnosis = injury, from, until,
              duration_days = as.integer(until - from + 1L),
              games_missed)
}

injuries <- purrr::imap_dfr(clubs, ~purrr::map_dfr(seasons, ~fetch_team_season(.y, .x, .x2)))
# write
if (!dir.exists("data/raw/transfermarkt")) dir.create("data/raw/transfermarkt", recursive = TRUE)
readr::write_csv(injuries, "data/raw/transfermarkt/injury_history_10clubs_2022_2025.csv")
```
