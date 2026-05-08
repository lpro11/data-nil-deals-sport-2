# NIL Research Project — Data Codebook

**Research question:** Do social media deals command higher average NIL compensation, conditional on school, sport, and year?

**Unit:** ECC3479 — Data and Evidence in Economics, Monash University
**Group:** Elena Langridge, Teah Papageorgiou, Lily Proposch
**Date:** May 2026

---

## Dataset 1: `data/clean/cleaned_nil.csv` (transaction-level)

**Description:** One row per individual NIL transaction, after cleaning and standardisation.
**Source:** `code/01_cleaning_script.py` applied to `data/raw/`
**Rows:** ~2,347 (positive-amount transactions only)

| Variable | Type | Description | Notes |
|----------|------|-------------|-------|
| `school` | String | Source university identifier | Derived from raw filename (e.g., `ucla`, `ucberkeley`) |
| `date` | Integer | Year of transaction | Derived from date columns in raw files; randomly assigned from {2021–2024} when unparseable |
| `amount` | Float | NIL deal value (USD) | Cleaned from strings with `$` and `,` removed; coerced to numeric. Rows with `amount ≤ 0` dropped |
| `sport` | String | Standardised sport label | Mapped from 30+ raw variants (e.g., `MBB` → `Basketball`, `WSW` → `Soccer`) |
| `deal_description` | String | Deal type / description | Missing or blank values filled with `"Other"` |
| `is_social_media` | Integer (0/1) | Treatment indicator | 1 if `deal_description` contains: `social media`, `instagram`, `tiktok`, `post`, `tweet` (case-insensitive). Special rule: UC San Diego (`ucsandiego1`) blank notes → 1 |

**Excluded schools:**
- `fresnostate1`, `sandiegostate1` — missing `amount` column (manually excluded)
- `ucirvine1`, `ucirvine2`, `sandiegostate2` — missing mappable `amount` column

---

## Dataset 2: `data/clean/nil_merged_analysis.csv` (group-level, analysis-ready)

**Description:** One row per (school × sport × year × deal-type) group. Each row represents the mean of all transactions in that cell.
**Source:** `code/02_aggregation_script.py` applied to `cleaned_nil.csv`
**Rows:** 371 total; 358 after dropping partial-year 2025 observations (done inside the notebook)

| Variable | Type | Description | Notes |
|----------|------|-------------|-------|
| `school` | String | University identifier | 12 unique values: calpolyslo, csulongbeach, sacramentostate1–4, ucberkeley, ucdavis, ucla, ucriverside, ucsandiego1, ucsantabarbara |
| `sport` | String | Standardised sport category | 20 sports: Baseball, Basketball, Beach Volleyball, Cross Country, Equestrian, Field Hockey, Football, Golf, Gymnastics, Ice Hockey, Lacrosse, Other, Rowing, Rugby, Soccer, Softball, Swimming/Diving, Tennis, Track and Field, Volleyball, Water Polo, Wrestling |
| `year` | Integer | Calendar year of deal | 2021, 2022, 2023, 2024 (2025 dropped in notebook: N=13, partial-year reporting) |
| `is_social_media` | Integer (0/1) | Treatment indicator | 1 = social media deals group; 0 = other deal types. Inherited from `cleaned_nil.csv` |
| `avg_transaction_value` | Float | Mean NIL value (USD) for the group | Group-level average of `amount`. Range: $0.01–$84,290.29; median: $178.72; mean: $2,184.62; skewness ≈ 6.7 |

### Descriptive Statistics (analysis sample, N=358)

```
Social media groups (is_social_media=1):  156 (43.6%)
Non-social media groups:                  202 (56.4%)

avg_transaction_value:
  Mean:     $2,184.62
  Median:   $178.72
  SD:       $5,423.42
  Min:      $0.01
  Max:      $84,290.29
  Skewness: 6.749 (raw); 0.702 (log-transformed) → supports log transformation

log(1 + avg_transaction_value):
  Mean:   5.556
  Median: 5.191
  SD:     1.643
```

---

## Data Quality Notes

1. **UC San Diego assumption rule:** `ucsandiego1` rows with blank `Notes` fields are coded `is_social_media = 1` (social media by assumption). This inflates social media deal share for this school. A robustness check excluding UC San Diego is reported in `code/04_primary_analysis.ipynb` Section 8.

2. **Measurement error in treatment:** `is_social_media` is derived from keyword matching on free-text `deal_description`. This introduces false positives (e.g., "post" meaning campus appearance) and false negatives (e.g., vague descriptions). Classical measurement error attenuates the coefficient toward zero. See Section 7.2 of `04_primary_analysis.ipynb`.

3. **Date imputation:** For a minority of transactions, year was randomly assigned from {2021–2024} when no date could be parsed. This introduces noise in year fixed effects. See Section 7.5.

4. **Aggregation artifact:** The unit of analysis is a group average. Small groups may be dominated by outliers. See Section 7.4.

5. **Sample selection:** 12 reporting schools are included. Six California D-I schools had no usable data. Results generalise only to reporting schools.

---

## How to Reproduce

```bash
# From repository root:
python code/01_cleaning_script.py       # → data/clean/cleaned_nil.csv
python code/02_aggregation_script.py    # → data/clean/nil_merged_analysis.csv
```

Or load directly:
```python
import pandas as pd
df = pd.read_csv("data/clean/nil_merged_analysis.csv")
df = df[df["year"] <= 2024]   # Drop 13 partial-year 2025 rows
```

---

## Analysis Files Using This Dataset

| File | Purpose |
|------|---------|
| `code/03_EDA_notebook.ipynb` | Exploratory data analysis — variable distributions, correlations, Simpson's Paradox check |
| `code/04_primary_analysis.ipynb` | Primary econometric analysis — OLS with fixed effects, regression table, robustness checks |
| `results/regression_table.png` | Auto-generated: Table 1 from primary analysis |
| `results/coefficient_plot.png` | Auto-generated: coefficient stability across specifications |
| `results/residual_diagnostics.png` | Auto-generated: M4 residual diagnostics |

---

## References

- **CalMatters NIL repository:** https://github.com/calmatters/nil-disclosures
- **Data acquisition:** Individual university FOIA responses and public NIL registries
