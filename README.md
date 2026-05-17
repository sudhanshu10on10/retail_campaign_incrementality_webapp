# Retail Campaign Incrementality Analyzer

> Measuring the **true causal impact** of a retail promotional campaign using Difference-in-Differences, Marketing Mix Modeling, and RFM Segmentation.

🔗 **Live App:** [Launch on Streamlit](https://retailcampaignincrementality-gdc6w6lobjsrj8osqja9sx.streamlit.app/)

---

## The Business Problem

When a company runs a promotional campaign, the instinctive question is:
> *"Sales went up — did the campaign cause it?"*

The honest answer is: **not necessarily.** Sales might have grown anyway due to seasonality, macro trends, or regional differences. This project isolates the **true causal lift** of a campaign — separating what the campaign actually caused from what would have happened regardless.

---

## What It Does

### 1. Campaign Lift — Difference-in-Differences (DiD)
- Splits data into **Treatment regions** (East + West) and **Control regions** (Central + South)
- Compares pre vs post campaign trends across both groups
- Isolates true causal lift by removing the counterfactual trend
- **Result:** $4,260 monthly causal lift (p = 0.02, statistically significant)
- 95% CI entirely positive [$645, $7,876] — confirms real effect

### 2. Revenue Decomposition — Marketing Mix Modeling (MMM)
- Builds a Ridge regression model on monthly aggregated data
- Decomposes revenue variance by driver:
  - Order Value Effect
  - Post-Period Trend
  - Seasonality (sin/cos encoding)
  - Region Effect
- Identifies that **order value drives 76%** of revenue variation

### 3. Segment Response — RFM Segmentation
- Scores customers on Recency, Frequency, and Monetary value
- Assigns each customer to a segment: Champions, Loyal, At-Risk, Lost
- Measures campaign lift **per segment** in treatment regions
- **Key finding:** Loyal customers are the only segment with positive lift (+7.4%)

---

## Key Findings

| Segment | Pre-Campaign Avg ($) | Post-Campaign Avg ($) | Lift |
|---|---|---|---|
| Loyal | 244 | 262 | **+7.4%** ✅ |
| Lost | 131 | 122 | -6.8% |
| Champions | 280 | 248 | -11.4% |
| At-Risk | 178 | 145 | -18.4% |

**Business Recommendation:**
> Reallocate campaign budget toward the **Loyal segment** — the only group showing positive incrementality. Champions show negative lift, meaning they buy regardless of the campaign, making spend on them inefficient. At-Risk customers require personalised interventions beyond broad promotional spend.

---

## Analytical Framework

```
Raw transactional data (Superstore)
        │
        ├── Simulate campaign
        │   Treatment: East + West regions
        │   Control:   Central + South regions
        │   Campaign date: June 2017
        │
        ├── DiD Model
        │   Sales ~ treatment × post + controls
        │   → True causal lift: $4,260/month (p=0.02)
        │
        ├── MMM Layer
        │   Ridge regression on monthly aggregates
        │   → Revenue decomposition by driver
        │
        └── RFM Segmentation
            R + F + M scoring → 4 segments
            → Segment-level campaign response
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data processing |
| Statsmodels | DiD regression (OLS) |
| Scikit-learn | MMM (Ridge regression), RFM scoring |
| Plotly | Interactive charts |
| Streamlit | Dashboard deployment |

---

## Project Structure

```
retail-campaign-incrementality/
├── app.py              ← Streamlit dashboard (3 pages)
├── superstore.csv      ← Source dataset (Kaggle Superstore)
├── requirements.txt    ← Dependencies
└── README.md
```

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/sudhanshu10on10/retail-campaign-incrementality.git
cd retail-campaign-incrementality

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Dataset

**Kaggle Superstore Sales Dataset**
- 9,800 transactions across 4 US regions
- Date range: January 2015 — December 2018
- Features: Order date, region, category, segment, sales

---

## Concepts Demonstrated

- **Causal Inference** — DiD to isolate true treatment effect
- **Parallel Trends Assumption** — validated visually pre-campaign
- **Marketing Mix Modeling** — Ridge regression with seasonal encoding
- **RFM Analysis** — customer value segmentation
- **Incrementality Testing** — segment-level response measurement
- **Business Recommendation** — actionable budget reallocation output

---

## Author

**Sudhanshu Ayer** — Data Scientist  
[GitHub](https://github.com/sudhanshu10on10) · [LinkedIn](https://www.linkedin.com/in/sudhanshu-a-964734128/)
