# Online Retail Customer Analytics & RFM Segmentation
### Technical Project Report

> **Authors:** Phase 1 — Eng. Israa Hamdy · Phase 2 — Eng. Jumana Khaled · Phase 3 — Eng. Mahmoud Mansour
> **Dataset Period:** December 2010 – December 2011
> **Snapshot Date:** 2011-12-10
> **Environment:** Python 3.11 · pandas 2.2.3 · numpy 2.1.3 · scikit-learn · Windows

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Dataset Overview](#2-dataset-overview)
3. [Pipeline Summary](#3-pipeline-summary)
4. [Phase 1 — Data Engineering & EDA](#4-phase-1--data-engineering--eda)
5. [Phase 2 — RFM Analysis & Preprocessing](#5-phase-2--rfm-analysis--preprocessing)
6. [Phase 3 — Clustering & Business Strategy](#6-phase-3--clustering--business-strategy)
7. [Models Used & Results](#7-models-used--results)
8. [Final Segments & Business Recommendations](#8-final-segments--business-recommendations)
9. [Quality Checks & Known Limitations](#9-quality-checks--known-limitations)
10. [Key Takeaway](#10-key-takeaway)

---

## 1. Project Goal

**Segment customers into actionable marketing groups** based on their purchasing behaviour, so each group receives a different strategy: retention, re-engagement, onboarding, or suppression.

The segmentation is built on **RFM (Recency, Frequency, Monetary)** — three interpretable dimensions that translate raw transactions into customer value and engagement signals.

---

## 2. Dataset Overview

| Item | Value |
|---|---|
| Source | UCI Online Retail dataset — UK-based e-commerce gift/homeware retailer |
| Customer type | Mix of wholesalers (bulk buyers) and individual buyers |
| Raw rows | 541,909 |
| Clean sale lines | 522,537 |
| Unique invoices | 19,773 |
| Identified customers | 4,334 |
| Net revenue (after returns) | £9,771,318 |
| Gross revenue | £10,247,219 |
| Total returns value | £475,901 (4.64% of gross) |
| Period covered | ~13 months |

### Dataset Columns

| Column | Description | Handling |
|---|---|---|
| `InvoiceNo` | Invoice ID; prefix `C` = cancellation | Kept as text |
| `StockCode` | Product code; non-product codes also present | Upper-cased, non-product excluded |
| `Description` | Product name | Text only |
| `Quantity` | Units; negative on cancellations | Signed numeric |
| `InvoiceDate` | Transaction timestamp | Parsed to datetime |
| `UnitPrice` | Price per unit in GBP; ≤ 0 = adjustment | Numeric |
| `CustomerID` | Customer identifier; missing on guest orders | **Text index only — never a feature** |
| `Country` | Customer country | Text |

> **Grain:** one row = one transaction **line**, not one order. Orders are rebuilt from `InvoiceNo`.

---

## 3. Pipeline Summary

```
Raw Excel
    │
    ▼
Phase 1 ── Load → Audit → Missing-ID Check → Clean & Classify
    │         → Feature Engineering → EDA (Products / Revenue /
    │           Time / Geography / Baskets / Returns / Pareto /
    │           Dead Stock / Outliers) → Business Insights (16)
    │         → Save clean tables
    ▼
Phase 2 ── Load clean tables → Build customer-level RFM table
    │         → Quantile scoring (1–5) → Log1p transform
    │         → Outlier handling (cap if |z| > 4) → StandardScaler
    │         → PCA → Validate (17 checks PASS) → Save RFM tables
    ▼
Phase 3 ── Load RFM tables → Algorithm comparison
              (Hierarchical structure check, DBSCAN whale screen,
               K-Means evaluation K=2…10) → Select K=4
              → Fit final K-Means → Stability test
              → Profile segments → Name from RFM scores
              → Business recommendations → Export
```

---

## 4. Phase 1 — Data Engineering & EDA

**Author: Eng. Israa Hamdy**

### 4.1 Cleaning Decisions (Waterfall)

| Line type | Count | Revenue | Decision |
|---|---|---|---|
| **Sale** (valid) | 522,537 | £10,247,219 | Kept for all analysis |
| **Return** (C-invoices) | 8,668 | −£475,901 | Kept, matched to original sale, netted in Phase 2 |
| Excluded — non-product codes | 2,924 | −£23,187 | Postage, fees, manual entries — not merchandise |
| Excluded — price anomalies | 2,512 | −£22,124 | Zero/negative price: write-offs, giveaways, bad debt |
| Exact duplicates removed | 5,268 | — | Double-exported lines |
| Guest lines (no CustomerID) | 135,080 | £1,510,000 | Kept for EDA, excluded from RFM |

> Every exclusion is documented with a reason; nothing is silently dropped.

### 4.2 Key EDA Findings

| # | Finding |
|---|---|
| 1 | Top product by revenue: **22423** (REGENCY CAKESTAND 3 TIER) — £174,157 |
| 2 | Top product by units: **23843** (PAPER CRAFT, LITTLE BIRDIE) — 80,995 units ⚠️ gross only (order fully returned) |
| 3 | Top customer by revenue: **14646** — £279,138 |
| 4 | Top country: **United Kingdom** — £8,724,444 (85.1% of revenue) |
| 5 | UK lines = 91.6% of sale lines; Saturday = zero trading |
| 6 | Guest lines = 25.1% of sale lines; retained for EDA only |
| 7 | Returns = £475,901 (4.64% gross); inflated by two large orders (see §4.3) |
| 8 | Median basket: 152 units, 15 distinct products |
| 9 | Top 21.5% of StockCodes = 80% of revenue (slightly weaker than classic 80/20) |
| 10 | Bulk-like baskets (top 10% by units) = 10% of orders → 45% of revenue — wholesale behaviour |
| 11 | Dead-stock screen flagged 20 products (0.6%); 2 likely seasonal |
| 12 | Strongest weekday: **Thursday**; peak hours: 12:00–14:00 |
| 13 | Q4 (Oct–Dec) lift = **1.60×** vs non-Q4 average (one season — suggestive, not proven) |
| 14 | Peak month (complete): **November 2011** — £1,427,125 net |
| 15 | Top 10 customers = 17.4% of identified-customer revenue; top 1% = 32.2% |
| 16 | Median customer £663 vs mean £2,016 (3.0× ratio → wholesale skew) |

### 4.3 Important Gross-Data Caveat

> **⚠️ Phase 1 EDA uses gross figures (before netting returns).**
> Two customers placed very large orders that were subsequently cancelled in full:
> - **Customer 16446**: £168,473 gross → £2.90 net (100% returned)
> - **Customer 12346**: £77,184 gross → £0.00 net (100% returned)
>
> Impact: top-units products (23843, 23166) and top-10 customer rankings are inflated.
> The true return rate is ~2.3%, not 4.64%, when these two transactions are excluded.
> **Phase 2+ uses net Monetary — corrected.**

### 4.4 Return Matching

- 8,668 return lines matched back to the most-recent earlier purchase (same customer + product).
- 11.9% unmatched — purchase pre-dates the dataset window.
- 1.9% unmatched — guest returns (no CustomerID).
- Highest return month: **December 2011** (28.3% of that month's gross) — driven entirely by the 16446 cancellation.

---

## 5. Phase 2 — RFM Analysis & Preprocessing

**Author: Eng. Jumana Khaled**

### 5.1 Customer Population

| Step | Customers / Lines |
|---|---|
| Sale lines (Phase 1) | 522,537 |
| − Guest lines removed | 131,387 |
| = Lines used for RFM | 391,150 |
| Unique identified customers | 4,334 |
| − Net revenue ≤ 0 (fully returned) | 12 |
| **= Final RFM population** | **4,322** |

### 5.2 RFM Definitions

| Dimension | Formula | Direction |
|---|---|---|
| **Recency (R)** | Days from last purchase to snapshot (2011-12-10) | Lower = better |
| **Frequency (F)** | Count of unique invoices (not lines) | Higher = better |
| **Monetary (M)** | Net revenue = gross sales − returns | Higher = better |

> **Why net Monetary?** Customers 16446 and 12346 would rank in the top 5 on gross; their net values are £2.90 and £0.00. Net is the honest basis for value segmentation.

### 5.3 Typical Customer Profile (Medians)

| Metric | Value |
|---|---|
| Customers modelled | 4,322 |
| Median recency | 51 days |
| Median orders (F) | 2 |
| Median net spend | £650 |
| Median order value (AOV) | £281 |
| Median distinct products | 35 |
| One-time buyers | 34.6% |
| UK customers | 90.4% |
| Median tenure | 250 days |
| Mean / median Monetary ratio | 2.9× (wholesale skew) |

### 5.4 RFM Quantile Scoring (1–5)

- Each dimension split into 5 equal quantiles using `pd.qcut` on ranked values.
- R inverted: score 5 = most recent, score 1 = most lapsed.
- Scores combined into a 3-digit string (e.g., `555` = top-tier champion).

**Score boundaries:**

| Score | Recency (days) | Frequency (orders) | Monetary (net £) |
|---|---|---|---|
| 5 | 1 – 15 | ≥ 6 | ≥ £1,998 |
| 4 | 15 – 33 | 3 – 5 | £909 – £1,998 |
| 3 | 33 – 72 | 2 | £471 – £909 |
| 2 | 72 – 180 | 1 – 2 | £242 – £471 |
| 1 | 180 – 374 | 1 | £0 – £242 |

> **Tie-breaking caveat:** `rank(method="first")` distributes ties by CustomerID order. ~34.6% of customers have F=1; the tie-break places approximately half of them in F_score=2 rather than 1. The F dimension boundary between scores 1 and 2 is therefore soft — both indicate low-frequency customers.

**Early warning (churn risk):** 173 customers with F≥4 and M≥4 but R≤2 — holding 4.5% of net revenue. These are historically valuable accounts that have gone quiet.

### 5.5 Preprocessing Pipeline

| Step | Method | Why |
|---|---|---|
| Log transform | `log1p(x)` on R, F, M | Compresses extreme right skew (Monetary skew: 21.5 → 0.30; Frequency: 11.9 → 1.21) |
| Outlier cap | Winsorise at 0.5th–99.5th pct on transformed scale | Applied because max\|z\| = 5.85 > trigger (4.0); 66 values capped |
| Scaling | `StandardScaler` → mean=0, std=1 | Prevents Monetary from dominating Euclidean distance |
| PCA | 2 components (94.0% variance retained) | Used for **visualisation only**; clustering ran on all 3 scaled features |

**Skewness before and after log1p:**

| Feature | Raw skew | After log1p |
|---|---|---|
| Recency | 1.245 | −0.325 |
| Frequency | 11.938 | 1.211 ⚠️ |
| Monetary | 21.548 | 0.300 |

> **Note on Frequency:** skew remains above 1.0 after log1p because ~35% of customers bought only once, creating a discrete spike that no continuous transform can fully smooth. A StandardScaler `symmetric: False` warning is expected and noted.

**Outlier strategy comparison:**

| Approach | Max \|z\| | Top-1% dominance | Spearman(M) |
|---|---|---|---|
| A — Keep raw | 33.35 | 54.1% | 1.000 |
| B — Winsorise raw 1–99 | 6.18 | 21.1% | 1.000 |
| C — Transform only | 5.85 | 9.4% | 1.000 |
| **D — Transform + cap ✅** | **3.61** | **7.7%** | **1.000** |

**All 17 Phase 2 validation checks: PASS**

---

## 6. Phase 3 — Clustering & Business Strategy

**Author: Eng. Mahmoud Mansour**

### 6.1 Clustering Input

- Matrix: `rfm_scaled` — 4,322 rows × 3 columns (`Recency_s`, `Frequency_s`, `Monetary_s`)
- All features numeric, mean ≈ 0, std = 1, CustomerID is index only.

---

## 7. Models Used & Results

### 7.1 Hierarchical Clustering (Ward) — Structure Check

Run on a reproducible sample of **1,500 customers** (full dataset is computationally expensive).

**Dendrogram height gaps (largest = most natural cut):**

| Merge | Gap |
|---|---|
| K=2 → K=1 | **36.623** ← largest |
| K=3 → K=2 | 6.756 |
| K=4 → K=3 | 3.788 |
| K=5 → K=4 | 3.231 |

The tree strongly prefers K=2 statistically. This is too coarse for four marketing actions, but it confirms that the data has two broad natural groupings (active vs. lapsed).

**Silhouette per cut (on the sample):**

| K | Silhouette | Cluster Sizes |
|---|---|---|
| 3 | 0.327 | [843, 219, 438] |
| 4 | 0.253 | [450, 393, 219, 438] |
| 5 | 0.244 | [450, 393, 219, 204, 234] |
| 6 | 0.234 | [450, 155, 238, 219, 204, 234] |

**Role:** structure validation only — not the final segmentation.

---

### 7.2 DBSCAN — Whale / Outlier Screen

- **Parameters:** `eps = 0.35` (knee of k-distance curve), `min_samples = 6` (= 2 × 3 features)

**Sensitivity to eps:**

| eps | Clusters | Noise customers | Noise % |
|---|---|---|---|
| 0.30 | 7 | 263 | 6.1% |
| **0.35** | **4** | **144** | **3.3%** |
| 0.50 | 2 | 36 | 0.8% |
| 0.75 | 1 | 8 | 0.2% |
| 1.00 | 1 | 3 | 0.07% |

**Who are the 144 noise customers?**

| Metric | Core Customers | DBSCAN Noise (Whales) |
|---|---|---|
| Recency (median) | 52 days | 22 days |
| Frequency (median) | 2 orders | 8 orders |
| Monetary (median) | £632 | £6,683 |
| AOV (median) | £277 | £655 |
| Revenue share | 76.5% | **23.5%** |
| Customer share | 96.7% | **3.3%** |

These are the **whale buyers** — extremely high-value, high-frequency, recent customers who sit too far from any density cluster to be grouped with ordinary customers.

**Role:** outlier identification only — not the final segmentation. These customers are profiled inside the Champions segment in K-Means.

---

### 7.3 K-Means — Final Segmentation Model

**Evaluation across K=2 to K=10:**

| K | WCSS | Silhouette ↑ | Davies-Bouldin ↓ | Smallest Cluster % | WCSS Drop % |
|---|---|---|---|---|---|
| 2 | 6,356 | **0.434** | **0.886** | 38.5% | — |
| 3 | 4,736 | 0.339 | 1.041 | 17.9% | 25.5% |
| **4** | **3,799** | **0.335** | **1.017** | **16.8%** | **19.8%** |
| 5 | 3,161 | 0.317 | 0.988 | 9.3% | 16.8% |
| 6 | 2,731 | 0.313 | 1.014 | 9.3% | 13.6% |
| 7 | 2,437 | 0.305 | 0.987 | 5.0% | 10.7% |

**K selection logic:**

| Criterion | Result |
|---|---|
| Elbow (chord method) | K=5 |
| Best Silhouette — unconstrained | K=2 (0.434) |
| Best Davies-Bouldin — unconstrained | K=2 |
| Business range constraint | K ∈ [4, 6] |
| Best Silhouette inside range | **K=4 ✅ (0.335)** |
| Silhouette cost vs unconstrained best | −0.099 |

> **Silhouette caveat:** all scores in the business range are in the 0.31–0.34 band — below the 0.50 threshold for clear cluster structure. RFM is a continuum; there are no sharp natural boundaries. The Ward dendrogram's K=2 preference confirms this. **K=4 is a business decision** that produces the four actionable archetypes the marketing team needs, not a statistical discovery. Segment boundaries are soft zones.

**Stability tests (K=4):**

| Test | ARI | Interpretation |
|---|---|---|
| Random starts (10 runs) | mean=1.000, min=0.999 | Near-perfect reproducibility |
| 80% subsamples (10 runs) | mean=0.930, min=0.793 | Robust across different customer samples |
| K-Means vs Hierarchical (sample) | 0.420 | Expected — different algorithms, different assumptions |
| **Verdict** | | **STABLE** |

> The K-Means vs Hierarchical ARI of 0.420 is expected and does not invalidate the solution. The two algorithms have fundamentally different assumptions (centroid vs. linkage). Random-start and subsample ARIs are the primary stability indicators.

---

## 8. Final Segments & Business Recommendations

**Model:** K-Means, K=4, clustering on `rfm_scaled` (3 standardised features)
**Population:** 4,322 identified customers with net Monetary > 0

### 8.1 Segment Overview

| Cluster | Segment Name | Customers | % Customers | % Revenue |
|---|---|---|---|---|
| 0 | **Champions** | 728 | 16.8% | **65.1%** |
| 1 | **Developing** (R:Mid F:High M:High) | 1,182 | 27.4% | 23.6% |
| 2 | **New Inflow** | 882 | 20.4% | 5.0% |
| 3 | **Hibernating / Lost** | 1,530 | 35.4% | 6.4% |

### 8.2 Segment Profiles

#### 🏆 Cluster 0 — Champions

| Metric | Value |
|---|---|
| Customers | 728 (16.8%) |
| Revenue share | **65.1%** |
| Recency (median) | 9 days |
| Frequency (median) | 10 orders |
| Monetary (median) | £3,481 |
| R / F / M scores | 4.65 / 4.87 / 4.78 |

The highest-value segment by a large margin. Recent, frequent, and high-spending. Includes most of the DBSCAN whale buyers. 728 customers generating nearly two-thirds of all modelled revenue.

**Strategy:** Reward with loyalty perks, exclusive early access to new product lines, and a brand referral programme. **Do not offer blanket discounts** — they are already buying without incentive.

---

#### 📈 Cluster 1 — Developing (R:Mid F:High M:High)

| Metric | Value |
|---|---|
| Customers | 1,182 (27.4%) |
| Revenue share | 23.6% |
| Recency (median) | 53 days |
| Frequency (median) | 4 orders |
| Monetary (median) | £1,304 |
| R / F / M scores | 2.97 / 3.80 / 3.97 |

Strong purchase history (high F and M) but recency is drifting — these accounts are starting to cool. **This is the highest-priority retention segment.**

**Strategy:**
1. Trigger a personalised re-engagement email within 7 days, featuring their most-purchased categories.
2. Offer a time-limited loyalty incentive (e.g. free shipping on next order) — not a blanket discount.
3. For customers with net spend above the segment median (£1,304), assign to a KAM review.
4. Validate impact against a holdout group before scaling.

---

#### 🆕 Cluster 2 — New Inflow

| Metric | Value |
|---|---|
| Customers | 882 (20.4%) |
| Revenue share | 5.0% |
| Recency (median) | 20 days |
| Frequency (median) | 2 orders |
| Monetary (median) | £399 |
| R / F / M scores | 4.10 / 2.43 / 2.25 |

Recent first- or second-time buyers. The pipeline of future value — conversion rate from first to second purchase is the key metric to track.

**Strategy:** Welcome sequence with an incentive for the second purchase. Track first-to-second-order conversion. Low cost to serve; high potential upside if they become regulars.

---

#### 😴 Cluster 3 — Hibernating / Lost

| Metric | Value |
|---|---|
| Customers | 1,530 (35.4%) |
| Revenue share | 6.4% |
| R / F / M scores | 1.61 / low / low |

Largest segment by count, lowest by revenue. All three RFM dimensions are low — these customers have not bought recently, not frequently, and not much.

**Strategy:** One low-cost reactivation email. If no response, suppress from paid campaigns. Do not invest disproportionate resources in this segment given the low revenue contribution.

---

### 8.3 Segment Naming Rules

Segments are named from their **measured mean RFM scores** using the following rules, not assigned labels upfront:

| Name | Rule |
|---|---|
| Champions | R ≥ 4, F ≥ 4, M ≥ 4 |
| Loyal Customers | R, F, M all ≥ 3.5 (not Champions) |
| At-Risk / High-Value Churn | R ≤ 2.5, F ≥ 3, M ≥ 3 |
| New Inflow | R ≥ 3.5, F ≤ 2.5, M ≤ 2.5 |
| Hibernating / Lost | R, F, M all ≤ 2.5 |
| Developing (R:X F:X M:X) | Fallback — descriptive label from measured levels |

---

## 9. Quality Checks & Known Limitations

### 9.1 Validation (All 17 PASS)

- One row per customer (unique index) ✅
- No duplicate CustomerIDs ✅
- Row count matches population ✅
- No missing RFM values in any dataset ✅
- No NaN/inf anywhere ✅
- All datasets share same CustomerID order ✅
- CustomerID not a feature column ✅
- All clustering features numeric ✅
- Raw R≥1, F≥1, M>0 ✅
- Every customer from identified sale lines ✅
- No guest lines in RFM population ✅
- No cancellation invoice counted as an order ✅
- No transaction after snapshot date ✅
- Zero customers deleted as outliers ✅
- Scaled features centred (|mean| < 1e-6) ✅
- RFM scores integers 1–5 ✅
- RFM combined score is 3-digit string ✅

### 9.2 Known Limitations

| # | Limitation | Impact |
|---|---|---|
| 1 | Silhouette ~0.33 — no sharp cluster boundaries | Segment edges are soft; interpret ranges, not hard cutoffs |
| 2 | Ward tree prefers K=2; K=4 is a business decision | Statistically, data has two natural groups; four is a marketing choice |
| 3 | Frequency skew after log1p = 1.21 | StandardScaler `symmetric: False` warning; F dimension boundaries are approximate |
| 4 | Tie-breaking in F_score by CustomerID order | ~630 one-time buyers receive F_score=2 instead of 1 by artefact |
| 5 | K-Means vs Hierarchical ARI = 0.420 | Two algorithms don't perfectly agree; boundaries confirmed soft |
| 6 | Phase 1 EDA uses gross figures | Top-units and top-customer rankings are inflated; corrected in Phase 2+ |
| 7 | Dataset = ~13 months, one holiday season | Seasonality is suggestive, not proven |
| 8 | Return matching is approximate | Complex partial-return sequences may stay ambiguous |
| 9 | Static snapshot | Customers migrate between segments over time; re-run quarterly |
| 10 | Only R/F/M clustered | Product mix, seasonality, discount sensitivity not in the model |
| 11 | Descriptive analysis | No causal claims — segment strategies are hypotheses to validate |

### 9.3 Recommended Next Steps

1. Re-run with `MONETARY_BASIS="gross"` and cap on/off to verify segment stability.
2. Build monthly RFM snapshots to measure segment migration over time.
3. Add product category mix and seasonality features for a richer model.
4. Run controlled experiments (A/B tests) per segment before attributing revenue lift.
5. Save the fitted StandardScaler and re-apply it when scoring new customers.

---

## 10. Key Takeaway

> **728 customers (16.8%) generate 65.1% of net revenue.**
>
> The Champions segment is where the business lives. Losing even a handful of these accounts has an outsized revenue impact. The Developing segment (27.4% of customers, 23.6% of revenue) is the highest-priority retention target — historically strong accounts that are beginning to cool and can still be saved with a timely, personalised intervention.

---

*Report generated from `RFM_Segmentation_Final_v2.ipynb` — run on the full Online Retail dataset. All figures are data-derived; no values are hard-coded.*
