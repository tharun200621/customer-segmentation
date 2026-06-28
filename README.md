# 🛒 SmartCart — Customer Segmentation

> Segmenting ~2,200 retail customers into actionable marketing personas with unsupervised learning.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-clustering-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-EDA-150458?logo=pandas&logoColor=white)

A complete customer-segmentation analysis: clean the data, engineer behavioural
features, reduce dimensionality with PCA, choose the number of clusters with the
elbow method + silhouette score, then **profile each cluster into a named persona
with concrete marketing recommendations.**

## 🎯 Result: 4 customer segments

The clusters separate cleanly along **income/spending × living situation**, with
campaign-response rate ranging from 10% to **33%**:

| Segment | Persona | Income | Spending | Response | Size |
|---------|---------|--------|----------|----------|------|
| 💎 **High-Value Singles** | Affluent, no kids, live alone | ~$72K | ~$1,228 | **33%** | 332 |
| 🏡 **Affluent Families** | Wealthy, partnered, with kids | ~$72K | ~$1,200 | 20% | 573 |
| 🛒 **Budget Singles** | Lower income, live alone | ~$38K | ~$186 | 14% | 463 |
| 👨‍👩‍👧 **Budget Families** | Lower income, partnered, more kids | ~$39K | ~$200 | 10% | 868 |

**Clustering quality:** silhouette ≈ **0.41** (KMeans) / **0.40** (Agglomerative) —
consistent across both methods, so the 4-way structure is stable.

## 💡 Business takeaways

- **Target campaigns at High-Value Singles** — they spend the most *and* respond at 33% (~3× budget families). Highest marketing ROI.
- **Budget segments visit the website most but spend least** — they browse without buying. Better served by web conversion nudges (discounts, retargeting) than expensive catalog/in-store campaigns.
- **Income + living situation drive behaviour** far more than age or education, which are nearly flat across segments.

## 🧠 Method

```
clean + feature-engineer → one-hot encode → standardize → PCA(3)
   → elbow + silhouette to pick k=4
   → KMeans & Agglomerative (compared) → profile segments → personas
```

| Step | Choice | Why |
|------|--------|-----|
| Feature engineering | Age, tenure, total spend, total children, living-with | Turns raw fields into behaviour-relevant signals |
| Encoding | One-hot (Education, Living_With) | Categorical → numeric for distance metrics |
| Scaling | StandardScaler | Clustering is distance-based; features must share a scale |
| Reduction | PCA to 3 components | Denoise + decorrelate before clustering; interpret on original features |
| Choosing k | Elbow (kneed) + silhouette | Two independent signals both point to k=4 |
| Clustering | KMeans vs Agglomerative (ward) | Cross-check stability; near-identical results |

> **Note on PCA:** clustering runs on PCA-reduced features (better for distance-based
> methods), but segments are *interpreted* on the original human-readable features
> (income, spending, response) — i.e. reduce to cluster, interpret in original units.

## 🚀 Run it

```bash
pip install -r requirements.txt
jupyter notebook SMART_CART_CLUSTER.ipynb
```

The notebook saves the fitted encoder, scaler, PCA, model, and segment profiles to `models/`.

## 🗂️ Files

```
SMART_CART_CLUSTER.ipynb   full analysis
smartcart_customers.csv    dataset (2,240 customers × 22 features)
models/                    saved encoder / scaler / PCA / model + segment_profiles.csv
requirements.txt
```

## 📊 Dataset

Customer Personality Analysis — 2,240 customers with demographics (age, education,
marital status, income, children) and behaviour (spend per category, purchases by
channel, web visits, campaign response).
