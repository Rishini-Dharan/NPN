# NPN — Multi-Model Winning Strategy for M5 Forecasting

**Goal:** Predict 28 days of daily sales for 30,490 product-store series.
**Metric:** WRMSSE (Weighted Root Mean Square Scaled Error).
**Scoring:** Hierarchical — predictions evaluated at 12 aggregation levels simultaneously.

---

## 1. Competition scoring explained (simple)

WRMSSE compares your forecast against a seasonal-naive baseline (last week's sales).

- WRMSSE = 1.0 means you did **no better** than guessing "same as last week"
- WRMSSE < 1.0 means you're **better** than the baseline
- Lower is better. **Winning = WRMSSE as far below 1.0 as possible**

**12 scoring levels** (you need to be good at ALL of them):
1. Item × Store (most granular)
2. Department × Store
3. Category × Store
4. All items × Store
5. Item × State
6. Department × State
7. Category × State
8. All items × State
9. Item × All stores
10. Department × All stores
11. Category × All stores
12. All items × All stores (total)

**Implication:** A model that forecasts each item-store separately — then reconciles up the hierarchy — usually beats one that forecasts at the total level and disaggregates.

---

## 2. The 4-Arm Parallel Model Strategy

Train **four model families** in parallel, then sync (ensemble) their predictions:

| Arm | Approach | Strengths | Weaknesses |
|---|---|---|---|
| **A — Gradient Boosting** | LightGBM on item-store series with rich features | Fast, handles nonlinearity, strong on tabular features | Struggles with long-horizon and rare items |
| **B — Deep Learning** | N-BEATS / TFT / Seq2Seq on sequences | Captures complex temporal patterns, good for trends | Slow to train, needs careful tuning |
| **C — Statistical** | ETS, Theta, Croston, SARIMA | Excellent baselines, interpretable, robust on sparse items | Can't use rich features, limited flexibility |
| **D — Hierarchical** | Bottom-up + MinT reconciliation | Ensures all levels add up, strong on aggregate levels | Depends on quality of base forecasts |

### 2.1 Arm A: LightGBM (primary workhorse)

**Setup:**
- One model per **category** (FOODS, HOBBIES, HOUSEHOLD) — 3 models
- OR one model per **department** (7 models) — finer granularity
- Input: item-store-day panel with engineered features

**Key features:**
- Lag features: sales at lags 1, 2, 3, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84
- Rolling means: 7, 14, 28, 56-day rolling means (and std) shifted by 28+ days to avoid leakage
- Price features: current price, price lag, price diff, price rolling mean, price momentum
- Calendar: dayofweek, month, year, quarter, is_weekend, is_month_start/end
- Events: event_name one-hot, event_type, SNAP flag (per state)
- Time index: day counter, sine/cosine day-of-week, sine/cosine month

**Training approach:**
- Train on first ~1,885 days
- Validate on last ~28 days (the "validation" window)
- Use first 28-day validation to tune, then retrain on full data for final predictions

### 2.2 Arm B: Deep Learning

**Option 1: N-BEATS (recommended)**
- Generic architecture: fully connected, interpretable basis expansion
- Input: normalized sales history (last ~100 days) + static features (price, category) + dynamic features (calendar)
- Output: 28-day forecast directly
- Trained per category to manage compute

**Option 2: Temporal Fusion Transformer (TFT)**
- Attention-based, handles known and unknown future inputs
- Known future: calendar, events, prices (known ahead of time)
- Unknown future: actual sales (autoregressive)
- Use GluonTS or PyTorch Forecasting

**Hybrid approach:** Train N-BEATS for the bulk of items (good at long sequences), TFT for items with rich event/promo signals.

### 2.3 Arm C: Statistical models

- **ETS** (Error Trend Seasonal) — strong baseline, handles trend + weekly seasonality well
- **Theta** — simple, consistently good on many M5 items; the 2018 M4 winner uses it
- **Croston** — designed specifically for intermittent demand (the 68%-zero problem)
- **SARIMA** — for items with clear seasonal patterns (subset of items, too slow for all 30k)

Use these primarily as **ensemble components** and to validate the ML/DL arms. For highly intermittent items, Croston may actually be the best single model.

### 2.4 Arm D: Hierarchical reconciliation

1. Generate base forecasts from Arms A-C at the **item-store level** (most granular)
2. Aggregate to all 12 levels
3. Apply **MinT (Minimum Trace) reconciliation** — optimally combines forecasts across levels so the final result is consistent and minimizes overall error

Tools: `scikit-hts`, `hierarchicalforecast` (by Niño B. García), or `mlforecast` from Nixtla.

---

## 3. Feature engineering plan (detailed)

### 3.1 Lag features
- Short lags: 1, 2, 3 days
- Weekly lags: 7, 14, 21, 28
- Long lags: 35, 42, 49, 56, 63, 70, 77, 84
- **Leakage rule:** Only use lags that are available *before* the prediction date.

### 3.2 Rolling statistics
- Rolling mean/stddev: 7, 14, 28, 56 days
- Rolling min/max: 7, 28 days (detect outliers)
- **Shift by 28+ days** so rolling windows don't leak future info

### 3.3 Price features
- Current price (per item-store-week)
- Price change: (current - lag_1) / lag_1
- Price rolling mean (7, 28 days)
- Price momentum: is price trending up or down?
- Price relative to min/max: (price - min) / (max - min)

### 3.4 Calendar features
- dayofweek (0–6), dayofmonth (1–31), month (1–12), year
- week of year, quarter
- is_weekend (Sat/Sun)
- sine/cosine encodings for cyclical features

### 3.5 Event features
- event_name_1, event_name_2, event_type_1, event_type_2
- One-hot encode event names
- Event "days before/after" (events often have lead/lag effects)
- SNAP flag (per state — CA, TX, WI have different schedules)

### 3.6 Cumulative / stock features
- Cumulative sales over the year (for trend detection)
- Days since last sale (for intermittent items)
- Sell-on-last-X-days (binary: did this item sell in last 7 days?)

---

## 4. Ensemble & sync strategy

### 4.1 Blending weights (determined via cross-validation)
| Model | Weight (typical) |
|---|---|
| LightGBM (per category) | 40-50% |
| N-BEATS | 20-25% |
| Statistical (ETS + Theta + Croston avg) | 15-20% |
| Hierarchical reconciliation | Applied to final blend |

**How to find optimal weights:** Use a rolling-origin CV window over the last ~40 days. For each candidate weight combination, compute WRMSSE across all 12 levels. Pick the combination with the best weighted average.

### 4.2 Stacked generalization (advanced)
- Use LightGBM as a meta-learner
- Input to meta-model: predictions from N-BEATS, ETS, Theta, Croston
- Train meta-model to output residuals correction
- This captures interactions between arms that simple averaging misses

### 4.3 Hierarchical reconciliation (final sync)
- After blending, apply MinT reconciliation across all 12 levels
- This is the **final sync** — ensures forecasts add up correctly at every level
- Tools: `hierarchicalforecast` library with `MinTrace` method

---

## 5. Cross-validation & backtesting

### 5.1 Rolling-origin evaluation
Don't use a single validation window. Use **multiple cutoffs**:

| Cutoff day | Validation window |
|---|---|
| Day 1,885 | Validate days 1,886–1,913 (28 days) |
| Day 1,857 | Validate days 1,858–1,885 (28 days) |
| Day 1,829 | Validate days 1,830–1,857 (28 days) |

This gives you 3 validation windows to average model performance — more robust than a single split.

### 5.2 WRMSSE computation
- For each item-store series, compute the scaling factor (mean absolute error of the seasonal naive baseline)
- Compute your model's MAE on the validation window
- WRMSSE = sqrt(your_MAE / baseline_MAE)
- Weight by total sales volume at each level

### 5.3 Backtesting checklist
- [ ] All lags and rolling features shift correctly (no leakage)
- [ ] Price features only use known-future information at prediction time
- [ ] Event features use the actual event calendar (not future sales)
- [ ] Hierarchical totals add up after reconciliation
- [ ] No negative predictions (clip to zero)

---

## 6. Execution roadmap

### Week 1: Setup & EDA replication
- Clone repo, set up Python environment (pandas, numpy, lightgbm, torch, gluonts)
- Load all 4 CSV files, verify shapes match study report
- Replicate the key EDA findings (weekday effect, SNAP effect, zero-rate)
- Set up the wide-to-long format conversion pipeline

### Week 2: Arm A — LightGBM
- Build feature engineering pipeline (lags, rolling, price, calendar, events)
- Train per-category LightGBM models with early stopping
- Validate on last 28 days, check WRMSSE
- Baseline: compare against seasonal naive

### Week 3: Arm C — Statistical + Arm B — Deep Learning
- Implement ETS, Theta, Croston for all 30,490 series
- Train N-BEATS (or TFT) on subset (start with FOODS, 1,671 items)
- Compare all arms against the seasonal-naive baseline

### Week 4: Ensemble & Reconciliation
- Blend Arm A + Arm C + Arm B predictions with CV-tuned weights
- Apply hierarchical reconciliation (MinT)
- Submit to Kaggle M5 competition
- Iterate on the weakest categories/items

### Ongoing: Tuning
- Feature ablation: which features help most?
- Hyperparameter search: LightGBM depth, learning rate, etc.
- Try different ensemble weights
- Debug high-error items (often intermittent/low-volume ones)

---

## 7. Winning tactics (from M5 & M4 competition winners)

### 7.1 Handle intermittent demand explicitly
- For items with >90% zeros, Croston or a zero-inflated model often beats LightGBM
- Consider modeling "will it sell?" (binary classifier) + "how much if it sells?" (regression) separately

### 7.2 Use the right loss function
- Train LightGBM with `quantile` regression objective (use multiple quantiles)
- Or use a custom WRMSSE-aware objective (advanced)
- At minimum, use RMSE or Poisson deviance (handles zeros better than MSE on raw counts)

### 7.3 Ensemble diversity matters more than individual strength
- A mediocre ensemble of 4 diverse models > one very strong model
- Ensure arms are uncorrelated: statistical vs. boosting vs. deep learning captures different patterns

### 7.4 Hierarchical reconciliation is non-negotiable
- Forecasting at item-store and then averaging up loses information
- MinT reconciliation typically gives 5–10% WRMSSE improvement
- This is a "free" win — always do this

### 7.5 Price is a modest but real signal
- Include price changes as features
- Don't expect it to be the dominant driver (correlation is -0.18 to -0.26)
- Price momentum + event interaction (price drop during a holiday) can add value

### 7.6 Per-category models
- One global model across all categories often underperforms
- Train separate models per category (3 models) or per department (7 models)
- This lets each model learn category-specific patterns (FOODS = high volume, flat; HOBBIES = low volume, spiky)

### 7.7 Clipping and post-processing
- Clip predictions to [0, max_observed] (no negative or impossible-large forecasts)
- Smooth extreme outliers (very common with boosting on sparse data)
- Round to integers (sales are integers) — optional but sometimes helps WRMSSE marginally

---

## 8. Tech stack recommendations

| Task | Tool |
|---|---|
| Feature engineering | pandas, polars (for speed) |
| LightGBM | lightgbm |
| Deep learning | PyTorch + GluonTS or PyTorch Forecasting |
| Statistical | statsmodels (ETS, Theta), prophet |
| Hierarchy reconciliation | hierarchicalforecast or scikit-hts |
| Cross-validation | sklearn TimeSeriesSplit or custom |
| Experiment tracking | MLflow or Weights & Biases |
| Submission generation | pandas (format per sample_submission) |

---

## 9. Key success factors (checklist)

- [ ] **68% zeros handled** — intermittent demand strategy in place
- [ ] **No data leakage** — all features only use knowable-at-prediction-time info
- [ ] **Day-of-week features** — weekend effect captured
- [ ] **SNAP flags per state** — state-specific benefit schedules used
- [ ] **Price features** — at least price, price change, price momentum
- [ ] **Event features** — specific event types, not generic "event day" flag
- [ ] **Hierarchical reconciliation** — MinT applied to all 12 levels
- [ ] **Ensemble** — at least 3 model arms blended with CV-tuned weights
- [ ] **Per-category models** — not one global model
- [ ] **Backtested** — at least 3 rolling-origin validation windows

---

## 10. What to avoid

1. **Don't train one global model** on all 30,490 series mixed together — categories are too different
2. **Don't ignore hierarchy** — item-store forecasts should sum to store, state, total
3. **Don't use random CV** — time series must use sequential/rolling splits
4. **Don't overfit to the validation window** — test on multiple cutoff dates
5. **Don't forget to clip** at zero — negative forecasts inflate WRMSSE

---

*Strategy document for the Cognizant iPN Hackathon — NPN team. Update this file as model arms are implemented and tested.*
