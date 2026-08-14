# M5 Retail Forecasting Study

**Before we build any forecasting model, we need to understand our data.** This document explains, in plain language, everything we found about the Walmart M5 dataset — what the data looks like, what patterns exist, and what challenges we face.

---

## What is this project about?

**The problem:** Walmart stocks thousands of products across 10 stores. Every day they must decide how much of each product to stock. Too little = empty shelves. Too much = wasted money. Our goal: predict how many units of each product will sell at each store over the next 28 days.

**The data:** The [M5 Forecasting Dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy), released by Walmart for a Kaggle competition. It contains 5+ years of real (anonymized) daily sales data.

---

## What's inside the data?

| File | What it contains |
|---|---|
| `sales_train_validation.csv` | Daily sales for each product at each store (Jan 2011 – Jun 2016) |
| `calendar.csv` | Calendar: dates, holidays, special events, SNAP days |
| `sell_prices.csv` | Selling price of each product at each store, tracked weekly |
| `sample_submission.csv` | Template showing the required prediction format |
| `sales_train_evaluation.csv` | Same as main sales file, plus 28 extra days (for scoring) |

---

## How big is the data?

| Measure | Value |
|---|---|
| Unique products | 3,049 |
| Stores | 10 (4 in California, 3 in Texas, 3 in Wisconsin) |
| Product × Store combinations | 30,490 |
| Days of history | 1,913 (~5.2 years) |
| Total sales records | ~58.3 million |

This is a genuinely large problem — 30,490 individual product-store series, each needing its own forecast.

---

## Is the data clean?

| File | Missing values |
|---|---|
| Sales records | 0 — every record exists |
| Prices | 0 — every price is present |
| Calendar | ~7,542 — expected, ordinary days simply have blank event columns |

**Good news:** No data repair needed. We can go straight to analysis.

---

## Key findings

### 1. Weekends sell ~35% more than weekdays

| Day | Average units sold |
|---|---|
| Monday | 32,853 |
| Tuesday | 30,369 |
| Wednesday | 30,010 |
| Thursday | 30,205 |
| Friday | 34,226 |
| **Saturday** | **41,547** |
| **Sunday** | **41,130** |

**What this means:** People shop more on weekends. Any forecasting model that ignores the day of the week will be badly wrong.

---

### 2. Monthly patterns are mild (not a big factor)

| Month | Avg daily units |
|---|---|
| January | 33,832 |
| April | 34,259 |
| June | 35,001 |
| **August (peak)** | **35,947** |
| **May & December (dip)** | **32,504 / 32,980** |

**What this means:** Month-to-month differences are small. The big driver is weekly rhythm (weekends), not seasonal monthly patterns.

---

### 3. Special events don't boost sales — on average

| Day type | Average units sold |
|---|---|
| Ordinary day (no event) | 34,489 |
| Event day | 32,655 |

**What this means:** Counter-intuitively, event days average *slightly lower* sales. This is because some events (like Christmas Day) mean store closures. The takeaway: a generic "event day" flag is misleading. Specific events (Super Bowl, Labor Day) do boost sales, so we'd need event-specific data — not a blanket flag.

---

### 4. SNAP days boost sales by ~8–10% consistently

SNAP = US government food-assistance program. Each state has its own schedule of benefit days.

| State | Non-SNAP avg | SNAP avg | Uplift |
|---|---|---|---|
| California | 33,409 | 36,242 | +8.5% |
| Texas | 33,263 | 36,538 | +9.8% |
| Wisconsin | 33,240 | 36,585 | +10.0% |

**What this means:** People buy more groceries when they have SNAP benefits. This is a real, reliable signal — the effect is nearly identical across all three states. We should include a SNAP flag as a model feature.

---

### 5. The data hierarchy is perfect — zero mismatches

We checked: do all store totals within a state add up exactly to that state's total?

| State | State total | Sum of stores | Difference |
|---|---|---|---|
| California | 28,675,547 | 28,675,547 | 0 |
| Texas | 18,899,006 | 18,899,006 | 0 |
| Wisconsin | 18,120,856 | 18,120,856 | 0 |

**What this means:** The numbers add up at every level. We can safely aggregate or break down forecasts without data inconsistency.

---

### 6. 68.2% of all records are ZERO — this is the core challenge

Out of 58.3 million product-store-day records:

- **68.2%** show zero units sold
- The busiest product sells almost every day (0.16% zeros)
- The rarest product sells on only 0.37% of days (99.63% zeros)
- The typical product (median) sells on only about 26% of days

**What this means:** This is the single most important finding. Most everyday prediction methods assume smooth, continuous data. Here, most records are zero with occasional small spikes. This is known as "intermittent demand" — a hard problem. Our model must be designed to handle this from the start, or it will fail.

---

### 7. Category breakdown: FOODS dominates

| Category | Total units sold | Share | Avg price |
|---|---|---|---|
| FOODS | 45,089,939 | ~69% | $3.25 |
| HOUSEHOLD | 14,480,670 | ~22% | $5.47 |
| HOBBIES | 6,124,800 | ~9% | $5.33 |

**What this means:** Groceries (FOODS) are the bulk of the business — cheap, bought often, with stable patterns. HOBBIES and HOUSEHOLD are smaller and sparser. The model should expect its best performance on FOODS and be evaluated separately for each category.

---

### 8. Price has a weak effect on sales

| Category | Price → demand correlation | Meaning |
|---|---|---|
| FOODS | -0.18 | Pricier food items sell *slightly* less |
| HOBBIES | -0.19 | Similar weak tendency |
| HOUSEHOLD | -0.26 | Clearer (but still moderate): pricier items sell less |

**What this means:** Cheaper items do sell slightly more, but price is a modest factor — not a dominant driver. It should be one input among several, not the primary signal.

---

### 9. Promotions exist, but they're rare

| Price characteristic | Finding |
|---|---|
| Items that never change price | ~85% of product-store combinations |
| Items with at least one price change | ~15% show real price drops |
| Maximum price drop observed | Up to 70% off the highest price |

**What this means:** Most items have stable prices. But the 15% that do change price — especially with steep drops (50–70%) — are genuine promotions. A promotion indicator will help specific items but won't move the needle for most products.

---

### 10. Product hierarchy: 3 categories → 7 departments → 3,049 items

| Category | Departments | Item count (approx.) |
|---|---|---|
| FOODS | food_1, food_2, food_3 | ~1,671 items |
| HOBBIES | hobbies_1, hobbies_2 | ~369 items |
| HOUSEHOLD | household_1, household_2, household_3 | ~1,009 items |

**What this means:** The data is nested — items belong to departments, departments to categories, all replicated across 10 stores. Forecasts at each level need to add up correctly (hierarchical forecasting). FOODS has dense data; HOBBIES and HOUSEHOLD are sparse.

---

## So what does this all mean for the model?

| Finding | Modeling implication |
|---|---|
| **68% of records are zero** | Must handle intermittent demand — standard methods will fail |
| **Weekends sell 35% more** | Must include day-of-week as a feature |
| **SNAP days boost sales 8–10%** | Include SNAP flag — reliable signal |
| **Event days are misleading** | Use event-specific flags, not a generic "event" flag |
| **Hierarchy adds up perfectly** | Reconcile forecasts across levels (item → store → state) |
| **FOODS dominates; HOBBIES/HOUSEHOLD are sparse** | Evaluate performance per category, not just overall |
| **Price has weak effect (-0.18 to -0.26)** | Include price as one input, don't over-rely on it |
| **Promotions exist but rare (~15%)** | Useful for specific items; not a universal signal |
| **Data is clean — zero missing values** | No cleaning needed; go straight to modeling |

---

## The bottom line

This dataset is **clean** but **hard**. The biggest risk is treating it like a smooth, predictable problem when in reality **most of it is zeros**. The model must be designed from the ground up to handle intermittent demand, while capturing the strong weekly and SNAP signals that clearly exist. Everything else — promotions, price effects, event flags — is useful refinement on top of those core requirements.

---

## Repository structure

```
m5-retail-forecasting-study/
├── README.md          ← You are here (study report)
├── data/              ← Dataset files (not uploaded — download from Kaggle)
│   ├── sales_train_validation.csv
│   ├── calendar.csv
│   ├── sell_prices.csv
│   ├── sample_submission.csv
│   └── sales_train_evaluation.csv
└── notebooks/         ← (Coming soon — EDA and model code)
```

## Getting the data

Download the M5 Forecasting - Accuracy dataset from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place the CSV files in the `data/` folder.

---

*For the Cognizant iPN Hackathon. This README covers Phase 1: data understanding. Model building and evaluation follow in subsequent phases.*
