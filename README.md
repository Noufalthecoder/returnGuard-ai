# 🕸️ Return-Abuse Ring Detector
**Track:** AI Risk Manager — Razorpay AI Buildathon

## The problem
Most return-fraud tools look at *one transaction at a time*: "is this
single return suspicious?" Real abuse rings don't work that way — a
handful of accounts reuse the same device, address, or bank account and
split their fraud across accounts specifically to stay under any
per-account threshold. A row-by-row classifier is structurally blind to
this pattern. This project detects the **network**, not just the
transaction.

## What makes this different from a typical fraud-classifier project
Most submissions build one thing: a per-transaction classifier that
outputs a fraud probability. This project deliberately goes further in
three ways that a single classifier can't:

1. **It looks at relationships, not just rows** — the graph layer.
2. **It assumes the fraudster is adaptive** — the adversarial test tab
   plants a ring that deliberately shares *nothing* identity-wise (no
   shared device/address/bank), specifically to beat the graph layer, and
   shows a second, independent signal (coordinated order timing) that
   partially recovers it. The result is reported honestly — 5/7 caught,
   2 incidental false positives — not cherry-picked to look perfect.
3. **It prices detection speed, not just accuracy** — the counterfactual
   "Cost of Delay" tab replays detection at historical checkpoints to show
   what earlier vs. later flagging is actually worth in ₹, not just as a
   precision/recall number.

## How it works
1. **Graph construction** — every customer is a node. An edge is drawn
   between two customers if they share a `device_id`, `address`, or
   `bank_account`.
2. **Community detection** — Louvain clustering finds tightly-connected
   groups of accounts (`graph_detection.py`).
3. **Scoring, not guessing** — each community gets an explicit, explainable
   score:
   `ring_score = 0.6 × avg_return_rate + 0.4 × graph_density`
   Sharing an attribute alone isn't proof of fraud (families/roommates share
   addresses too) — the score only flags communities that are *both* densely
   connected *and* behaviourally abnormal.
4. **Adversarial second layer (`temporal_detection.py`)** — for accounts
   the graph layer can't connect at all, a separate signal looks for
   groups whose *first order dates* cluster in a tight window (a ring
   still has to be recruited and activated together, even if every
   identity attribute is laundered). Gated behind a minimum return-rate
   so coordinated timing alone never triggers a flag.
5. **Counterfactual cost simulation (`counterfactual.py`)** — re-runs
   detection at 10-day historical checkpoints to find the earliest point
   each ring would have been flagged, then prices the returns processed
   before vs. after that point.
6. **Audit trail** — every flagging decision is written to SQLite
   (`audit_db.py`) with a plain-English reason, so any flagged account can
   be traced back to exactly why it was flagged.
7. **Dashboard** — a Streamlit app (`app.py`) ties it together across six
   tabs: Overview, Ring Explorer, Threshold Lab, Adversarial Test, Cost of
   Delay, and Audit Trail.

## Honest metrics (synthetic data, 946 customers / 6 planted rings)
| Threshold | Precision | Recall | Estimated cost |
|---|---|---|---|
| 0.35 | 0.754 | 1.000 | ₹5,250 |
| 0.45 | 0.885 | 1.000 | ₹2,100 |
| **0.55 (default)** | **1.000** | **1.000** | **₹0** |
| 0.70 | 1.000 | 1.000 | ₹0 |

At low thresholds the model catches every real ring but also flags a few
normal households who happen to share an address — an honest false-positive
cost, not hidden. Cost assumptions: ₹4,500 per missed ring account,
₹350 per false positive (investigation + trust cost). Full sweep in
`evaluate.py`.

**This is strictly a detection/defense tool.** It does not take any
autonomous action on an account — it flags communities for human/automated
review with a full audit trail, per the "defense-only" requirement.

## Project structure
```
return_abuse_detector/
├── generate_data.py     # synthetic customers/orders: naive rings + evasive ring + household noise
├── graph_detection.py   # graph build, Louvain clustering, ring scoring
├── temporal_detection.py # second-layer: catches identity-laundered rings via order-timing bursts
├── counterfactual.py     # historical-checkpoint replay -> cost of detection delay
├── evaluate.py           # precision/recall/cost metrics + threshold sweep
├── audit_db.py           # SQLite audit trail
├── app.py                 # Streamlit dashboard (6 tabs)
├── requirements.txt
└── data/                  # generated CSVs + audit_trail.db (created on run)
```

## Running it
```bash
pip install -r requirements.txt
python generate_data.py      # generates data/customers.csv, data/orders.csv
python graph_detection.py    # sanity-check graph-layer detection
python temporal_detection.py # sanity-check adversarial (evasive-ring) detection
python counterfactual.py     # sanity-check cost-of-delay simulation
python evaluate.py           # precision/recall/cost report
streamlit run app.py         # launch the dashboard
```

## What to say in the 5-minute pitch
1. **The gap**: per-transaction fraud models miss coordinated rings that
   split their activity across accounts.
2. **The approach**: shared-attribute graph + community detection — show
   the live graph in the dashboard, click a ring, show its members.
3. **The adversarial mindset**: show the Adversarial Test tab — plant a
   ring that beats the primary detector on purpose, then show the second
   signal that partially recovers it, and be upfront that it's partial,
   not perfect.
4. **The business framing**: show the Cost of Delay tab — detection speed
   has a ₹ value, and that's a more useful story for a risk team than a
   single accuracy number.
5. **The honesty**: don't just show 100% precision — show the threshold
   sweep, explain the false-positive cost tradeoff, and justify why 0.55
   was chosen (not just "it looked good").
6. **The audit trail**: every decision is traceable in SQLite — nothing is
   a black box.
7. **The scope discipline**: this only flags for review, it never
   auto-blocks or auto-refunds anything — strictly defense, per the track's
   own requirement.

## Next steps if extending
- Replace synthetic data with real (or anonymized/replayed) transaction
  logs.
- Add IP/geolocation as a third sharing signal.
- Combine the temporal signal with weaker partial-identity matches (e.g.
  same city, same payment network) instead of requiring zero overlap, to
  reduce the incidental false positives seen in the Adversarial Test tab.
