"""
counterfactual.py
------------------
"What if we had detected this ring earlier / later?"

Re-runs the graph detection pipeline at successive historical checkpoints
(using only orders that existed up to that date) to find the EARLIEST
point each planted ring would have crossed the flagging threshold. Then
translates detection delay into ₹ — every return processed by a ring
member before it gets flagged is treated as a loss that earlier detection
would have prevented.

This is what turns "we have 100% precision" into a business argument:
speed of detection has a ₹ value, not just accuracy.
"""

import pandas as pd
from graph_detection import (
    load_data, build_shared_attribute_graph, detect_communities,
    score_communities, flag_rings,
)

SCORE_THRESHOLD = 0.55
MIN_SIZE = 3
CHECKPOINT_STEP_DAYS = 10


def compute_stats_up_to(orders: pd.DataFrame, cutoff) -> pd.DataFrame:
    subset = orders[orders["order_date"] <= cutoff]
    stats = subset.groupby("customer_id").agg(
        n_orders=("order_id", "count"),
        n_returns=("returned", "sum"),
        total_spend=("amount", "sum"),
    ).reset_index()
    stats["return_rate"] = (stats["n_returns"] / stats["n_orders"]).round(3)
    return stats


def run_checkpoints():
    customers, orders = load_data()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    G = build_shared_attribute_graph(customers)  # attribute graph doesn't change over time
    partition = detect_communities(G)

    start = orders["order_date"].min()
    end = orders["order_date"].max()
    checkpoints = pd.date_range(start, end, freq=f"{CHECKPOINT_STEP_DAYS}D")

    ring_ids = [r for r in customers["ring_id"].dropna().unique() if r]
    first_flagged_at = {r: None for r in ring_ids}

    for cp in checkpoints:
        stats_cp = compute_stats_up_to(orders, cp)
        customers_scored, ring_df = score_communities(customers, stats_cp, partition, G)
        flagged = flag_rings(ring_df, SCORE_THRESHOLD, MIN_SIZE)

        flagged_members = set()
        for m in flagged["members"]:
            flagged_members.update(m)

        for r in ring_ids:
            if first_flagged_at[r] is not None:
                continue
            ring_members = set(customers[customers["ring_id"] == r]["customer_id"])
            # consider a ring "caught" once at least half its members are in a flagged community
            if len(ring_members & flagged_members) >= max(2, len(ring_members) // 2):
                first_flagged_at[r] = cp

    return first_flagged_at, orders, customers


def compute_cost_of_delay(first_flagged_at, orders, customers):
    """
    For each ring, sum the ₹ value of fraudulent RETURNS its members
    processed before the ring was (or would have been) flagged. If never
    flagged within the observed window, all its returns count as loss.
    """
    rows = []
    for ring_id, flagged_date in first_flagged_at.items():
        members = customers[customers["ring_id"] == ring_id]["customer_id"]
        ring_orders = orders[orders["customer_id"].isin(members) & (orders["returned"] == 1)]

        if flagged_date is not None:
            pre_detection = ring_orders[ring_orders["order_date"] < flagged_date]
            post_detection = ring_orders[ring_orders["order_date"] >= flagged_date]
        else:
            pre_detection = ring_orders
            post_detection = ring_orders.iloc[0:0]

        loss_before = pre_detection["amount"].sum()
        prevented = post_detection["amount"].sum()

        rows.append({
            "ring_id": ring_id,
            "size": len(members),
            "flagged_date": flagged_date.date() if flagged_date is not None else "not caught",
            "fraudulent_returns_before_detection": len(pre_detection),
            "loss_before_detection": round(loss_before, 0),
            "loss_prevented_by_flagging": round(prevented, 0),
        })
    return pd.DataFrame(rows).sort_values("loss_before_detection", ascending=False)


if __name__ == "__main__":
    first_flagged_at, orders, customers = run_checkpoints()
    df = compute_cost_of_delay(first_flagged_at, orders, customers)
    print(df.to_string(index=False))
    print(f"\nTotal loss incurred before detection across all rings: ₹{df['loss_before_detection'].sum():,.0f}")
    print(f"Total loss prevented by flagging (had action been taken immediately): ₹{df['loss_prevented_by_flagging'].sum():,.0f}")
