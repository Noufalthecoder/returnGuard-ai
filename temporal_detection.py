"""
temporal_detection.py
----------------------
ADVERSARIAL-RESISTANT LAYER.

graph_detection.py catches rings that reuse device/address/bank accounts.
But a ring that KNOWS this check exists will simply use a different
device, address, and bank account per member ("identity laundering").
To the shared-attribute graph, laundered members look like total
strangers — zero edges, zero signal.

What's much harder to launder: WHEN accounts are used. Rings are
recruited and activated together, so their *first* orders tend to land
in an unnaturally tight time window. This module finds groups of
customers — with NO shared device/address/bank — whose first orders
still cluster suspiciously close together in time, and treats that as
an independent second detection signal.

This is intentionally a separate, simple, explainable model (not a
fancier version of graph_detection) — the point is layered defense:
different evasion tactics get caught by different signals.
"""

import pandas as pd
import networkx as nx
from itertools import combinations

BURST_WINDOW_DAYS = 3        # first orders within this many days = suspicious
MIN_CLUSTER_SIZE = 4         # need at least this many co-bursting customers
MIN_RETURN_RATE = 0.4        # burst alone isn't enough — must ALSO have abnormal returns


def compute_first_order_dates(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    first_orders = orders.groupby("customer_id")["order_date"].min().reset_index()
    first_orders.columns = ["customer_id", "first_order_date"]
    return first_orders


def build_burst_graph(first_orders: pd.DataFrame, already_flagged_ids: set) -> nx.Graph:
    """
    Kept for visualization purposes (pairwise "close in time" edges), but
    NOTE: connected components of this graph over-chain transitively
    (A~B~C can span far more than BURST_WINDOW_DAYS end-to-end). The actual
    cluster detection uses find_tight_clusters() below, which enforces
    that every reported group's full date SPAN (max - min) stays within
    BURST_WINDOW_DAYS, not just consecutive gaps.
    """
    candidates = first_orders[~first_orders["customer_id"].isin(already_flagged_ids)].copy()
    candidates = candidates.sort_values("first_order_date").reset_index(drop=True)

    G = nx.Graph()
    G.add_nodes_from(candidates["customer_id"])
    dates = candidates["first_order_date"].tolist()
    ids = candidates["customer_id"].tolist()
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            gap = (dates[j] - dates[i]).days
            if gap > BURST_WINDOW_DAYS:
                break
            G.add_edge(ids[i], ids[j], gap_days=gap)
    return G


def find_tight_clusters(first_orders: pd.DataFrame, already_flagged_ids: set):
    """
    Proper burst detection: greedy sliding window over sorted first-order
    dates. A cluster is only reported if its FULL span (last date - first
    date) fits within BURST_WINDOW_DAYS — this avoids the "telephone game"
    problem where loosely-chained pairs (A~B~C~D...) get transitively
    merged into one meaningless giant group spanning months.
    """
    candidates = first_orders[~first_orders["customer_id"].isin(already_flagged_ids)].copy()
    candidates = candidates.sort_values("first_order_date").reset_index(drop=True)
    ids = candidates["customer_id"].tolist()
    dates = candidates["first_order_date"].tolist()

    clusters = []
    left = 0
    n = len(dates)
    while left < n:
        right = left
        while right + 1 < n and (dates[right + 1] - dates[left]).days <= BURST_WINDOW_DAYS:
            right += 1
        group = ids[left:right + 1]
        if len(group) >= MIN_CLUSTER_SIZE:
            clusters.append(group)
            left = right + 1  # move past this group, no overlap
        else:
            left += 1  # slide forward by one and try again
    return clusters


def run_temporal_pipeline(customers, orders, already_flagged_ids):
    first_orders = compute_first_order_dates(orders)
    clusters = find_tight_clusters(first_orders, already_flagged_ids)
    G = build_burst_graph(first_orders, already_flagged_ids)  # kept for visualization

    rate_map = dict(zip(customers["customer_id"], customers["return_rate"]))
    rows = []
    for i, members in enumerate(clusters):
        avg_return_rate = sum(rate_map.get(m, 0) for m in members) / len(members)
        if avg_return_rate < MIN_RETURN_RATE:
            continue
        rows.append({
            "burst_cluster_id": f"BURST-{i}",
            "size": len(members),
            "avg_return_rate": round(avg_return_rate, 3),
            "members": members,
            "flag_reason": (
                f"{len(members)} accounts with NO shared device/address/bank all placed "
                f"their first order within a {BURST_WINDOW_DAYS}-day window, and average "
                f"{avg_return_rate*100:.0f}% return rate — consistent with a coordinated "
                f"ring using laundered identities."
            ),
        })
    burst_df = pd.DataFrame(rows)
    return G, burst_df


if __name__ == "__main__":
    from graph_detection import run_pipeline

    result = run_pipeline()
    customers = result["customers"]
    orders = result["orders"]

    flagged_ids = set()
    for m in result["flagged"]["members"]:
        flagged_ids.update(m)

    G, burst_df = run_temporal_pipeline(customers, orders, flagged_ids)

    print(f"Burst graph: {G.number_of_nodes()} candidate nodes, {G.number_of_edges()} temporal edges")
    print(f"\nBurst clusters found (missed by graph-only detection):")
    if burst_df.empty:
        print("  none")
    else:
        print(burst_df[["burst_cluster_id", "size", "avg_return_rate", "flag_reason"]].to_string())

        # sanity check against ground truth
        evasive_ids = set(customers[customers["ring_id"] == "RING-EVASIVE"]["customer_id"])
        caught = set()
        for m in burst_df["members"]:
            caught.update(m)
        print(f"\nGround truth evasive ring members: {len(evasive_ids)}")
        print(f"Caught by temporal layer: {len(evasive_ids & caught)} / {len(evasive_ids)}")
