"""
graph_detection.py
-------------------
Core detection engine.

Idea: build a graph where each customer is a node, and we draw an edge
between two customers if they share a device_id, address, or bank_account.
Real fraud rings show up as small dense clusters (because they reuse a
small pool of devices/addresses/accounts) that ALSO have an abnormally
high return rate. Two independent signals -> combined ring_score.

This is deliberately NOT a black-box row-by-row classifier: it looks at
*relationships between customers*, which is what a naive per-transaction
model misses.
"""

import os
import pandas as pd
import networkx as nx
from itertools import combinations
import community as community_louvain  # python-louvain


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_data():
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    return customers, orders


def build_shared_attribute_graph(customers: pd.DataFrame) -> nx.Graph:
    """Edge between two customers if they share device_id, address, or bank_account."""
    G = nx.Graph()
    G.add_nodes_from(customers["customer_id"])

    for attr in ["device_id", "address", "bank_account"]:
        groups = customers.groupby(attr)["customer_id"].apply(list)
        for _, members in groups.items():
            if len(members) < 2:
                continue
            # cap combinations for very large shared groups (e.g. generic addresses)
            members = members[:25]
            for a, b in combinations(members, 2):
                if G.has_edge(a, b):
                    G[a][b]["shared_attrs"] += 1
                else:
                    G.add_edge(a, b, shared_attrs=1)
    return G


def compute_customer_stats(orders: pd.DataFrame) -> pd.DataFrame:
    stats = orders.groupby("customer_id").agg(
        n_orders=("order_id", "count"),
        n_returns=("returned", "sum"),
        total_spend=("amount", "sum"),
    ).reset_index()
    stats["return_rate"] = (stats["n_returns"] / stats["n_orders"]).round(3)
    return stats


def detect_communities(G: nx.Graph) -> dict:
    """Louvain community detection -> {customer_id: community_id}."""
    if G.number_of_edges() == 0:
        return {n: -1 for n in G.nodes()}
    partition = community_louvain.best_partition(G, weight="shared_attrs", random_state=42)
    return partition


def score_communities(customers, stats, partition, G):
    """
    For every community, compute:
      - size
      - avg return rate of its members
      - density (how tightly connected -> stronger sharing signal)
    ring_score = normalized blend of return_rate and density, only for
    communities of size >= 2 (a size-1 'community' can't be a ring).
    """
    customers = customers.merge(stats, on="customer_id", how="left").fillna(
        {"n_orders": 0, "n_returns": 0, "total_spend": 0, "return_rate": 0}
    )
    customers["community_id"] = customers["customer_id"].map(partition)

    rows = []
    for comm_id, group in customers.groupby("community_id"):
        if comm_id == -1 or len(group) < 2:
            continue
        members = list(group["customer_id"])
        sub = G.subgraph(members)
        possible_edges = len(members) * (len(members) - 1) / 2
        density = sub.number_of_edges() / possible_edges if possible_edges > 0 else 0
        avg_return_rate = group["return_rate"].mean()
        avg_orders = group["n_orders"].mean()

        # simple, explainable scoring — not a black box
        ring_score = round(0.6 * avg_return_rate + 0.4 * density, 3)

        rows.append({
            "community_id": comm_id,
            "size": len(members),
            "avg_return_rate": round(avg_return_rate, 3),
            "density": round(density, 3),
            "avg_orders_per_member": round(avg_orders, 2),
            "ring_score": ring_score,
            "members": members,
        })

    ring_df = pd.DataFrame(rows).sort_values("ring_score", ascending=False).reset_index(drop=True)
    return customers, ring_df


def flag_rings(ring_df: pd.DataFrame, score_threshold: float = 0.45, min_size: int = 3):
    """Apply a threshold to decide which communities get flagged as suspected rings."""
    flagged = ring_df[(ring_df["ring_score"] >= score_threshold) & (ring_df["size"] >= min_size)].copy()
    flagged["flag_reason"] = flagged.apply(
        lambda r: f"{r['size']} accounts share device/address/bank (density={r['density']}), "
                  f"avg return rate {r['avg_return_rate']*100:.0f}%",
        axis=1,
    )
    return flagged


def run_pipeline(score_threshold: float = 0.55, min_size: int = 3):
    customers, orders = load_data()
    G = build_shared_attribute_graph(customers)
    stats = compute_customer_stats(orders)
    partition = detect_communities(G)
    customers_scored, ring_df = score_communities(customers, stats, partition, G)
    flagged = flag_rings(ring_df, score_threshold, min_size)
    return {
        "graph": G,
        "customers": customers_scored,
        "orders": orders,
        "ring_df": ring_df,
        "flagged": flagged,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print(f"Graph: {result['graph'].number_of_nodes()} nodes, {result['graph'].number_of_edges()} edges")
    print(f"Communities of size >= 2: {(result['ring_df']['size'] >= 2).sum()}")
    print(f"\nTop suspected rings:\n{result['flagged'][['community_id','size','avg_return_rate','density','ring_score']]}")
