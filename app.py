"""
app.py
------
Streamlit dashboard for the Return-Abuse Ring Detector.

Tabs:
  1. Overview           - headline metrics, cost savings story
  2. Ring Explorer       - interactive network graph, click a ring to inspect it
  3. Threshold Lab        - live precision/recall/cost tradeoff as you move the slider
  4. Adversarial Test      - evasive ring that beats the graph detector, caught by a second signal
  5. Cost of Delay          - counterfactual: what earlier/later detection is worth in ₹
  6. Audit Trail              - every flagging decision, explainable and traceable

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import os

from graph_detection import (
    load_data, build_shared_attribute_graph, compute_customer_stats,
    detect_communities, score_communities, flag_rings,
)
from audit_db import log_flags, get_audit_history, init_db
from temporal_detection import compute_first_order_dates, find_tight_clusters, MIN_RETURN_RATE, BURST_WINDOW_DAYS
from counterfactual import run_checkpoints, compute_cost_of_delay

st.set_page_config(page_title="Return-Abuse Ring Detector", layout="wide", page_icon="🕸️")

COST_PER_MISSED_FRAUD = 4500
COST_PER_FALSE_POSITIVE = 350


@st.cache_data
def load_pipeline():
    customers, orders = load_data()
    G = build_shared_attribute_graph(customers)
    stats = compute_customer_stats(orders)
    partition = detect_communities(G)
    customers_scored, ring_df = score_communities(customers, stats, partition, G)
    return customers_scored, ring_df, orders


def compute_metrics(customers_scored, ring_df, threshold, min_size):
    flagged = flag_rings(ring_df, score_threshold=threshold, min_size=min_size)
    flagged_ids = set()
    for m in flagged["members"]:
        flagged_ids.update(m)
    c = customers_scored.copy()
    c["predicted_flag"] = c["customer_id"].isin(flagged_ids).astype(int)
    tp = int(((c["predicted_flag"] == 1) & (c["is_ring_member"] == 1)).sum())
    fp = int(((c["predicted_flag"] == 1) & (c["is_ring_member"] == 0)).sum())
    fn = int(((c["predicted_flag"] == 0) & (c["is_ring_member"] == 1)).sum())
    tn = int(((c["predicted_flag"] == 0) & (c["is_ring_member"] == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    cost = fp * COST_PER_FALSE_POSITIVE + fn * COST_PER_MISSED_FRAUD
    naive_cost = int(c["is_ring_member"].sum()) * COST_PER_MISSED_FRAUD
    return flagged, dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                          cost=cost, naive_cost=naive_cost, savings=naive_cost - cost)


def draw_ring_graph(G, ring_df, customers_scored, highlight_community=None):
    sub_nodes = set()
    for members in ring_df[ring_df["size"] >= 2]["members"]:
        sub_nodes.update(members)
    if not sub_nodes:
        return go.Figure()

    subG = G.subgraph(sub_nodes)
    pos = nx.spring_layout(subG, seed=42, k=0.6)

    comm_map = dict(zip(customers_scored["customer_id"], customers_scored["community_id"]))
    ring_score_map = {}
    for _, row in ring_df.iterrows():
        for m in row["members"]:
            ring_score_map[m] = row["ring_score"]

    edge_x, edge_y = [], []
    for u, v in subG.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#999"),
                             hoverinfo="none", mode="lines")

    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
    for n in subG.nodes():
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        comm = comm_map.get(n, -1)
        score = ring_score_map.get(n, 0)
        is_highlight = (highlight_community is not None and comm == highlight_community)
        node_color.append("#ff4b4b" if is_highlight else "#636efa")
        node_size.append(22 if is_highlight else 14)
        node_text.append(f"{n}<br>community: {comm}<br>ring_score: {score}")

    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers", hoverinfo="text",
                             text=node_text, marker=dict(size=node_size, color=node_color,
                                                          line=dict(width=1, color="white")))

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=500, plot_bgcolor="white")
    return fig


# ---------------- App layout ----------------

st.title("🕸️ Return-Abuse Ring Detector")
st.caption("AI Risk Manager — Razorpay Buildathon | Graph-based fraud ring detection with explainable audit trail")

customers_scored, ring_df, orders = load_pipeline()

with st.sidebar:
    st.header("Detection Settings")
    threshold = st.slider("Ring score threshold", 0.0, 1.0, 0.55, 0.01)
    min_size = st.slider("Minimum ring size", 2, 6, 3)
    st.markdown("---")
    st.markdown(
        "**How scoring works**\n\n"
        "`ring_score = 0.6 × avg_return_rate + 0.4 × density`\n\n"
        "Communities are found via Louvain clustering on a graph where edges "
        "connect customers sharing a device, address, or bank account."
    )

flagged, metrics = compute_metrics(customers_scored, ring_df, threshold, min_size)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🕸️ Ring Explorer", "🎚️ Threshold Lab",
    "🥷 Adversarial Test", "⏱️ Cost of Delay", "📜 Audit Trail",
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers analyzed", f"{len(customers_scored):,}")
    c2.metric("Rings flagged", f"{len(flagged)}")
    c3.metric("Accounts flagged", f"{sum(len(m) for m in flagged['members'])}")
    c4.metric("Estimated savings", f"₹{metrics['savings']:,.0f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{metrics['precision']:.1%}")
    c2.metric("Recall", f"{metrics['recall']:.1%}")
    c3.metric("Est. cost with detector", f"₹{metrics['cost']:,.0f}",
              delta=f"-₹{metrics['savings']:,.0f} vs flagging nobody", delta_color="normal")

    st.markdown("### Confusion Matrix")
    cm_df = pd.DataFrame(
        [[metrics["tp"], metrics["fn"]], [metrics["fp"], metrics["tn"]]],
        index=["Actually ring member", "Actually normal"],
        columns=["Flagged", "Not flagged"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.info(
        "**Why this isn't a black box:** every flagged customer belongs to a graph "
        "community with an explicit density + return-rate score. Nobody gets flagged "
        "on a single opaque model output — you can always point to *which* accounts "
        "they're connected to and *why* the cluster looks abnormal."
    )

with tab2:
    st.subheader("Suspected rings (this run)")
    if flagged.empty:
        st.warning("No rings meet the current threshold/size settings — try lowering them.")
    else:
        display_df = flagged[["community_id", "size", "avg_return_rate", "density", "ring_score", "flag_reason"]]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        selected = st.selectbox("Select a community to inspect", flagged["community_id"].tolist())
        row = flagged[flagged["community_id"] == selected].iloc[0]
        st.markdown(f"**Ring {selected}** — {row['flag_reason']}")

        G = build_shared_attribute_graph(load_data()[0])
        fig = draw_ring_graph(G, ring_df, customers_scored, highlight_community=selected)
        st.plotly_chart(fig, use_container_width=True)

        members_df = customers_scored[customers_scored["community_id"] == selected][
            ["customer_id", "name", "device_id", "address", "bank_account", "n_orders", "n_returns", "return_rate"]
        ]
        st.markdown("**Members of this ring**")
        st.dataframe(members_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Threshold sensitivity")
    st.caption("Move the sidebar slider and watch precision/recall/cost respond live.")

    rows = []
    for th in [round(x * 0.05, 2) for x in range(4, 16)]:
        f, m = compute_metrics(customers_scored, ring_df, th, min_size)
        rows.append({"threshold": th, "precision": m["precision"], "recall": m["recall"], "cost": m["cost"]})
    sweep_df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep_df["threshold"], y=sweep_df["precision"], name="Precision", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=sweep_df["threshold"], y=sweep_df["recall"], name="Recall", mode="lines+markers"))
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="current")
    fig.update_layout(height=400, yaxis_title="score", xaxis_title="ring_score threshold")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=sweep_df["threshold"], y=sweep_df["cost"], name="Estimated cost (₹)"))
    fig2.add_vline(x=threshold, line_dash="dash", line_color="red")
    fig2.update_layout(height=350, yaxis_title="₹ cost", xaxis_title="ring_score threshold")
    st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.subheader("🥷 Adversarial Test: what if fraudsters know about the graph detector?")
    st.markdown(
        "The 6 rings above all reused a device, address, or bank account across members — "
        "a real fraud ring might know that's checked, and simply **not reuse anything**. "
        "This dataset includes one such **evasive ring**: 7 accounts with completely unique "
        "device/address/bank per member. The graph detector sees them as total strangers."
    )

    customers_raw, orders_raw = load_data()
    orders_raw["order_date"] = pd.to_datetime(orders_raw["order_date"])
    evasive_ids = set(customers_scored[customers_scored["ring_id"] == "RING-EVASIVE"]["customer_id"])
    graph_caught = evasive_ids & set(
        cid for members in flagged["members"] for cid in members
    )

    c1, c2 = st.columns(2)
    c1.metric("Evasive ring members", len(evasive_ids))
    c2.metric("Caught by graph detector alone", f"{len(graph_caught)} / {len(evasive_ids)}",
               delta="the exact evasion this ring was designed for" if len(graph_caught) == 0 else None,
               delta_color="off")

    st.markdown(
        "**Their one remaining tell:** identity can be laundered, but coordinating a ring still "
        "means recruiting and activating members together. Their *first orders* cluster in a "
        f"tight {BURST_WINDOW_DAYS}-day window even though nothing else about them is shared."
    )

    already_flagged_ids = set(cid for members in flagged["members"] for cid in members)
    first_orders = compute_first_order_dates(orders_raw)
    clusters = find_tight_clusters(first_orders, already_flagged_ids)
    rate_map = dict(zip(customers_scored["customer_id"], customers_scored["return_rate"]))

    burst_rows = []
    for i, members in enumerate(clusters):
        avg_rr = sum(rate_map.get(m, 0) for m in members) / len(members)
        if avg_rr >= MIN_RETURN_RATE:
            burst_rows.append({"burst_cluster_id": f"BURST-{i}", "size": len(members),
                                "avg_return_rate": round(avg_rr, 3), "members": members})
    burst_df = pd.DataFrame(burst_rows)

    if burst_df.empty:
        st.warning("No burst clusters found at current settings.")
    else:
        temporal_caught = set()
        for m in burst_df["members"]:
            temporal_caught.update(m)
        c1, c2, c3 = st.columns(3)
        c1.metric("Burst clusters found", len(burst_df))
        c2.metric("Caught by temporal layer", f"{len(evasive_ids & temporal_caught)} / {len(evasive_ids)}")
        c3.metric("Incidental false positives", len(temporal_caught - evasive_ids))

        st.dataframe(burst_df[["burst_cluster_id", "size", "avg_return_rate"]], use_container_width=True, hide_index=True)
        st.info(
            "**Honest result, not a cherry-picked one:** the temporal layer catches most — not "
            "all — of the evasive ring, and picks up a couple of unrelated customers who happened "
            "to order in the same window by chance. That's the real tradeoff of a second, weaker "
            "signal: it recovers some of what the primary detector misses, at the cost of some "
            "extra noise — which is exactly why this stays a *second, gated* layer feeding review, "
            "not an auto-block rule on its own."
        )

    st.caption(
        "Defense-only, by design: neither layer takes any autonomous action on an account. "
        "Both only produce a flagged community + reason for a human/queue to review."
    )

with tab5:
    st.subheader("⏱️ Cost of Delay: what earlier detection is worth")
    st.markdown(
        "Precision and recall describe *whether* a ring gets caught. They say nothing about "
        "*when*. This replays the detector at 10-day checkpoints through the transaction history "
        "to find the earliest point each ring would have crossed the flagging threshold, then "
        "prices the gap between 'ring starts' and 'ring gets flagged'."
    )

    if st.button("Run counterfactual simulation"):
        with st.spinner("Replaying detection at historical checkpoints..."):
            first_flagged_at, orders_cf, customers_cf = run_checkpoints()
            cost_df = compute_cost_of_delay(first_flagged_at, orders_cf, customers_cf)
        st.session_state["cost_df"] = cost_df

    cost_df = st.session_state.get("cost_df")
    if cost_df is not None:
        display_df = cost_df.rename(columns={
            "ring_id": "Ring", "size": "Size", "flagged_date": "Flagged on",
            "fraudulent_returns_before_detection": "Returns before flag",
            "loss_before_detection": "Loss before detection (₹)",
            "loss_prevented_by_flagging": "Loss prevented by flagging (₹)",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        c1.metric("Total loss incurred before detection", f"₹{cost_df['loss_before_detection'].sum():,.0f}")
        c2.metric("Total loss prevented by flagging", f"₹{cost_df['loss_prevented_by_flagging'].sum():,.0f}")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=cost_df["ring_id"], y=cost_df["loss_before_detection"],
                              name="Loss before detection", marker_color="#ff4b4b"))
        fig.add_trace(go.Bar(x=cost_df["ring_id"], y=cost_df["loss_prevented_by_flagging"],
                              name="Loss prevented", marker_color="#00cc96"))
        fig.update_layout(barmode="stack", height=400, yaxis_title="₹")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "The naive rings get flagged within ~2 weeks of forming — most of the potential loss "
            "is prevented. The evasive ring is never caught by this (graph-only) view, which is "
            "exactly the gap the temporal layer in the Adversarial Test tab exists to close."
        )
    else:
        st.caption("Click the button above to run the checkpoint replay (takes a few seconds).")

with tab6:
    st.subheader("Audit Trail")
    st.caption("Every flag decision is written to SQLite with a plain-English reason — click 'Log this run' to persist it.")

    if st.button("Log current run to audit DB"):
        log_flags(flagged, threshold_used=threshold)
        st.success(f"Logged {len(flagged)} ring flags to audit_trail.db")

    cols, rows = get_audit_history()
    if rows:
        hist_df = pd.DataFrame(rows, columns=cols)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No audit records yet — click the button above to log this run.")

    st.markdown("---")
    st.markdown("**Trace a specific customer**")
    cust_id = st.text_input("Customer ID (e.g. C00123)")
    if cust_id:
        from audit_db import get_customer_history
        hist = get_customer_history(cust_id)
        if hist:
            st.write(pd.DataFrame(hist, columns=["customer_id", "community_id", "reason", "flagged_at"]))
        else:
            st.write("No flag history for this customer.")
