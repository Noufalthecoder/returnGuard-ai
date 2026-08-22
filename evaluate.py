"""
evaluate.py
-----------
Honest metrics against the planted ground truth (is_ring_member).
This is what Razorpay explicitly asks for: precision, recall, and the
COST of false positives — not just an accuracy number.
"""

from graph_detection import run_pipeline

# Assumed business costs (tune these — they're the whole point of the
# false-positive-cost analysis, not a throwaway number)
COST_PER_MISSED_FRAUD = 4500      # avg loss when a real ring account is NOT flagged
COST_PER_FALSE_POSITIVE = 350     # support/investigation cost + trust damage when a
                                   # genuine customer is wrongly flagged


def evaluate():
    result = run_pipeline()
    customers = result["customers"]
    flagged = result["flagged"]

    flagged_member_ids = set()
    for members in flagged["members"]:
        flagged_member_ids.update(members)

    customers["predicted_flag"] = customers["customer_id"].isin(flagged_member_ids).astype(int)

    tp = ((customers["predicted_flag"] == 1) & (customers["is_ring_member"] == 1)).sum()
    fp = ((customers["predicted_flag"] == 1) & (customers["is_ring_member"] == 0)).sum()
    fn = ((customers["predicted_flag"] == 0) & (customers["is_ring_member"] == 1)).sum()
    tn = ((customers["predicted_flag"] == 0) & (customers["is_ring_member"] == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    total_cost = fp * COST_PER_FALSE_POSITIVE + fn * COST_PER_MISSED_FRAUD
    naive_cost = customers["is_ring_member"].sum() * COST_PER_MISSED_FRAUD  # cost of flagging nobody

    print("=== Confusion Matrix ===")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print("\n=== Metrics ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print("\n=== Cost Analysis (assumed ₹{}/missed ring account, ₹{}/false positive) ===".format(
        COST_PER_MISSED_FRAUD, COST_PER_FALSE_POSITIVE))
    print(f"Estimated cost with detector: ₹{total_cost:,.0f}")
    print(f"Cost if we flagged nobody:    ₹{naive_cost:,.0f}")
    print(f"Estimated savings:            ₹{naive_cost - total_cost:,.0f}")

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "total_cost": total_cost, "naive_cost": naive_cost,
    }


def threshold_sweep():
    """Shows the precision/recall/cost tradeoff across thresholds — proves
    the default wasn't picked arbitrarily."""
    from graph_detection import load_data, build_shared_attribute_graph, compute_customer_stats, \
        detect_communities, score_communities, flag_rings

    customers, orders = load_data()
    G = build_shared_attribute_graph(customers)
    stats = compute_customer_stats(orders)
    partition = detect_communities(G)
    customers_scored, ring_df = score_communities(customers, stats, partition, G)

    print("\n=== Threshold sweep ===")
    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'cost (₹)':>12}")
    for th in [0.35, 0.45, 0.50, 0.55, 0.60, 0.70]:
        flagged = flag_rings(ring_df, score_threshold=th, min_size=3)
        flagged_ids = set()
        for m in flagged["members"]:
            flagged_ids.update(m)
        c = customers_scored.copy()
        c["predicted_flag"] = c["customer_id"].isin(flagged_ids).astype(int)
        tp = ((c["predicted_flag"] == 1) & (c["is_ring_member"] == 1)).sum()
        fp = ((c["predicted_flag"] == 1) & (c["is_ring_member"] == 0)).sum()
        fn = ((c["predicted_flag"] == 0) & (c["is_ring_member"] == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        cost = fp * COST_PER_FALSE_POSITIVE + fn * COST_PER_MISSED_FRAUD
        print(f"{th:>10} {precision:>10.3f} {recall:>10.3f} {cost:>12,.0f}")


if __name__ == "__main__":
    evaluate()
    threshold_sweep()
