"""Freeze the A3/RQ2a detection corpus before any model calls."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
A2 = HERE / "a2_full_manifest.json"
OUT = HERE / "a3_manifest.json"


def uid(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


HANDWRITTEN = [
    {
        "pair_id": "olist_seller_value_availability",
        "provenance": "rel/adapters/olist.py: programs()",
        "context": """A flat Olist event table has row time ts. review_score becomes known at review_ts; late and delay_days become known at deliv_ts. Features are computed at seed for sellers.""",
        "leak": """def features(db, seed, sellers):
    h = db['flat']
    h = h[(h.ts <= seed) & h.seller_id.isin(sellers)]
    return h.groupby('seller_id').agg({'price':'mean', 'review_score':'mean', 'late':'mean'}).reindex(sellers)""",
        "clean": """def features(db, seed, sellers):
    h = db['flat']
    h = h[(h.ts <= seed) & h.seller_id.isin(sellers)].copy()
    h.loc[~(h.review_ts <= seed), 'review_score'] = np.nan
    h.loc[~(h.deliv_ts <= seed), ['late']] = np.nan
    return h.groupby('seller_id').agg({'price':'mean', 'review_score':'mean', 'late':'mean'}).reindex(sellers)""",
    },
    {
        "pair_id": "olist_product_join_availability",
        "provenance": "rel/fix_c.py: nbr_features()",
        "context": """Olist rows have row time ts; review_score and late become known only at review_ts and deliv_ts. For each product, features aggregate the full history of its associated seller at seed.""",
        "leak": """def seller_features(db, seed, product_ids, product_seller):
    h = db['flat'][db['flat'].ts <= seed]
    s = h.groupby('seller_id').agg({'price':'sum', 'review_score':'mean', 'late':'mean'})
    sellers = product_seller.reindex(product_ids)
    out = s.reindex(sellers.values); out.index = product_ids
    return out""",
        "clean": """def seller_features(db, seed, product_ids, product_seller):
    h = db['flat'][db['flat'].ts <= seed].copy()
    h.loc[~(h.review_ts <= seed), 'review_score'] = np.nan
    h.loc[~(h.deliv_ts <= seed), 'late'] = np.nan
    s = h.groupby('seller_id').agg({'price':'sum', 'review_score':'mean', 'late':'mean'})
    sellers = product_seller.reindex(product_ids)
    out = s.reindex(sellers.values); out.index = product_ids
    return out""",
    },
    {
        "pair_id": "olist_delivery_date",
        "provenance": "pilot/pilot_bc_density_canary.py: delivery_date_feature()",
        "context": """orders.order_purchase_timestamp is row time. order_delivered_customer_date is filled only when delivery occurs. Product features are computed at seed_time.""",
        "leak": """def get_features(db, entity_ids, seed_time):
    o = db['orders'][db['orders'].order_purchase_timestamp <= seed_time]
    x = db['order_items'].merge(o[['order_id','order_purchase_timestamp','order_delivered_customer_date']], on='order_id')
    x['delivery_days'] = (x.order_delivered_customer_date-x.order_purchase_timestamp).dt.days
    return x.groupby('product_id').delivery_days.mean().to_frame().reindex(entity_ids)""",
        "clean": """def get_features(db, entity_ids, seed_time):
    o = db['orders'][(db['orders'].order_purchase_timestamp <= seed_time) & (db['orders'].order_delivered_customer_date <= seed_time)]
    x = db['order_items'].merge(o[['order_id','order_purchase_timestamp','order_delivered_customer_date']], on='order_id')
    x['delivery_days'] = (x.order_delivered_customer_date-x.order_purchase_timestamp).dt.days
    return x.groupby('product_id').delivery_days.mean().to_frame().reindex(entity_ids)""",
    },
    {
        "pair_id": "rel_f1_outcome_availability",
        "provenance": "rel/adapters/f1.py: programs()",
        "context": """In a Formula 1 table, ts is race start. position, points and dnf become known at avail_ts, strictly after the start. Features are computed at seed.""",
        "leak": """def features(db, seed, drivers):
    h = db['flat']
    h = h[(h.ts <= seed) & h.driverId.isin(drivers)]
    return h.groupby('driverId').agg({'position':'mean','points':'sum','dnf':'mean'}).reindex(drivers)""",
        "clean": """def features(db, seed, drivers):
    h = db['flat']
    h = h[(h.ts <= seed) & h.driverId.isin(drivers)].copy()
    h.loc[~(h.avail_ts <= seed), ['position','points','dnf']] = np.nan
    return h.groupby('driverId').agg({'position':'mean','points':'sum','dnf':'mean'}).reindex(drivers)""",
    },
    {
        "pair_id": "featuretools_cutoff",
        "provenance": "prestudy/p1_featuretools.py: run_dfs() D1/D3",
        "context": """Featuretools builds features for entities at per-row prediction times in cutoff_table. Events at or after a prediction time must not contribute. es_with_lti has last-time indexes.""",
        "leak": """def build_features(es, target, cutoff_table):
    fm, defs = ft.dfs(entityset=es, target_dataframe_name=target,
                      agg_primitives=AGG, max_depth=2)
    return fm""",
        "clean": """def build_features(es_with_lti, target, cutoff_table):
    fm, defs = ft.dfs(entityset=es_with_lti, target_dataframe_name=target,
                      agg_primitives=AGG, max_depth=2,
                      cutoff_time=cutoff_table, include_cutoff_time=False)
    return fm""",
    },
    {
        "pair_id": "sql_event_cutoff",
        "provenance": "prestudy/test_p3_sql_oracle.py: CORRECT_Q/LEAKY_Q",
        "context": """eval_table contains UserId and a per-row seed_time. events contains UserId, EventId and EventTime. The feature is the number of events known before each seed_time.""",
        "leak": """SELECT e.UserId, COUNT(ev.EventId) AS n_events
FROM eval_table e LEFT JOIN events ev ON ev.UserId = e.UserId
GROUP BY e.UserId""",
        "clean": """SELECT e.UserId, COUNT(ev.EventId) AS n_events
FROM eval_table e LEFT JOIN events ev
  ON ev.UserId = e.UserId AND ev.EventTime < e.seed_time
GROUP BY e.UserId""",
    },
]


def main():
    a2 = json.loads(A2.read_text(encoding="utf-8"))
    items = []
    for p in a2["programs"]:
        if p.get("status") != "clean":
            continue
        pair = "a2-" + p["program_uid"]
        context = "Olist product-demand feature code. " + (
            "Tables and task follow prestudy/p3_baseline_run.py NEUTRAL_PROMPT. "
            "Each table field is usable only when it is available at seed_time."
        )
        for truth, key in (("LEAK", "source_code"), ("CLEAN", "current_code")):
            code = p[key]
            items.append({"item_id": uid(pair + truth + code), "pair_id": pair,
                          "category": "checker_pair", "truth": truth,
                          "context": context, "code": code,
                          "provenance": f"pilot/a2_full_manifest.json:{p['program_uid']}"})
    for pair in HANDWRITTEN:
        for truth, key in (("LEAK", "leak"), ("CLEAN", "clean")):
            code = pair[key]
            items.append({"item_id": uid(pair["pair_id"] + truth + code),
                          "pair_id": pair["pair_id"], "category": "handwritten_pair",
                          "truth": truth, "context": pair["context"], "code": code,
                          "provenance": pair["provenance"]})
    if len({x["item_id"] for x in items}) != len(items):
        raise ValueError("duplicate item id")
    manifest = {
        "experiment": "A3 RQ2a LLM code-reading detection",
        "frozen_before_calls": True,
        "model": "z-ai/glm-5.3-flash",
        "arms": ["F0", "F1"],
        "primary_arm": "F1",
        "decision_rule": {"checker_pair_recall_min": 0.9,
                          "checker_pair_false_positive_rate_max": 0.1,
                          "invalid_response_policy": "conservatively wrong"},
        "category_meaning": {
            "handwritten_pair": "truth by construction; small precision illustration",
            "checker_pair": "A2 leaking/repaired-and-verified pairs; checker agreement, not precision",
        },
        "items": items,
    }
    OUT.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print("items", len(items), "checker pairs", sum(x["category"] == "checker_pair" for x in items)//2,
          "handwritten pairs", sum(x["category"] == "handwritten_pair" for x in items)//2)


if __name__ == "__main__":
    main()
