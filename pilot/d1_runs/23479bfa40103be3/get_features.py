import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Unpack dataframes
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]

    # Normalize seed_time
    if not isinstance(seed_time, pd.Timestamp):
        seed_time = pd.Timestamp(seed_time)

    # Filter orders placed strictly before seed_time
    orders_before = orders[orders["order_purchase_timestamp"] < seed_time]

    # Keep only products of interest
    items_sel = order_items[order_items["product_id"].isin(entity_ids)].copy()

    # If there are no relevant order_items at all, return a zero-filled DataFrame
    if items_sel.empty:
        cols = [
            "item_count", "num_orders", "total_revenue", "total_freight",
            "avg_price", "avg_freight", "unique_sellers", "days_since_last",
            "review_mean", "review_count",
            "payment_total", "payment_count", "payment_avg", "payment_types",
            "avg_installments",
            "orders_30d", "items_30d", "revenue_30d",
            "orders_60d", "items_60d", "revenue_60d",
            "orders_90d", "items_90d", "revenue_90d",
        ]
        res = pd.DataFrame(0, index=entity_ids, columns=cols)
        res["days_since_last"] = 999   # no sales -> artificially large number
        return res

    # Merge with order information (keeps only orders that happened before seed_time)
    merged = items_sel.merge(
        orders_before[["order_id", "order_purchase_timestamp"]],
        on="order_id",
        how="inner"
    )

    # If after the merge no rows remain (no product purchased before seed_time)
    if merged.empty:
        cols = [
            "item_count", "num_orders", "total_revenue", "total_freight",
            "avg_price", "avg_freight", "unique_sellers", "days_since_last",
            "review_cnt", "review_count",
            "payment_total", "payment_count", "payment_avg", "payment_types",
            "avg_installments",
            "orders_30d", "items_30d", "revenue_30d",
            "orders_60d", "items_60d", "revenue_60d",
            "orders_90d", "items_90d", "revenue_90d",
        ]
        res = pd.DataFrame(0, index=entity_ids, columns=cols)
        res["days_since_last"] = 999
        return res

    # ---- Basic per‑product aggregates (using all historical orders) ----
    grp = merged.groupby("product_id")
    agg = grp.agg(
        item_count=("price", "size"),
        num_orders=("order_id", "nunique"),
        total_revenue=("price", "sum"),
        total_freight=("freight_value", "sum"),
        avg_price=("price", "mean"),
        avg_freight=("freight_value", "mean"),
        unique_sellers=("seller_id", "nunique"),
        last_purchase=("order_purchase_timestamp", "max"),
    )
    agg["days_since_last"] = (seed_time - agg.pop("last_purchase")).dt.days

    # ---- Reviews (linked through order_id) ----
    # Only reviews that already exist at seed_time are usable:
    # a review becomes known at review_creation_date.
    reviews_known = reviews[reviews["review_creation_date"] <= seed_time]
    product_orders = merged[["product_id", "order_id"]].drop_duplicates()
    rev = product_orders.merge(
        reviews_known[["order_id", "review_score"]], on="order_id", how="left"
    )
    rev_agg = rev.groupby("product_id").agg(
        review_mean=("review_score", "mean"),
        review_count=("review_score", "count"),   # count of non-null values
    )

    # ---- Payments (linked through order_id) ----
    # Keep only payment records already available at seed_time.
    payments_known = payments[payments["ts"] <= seed_time]
    pay = product_orders.merge(
        payments_known[["order_id", "payment_type", "payment_installments", "payment_value"]],
        on="order_id",
        how="left",
    )
    pay_agg = pay.groupby("product_id").agg(
        payment_total=("payment_value", "sum"),
        payment_count=("payment_value", "size"),
        payment_avg=("payment_value", "mean"),
        payment_types=("payment_type", "nunique"),
        avg_installments=("payment_installments", "mean"),
    )

    # ---- Features for fixed look‑back windows (30 / 60 / 90 days) ----
    win_feats = {}
    for window in [30, 60, 90]:
        mask = merged["order_purchase_timestamp"] >= seed_time - pd.Timedelta(days=window)
        sub = merged[mask]
        if len(sub) == 0:
            # No sales in this window → zeros for all products
            wdf = pd.DataFrame(index=agg.index)
            wdf[f"orders_{window}d"] = 0
            wdf[f"items_{window}d"] = 0
            wdf[f"revenue_{window}d"] = 0.0
        else:
            wg = sub.groupby("product_id").agg(
                orders=("order_id", "nunique"),
                items=("price", "size"),
                revenue=("price", "sum"),
            )
            wg.columns = [
                f"orders_{window}d",
                f"items_{window}d",
                f"revenue_{window}d",
            ]
            wdf = wg
        win_feats[window] = wdf

    # ---- Combine all parts ----
    feats = agg.join(rev_agg, how="left")
    feats = feats.join(pay_agg, how="left")
    for wdf in win_feats.values():
        feats = feats.join(wdf, how="left")

    # Reindex by the original entity_ids (ensures every requested product appears)
    feats = feats.reindex(entity_ids)

    # Fill missing values with zeros (except days_since_last, use large sentinel)
    feats = feats.fillna(0)
    feats["days_since_last"] = feats["days_since_last"].fillna(999)

    return feats