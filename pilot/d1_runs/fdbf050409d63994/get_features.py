def get_features(db, entity_ids, seed_time):
    import pandas as pd
    import numpy as np

    if not isinstance(seed_time, pd.Timestamp):
        seed_time = pd.Timestamp(seed_time)

    features = pd.DataFrame(index=entity_ids)

    # -----------------------------------------------------------------
    # 1. Filter order_items by purchase time (<= seed_time)
    oi = db["order_items"].copy()
    oi = oi[oi["ts"] <= seed_time]

    if len(oi) == 0:
        # No history for any product → return all zero features
        return features

    # -----------------------------------------------------------------
    # 2. Merge reviews: only reviews already created by seed_time
    reviews = db["reviews"][["order_id", "review_score", "review_creation_date"]]
    reviews = reviews[reviews["review_creation_date"] <= seed_time]
    reviews = reviews[["order_id", "review_score"]]
    oi = oi.merge(reviews, on="order_id", how="left")

    # -----------------------------------------------------------------
    # 3. Add time windows
    oi["days_diff"] = (seed_time - oi["ts"]).dt.days
    oi["last_7"]   = (oi["days_diff"] <= 7)   & (oi["days_diff"] >= 0)
    oi["last_30"]  = (oi["days_diff"] <= 30)  & (oi["days_diff"] >= 0)
    oi["last_60"]  = (oi["days_diff"] <= 60)  & (oi["days_diff"] >= 0)
    oi["last_90"]  = (oi["days_diff"] <= 90)  & (oi["days_diff"] >= 0)

    # -----------------------------------------------------------------
    # 4. Item-level aggregation (per product)
    item_agg = oi.groupby("product_id").agg(
        n_orders        = ("order_id", "nunique"),
        n_items         = ("price", "count"),
        total_price     = ("price", "sum"),
        avg_price       = ("price", "mean"),
        median_price    = ("price", "median"),
        std_price       = ("price", "std"),
        total_freight   = ("freight_value", "sum"),
        avg_freight     = ("freight_value", "mean"),
        std_freight     = ("freight_value", "std"),
        n_sellers       = ("seller_id", "nunique"),
        avg_review      = ("review_score", "mean"),
        n_reviews       = ("review_score", "count"),
        cnt_last_7      = ("last_7", "sum"),
        cnt_last_30     = ("last_30", "sum"),
        cnt_last_60     = ("last_60", "sum"),
        cnt_last_90     = ("last_90", "sum"),
        last_order_ts   = ("ts", "max"),
        first_order_ts  = ("ts", "min"),
    ).reset_index()

    # Recency features
    item_agg["days_since_last"] = (seed_time - item_agg["last_order_ts"]).dt.days
    item_agg["days_since_first"] = (seed_time - item_agg["first_order_ts"]).dt.days
    item_agg.drop(columns=["last_order_ts", "first_order_ts"], inplace=True)

    # Average items per order (only where n_orders > 0) – replace inf with 0
    item_agg["avg_items_per_order"] = item_agg["n_items"] / item_agg["n_orders"].replace(0, np.nan)
    item_agg["avg_items_per_order"] = item_agg["avg_items_per_order"].fillna(0)

    # -----------------------------------------------------------------
    # 5. Order-level payment features
    # A payment exists for the model only if the order was approved by
    # seed_time; where approval is missing fall back to the payment row's
    # own purchase timestamp.
    order_prod = oi[["order_id", "product_id"]].drop_duplicates()

    orders = db["orders"][["order_id", "order_approved_at"]]
    payments = db["payments"][["order_id", "payment_value", "ts"]]
    payments = payments.merge(orders, on="order_id", how="left")
    avail = payments["order_approved_at"].fillna(payments["ts"])
    payments = payments[avail <= seed_time]
    payments = payments[["order_id", "payment_value"]]

    pay_by_order = payments.groupby("order_id")["payment_value"].sum().reset_index().rename(
        columns={"payment_value": "order_total_payment"}
    )
    order_prod = order_prod.merge(pay_by_order, on="order_id", how="left")

    pay_agg = order_prod.groupby("product_id").agg(
        n_orders_with_payment = ("order_total_payment", "count"),
        sum_order_payment     = ("order_total_payment", "sum"),
        avg_payment_per_order = ("order_total_payment", "mean"),
    ).reset_index()

    # -----------------------------------------------------------------
    # 6. Merge & reindex
    features = item_agg.merge(pay_agg, on="product_id", how="left")
    features.set_index("product_id", inplace=True)

    # Reindex to entity_ids and fill missing/NaN
    features = features.reindex(entity_ids, fill_value=0)
    features = features.fillna(0)

    return features
