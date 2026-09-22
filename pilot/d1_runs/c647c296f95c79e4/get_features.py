def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    order_items = db["order_items"].copy()
    reviews = db["reviews"].copy()
    payments = db["payments"].copy()

    # Convert seed_time to datetime if needed
    if not isinstance(seed_time, pd.Timestamp):
        seed_time = pd.Timestamp(seed_time)
    entity_ids = list(entity_ids)

    # Only items of orders purchased strictly before seed_time are known at
    # seed_time; order_items.ts is the purchase moment of the order.
    items_before = order_items[order_items["ts"] < seed_time]
    items_before = items_before[items_before["product_id"].isin(entity_ids)]

    feature_frame = pd.DataFrame(index=entity_ids)

    # 1. Number of purchases (total items) in the last 30, 60, 90, 180 days before seed_time
    for window in [30, 60, 90, 180]:
        cutoff = seed_time - pd.Timedelta(days=window)
        temp = items_before[items_before["ts"] >= cutoff]
        counts = temp.groupby("product_id").size().rename(f"sales_last_{window}d")
        feature_frame[f"sales_last_{window}d"] = counts.reindex(entity_ids).fillna(0)

    # 2. Total quantity sold and total revenue up to seed_time
    if len(items_before) > 0:
        temp_sum = items_before.groupby("product_id").agg(
            total_sold=("order_item_id", "count"),
            total_revenue=("price", "sum"),
            avg_price=("price", "mean"),
            total_freight=("freight_value", "sum"),
            avg_freight=("freight_value", "mean"),
        )
        for col in ["total_sold", "total_revenue", "avg_price", "total_freight", "avg_freight"]:
            feature_frame[col] = temp_sum[col].reindex(entity_ids).fillna(0)
    else:
        for col in ["total_sold", "total_revenue", "avg_price", "total_freight", "avg_freight"]:
            feature_frame[col] = 0

    # 3. Number of distinct sellers selling this product (ever before seed_time)
    sellers = items_before.groupby("product_id")["seller_id"].nunique().rename("distinct_sellers")
    feature_frame["distinct_sellers"] = sellers.reindex(entity_ids).fillna(0)

    # 4. Number of distinct orders containing the product
    distinct_orders = items_before.groupby("product_id")["order_id"].nunique().rename("distinct_orders")
    feature_frame["distinct_orders"] = distinct_orders.reindex(entity_ids).fillna(0)

    # Orders containing the product (all purchased before seed_time)
    prod_orders = items_before[["product_id", "order_id"]].drop_duplicates()

    # 5-6. Reviews: a review becomes known only when it is created
    # (review_creation_date), which can be long after the purchase, so keep
    # only reviews created up to seed_time.
    reviews_known = reviews[reviews["review_creation_date"] <= seed_time]
    rev_merged = prod_orders.merge(reviews_known[["order_id", "review_score"]], on="order_id", how="left")
    avg_review = rev_merged.groupby("product_id")["review_score"].mean().rename("avg_review_score")
    feature_frame["avg_review_score"] = avg_review.reindex(entity_ids).fillna(0)
    rev_count = rev_merged.groupby("product_id")["review_score"].count().rename("review_count")
    feature_frame["review_count"] = rev_count.reindex(entity_ids).fillna(0)

    # 7. Payment info: a payment row is known only from its ts, keep only
    # payments with ts <= seed_time.
    payments_known = payments[payments["ts"] <= seed_time]
    pay_orders = prod_orders.merge(
        payments_known[["order_id", "payment_installments", "payment_value"]], on="order_id", how="left"
    )
    if len(pay_orders) > 0:
        pay_agg = pay_orders.groupby("product_id").agg(
            avg_installments=("payment_installments", "mean"),
            avg_payment_value=("payment_value", "mean"),
            total_payment_value=("payment_value", "sum"),
        )
        for col in ["avg_installments", "avg_payment_value", "total_payment_value"]:
            feature_frame[col] = pay_agg[col].reindex(entity_ids).fillna(0)
    else:
        for col in ["avg_installments", "avg_payment_value", "total_payment_value"]:
            feature_frame[col] = 0

    # 8. Time since first purchase of this product (in days) up to seed_time
    if len(items_before) > 0:
        first_purchase = items_before.groupby("product_id")["ts"].min().rename("first_purchase_ts")
        feature_frame["first_purchase_ts"] = first_purchase.reindex(entity_ids)
        feature_frame["days_since_first_purchase"] = (seed_time - feature_frame["first_purchase_ts"]).dt.days.fillna(seed_time.toordinal())
    else:
        feature_frame["days_since_first_purchase"] = 0

    # 9. Recency: how many days since last purchase (for product) up to seed_time
    if len(items_before) > 0:
        last_purchase = items_before.groupby("product_id")["ts"].max().rename("last_purchase_ts")
        feature_frame["last_purchase_ts"] = last_purchase.reindex(entity_ids)
        feature_frame["days_since_last_purchase"] = (seed_time - feature_frame["last_purchase_ts"]).fillna(pd.Timedelta(days=0)).dt.days
    else:
        feature_frame["days_since_last_purchase"] = 0

    # Drop helper columns
    drop_cols = ["first_purchase_ts", "last_purchase_ts"]
    feature_frame = feature_frame.drop(columns=drop_cols, errors="ignore")

    # Ensure all columns numeric, fill any remaining NaNs with 0
    feature_frame = feature_frame.fillna(0)

    # Reindex to entity_ids and return
    feature_frame = feature_frame.reindex(entity_ids)
    return feature_frame
