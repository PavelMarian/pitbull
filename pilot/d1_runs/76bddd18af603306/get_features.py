def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    orders = db["orders"]
    items = db["order_items"]
    reviews = db["reviews"]

    seed_time = pd.Timestamp(seed_time)
    past_items = items[items["ts"] < seed_time].copy()

    result_index = pd.Index(entity_ids)
    result = pd.DataFrame(index=result_index)

    if len(past_items) == 0:
        zero_cols = [
            "total_items_sold", "total_orders_sold", "total_customers",
            "avg_price", "avg_freight", "min_price", "max_price", "std_price",
            "total_price", "total_revenue", "days_since_last", "span_days",
            "sales_last_30d", "orders_last_30d",
            "sales_last_60d", "orders_last_60d",
            "sales_last_90d", "orders_last_90d",
            "sales_last_180d", "orders_last_180d",
            "avg_review_score", "review_count", "avg_interval_days",
            "share_last_90", "had_sales", "sales_per_day"
        ]
        for col in zero_cols:
            result[col] = 0
        return result

    orders_sub = orders[["order_id", "customer_id"]]
    merged = past_items.merge(orders_sub, on="order_id", how="left")
    merged["price_freight"] = merged["price"] + merged["freight_value"]

    grp = merged.groupby("product_id").agg(
        total_items_sold=("price_freight", "count"),
        total_orders_sold=("order_id", "nunique"),
        total_customers=("customer_id", "nunique"),
        avg_price=("price", "mean"),
        avg_freight=("freight_value", "mean"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        std_price=("price", "std"),
        total_price=("price", "sum"),
        total_revenue=("price_freight", "sum"),
        first_ts=("ts", "min"),
        last_ts=("ts", "max"),
    )

    for window in [30, 60, 90, 180]:
        start = seed_time - pd.Timedelta(days=window)
        mask = (merged["ts"] >= start) & (merged["ts"] < seed_time)
        win = merged[mask]
        grp[f"sales_last_{window}d"] = win.groupby("product_id").size()
        grp[f"orders_last_{window}d"] = win.groupby("product_id")["order_id"].nunique()

    review_joined = past_items[["order_id", "product_id"]].merge(
        reviews[["order_id", "review_score"]], on="order_id", how="left"
    )
    # Отзыв появляется в базе позже покупки, в момент review_creation_date.
    # Берём только отзывы, уже существующие на seed_time.
    past_reviews = reviews[reviews["review_creation_date"] < seed_time]
    review_joined = past_items[["order_id", "product_id"]].merge(
        past_reviews[["order_id", "review_score"]], on="order_id", how="left"
    )
    if len(review_joined) > 0:
        scored = review_joined.dropna(subset=["review_score"])
        rev_agg = scored.groupby("product_id").agg(
            avg_review_score=("review_score", "mean"),
            review_count=("review_score", "count"),
        )
        rev_agg = rev_agg.reindex(grp.index)
    else:
        rev_agg = pd.DataFrame(
            columns=["avg_review_score", "review_count"], index=grp.index
        )
    grp = grp.join(rev_agg, how="left")

    grp["days_since_last"] = (seed_time - grp["last_ts"]).dt.days
    grp["span_days"] = (grp["last_ts"] - grp["first_ts"]).dt.days

    def avg_interval(s):
        s = s.drop_duplicates().sort_values()
        if len(s) < 2:
            return np.nan
        diffs = s.diff().dropna().dt.days
        return diffs.mean()

    grp["avg_interval_days"] = merged.groupby("product_id")["ts"].apply(
        lambda x: avg_interval(x)
    )

    grp["sales_per_day"] = grp["total_orders_sold"] / grp["span_days"]
    grp["sales_per_day"] = grp["sales_per_day"].replace([np.inf, -np.inf], np.nan)
    grp["share_last_90"] = grp["orders_last_90d"] / grp["total_orders_sold"]

    result = grp.reindex(result_index).fillna(0)
    result["had_sales"] = result["total_orders_sold"] > 0

    return result