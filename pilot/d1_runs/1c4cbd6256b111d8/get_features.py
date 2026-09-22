def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    if not isinstance(seed_time, pd.Timestamp):
        seed_time = pd.Timestamp(seed_time)

    orders = db["orders"]
    items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]

    features = pd.DataFrame(index=entity_ids)

    # базовые колонки
    defaults = {
        "count_orders": 0,
        "count_items": 0,
        "count_sellers": 0,
        "total_price": 0.0,
        "total_freight": 0.0,
        "avg_price": 0.0,
        "avg_freight": 0.0,
        "std_price": 0.0,
        "std_freight": 0.0,
        "min_price": 0.0,
        "max_price": 0.0,
        "days_since_first": 0,
        "days_since_last": 0,
        "avg_items_per_order": 0.0,
        "has_sales": 0,
    }
    for k, v in defaults.items():
        features[k] = v

    for h in (30, 90):
        features[f"orders_last_{h}"] = 0
        features[f"items_last_{h}"] = 0

    features["review_count"] = 0
    features["avg_review_score"] = 0.0
    features["payment_count"] = 0

    # Покупки товаров до seed_time
    items_with_ts = items.merge(
        orders[["order_id", "order_purchase_timestamp"]],
        on="order_id",
        how="inner"
    )
    items_past = items_with_ts[
        (items_with_ts["order_purchase_timestamp"] < seed_time) &
        (items_with_ts["product_id"].isin(entity_ids))
    ]

    # Платежи: только строки, которые уже существуют на момент seed_time
    payments_past = payments
    if "ts" in payments_past.columns:
        payments_past = payments_past[payments_past["ts"] <= seed_time]

    payment_types = (
        payments_past["payment_type"].unique()
        if "payment_type" in payments_past.columns
        else []
    )
    for pt in payment_types:
        features[f"payment_type_{pt}"] = 0

    if len(items_past) > 0:
        g = items_past.groupby("product_id")

        cnt_orders = g["order_id"].nunique()
        cnt_items = g["order_id"].size()
        cnt_sellers = g["seller_id"].nunique()
        total_price = g["price"].sum()
        total_freight = g["freight_value"].sum()
        avg_price = g["price"].mean()
        avg_freight = g["freight_value"].mean()
        std_price = g["price"].std(ddof=0)
        std_freight = g["freight_value"].std(ddof=0)
        min_price = g["price"].min()
        max_price = g["price"].max()
        first_date = g["order_purchase_timestamp"].min()
        last_date = g["order_purchase_timestamp"].max()

        features.loc[cnt_orders.index, "has_sales"] = 1
        features.loc[cnt_orders.index, "count_orders"] = cnt_orders
        features.loc[cnt_items.index, "count_items"] = cnt_items
        features.loc[cnt_sellers.index, "count_sellers"] = cnt_sellers
        features.loc[total_price.index, "total_price"] = total_price
        features.loc[total_freight.index, "total_freight"] = total_freight
        features.loc[avg_price.index, "avg_price"] = avg_price
        features.loc[avg_freight.index, "avg_freight"] = avg_freight
        features.loc[std_price.index, "std_price"] = std_price
        features.loc[std_freight.index, "std_freight"] = std_freight
        features.loc[min_price.index, "min_price"] = min_price
        features.loc[max_price.index, "max_price"] = max_price
        features.loc[first_date.index, "days_since_first"] = (seed_time - first_date).dt.days
        features.loc[last_date.index, "days_since_last"] = (seed_time - last_date).dt.days

        items_per_order = cnt_items / cnt_orders
        features.loc[items_per_order.index, "avg_items_per_order"] = items_per_order

        # Недавние продажи
        for h in (30, 90):
            cutoff = seed_time - pd.Timedelta(days=h)
            recent = items_past[items_past["order_purchase_timestamp"] >= cutoff]
            if len(recent) > 0:
                rg = recent.groupby("product_id")
                r_orders = rg["order_id"].nunique()
                r_items = rg["order_id"].size()
                features.loc[r_orders.index, f"orders_last_{h}"] = r_orders
                features.loc[r_items.index, f"items_last_{h}"] = r_items

    # Отзывы: только отзывы, созданные до seed_time
    if "review_creation_date" in reviews.columns:
        reviews_past = reviews[reviews["review_creation_date"] <= seed_time]
    else:
        reviews_past = reviews

    mapping = items_past[["product_id", "order_id"]].drop_duplicates()
    if len(mapping) > 0 and len(reviews_past) > 0:
        joined = reviews_past.merge(mapping, on="order_id", how="inner")
        if len(joined) > 0:
            rev_agg = joined.groupby("product_id").agg(
                review_count=("review_score", "count"),
                avg_review_score=("review_score", "mean")
            )
            features.loc[rev_agg.index, "review_count"] = rev_agg["review_count"]
            features.loc[rev_agg.index, "avg_review_score"] = rev_agg["avg_review_score"]

    # Платежи: merge только с заказами, купленными до seed_time
    if len(mapping) > 0 and len(payments_past) > 0:
        pay_merged = payments_past.merge(mapping, on="order_id", how="inner")
        if len(pay_merged) > 0:
            pay_count = pay_merged.groupby("product_id").size()
            features.loc[pay_count.index, "payment_count"] = pay_count

            for pt in payment_types:
                sub = pay_merged[pay_merged["payment_type"] == pt]
                if len(sub) > 0:
                    ptype_count = sub.groupby("product_id").size()
                    features.loc[ptype_count.index, f"payment_type_{pt}"] = ptype_count

    features = features.reindex(entity_ids)
    features = features.fillna(0)
    return features
