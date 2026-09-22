def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]

    feature_cols = [
        "order_count", "unique_orders", "unique_customers", "unique_sellers",
        "total_price", "avg_price", "std_price", "max_price", "min_price",
        "total_freight", "avg_freight",
        "cnt_30", "cnt_60", "cnt_90",
        "delivered_share",
        "avg_review", "std_review", "total_reviews",
        "avg_installments", "avg_payment_value",
        "credit_share", "debit_share", "voucher_share",
        "last_purchase_days", "first_purchase_days", "avg_daily_orders",
    ]

    # Только позиции, купленные строго до seed_time
    product_set = set(entity_ids)
    mask = order_items["product_id"].isin(product_set) & (order_items["ts"] < seed_time)
    items = order_items.loc[mask, ["order_id", "product_id", "seller_id", "price", "freight_value", "ts"]].copy()

    if items.empty:
        result = pd.DataFrame(index=entity_ids)
        for c in feature_cols:
            result[c] = 0.0
        return result

    # Заказы: нужны customer_id и дата доставки (факт доставки на момент seed_time)
    orders_sub = orders[["order_id", "customer_id", "order_delivered_customer_date"]]
    items = items.merge(orders_sub, on="order_id", how="left")

    # Отзывы: доступны только те, что уже созданы к моменту seed_time
    rev = reviews[reviews["review_creation_date"] < seed_time]
    rev_agg = rev.groupby("order_id").agg(
        rev_count=("review_id", "count"),
        rev_avg=("review_score", "mean"),
        rev_std=("review_score", "std"),
    )

    # Платежи проходят в момент покупки, а покупки уже отфильтрованы по ts < seed_time
    pay_agg = payments.groupby("order_id").agg(
        install_mean=("payment_installments", "mean"),
        total_pay=("payment_value", "sum"),
        credit_share=("payment_type", lambda s: (s == "credit_card").mean()),
        debit_share=("payment_type", lambda s: (s == "debit_card").mean()),
        voucher_share=("payment_type", lambda s: (s == "voucher").mean()),
    )

    items = items.merge(pay_agg, on="order_id", how="left")
    items = items.merge(rev_agg, on="order_id", how="left")

    # Окна активности
    items["cnt_30"] = (items["ts"] >= seed_time - pd.Timedelta(days=30)).astype(int)
    items["cnt_60"] = (items["ts"] >= seed_time - pd.Timedelta(days=60)).astype(int)
    items["cnt_90"] = (items["ts"] >= seed_time - pd.Timedelta(days=90)).astype(int)

    # Доставлен = дата доставки уже наступила к seed_time (статус из выгрузки не используем)
    delivered = items["order_delivered_customer_date"]
    items["is_delivered"] = (delivered.notna() & (delivered <= seed_time)).astype(int)

    g = items.groupby("product_id")
    feat = g.agg(
        order_count=("order_id", "count"),
        unique_orders=("order_id", "nunique"),
        unique_customers=("customer_id", "nunique"),
        unique_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        avg_price=("price", "mean"),
        std_price=("price", "std"),
        max_price=("price", "max"),
        min_price=("price", "min"),
        total_freight=("freight_value", "sum"),
        avg_freight=("freight_value", "mean"),
        cnt_30=("cnt_30", "sum"),
        cnt_60=("cnt_60", "sum"),
        cnt_90=("cnt_90", "sum"),
        delivered_share=("is_delivered", "mean"),
        avg_review=("rev_avg", "mean"),
        std_review=("rev_std", "mean"),
        total_reviews=("rev_count", "sum"),
        avg_installments=("install_mean", "mean"),
        avg_payment_value=("total_pay", "mean"),
        credit_share=("credit_share", "mean"),
        debit_share=("debit_share", "mean"),
        voucher_share=("voucher_share", "mean"),
        max_ts=("ts", "max"),
        min_ts=("ts", "min"),
    )

    last_purchase_days = (seed_time - feat["max_ts"]) / pd.Timedelta(days=1)
    first_purchase_days = (seed_time - feat["min_ts"]) / pd.Timedelta(days=1)
    feat["last_purchase_days"] = last_purchase_days
    feat["first_purchase_days"] = first_purchase_days
    feat["avg_daily_orders"] = feat["order_count"] / first_purchase_days.clip(lower=1)
    feat = feat.drop(columns=["max_ts", "min_ts"])

    result = pd.DataFrame(index=entity_ids)
    result = result.join(feat, how="left")
    result = result.fillna(0.0)
    result = result.reindex(entity_ids)
    return result
