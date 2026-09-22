def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    orders = db["orders"]
    items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]

    seed_time = pd.Timestamp(seed_time)
    product_ids = list(entity_ids)

    # ----- 历史订单（种子时刻之前） -----
    orders_before = orders[orders["order_purchase_timestamp"] < seed_time]

    # 空数据集, 直接返回全 0 特征
    if orders_before.empty:
        columns = [
            "num_items_total", "num_orders_total", "total_price", "avg_price",
            "std_price", "total_freight", "avg_freight", "avg_review_score",
            "num_reviews", "avg_total_paid", "avg_installments",
            "avg_credit_card_share", "recency_days", "avg_days_ago",
            "items_7d", "items_30d", "items_60d", "items_90d"
        ]
        return pd.DataFrame(0.0, index=product_ids, columns=columns)

    # 将订单时间附到 item 上
    order_time = orders_before[["order_id", "order_purchase_timestamp"]].rename(
        columns={"order_purchase_timestamp": "purchase_time"}
    )
    items_needed = items[["order_id", "product_id", "price", "freight_value"]]

    hist = items_needed.merge(order_time, on="order_id", how="inner")

    # 只保留需要预测的 product
    hist = hist[hist["product_id"].isin(product_ids)]
    if hist.empty:
        columns = [
            "num_items_unique", "num_orders_total", "total_price", "avg_price",
            "std_price", "total_freight", "avg_freight", "avg_review_score",
            "num_reviews", "avg_total_paid", "avg_installments",
            "avg_credit_card_share", "recency_days", "avg_days_ago",
            "items_7d", "items_30d", "items_60d", "items_90d"
        ]
        return pd.DataFrame(0.0, index=product_ids, columns=columns)

    # 加入距 seed_time 的天数
    hist["days_ago"] = (seed_time - hist["purchase_time"]).dt.days

    # 时间窗口指示器
    for window in [7, 30, 60, 90]:
        hist[f"in_{window}d"] = (hist["days_ago"] <= window).astype(int)

    # 每个订单的平均评分和评论数
    # 只使用 seed_time 之前已经创建的评论
    if not reviews.empty:
        reviews_before = reviews[reviews["review_creation_date"] < seed_time]
        order_rev = reviews_before.groupby("order_id")["review_score"].agg(
            ["mean", "count"]
        ).reset_index()
        order_rev.columns = ["order_id", "order_avg_score", "order_num_reviews"]
        hist = hist.merge(order_rev, on="order_id", how="left")
    else:
        hist["order_avg_score"] = np.nan
        hist["order_num_reviews"] = 0

    # 每个订单的支付概况
    if not payments.empty:
        pay = payments[["order_id", "payment_type", "payment_installments", "payment_value"]].copy()
        pay["is_credit"] = (pay["payment_type"] == "credit_card").astype(int)
        order_pay = pay.groupby("order_id").agg(
            order_total_paid=("payment_value", "sum"),
            order_avg_installments=("payment_installments", "mean"),
            order_credit_share=("is_credit", "mean"),
            order_num_payments=("payment_value", "size")
        ).reset_index()
        hist = hist.merge(order_pay, on="order_id", how="left")
    else:
        for col in ["order_total_paid", "order_avg_installments", "order_credit_share", "order_num_payments"]:
            hist[col] = np.nan

    # --------- 聚合特征 ----------
    grouped = hist.groupby("product_id")

    features = pd.DataFrame({
        "num_items_unique": grouped["price"].count(),
        "num_orders_total": grouped["order_id"].nunique(),
        "total_price": grouped["price"].sum(),
        "avg_price": grouped["price"].mean(),
        "std_price": grouped["price"].std().fillna(0.0),
        "total_freight": grouped["freight_value"].sum(),
        "avg_freight": grouped["freight_value"].mean(),
        "avg_review_score": grouped["order_avg_score"].mean().fillna(0.0),
        "num_reviews": grouped["order_num_reviews"].sum().fillna(0),
        "avg_total_paid": grouped["order_total_paid"].mean().fillna(0.0),
        "avg_installments": grouped["order_avg_installments"].mean().fillna(0.0),
        "avg_credit_card_share": grouped["order_credit_share"].mean().fillna(0.0),
        "recency_days": grouped["days_ago"].min(),
        "avg_days_ago": grouped["days_ago"].mean(),
        "items_7d": grouped["in_7d"].sum(),
        "items_30d": grouped["in_30d"].sum(),
        "items_60d": grouped["in_60d"].sum(),
        "items_90d": grouped["in_90d"].sum(),
    })

    # 处理缺失商品（完全没有历史）
    features = features.reindex(product_ids, fill_value=-1.0)

    # 不同的缺失值处理：历史长度类 -> 0，时间相关 -> 9999
    for col in features.columns:
        if col in ["recency_days", "avg_days_ago"]:
            features[col] = features[col].replace(-1.0, 9999.0)
        else:
            features[col] = features[col].replace(-1.0, 0.0)

    # 消除可能的其他 NaN（例如单样本 std）
    return features.fillna(0.0)
