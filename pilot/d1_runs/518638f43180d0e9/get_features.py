def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    seed_time = pd.Timestamp(seed_time)

    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]

    # Point-in-time: каждая таблица усекается по времени появления строки.
    # Признаки строятся только по данным, которые уже существуют на seed_time.
    orders = orders[orders["order_purchase_timestamp"] <= seed_time]
    items = order_items[order_items["ts"] <= seed_time]
    pays = payments[payments["ts"] <= seed_time]
    revs = reviews[reviews["review_creation_date"] <= seed_time]

    PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card", "not_defined"]

    features = pd.DataFrame(index=pd.Index(list(entity_ids)))

    # ----- Заказы и позиции -----

    g_it = items.groupby("product_id")
    features["item_count"] = g_it.size()
    features["order_count"] = g_it["order_id"].nunique()
    features["seller_count"] = g_it["seller_id"].nunique()
    features["price_sum"] = g_it["price"].sum()
    features["price_mean"] = g_it["price"].mean()
    features["price_std"] = g_it["price"].std()
    features["price_max"] = g_it["price"].max()
    features["freight_sum"] = g_it["freight_value"].sum()
    features["freight_mean"] = g_it["freight_value"].mean()

    features["price_per_order"] = features["price_sum"] / features["order_count"]
    features["freight_to_price"] = features["freight_sum"] / (features["price_sum"] + 1.0)

    last_ts = g_it["ts"].max()
    features["days_since_last_item"] = (seed_time - last_ts).dt.days
    first_ts = g_it["ts"].min()
    features["days_since_first_item"] = (seed_time - first_ts).dt.days

    # Средний интервал между покупками (по уникальным заказам продукта)
    ot = items[["product_id", "order_id", "ts"]].drop_duplicates()
    ot = ot.sort_values(["product_id", "ts"], kind="mergesort")
    ot["prev_ts"] = ot.groupby("product_id")["ts"].shift()
    ot["gap_days"] = (ot["ts"] - ot["prev_ts"]).dt.days
    features["avg_days_between_orders"] = ot.groupby("product_id")["gap_days"].mean()

    # Окна истории
    for window in (30, 60, 90, 180, 365):
        start = seed_time - pd.Timedelta(days=window)
        win = items[items["ts"] >= start]
        g_win = win.groupby("product_id")
        features[f"items_last_{window}d"] = g_win.size()
        features[f"orders_last_{window}d"] = g_win["order_id"].nunique()
        features[f"price_sum_last_{window}d"] = g_win["price"].sum()

    features["items_momentum_30_90"] = (
        features["items_last_30d"] / (features["items_last_90d"] + 1.0)
    )
    features["orders_momentum_90_365"] = (
        features["orders_last_90d"] / (features["orders_last_365d"] + 1.0)
    )

    # ----- Платежи (агрегаты на уровне заказа) -----

    pay_order = pays.groupby("order_id").agg(
        pay_value_sum=("payment_value", "sum"),
        pay_value_mean=("payment_value", "mean"),
        pay_installments_mean=("payment_installments", "mean"),
        pay_count=("payment_value", "count"),
    ).reset_index()

    if len(pays):
        pt = pays.groupby(["order_id", "payment_type"]).size().unstack(fill_value=0)
    else:
        pt = pd.DataFrame(columns=PAYMENT_TYPES)
    pt = pt.reindex(columns=PAYMENT_TYPES, fill_value=0).reset_index()
    pt.columns = ["order_id"] + ["pay_type_" + c for c in PAYMENT_TYPES]
    pay_order = pay_order.merge(pt, on="order_id", how="left")

    # Уникальные пары заказ-продукт: платежи и отзывы считаются на уровне заказа,
    # чтобы не умножать их на число позиций в заказе.
    op = items[["product_id", "order_id"]].drop_duplicates()

    opay = op.merge(pay_order, on="order_id", how="left")
    g_pay = opay.groupby("product_id")
    features["pay_value_sum"] = g_pay["pay_value_sum"].sum()
    features["pay_value_mean"] = g_pay["pay_value_mean"].mean()
    features["pay_installments_mean"] = g_pay["pay_installments_mean"].mean()
    features["pay_count"] = g_pay["pay_count"].sum()
    for c in ["pay_type_" + t for t in PAYMENT_TYPES]:
        features[c] = g_pay[c].sum()

    # ----- Отзывы (только те, что уже созданы к seed_time) -----

    rev_order = revs.groupby("order_id").agg(
        rev_score_mean=("review_score", "mean"),
        rev_count=("review_score", "count"),
    ).reset_index()

    orev = op.merge(rev_order, on="order_id", how="left")
    g_rev = orev.groupby("product_id")
    features["review_score_mean"] = g_rev["rev_score_mean"].mean()
    features["review_count"] = g_rev["rev_count"].sum()
    features["review_rate"] = features["review_count"] / (features["order_count"] + 1.0)

    # ----- Статусы исполнения заказов через маску доступности -----

    # Даты одобрения и доставки заполняются после покупки. Признак учитывает
    # их только если значение уже известно на seed_time (не пусто и не позже).
    avail = orders[["order_id"]].copy()
    for col, flag in (
        ("order_approved_at", "approved"),
        ("order_delivered_carrier_date", "carrier"),
        ("order_delivered_customer_date", "delivered"),
    ):
        v = orders[col]
        avail[flag + "_by_seed"] = (v.notna() & (v <= seed_time)).astype(int)

    oavl = op.merge(avail, on="order_id", how="left")
    g_avl = oavl.groupby("product_id")
    features["approved_share"] = g_avl["approved_by_seed"].mean()
    features["carrier_share"] = g_avl["carrier_by_seed"].mean()
    features["delivered_share"] = g_avl["delivered_by_seed"].mean()

    features = features.reindex(list(entity_ids))
    features = features.fillna(0).astype("float64")
    return features
