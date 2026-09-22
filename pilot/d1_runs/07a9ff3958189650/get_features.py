def get_features(db, entity_ids, seed_time):
    # данные
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]

    product_ids = list(entity_ids)
    seed = pd.Timestamp(seed_time)

    # обрезаем данные до seed_time
    sub = order_items[order_items["product_id"].isin(product_ids)].copy()
    if "ts" in sub.columns and not pd.api.types.is_datetime64_any_dtype(sub["ts"]):
        sub["ts"] = pd.to_datetime(sub["ts"])
    sub = sub[sub["ts"] <= seed]

    # отзывы: берём только те, что уже созданы к моменту seed_time
    rev = reviews.copy()
    if "review_creation_date" in rev.columns:
        if not pd.api.types.is_datetime64_any_dtype(rev["review_creation_date"]):
            rev["review_creation_date"] = pd.to_datetime(rev["review_creation_date"])
        rev = rev[rev["review_creation_date"] <= seed]
    rev = rev.groupby("order_id")["review_score"].mean().reset_index()
    sub = sub.merge(rev, on="order_id", how="left")

    # платёжная статистика заказа: только платежи, доступные к seed_time
    pay = payments.copy()
    if "ts" in pay.columns:
        if not pd.api.types.is_datetime64_any_dtype(pay["ts"]):
            pay["ts"] = pd.to_datetime(pay["ts"])
        pay = pay[pay["ts"] <= seed]
    pay = pay.groupby("order_id").agg(
        total_paid=("payment_value", "sum"),
        n_payments=("payment_sequential", "count"),
        avg_installments=("payment_installments", "mean")
    ).reset_index()
    sub = sub.merge(pay, on="order_id", how="left")

    # основные агрегаты за всё время
    agg = sub.groupby("product_id").agg(
        order_count=("order_id", "count"),
        unique_orders=("order_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        sum_price=("price", "sum"),
        mean_price=("price", "mean"),
        median_price=("price", "median"),
        std_price=("price", "std"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        sum_freight=("freight_value", "sum"),
        mean_freight=("freight_value", "mean"),
        review_mean=("review_score", "mean"),
        total_paid_all=("total_paid", "sum"),
        avg_total_paid=("total_paid", "mean"),
        avg_n_payments=("n_payments", "mean"),
        avg_installments=("avg_installments", "mean"),
        first_purchase=("ts", "min"),
        last_purchase=("ts", "max")
    ).reset_index()

    # счётчики за последние периоды
    for days in (30, 60, 90, 180):
        cutoff = seed - pd.Timedelta(days=days)
        cnt = sub[sub["ts"] >= cutoff].groupby("product_id").size()
        agg[f"count_last_{days}"] = agg["product_id"].map(cnt).fillna(0)

    # статистика за последние 90 дней
    cutoff90 = seed - pd.Timedelta(days=90)
    sub90 = sub[sub["ts"] >= cutoff90]
    if not sub90.empty:
        g90 = sub90.groupby("product_id").agg(
            sum_price_90=("price", "sum"),
            review_mean_90=("review_score", "mean")
        ).reset_index()
        agg = agg.merge(g90, on="product_id", how="left")
    else:
        agg["sum_price_90"] = 0.0
        agg["review_mean_90"] = np.nan

    # временные признаки
    agg["days_since_first"] = (seed - agg["first_purchase"]).dt.days
    agg["days_since_last"] = (seed - agg["last_purchase"]).dt.days
    agg["span_days"] = (agg["last_purchase"] - agg["first_purchase"]).dt.days

    agg["has_sales"] = (agg["order_count"] > 0).astype(int)
    span_safe = np.maximum(agg["span_days"].fillna(0), 1)
    agg["sales_per_day"] = agg["order_count"] / span_safe

    # доля продаж в последние 90 дней
    agg["share_last_90"] = agg["count_last_90"] / (agg["order_count"] + 1e-6)

    # удаляем ненужные временные столбцы
    agg.drop(columns=["first_purchase", "last_purchase"], inplace=True)

    # заполняем пропуски (кроме дней пустых продаж)
    agg.fillna(0, inplace=True)
    agg.loc[agg["has_sales"] == 0, "days_since_first"] = 9999.0
    agg.loc[agg["has_sales"] == 0, "days_since_last"] = 9999.0

    # реиндексация по entity_ids
    agg = agg.set_index("product_id").reindex(product_ids)
    agg.fillna(0, inplace=True)
    agg.loc[agg["has_sales"] == 0, "days_since_first"] = 9999.0
    agg.loc[agg["has_sales"] == 0, "days_since_last"] = 9999.0

    return agg
