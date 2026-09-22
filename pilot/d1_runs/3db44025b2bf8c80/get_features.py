def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Извлечение таблиц
    orders = db["orders"]
    items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]

    seed_time = pd.Timestamp(seed_time)

    # Заказы, совершённые строго до seed_time
    orders_before = orders[orders["order_purchase_timestamp"] <= seed_time]

    # Объединяем заказы и товарные позиции
    merged = items.merge(
        orders_before[["order_id", "order_purchase_timestamp"]],
        on="order_id",
        how="inner"
    )
    merged = merged[merged["product_id"].isin(entity_ids)].copy()

    cols_names = [
        "total_orders", "total_unique_orders", "total_price", "total_freight",
        "avg_price", "std_price", "avg_freight", "max_price",
        "total_orders_30", "total_price_30",
        "total_orders_90", "total_price_90",
        "total_orders_180", "total_price_180",
        "days_since_first", "days_since_last", "avg_interval",
        "reviews_count", "avg_review_score", "positive_review_ratio",
        "total_payment_value", "avg_payment_installments", "n_payment_records",
        "ever_sold"
    ]

    # Если нет ни одного заказа для данных товаров – возвращаем нулевую матрицу
    if merged.empty:
        return pd.DataFrame(
            0,
            index=pd.Index(entity_ids, name="product_id"),
            columns=cols_names
        )

    # Вычисляем количество дней до seed_time
    merged["days_before"] = (seed_time - merged["order_purchase_timestamp"]).dt.days

    # --- Общие агрегаты ---
    agg = merged.groupby("product_id").agg(
        total_orders=("order_id", "size"),
        total_unique_orders=("order_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        avg_price=("price", "mean"),
        std_price=("price", "std"),
        avg_freight=("freight_value", "mean"),
        max_price=("price", "max")
    ).reset_index()
    agg["std_price"] = agg["std_price"].fillna(0)
    agg["ever_sold"] = 1

    # --- Агрегаты за различные окна ---
    for period, suffix in [(30, "30"), (90, "90"), (180, "180")]:
        period_df = merged[merged["days_before"] <= period].groupby("product_id").agg(
            total_orders=("order_id", "size"),
            total_price=("price", "sum")
        ).reset_index()
        period_df = period_df.rename(columns={
            "total_orders": f"total_orders_{suffix}",
            "total_price": f"total_price_{suffix}"
        })
        agg = agg.merge(period_df, on="product_id", how="left")

    # --- Временные характеристики (для каждого товара) ---
    # Уникальные пары (product_id, order_id) для корректных дат заказов
    sales_dates = merged[["product_id", "order_id", "order_purchase_timestamp"]].drop_duplicates(
        subset=["product_id", "order_id"]
    )
    date_info = sales_dates.groupby("product_id").agg(
        first_date=("order_purchase_timestamp", "min"),
        last_date=("order_purchase_timestamp", "max"),
        n_orders=("order_id", "nunique")
    ).reset_index()

    date_info["days_since_first"] = (seed_time - date_info["first_date"]).dt.days
    date_info["days_since_last"] = (seed_time - date_info["last_date"]).dt.days

    # Средний интервал между заказами
    date_info["diff_days"] = (date_info["last_date"] - date_info["first_date"]).dt.days
    date_info["avg_interval"] = date_info["diff_days"] / (date_info["n_orders"] - 1).clip(lower=1)

    agg = agg.merge(
        date_info[["product_id", "days_since_first", "days_since_last", "avg_interval"]],
        on="product_id",
        how="left"
    )

    # --- Отзывы ---
    # Утечка: отзыв появляется в базе позже покупки (review_creation_date),
    # поэтому на момент seed_time доступны только отзывы, уже созданные к нему.
    reviews_before = reviews[reviews["review_creation_date"] <= seed_time]
    order_product = merged[["order_id", "product_id"]].drop_duplicates()
    rev_merge = order_product.merge(
        reviews_before[["order_id", "review_score"]],
        on="order_id",
        how="left"
    )
    rev_agg = rev_merge.groupby("product_id").agg(
        reviews_count=("review_score", "count"),
        avg_review_score=("review_score", "mean"),
        pos_count=("review_score", lambda x: (x >= 4).sum())
    ).reset_index()
    rev_agg["positive_review_ratio"] = rev_agg["pos_count"] / rev_agg["reviews_count"].clip(lower=1)
    agg = agg.merge(
        rev_agg[["product_id", "reviews_count", "avg_review_score", "positive_review_ratio"]],
        on="product_id",
        how="left"
    )

    # --- Платежи ---
    pay_agg_by_order = payments.groupby("order_id").agg(
        total_payment=("payment_value", "sum"),
        avg_install=("payment_installments", "mean"),
        n_pay=("payment_sequential", "count")
    ).reset_index()

    pay_merge = order_product.merge(pay_agg_by_order, on="order_id", how="left")
    pay_prod_agg = pay_merge.groupby("product_id").agg(
        total_payment_value=("total_payment", "sum"),
        avg_payment_installments=("avg_install", "mean"),
        n_payment_records=("n_pay", "sum")
    ).reset_index()
    agg = agg.merge(pay_prod_agg, on="product_id", how="left")

    # --- Реиндексация и заполнение пропусков ---
    agg = agg.set_index("product_id").reindex(entity_ids)
    agg = agg.fillna(0)

    # Бинарный признак наличия продаж (для товаров без истории = 0)
    agg["ever_sold"] = (agg["total_orders"] > 0).astype(int)

    # Выбранные столбцы (порядок фиксированный)
    agg = agg[cols_names]

    return agg