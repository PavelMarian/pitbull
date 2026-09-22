def get_features(db, entity_ids, seed_time):
    # Извлекаем таблицы
    order_items = db["order_items"]
    orders = db["orders"]
    reviews = db["reviews"]
    payments = db["payments"]  # может пригодиться, но здесь не используем

    entities = list(entity_ids)

    # Отбираем только прошлые продажи (до seed_time)
    items_past = order_items[
        (order_items["product_id"].isin(entities)) &
        (order_items["ts"] <= seed_time)
    ]

    # Строим feature-датафрейм с индексом = entity_ids
    features = pd.DataFrame(index=entities)

    # 1. Агрегаты за всю предшествующую историю
    if not items_past.empty:
        agg_all = items_past.groupby("product_id").agg(
            n_orders=("order_id", "nunique"),
            n_items=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            mean_price=("price", "mean"),
            mean_freight=("freight_value", "mean"),
            std_price=("price", "std"),
            std_freight=("freight_value", "std"),
            n_sellers=("seller_id", "nunique"),
            min_ts=("ts", "min"),
            max_ts=("ts", "max")
        )
        features = features.join(agg_all)
    else:
        # Пустой датафрейм – всё останется NaN
        pass

    # 2. Признаки, связанные со временем
    if "max_ts" in features.columns:
        features["days_since_last"] = (seed_time - features["max_ts"]).dt.days
        features["days_since_first"] = (seed_time - features["min_ts"]).dt.days
        features["active_days"] = (features["max_ts"] - features["min_ts"]).dt.days

    # 3. Агрегаты по временным окнам (30, 60, 90, 180 дней)
    windows = [30, 60, 90, 180]
    for win in windows:
        cutoff = seed_time - pd.Timedelta(days=win)
        items_win = items_past[items_past["ts"] > cutoff]  # строго последние win дней
        if items_win.empty:
            # Заполняем нулями (нет продаж в окне)
            features[f"cnt_orders_{win}"] = 0
            features[f"cnt_items_{win}"] = 0
            features[f"sum_price_{win}"] = 0.0
            features[f"has_sales_{win}"] = 0
        else:
            agg_win = items_win.groupby("product_id").agg(
                cnt_orders=("order_id", "nunique"),
                cnt_items=("order_item_id", "count"),
                sum_price=("price", "sum"),
                mean_price=("price", "mean")
            ).add_prefix(f"win{win}_")
            features = features.join(agg_win)

    # 4. Признаки на основе отзывов
    if not items_past.empty and not reviews.empty:
        # Используем только отзывы, которые уже существуют на момент seed_time
        reviews_past = reviews[reviews["review_creation_date"] <= seed_time]

        # Средняя оценка по каждому заказу (может быть несколько отзывов на заказ)
        order_rating = reviews_past.groupby("order_id")["review_score"].mean().rename("avg_review").reset_index()

        # Связываем заказы и товары
        prod_orders = items_past[["product_id", "order_id"]].drop_duplicates()
        prod_orders = prod_orders.merge(order_rating, on="order_id", how="left")

        if prod_orders["avg_review"].notna().any():
            rating_agg = prod_orders.groupby("product_id")["avg_review"].agg(
                ["mean", "count", lambda x: (x >= 4).mean() * 100]  # процент положительных отзывов
            ).rename(columns={"mean": "avg_review_score",
                              "count": "n_reviews",
                              "<lambda_0>": "pct_positive_reviews"})
            features = features.join(rating_agg)

    # 5. Дополнительные признаки: число платежей, доля способов оплаты и т.д.
    # Это можно добавить, но для простоты пропустим.

    # 6. Глобальные временные признаки
    features["month"] = seed_time.month
    features["quarter"] = seed_time.quarter
    features["day_of_week"] = seed_time.dayofweek
    features["year"] = seed_time.year

    # 7. Признак "есть ли вообще продажи" – полезен, чтобы отличить 0 из-за малого периода
    if "n_orders" in features.columns:
        features["has_sales"] = (features["n_orders"].fillna(0) > 0).astype(int)
    else:
        features["has_sales"] = 0

    # 8. Заполняем оставшиеся пропуски нулями
    features = features.fillna(0.0)

    # 9. Переиндексируем в порядок входных entity_ids (они уже в features.index)
    features = features.reindex(entity_ids, fill_value=0.0)
    features.index.name = "product_id"

    return features