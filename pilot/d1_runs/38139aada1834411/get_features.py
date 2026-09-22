def get_features(db: dict, entity_ids, seed_time) -> "pd.DataFrame":
    import pandas as pd
    import numpy as np

    orders = db["orders"].copy()
    order_items = db["order_items"].copy()
    reviews = db["reviews"].copy()
    payments = db["payments"].copy()

    # 1. Фильтруем заказы до seed_time
    orders = orders[orders["order_purchase_timestamp"] < seed_time]

    # 2. Базовая объединенная таблица (product_id, order_id, покупатель, время покупки)
    merged = order_items.merge(
        orders[["order_id", "customer_id", "order_purchase_timestamp"]],
        on="order_id",
        how="inner"
    )

    # 3. Агрегация по товару: основные продажи
    feat_agg = merged.groupby("product_id").agg(
        total_orders=("order_id", "nunique"),          # сколько уникальных заказов
        total_items=("order_item_id", "count"),        # суммарное количество товара
        price_sum=("price", "sum"),
        price_mean=("price", "mean"),
        freight_sum=("freight_value", "sum"),
        freight_mean=("freight_value", "mean"),
        n_customers=("customer_id", "nunique")         # количество разных покупателей
    ).reset_index()

    # 4. Время первой и последней покупки
    time_agg = merged.groupby("product_id")["order_purchase_timestamp"].agg(["min", "max"]).reset_index()
    time_agg.columns = ["product_id", "first_purchase", "last_purchase"]

    # 5. Помесячная статистика продаж
    monthly = merged.copy()
    monthly["purchase_month"] = monthly["order_purchase_timestamp"].dt.to_period("M")
    monthly_g = monthly.groupby(["product_id", "purchase_month"]).size().reset_index(name="cnt")
    monthly_agg = monthly_g.groupby("product_id").agg(
        n_months=("purchase_month", "count"),          # количество активных месяцев
        avg_orders_per_month=("cnt", "mean")          # среднее количество заказов в месяц
    ).reset_index()

    # 6. Отзывы (только созданные до seed_time)
    reviews_f = reviews[reviews["review_creation_date"] < seed_time]
    rev_tmp = reviews_f[["order_id", "review_score"]].merge(
        merged[["product_id", "order_id"]].drop_duplicates(),
        on="order_id",
        how="inner"
    )
    review_agg = rev_tmp.groupby("product_id").agg(
        n_reviews=("review_score", "count"),
        mean_review_score=("review_score", "mean")
    ).reset_index()

    # 7. Платежи (только те, что произошли до seed_time по ts)
    payments_f = payments[payments["ts"] < seed_time]
    pay_tmp = payments_f.merge(
        merged[["product_id", "order_id"]].drop_duplicates(),
        on="order_id",
        how="inner"
    )
    pay_base = pay_tmp.groupby("product_id").agg(
        n_payments=("payment_value", "count"),
        total_payment=("payment_value", "sum"),
        avg_payment=("payment_value", "mean")
    ).reset_index()

    # Доля кредитных карт и среднее количество платежей
    pay_tmp["is_credit"] = (pay_tmp["payment_type"] == "credit_card").astype(int)
    pay_tmp["is_debit"] = (pay_tmp["payment_type"] == "debit_card").astype(int)
    pay_type_agg = pay_tmp.groupby("product_id").agg(
        credit_rate=("is_credit", "mean"),
        debit_rate=("is_debit", "mean"),
        avg_installments=("payment_installments", "mean")
    ).reset_index()

    # 8. Средняя задержка доставки (по заказам с фактической датой доставки)
    # Уникальные (product_id, order_id)
    tmp_del = merged[["product_id", "order_id"]].drop_duplicates()
    delivered_orders = orders[["order_id", "order_purchase_timestamp", "order_delivered_customer_date"]]
    delivered_orders = delivered_orders.dropna(subset=["order_delivered_customer_date"])
    # Дата доставки заполняется позже покупки и может быть позже seed_time:
    # учитываем только доставки, которые уже произошли к моменту предсказания.
    delivered_orders = delivered_orders[delivered_orders["order_delivered_customer_date"] <= seed_time]
    del_join = tmp_del.merge(delivered_orders, on="order_id", how="inner")
    del_join["delivery_days"] = (del_join["order_delivered_customer_date"] - del_join["order_purchase_timestamp"]).dt.days
    delivery_agg = del_join.groupby("product_id")["delivery_days"].mean().reset_index(name="avg_delivery_days")

    # 9. Объединение всех агрегатов
    features = feat_agg \
        .merge(time_agg, on="product_id", how="left") \
        .merge(monthly_agg, on="product_id", how="left") \
        .merge(review_agg, on="product_id", how="left") \
        .merge(pay_base, on="product_id", how="left") \
        .merge(pay_type_agg, on="product_id", how="left") \
        .merge(delivery_agg, on="product_id", how="left")

    # 10. Относительное время от seed_time
    features["days_since_first_sale"] = (seed_time - features["first_purchase"]).dt.days
    features["days_since_last_sale"] = (seed_time - features["last_purchase"]).dt.days

    # 11. Убираем нечисловые колонки, превращаем в нужный набор
    selected = [
        "total_orders", "total_items", "price_sum", "price_mean",
        "freight_sum", "freight_mean", "n_customers",
        "n_months", "avg_orders_per_month",
        "n_reviews", "mean_review_score",
        "n_payments", "total_payment", "avg_payment",
        "credit_rate", "debit_rate", "avg_installments",
        "avg_delivery_days",
        "days_since_first_sale", "days_since_last_sale"
    ]
    # Убедимся, что все колонки присутствуют; заполняем пропуски нулями
    features = features.set_index("product_id")[selected]
    features = features.reindex(entity_ids, fill_value=0.0)
    features = features.fillna(0.0)

    return features