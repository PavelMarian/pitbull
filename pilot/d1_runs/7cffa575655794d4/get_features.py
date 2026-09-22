import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Извлекаем таблицы
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]

    # Фильтруем заказы до seed_time
    mask = orders["order_purchase_timestamp"] <= seed_time
    orders_before = orders.loc[mask, ["order_id", "customer_id", "order_purchase_timestamp"]]

    # Соединяем order_items с заказами
    merged = order_items.merge(orders_before, on="order_id", how="inner")

    # Агрегируем отзывы по заказам: берём только отзывы, уже созданные к seed_time
    reviews_avail = reviews.loc[reviews["review_creation_date"] <= seed_time]
    review_agg = reviews_avail.groupby("order_id")["review_score"].mean().reset_index()
    review_agg.columns = ["order_id", "review_score_mean"]

    # Агрегируем платежи по заказам (без мутации входной таблицы)
    payments = payments.assign(is_credit=(payments["payment_type"] == "credit_card").astype(int))
    payment_agg = payments.groupby("order_id").agg(
        payment_count=("payment_sequential", "count"),
        payment_value_mean=("payment_value", "mean"),
        payment_value_sum=("payment_value", "sum"),
        payment_installments_mean=("payment_installments", "mean"),
        credit_card_share=("is_credit", "mean")
    ).reset_index()

    # Присоединяем агрегированные данные к merged
    merged = merged.merge(review_agg, on="order_id", how="left")
    merged = merged.merge(payment_agg, on="order_id", how="left")

    # Вычисляем разницу во времени (в днях) от заказа до seed_time
    merged["days_since_order"] = (seed_time - merged["order_purchase_timestamp"]).dt.days

    # Добавляем признак: количество дней с момента заказа (уже есть)
    # Добавляем признак: общая выручка по позиции
    merged["revenue"] = merged["price"] + merged["freight_value"]

    # Группируем по product_id и считаем признаки
    grouped = merged.groupby("product_id").agg(
        count_orders=("order_id", "nunique"),
        count_items=("order_id", "count"),
        unique_customers=("customer_id", "nunique"),
        avg_price=("price", "mean"),
        std_price=("price", "std"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        sum_price=("price", "sum"),
        avg_freight=("freight_value", "mean"),
        sum_freight=("freight_value", "sum"),
        total_revenue=("revenue", "sum"),
        avg_review_score=("review_score_mean", "mean"),
        count_reviews=("review_score_mean", "count"),
        payment_count_mean=("payment_count", "mean"),
        payment_value_mean=("payment_value_mean", "mean"),
        payment_installments_mean=("payment_installments_mean", "mean"),
        credit_card_share=("credit_card_share", "mean"),
        days_since_last_order=("days_since_order", "min"),
        days_since_first_order=("days_since_order", "max"),
    ).reset_index()

    # Добавляем признаки за последние 30, 90, 180 дней
    # Сначала создадим маску для каждого периода
    for days in [30, 90, 180]:
        tmp = merged[merged["days_since_order"] <= days].groupby("product_id").agg(
            **{f"count_orders_{days}d": ("order_id", "nunique"),
               f"revenue_{days}d": ("revenue", "sum")}
        ).reset_index()
        grouped = grouped.merge(tmp, on="product_id", how="left")

    # Вычисляем частоту заказов (заказов в день)
    # Период активности = days_since_first_order - days_since_last_order
    grouped["active_period"] = grouped["days_since_first_order"] - grouped["days_since_last_order"]
    grouped["order_frequency"] = np.where(
        grouped["active_period"] > 0,
        grouped["count_orders"] / grouped["active_period"],
        0
    )

    # Заполняем пропуски нулями (для товаров без истории)
    grouped = grouped.fillna(0)

    # Устанавливаем индекс product_id
    grouped = grouped.set_index("product_id")

    # Реиндексируем по entity_ids
    result = grouped.reindex(entity_ids, fill_value=0)

    # Убираем вспомогательные столбцы, если нужно оставить только числовые
    # Удаляем столбец active_period, если он не нужен (можно оставить)
    # result = result.drop(columns=["active_period"])

    return result