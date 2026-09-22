import pandas as pd
import numpy as np

def get_features(db, entity_ids, seed_time):
    # Приведём entity_ids к списку, чтобы работать с pandas
    if isinstance(entity_ids, (list, np.ndarray, pd.Series)):
        product_ids = list(entity_ids)
    else:
        product_ids = [entity_ids]

    seed_time = pd.Timestamp(seed_time)

    # --- Таблицы из БД ---
    orders = db["orders"]
    items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]

    # --- PIT-фильтры: в каждой таблице берём только строки, уже существующие
    # на момент seed_time. У каждой таблицы своя временная колонка:
    # order_items / payments — ts (момент покупки), reviews — review_creation_date
    # (отзыв появляется позже покупки), orders — order_purchase_timestamp.
    items_f = items[items["ts"] <= seed_time]
    items_f = items_f[items_f["product_id"].isin(product_ids)].copy()

    pay_f = payments[payments["ts"] <= seed_time]
    rev_f = reviews[reviews["review_creation_date"] <= seed_time]

    # Присоединяем заказы. Поскольку items_f уже ограничен покупками <= seed_time,
    # в merge попадают только заказы, купленные до seed_time.
    merged = items_f.merge(
        orders[["order_id", "order_delivered_customer_date"]],
        on="order_id", how="left",
    )

    # Дата доставки известна только если доставка уже состоялась к seed_time.
    # Заказы, купленные раньше, но ещё не доставленные, считаем недоставленными.
    deliv = merged["order_delivered_customer_date"]
    deliv_known = deliv.notna() & (deliv <= seed_time)
    merged["delivery_days"] = np.where(
        deliv_known, (deliv - merged["ts"]).dt.days, np.nan
    )
    merged["delivered"] = deliv_known.astype(float)

    # ---- Агрегаты для платежей по заказам (только платежи, сделанные к seed_time) ----
    pay_agg = pay_f.groupby("order_id").agg(
        total_payment=("payment_value", "sum"),
        count_payments=("payment_value", "count"),
        avg_installments=("payment_installments", "mean")
    ).reset_index()

    # ---- Агрегаты для отзывов по заказам (только отзывы, созданные к seed_time) ----
    rev_temp = rev_f.copy()
    rev_temp["positive"] = (rev_temp["review_score"] >= 4).astype(int)
    rev_agg = rev_temp.groupby("order_id").agg(
        avg_review=("review_score", "mean"),
        count_reviews=("review_score", "count"),
        positive_count=("positive", "sum")
    ).reset_index()
    rev_agg["positive_ratio"] = rev_agg["positive_count"] / rev_agg["count_reviews"]

    # ---- Присоединяем агрегаты к основной таблице ----
    merged = merged.merge(pay_agg, on="order_id", how="left")
    merged = merged.merge(rev_agg, on="order_id", how="left")

    # ============================================================
    # Признаки, считаемые по позициям (строкам item)
    # ============================================================
    pos_agg = merged.groupby("product_id").agg(
        total_items=("price", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        avg_price=("price", "mean"),
        std_price=("price", "std"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        avg_freight=("freight_value", "mean"),
        std_freight=("freight_value", "std"),
        avg_delivery_days=("delivery_days", "mean"),
        delivered_share=("delivered", "mean")
    ).reset_index()

    # ============================================================
    # Признаки по уникальным заказам для каждого продукта
    # ============================================================
    order_level = merged[["product_id", "order_id", "ts",
                          "total_payment", "count_payments", "avg_installments",
                          "avg_review", "count_reviews", "positive_ratio",
                          "delivered"]].drop_duplicates(
                              subset=["product_id", "order_id"])

    order_agg = order_level.groupby("product_id").agg(
        num_orders=("order_id", "nunique"),
        total_payment_all=("total_payment", "sum"),
        avg_payment_per_order=("total_payment", "mean"),
        std_payment_per_order=("total_payment", "std"),
        avg_payments_per_order=("count_payments", "mean"),
        avg_install_mean=("avg_installments", "mean"),
        avg_review_score=("avg_review", "mean"),
        avg_review_count=("count_reviews", "mean"),
        avg_positive_ratio=("positive_ratio", "mean"),
        delivered_order_share=("delivered", "mean")
    ).reset_index()

    # Временные признаки
    temp_agg = order_level.groupby("product_id").agg(
        days_since_last=("ts", lambda x: (seed_time - x.max()).days),
        days_since_first=("ts", lambda x: (seed_time - x.min()).days),
        date_span=("ts", lambda x: (x.max() - x.min()).days)
    ).reset_index()

    # объединяем все признаки по продуктам
    features = pos_agg.merge(order_agg, on="product_id", how="outer")
    features = features.merge(temp_agg, on="product_id", how="outer")

    # Устанавливаем индекс product_id
    features = features.set_index("product_id")

    # Создаём пустой DataFrame для всех требуемых продуктов
    result = pd.DataFrame(index=pd.Index(product_ids, name="product_id"))

    # Присоединяем вычисленные признаки
    result = result.join(features, how="left")

    # Добавляем признак, есть ли продажи вообще
    result["has_sales"] = result["num_orders"].notna().astype(float)

    # Заполняем отсутствующие значения нулями (для отсутствия истории)
    result = result.fillna(0)

    # Только числовые столбцы
    result = result.astype(float)

    # Возвращаем результат с реиндексацией под entity_ids
    result = result.loc[product_ids]
    return result