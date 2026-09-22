import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    result = pd.DataFrame(index=entity_ids)
    result.index.name = "product_id"

    order_items = db["order_items"].query("ts < @seed_time").copy()
    valid_orders = order_items["order_id"].unique()

    # Аггрегация по всей истории продаж
    product_sales = order_items.groupby("product_id").agg(
        total_revenue=("price", "sum"),
        total_units=("order_id", "count"),
        avg_price=("price", "mean"),
        avg_freight=("freight_value", "mean")
    )
    product_sales["has_sales_history"] = 1

    # Аггрегация за последние 90 дней
    cutoff_90 = seed_time - pd.Timedelta(days=90)
    recent_sales = order_items.query("ts >= @cutoff_90").groupby("product_id").agg(
        recent_revenue=("price", "sum"),
        recent_units=("order_id", "count"),
        last_sale_days=("ts", lambda x: (seed_time - x.max()).days)
    )

    # Аггрегация по отзывам: используем только отзывы, существующие на момент seed_time
    reviews = db["reviews"].query("review_creation_date < @seed_time")
    reviews = reviews.query("order_id in @valid_orders").copy()
    reviews_w_products = reviews.merge(order_items[["order_id", "product_id"]], on="order_id", how="left")
    product_reviews = reviews_w_products.groupby("product_id").agg(
        total_reviews=("review_id", "count"),
        avg_review_score=("review_score", "mean"),
        pct_negative_reviews=("review_score", lambda x: (x <= 2).mean())
    )

    # Объединение всех признаков
    features = product_sales.join(recent_sales, how="outer")
    features = features.join(product_reviews, how="outer")

    # Обработка пропусков
    count_cols = ["total_units", "total_revenue", "total_reviews", "recent_units", "recent_revenue"]
    for col in count_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    if "has_sales_history" in features.columns:
        features["has_sales_history"] = features["has_sales_history"].fillna(0)

    if "last_sale_days" in features.columns:
        features["last_sale_days"] = features["last_sale_days"].fillna(3650)

    avg_cols = ["avg_price", "avg_freight", "avg_review_score", "pct_negative_reviews"]
    for col in avg_cols:
        if col in features.columns:
            mean_val = features[col].mean()
            if pd.isna(mean_val):
                mean_val = 0
            features[col] = features[col].fillna(mean_val)

    # Приведение к числовому типу
    for col in features.columns:
        if not np.issubdtype(features[col].dtype, np.number):
            features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)

    # Реиндексация под запрошенные товары
    result = result.join(features, how="left")
    result = result.fillna(0)

    # Единый тип для всех признаков
    for col in result.columns:
        result[col] = result[col].astype("float64")

    return result