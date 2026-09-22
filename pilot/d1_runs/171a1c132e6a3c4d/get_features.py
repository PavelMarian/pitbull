import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Извлекаем таблицы
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]

    # Фильтруем позиции по интересующим товарам
    items = order_items[order_items["product_id"].isin(entity_ids)].copy()

    # Соединяем с заказами, чтобы получить дату покупки
    merged = items.merge(
        orders[["order_id", "order_purchase_timestamp"]],
        on="order_id",
        how="left"
    )

    # Оставляем только заказы, совершённые до seed_time
    merged = merged[merged["order_purchase_timestamp"] < seed_time]

    # Если нет ни одного исторического заказа — возвращаем нулевые признаки
    if merged.empty:
        features = pd.DataFrame(index=entity_ids)
        features["has_history"] = 0
        return features

    # Количество дней от заказа до seed_time (положительное, чем меньше, тем ближе)
    merged["days_to_seed"] = (seed_time - merged["order_purchase_timestamp"]).dt.days

    # Индикаторы окон для последних 30/60/90 дней
    merged["in_30"] = (merged["days_to_seed"] <= 30).astype(int)
    merged["in_60"] = (merged["days_to_seed"] <= 60).astype(int)
    merged["in_90"] = (merged["days_to_seed"] <= 90).astype(int)

    # Основные агрегации по товарам
    group = merged.groupby("product_id")

    total_orders = group["order_id"].nunique()                 # уникальных заказов
    total_items = group.size()                                 # всего позиций
    total_revenue = group.apply(lambda x: (x["price"] + x["freight_value"]).sum())
    avg_price = group["price"].mean()
    avg_freight = group["freight_value"].mean()
    median_price = group["price"].median()
    median_freight = group["freight_value"].median()
    std_price = group["price"].std()
    std_freight = group["freight_value"].std()
    min_price = group["price"].min()
    max_price = group["price"].max()
    n_sellers = group["seller_id"].nunique()

    avg_days = group["days_to_seed"].mean()
    min_days = group["days_to_seed"].min()   # давность последнего заказа
    max_days = group["days_to_seed"].max()   # возраст первого заказа (от seed)
    median_days = group["days_to_seed"].median()
    std_days = group["days_to_seed"].std()

    # Частота заказов (заказов в день между первым и последним)
    span_days = (max_days - min_days + 1).replace({0: np.nan})
    order_freq = total_orders / span_days

    # Среднее количество позиций и средний чек на заказ
    avg_items_per_order = total_items / total_orders
    avg_revenue_per_order = total_revenue / total_orders

    # Собираем все в DataFrame
    features = pd.DataFrame({
        "total_orders": total_orders,
        "total_items": total_items,
        "total_revenue": total_revenue,
        "avg_price": avg_price,
        "avg_freight": avg_freight,
        "median_price": median_price,
        "median_freight": median_freight,
        "std_price": std_price,
        "std_freight": std_freight,
        "min_price": min_price,
        "max_price": max_price,
        "n_sellers": n_sellers,
        "avg_days_to_seed": avg_days,
        "min_days_to_seed": min_days,
        "max_days_to_seed": max_days,
        "median_days_to_seed": median_days,
        "std_days_to_seed": std_days,
        "order_freq": order_freq,
        "avg_items_per_order": avg_items_per_order,
        "avg_revenue_per_order": avg_revenue_per_order,
        "has_history": 1
    })

    # Признаки активности в окнах (уникальные заказы)
    for window, col in [(30, "orders_30"), (60, "orders_60"), (90, "orders_90")]:
        mask = merged["days_to_seed"] <= window
        sub = merged[mask]
        if not sub.empty:
            features[col] = sub.groupby("product_id")["order_id"].nunique()
        else:
            features[col] = 0

    # Присоединяем только те отзывы, которые уже написаны на момент seed_time
    reviews_subset = reviews.loc[
        reviews["review_creation_date"] <= seed_time,
        ["order_id", "review_score"]
    ].drop_duplicates()
    merged_rev = merged.merge(reviews_subset, on="order_id", how="left")

    if not merged_rev["review_score"].isna().all():
        group_rev = merged_rev.groupby("product_id")
        avg_review = group_rev["review_score"].mean()
        num_reviews = group_rev["review_score"].count()
        # доля положительных (рейтинг >= 4)
        merged_rev["positive"] = (merged_rev["review_score"] >= 4).astype(int)
        pos_reviews = group_rev["positive"].sum()
        positive_ratio = pos_reviews / num_reviews.replace(0, np.nan)

        features["avg_review_score"] = avg_review
        features["num_reviews"] = num_reviews
        features["positive_reviews"] = pos_reviews
        features["positive_ratio"] = positive_ratio
    else:
        # Если отзывов нет вообще, ставим нейтральные нули
        features["avg_review_score"] = 0
        features["num_reviews"] = 0
        features["positive_reviews"] = 0
        features["positive_ratio"] = 0

    # Переиндексируем по entity_ids и заполняем пропуски для товаров без истории
    features = features.reindex(entity_ids)
    features = features.fillna(0)

    return features
