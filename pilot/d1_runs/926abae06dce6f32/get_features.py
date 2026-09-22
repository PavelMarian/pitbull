import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Извлекаем таблицы из базы
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]
    
    # Фильтруем исторические данные до seed_time
    items_hist = order_items[order_items["ts"] <= seed_time].copy()
    # Создаем столбец общей стоимости позиции
    items_hist["total_value"] = items_hist["price"] + items_hist["freight_value"]
    
    # Определяем временные окна для агрегации
    windows = {
        "7d": 7,
        "30d": 30,
        "90d": 90,
        "365d": 365,
        "all_time": 100000  # Достаточно большое число для всех данных
    }
    
    # Создаем базовый фрейм с требуемыми product_id в качестве индекса
    base = pd.DataFrame(index=entity_ids)
    base.index.name = "product_id"
    
    # Агрегируем признаки по order_items для каждого окна
    for window_name, days in windows.items():
        if window_name == "all_time":
            window_data = items_hist
        else:
            cutoff = seed_time - pd.Timedelta(days=days)
            window_data = items_hist[items_hist["ts"] >= cutoff]
        
        # Основные агрегаты по товарным позициям
        agg_params = {
            "order_id": ["count", "nunique"],
            "price": ["mean", "sum", "min", "max"],
            "freight_value": ["mean", "sum"],
            "total_value": ["mean", "sum"],
            "seller_id": "nunique",
            "order_item_id": "max"
        }
        window_agg = window_data.groupby("product_id").agg(agg_params)
        # Приводим имена столбцов к единому формату
        window_agg.columns = [f"{window_name}_{'_'.join(col)}" for col in window_agg.columns]
        # Упрощаем имена столбцов для читаемости
        rename_map = {
            f"{window_name}_order_id_count": f"{window_name}_item_count",
            f"{window_name}_order_id_nunique": f"{window_name}_order_count",
            f"{window_name}_price_mean": f"{window_name}_avg_price",
            f"{window_name}_price_sum": f"{window_name}_total_price",
            f"{window_name}_price_min": f"{window_name}_min_price",
            f"{window_name}_price_max": f"{window_name}_max_price",
            f"{window_name}_freight_value_mean": f"{window_name}_avg_freight",
            f"{window_name}_freight_value_sum": f"{window_name}_total_freight",
            f"{window_name}_total_value_mean": f"{window_name}_avg_order_value",
            f"{window_name}_total_value_sum": f"{window_name}_total_revenue",
            f"{window_name}_seller_id_nunique": f"{window_name}_unique_sellers",
            f"{window_name}_order_item_id_max": f"{window_name}_max_units_per_order",
        }
        window_agg = window_agg.rename(columns=rename_map)
        # Добавляем агрегаты к базовому фрейму
        base = base.join(window_agg, how="left")
    
    # Временные признаки: первая и последняя продажа
    time_metrics = items_hist.groupby("product_id")["ts"].agg(["min", "max"])
    time_metrics = time_metrics.rename(columns={"min": "first_purchase", "max": "last_purchase"})
    time_metrics["days_since_first_purchase"] = (seed_time - time_metrics["first_purchase"]).dt.days
    time_metrics["days_since_last_purchase"] = (seed_time - time_metrics["last_purchase"]).dt.days
    base = base.join(time_metrics[["days_since_first_purchase", "days_since_last_purchase"]], how="left")
    
    # Признаки по отзывам
    # Оставляем только отзывы, уже существующие на момент seed_time:
    # отзыв пишется после покупки, поэтому без фильтра сюда попадает будущее.
    reviews_hist = reviews[reviews["review_creation_date"] <= seed_time]
    # Объединяем товары с отзывами через order_id
    product_reviews = items_hist[["product_id", "order_id"]].drop_duplicates().merge(reviews_hist, on="order_id", how="inner")
    # Агрегируем метрики отзывов
    review_agg = product_reviews.groupby("product_id")["review_score"].agg([
        "count", "mean", "min", "max", "std"
    ])
    # Добавляем долю крайних оценок
    review_agg["five_star_ratio"] = (product_reviews["review_score"] == 5).groupby(product_reviews["product_id"]).mean()
    review_agg["one_star_ratio"] = (product_reviews["review_score"] == 1).groupby(product_reviews["product_id"]).mean()
    review_agg.columns = [f"review_{col}" for col in review_agg.columns]
    base = base.join(review_agg, how="left")
    
    # Признаки по платежам
    # Объединяем товары с платежами через order_id
    product_payments = items_hist[["product_id", "order_id"]].merge(payments, on="order_id", how="inner")
    # Агрегируем метрики платежей
    pay_agg = product_payments.groupby("product_id").agg({
        "payment_sequential": "count",
        "payment_installments": ["mean", "max"],
        "payment_value": ["mean", "sum"]
    })
    # Добавляем долю кредитных платежей
    pay_agg["payment_credit_ratio"] = (product_payments["payment_type"] == "credit_card").groupby(product_payments["product_id"]).mean()
    # Приводим имена столбцов
    pay_agg.columns = [f"{'_'.join(col)}" if col[1] else col[0] for col in pay_agg.columns]
    pay_agg = pay_agg.rename(columns={
        "payment_sequential_count": "payment_count",
        "payment_installments_mean": "payment_avg_installments",
        "payment_installments_max": "payment_max_installments",
        "payment_value_mean": "payment_avg_value",
        "payment_value_sum": "payment_total_value",
    })
    base = base.join(pay_agg, how="left")
    
    # Заполняем пропуски нулями (отсутствие данных означает нулевые значения метрик
    base = base.fillna(0)
    
    # Удаляем возможные дубликаты столбцов
    base = base.loc[:, ~base.columns.duplicated()]
    
    # Реиндексируем под заданный список entity_ids
    base = base.reindex(entity_ids)
    
    return base