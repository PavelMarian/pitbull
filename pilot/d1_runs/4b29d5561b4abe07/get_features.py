import pandas as pd
import numpy as np


def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Извлечение таблиц из базы
    orders = db["orders"].copy()
    order_items = db["order_items"].copy()
    reviews = db["reviews"].copy()
    payments = db["payments"].copy()
    
    # Приведение всех временных столбцов к единому формату (на случай если не datetime)
    for col in ["order_purchase_timestamp", "order_approved_at", 
                "order_delivered_carrier_date", "order_delivered_customer_date", 
                "order_estimated_delivery_date"]:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")
    for col in ["shipping_limit_date", "ts"]:
        if col in order_items.columns:
            order_items[col] = pd.to_datetime(order_items[col], errors="coerce")
    for col in ["review_creation_date", "review_answer_timestamp"]:
        if col in reviews.columns:
            reviews[col] = pd.to_datetime(reviews[col], errors="coerce")
    if "ts" in payments.columns:
        payments["ts"] = pd.to_datetime(payments["ts"], errors="coerce")
    
    # Фильтрация исторических данных до seed_time (используем ts как основной момент покупки)
    order_items_hist = order_items[order_items["ts"] < seed_time].copy()
    orders_hist = orders[orders["order_purchase_timestamp"] < seed_time].copy()
    reviews_hist = reviews[reviews["review_creation_date"] < seed_time].copy()
    payments_hist = payments[payments["ts"] < seed_time].copy()
    
    # Объединение order_items с orders для получения статуса заказа и др. информации
    order_items_full = pd.merge(
        order_items_hist,
        orders_hist[["order_id", "order_status", "order_delivered_customer_date", "order_estimated_delivery_date"]],
        on="order_id",
        how="left"
    )
    
    # Добавление информации об отзывах
    order_items_reviews = pd.merge(
        order_items_full,
        reviews_hist[["order_id", "review_score"]],
        on="order_id",
        how="left"
    )
    
    # Добавление информации о платежах (агрегируем по order_id)
    payments_agg = payments_hist.groupby("order_id", as_index=False).agg({
        "payment_value": "sum",
        "payment_installments": "max"
    }).rename(columns={
        "payment_value": "total_payment_value",
        "payment_installments": "max_installments"
    })
    order_items_all = pd.merge(
        order_items_reviews,
        payments_agg,
        on="order_id",
        how="left"
    )
    
    # Создание временных окон для агрегации (в днях)
    time_windows = [30, 90, 180, 9999]  # 9999 ~ все время
    features_list = []
    base_date = pd.Timestamp(seed_time)
    
    for window in time_windows:
        window_start = base_date - pd.Timedelta(days=window)
        mask = (order_items_all["ts"] >= window_start) & (order_items_all["ts"] < base_date)
        window_data = order_items_all[mask].copy()
        
        # Основные агрегаты по товару
        agg_dict = {
            "order_id": "nunique",          # Количество уникальных заказов
            "order_item_id": "count",        # Количество проданных единиц товара
            "price": ["mean", "sum", "std"], # Средняя, суммарная, стандартное отклонение цены
            "freight_value": ["mean", "sum"],# Средняя и суммарная стоимость доставки
            "review_score": ["mean", "std"], # Средняя и стандартное отклонение оценки отзыва
            "total_payment_value": "sum",    # Суммарная выручка по товару
            "max_installments": "mean"       # Среднее число рассрочек
        }
        
        # Агрегируем данные по product_id
        product_agg = window_data.groupby("product_id").agg(agg_dict)
        # Уплощаем мультииндекс столбцов
        product_agg.columns = [f"{col[0]}_{col[1]}_w{window}" for col in product_agg.columns]
        features_list.append(product_agg)
    
    # Объединяем признаки из всех временных окон
    all_features = pd.concat(features_list, axis=1, join="outer")
    
    # Удаляем дубликаты столбцов (если есть)
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]
    
    # Дополнительные признаки: время с последней продажи
    last_sale = order_items_hist.groupby("product_id")["ts"].max()
    days_since_last_sale = (base_date - last_sale).dt.days
    days_since_last_sale.name = "days_since_last_sale"
    all_features = all_features.join(days_since_last_sale, how="outer")
    
    # Признак: сколько дней прошло с первой продажи
    first_sale = order_items_hist.groupby("product_id")["ts"].min()
    days_since_first_sale = (base_date - first_sale).dt.days
    days_since_first_sale.name = "days_since_first_sale"
    all_features = all_features.join(days_since_first_sale, how="outer")
    
    # Доставка: читаем только даты доставки, которые уже наступили до seed_time.
    # order_delivered_customer_date заполняется в момент доставки, поэтому
    # заказы, доставленные после seed_time, в признаки не попадают.
    delivered_orders = order_items_all[order_items_all["order_delivered_customer_date"].notna()].copy()
    delivered_orders = delivered_orders[delivered_orders["order_delivered_customer_date"] < base_date]
    delivered_orders["delivery_days"] = (
        delivered_orders["order_delivered_customer_date"] - delivered_orders["order_estimated_delivery_date"]
    ).dt.days
    
    delivery_stats = delivered_orders.groupby("product_id").agg({
        "delivery_days": ["mean", "std"]
    })
    delivery_stats.columns = ["delivery_vs_est_mean", "delivery_vs_est_std"]
    all_features = all_features.join(delivery_stats, how="outer")
    
    # Обработка пропусков: заполняем средними значениями по всем товарам
    # И добавляем флаг "новый товар" (не было продаж до seed_time)
    all_features["is_new_product"] = all_features.isna().all(axis=1).astype(int)
    
    # Заполняем пропуски глобальными средними (для стабильности модели)
    global_means = all_features.drop(columns=["is_new_product"]).mean()
    # Заполняем числовые столбцы
    all_features = all_features.fillna(global_means)
    # Для std которые остались NaN (если только 1 значение) заполняем 0
    all_features = all_features.fillna(0)
    
    # Дополнительный признак: доля товара в общей выручке за последние 90 дней
    if "price_sum_w90" in all_features.columns:
        total_rev_90 = all_features["price_sum_w90"].sum()
        all_features["revenue_share_w90"] = all_features["price_sum_w90"] / total_rev_90
        all_features["revenue_share_w90"] = all_features["revenue_share_w90"].fillna(0)
    
    # Приводим индексы к требуемому списку entity_ids
    result_df = all_features.reindex(entity_ids)
    # Убеждаемся, что все пропуски заполнены (для товаров, которых нет в обучающей выборке)
    # Заполняем оставшиеся NaN нулями и ставим is_new_product=1
    if "is_new_product" in result_df.columns:
        result_df["is_new_product"] = result_df["is_new_product"].fillna(1)
    else:
        result_df["is_new_product"] = 1
    result_df = result_df.fillna(0)
    
    # Удаляем возможные нечисловые столбцы (на всякий случай)
    result_df = result_df.select_dtypes(include=[np.number])
    
    return result_df