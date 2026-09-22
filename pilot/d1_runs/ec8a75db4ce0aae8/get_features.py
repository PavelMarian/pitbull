import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Извлекаем таблицы
    orders = db['orders'].copy()
    items = db['order_items'].copy()
    reviews = db['reviews'].copy()
    payments = db['payments'].copy()

    # Приводим к datetime
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    items['ts'] = pd.to_datetime(items['ts'])
    reviews['review_creation_date'] = pd.to_datetime(reviews['review_creation_date'])

    # Фильтруем товары
    items = items[items['product_id'].isin(entity_ids)]
    if items.empty:
        result = pd.DataFrame(index=entity_ids)
        for col in ['count_orders', 'sum_price', 'avg_freight', 'count_customers',
                    'avg_review_score', 'count_reviews', 'days_since_last_purchase',
                    'days_since_first_purchase', 'purchase_frequency_30d', 'purchase_frequency_90d',
                    'avg_basket_size', 'avg_payment_value', 'installment_ratio']:
            result[col] = 0
        return result.reindex(entity_ids)

    # Объединяем с заказами
    merged = items.merge(orders[['order_id', 'customer_id', 'order_purchase_timestamp']],
                         on='order_id', how='left')
    merged = merged[merged['order_purchase_timestamp'] < seed_time]

    if merged.empty:
        result = pd.DataFrame(index=entity_ids)
        for col in ['count_orders', 'sum_price', 'avg_freight', 'count_customers',
                    'avg_review_score', 'count_reviews', 'days_since_last_purchase',
                    'days_since_first_purchase', 'purchase_frequency_30d', 'purchase_frequency_90d',
                    'avg_basket_size', 'avg_payment_value', 'installment_ratio']:
            result[col] = 0
        return result.reindex(entity_ids)

    # Размер заказа
    order_sizes = items.groupby('order_id').size().rename('order_size')
    merged = merged.merge(order_sizes, on='order_id', how='left')

    # Отзывы: берём только те, что уже написаны к моменту seed_time
    reviews = reviews[reviews['review_creation_date'] <= seed_time]
    review_agg = reviews.groupby('order_id')['review_score'].mean().rename('review_score')
    merged = merged.merge(review_agg, on='order_id', how='left')

    # Платежи
    pay_agg = payments.groupby('order_id').agg(
        avg_payment_value=('payment_value', 'mean'),
        installment_ratio=('payment_installments', lambda x: (x > 1).mean())
    ).reset_index()
    merged = merged.merge(pay_agg, on='order_id', how='left')

    # Вспомогательные функции
    def days_since_last(series):
        if series.empty:
            return np.nan
        return (seed_time - series.max()).days

    def days_since_first(series):
        if series.empty:
            return np.nan
        return (seed_time - series.min()).days

    # Признаки для окон
    merged['is_30d'] = (seed_time - merged['order_purchase_timestamp']).dt.days <= 30
    merged['is_90d'] = (seed_time - merged['order_purchase_timestamp']).dt.days <= 90

    # Агрегация
    features = merged.groupby('product_id').agg(
        count_orders=('order_id', 'count'),
        sum_price=('price', 'sum'),
        avg_freight=('freight_value', 'mean'),
        count_customers=('customer_id', 'nunique'),
        avg_review_score=('review_score', 'mean'),
        count_reviews=('review_score', 'count'),
        days_since_last_purchase=('order_purchase_timestamp', days_since_last),
        days_since_first_purchase=('order_purchase_timestamp', days_since_first),
        purchase_frequency_30d=('is_30d', 'sum'),
        purchase_frequency_90d=('is_90d', 'sum'),
        avg_basket_size=('order_size', 'mean'),
        avg_payment_value=('avg_payment_value', 'mean'),
        installment_ratio=('installment_ratio', 'mean')
    ).reset_index()

    features = features.set_index('product_id')
    features = features.reindex(entity_ids)
    features = features.fillna(0)

    return features
