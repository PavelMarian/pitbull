import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    """
    Возвращает матрицу признаков для каждого product_id на основе истории до seed_time.
    Используются только данные, существующие на момент seed_time.
    """
    orders = db['orders']
    items = db['order_items']
    payments = db['payments']
    reviews = db['reviews']

    seed_time = pd.Timestamp(seed_time)

    feature_cols = [
        'order_count', 'total_revenue', 'avg_price', 'median_price', 'std_price',
        'unique_sellers', 'days_since_last_purchase', 'product_age_days',
        'avg_freight', 'avg_installments', 'avg_delivery_days',
        'avg_estimated_delivery_days', 'avg_review_score'
    ]

    # Только заказы, совершённые строго до seed_time
    orders_before = orders[orders['order_purchase_timestamp'] < seed_time].copy()

    if orders_before.empty:
        return pd.DataFrame(0, index=entity_ids, columns=feature_cols)

    # Соединяем items с заказами
    merged = items.merge(
        orders_before[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date',
                       'order_estimated_delivery_date']],
        on='order_id',
        how='inner'
    )

    if merged.empty:
        return pd.DataFrame(0, index=entity_ids, columns=feature_cols)

    # Фактическая доставка известна только для заказов, доставленных до seed_time
    delivered_mask = (
        merged['order_delivered_customer_date'].notna()
        & (merged['order_delivered_customer_date'] <= seed_time)
    )

    # Агрегация по товарам
    grp = merged.groupby('product_id')

    features = pd.DataFrame()
    features['order_count'] = grp.size()
    features['total_revenue'] = grp.apply(lambda x: (x['price'] + x['freight_value']).sum())
    features['avg_price'] = grp['price'].mean()
    features['median_price'] = grp['price'].median()
    features['std_price'] = grp['price'].std()
    features['unique_sellers'] = grp['seller_id'].nunique()
    features['avg_freight'] = grp['freight_value'].mean()

    # Время последней и первой покупки
    last_purchase = grp['order_purchase_timestamp'].max()
    first_purchase = grp['order_purchase_timestamp'].min()
    features['days_since_last_purchase'] = (seed_time - last_purchase).dt.days
    features['product_age_days'] = (seed_time - first_purchase).dt.days

    # Среднее время доставки: только по заказам, уже доставленным к seed_time
    delivered = merged[delivered_mask]
    grp_del = delivered.groupby('product_id')
    features['avg_delivery_days'] = grp_del.apply(
        lambda x: (x['order_delivered_customer_date'] - x['order_purchase_timestamp']).dt.days.mean()
    )

    # Оценка доставки планировалась на момент покупки: можно по всем заказам до seed_time
    features['avg_estimated_delivery_days'] = grp.apply(
        lambda x: (x['order_estimated_delivery_date'] - x['order_purchase_timestamp']).dt.days.mean()
    )

    # Признаки из платежей (среднее количество платежей в рассрочку)
    order_pay = payments.groupby('order_id')['payment_installments'].mean().rename('avg_installments')
    merged_pay = merged[['order_id', 'product_id']].drop_duplicates().merge(order_pay, on='order_id', how='left')
    features['avg_installments'] = merged_pay.groupby('product_id')['avg_installments'].mean()

    # Средний рейтинг отзывов: только отзывы, созданные до seed_time
    reviews_before = reviews[reviews['review_creation_date'] <= seed_time]
    order_rev = reviews_before.groupby('order_id')['review_score'].mean().rename('review_score')
    merged_rev = merged[['order_id', 'product_id']].drop_duplicates().merge(order_rev, on='order_id', how='left')
    features['avg_review_score'] = merged_rev.groupby('product_id')['review_score'].mean()

    # Убираем возможные дубликаты индексов
    features = features[~features.index.duplicated(keep='first')]

    # Реиндексация по entity_ids
    features = features.reindex(entity_ids)
    features = features.fillna(0)

    return features
