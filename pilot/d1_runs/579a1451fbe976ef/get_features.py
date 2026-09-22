import pandas as pd
import numpy as np

def get_features(db, entity_ids, seed_time):
    orders = db['orders']
    items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']

    if not isinstance(seed_time, pd.Timestamp):
        seed_time = pd.Timestamp(seed_time)

    # Только заказы, купленные до seed_time
    orders = orders[orders['order_purchase_timestamp'] <= seed_time]

    # Доставленные к seed_time: дата доставки уже известна
    delivered = orders[orders['order_delivered_customer_date'].notna()
                       & (orders['order_delivered_customer_date'] <= seed_time)]

    # Отзывы, уже созданные к seed_time
    reviews = reviews[reviews['review_creation_date'] <= seed_time]
    review_agg = reviews.groupby('order_id').agg(
        avg_review_score=('review_score', 'mean'),
    ).reset_index()

    # Платежи, доступные к seed_time
    payments = payments[payments['ts'] <= seed_time]
    pay_agg = payments.groupby('order_id').agg(
        total_payment=('payment_value', 'sum'),
        num_payments=('payment_sequential', 'count')
    ).reset_index()

    # Базовая витрина: позиции заказов, купленных до seed_time
    df = items.merge(orders[['order_id', 'order_purchase_timestamp']], on='order_id', how='inner')
    df['days_since_purchase'] = (seed_time - df['order_purchase_timestamp']).dt.days
    df['last_7'] = (df['days_since_purchase'] <= 7).astype(int)
    df['last_30'] = (df['days_since_purchase'] <= 30).astype(int)
    df['last_90'] = (df['days_since_purchase'] <= 90).astype(int)

    grouped = df.groupby('product_id').agg(
        num_orders=('order_id', 'nunique'),
        num_items=('order_id', 'size'),
        total_price=('price', 'sum'),
        avg_price=('price', 'mean'),
        median_price=('price', 'median'),
        std_price=('price', 'std'),
        total_freight=('freight_value', 'sum'),
        avg_freight=('freight_value', 'mean'),
        days_since_last_order=('days_since_purchase', 'min'),
        days_since_first_order=('days_since_purchase', 'max'),
        last_7_orders=('last_7', 'sum'),
        last_30_orders=('last_30', 'sum'),
        last_90_orders=('last_90', 'sum'),
    ).reset_index()
    grouped['order_frequency'] = grouped['num_orders'] / (grouped['days_since_first_order'] + 1)

    # Признаки доставки: только по заказам, доставленным до seed_time
    d_items = items.merge(delivered[['order_id', 'order_purchase_timestamp',
                                     'order_delivered_customer_date',
                                     'order_estimated_delivery_date']],
                          on='order_id', how='inner')
    d_items = d_items[d_items['order_delivered_customer_date'] <= seed_time]
    d_items['delivery_days'] = (d_items['order_delivered_customer_date']
                                - d_items['order_purchase_timestamp']).dt.days
    d_items['late'] = (d_items['order_delivered_customer_date']
                       > d_items['order_estimated_delivery_date']).astype(float)
    deliv_prod = d_items.groupby('product_id').agg(
        avg_delivery_days=('delivery_days', 'mean'),
        min_delivery_days=('delivery_days', 'min'),
        max_delivery_days=('delivery_days', 'max'),
        late_delivery_rate=('late', 'mean'),
        num_delivered=('delivery_days', 'size'),
    ).reset_index()

    # Отзывы по продукту: только отзывы, существующие на seed_time
    r_items = items.merge(orders[['order_id']], on='order_id', how='inner')
    r_items = r_items.merge(review_agg, on='order_id', how='inner')
    review_prod = r_items.groupby('product_id').agg(
        avg_review_score=('avg_review_score', 'mean'),
        num_reviews=('order_id', 'nunique'),
    ).reset_index()

    # Платежи по продукту
    p_items = items.merge(orders[['order_id']], on='order_id', how='inner')
    p_items = p_items.merge(pay_agg, on='order_id', how='inner')
    p_items = p_items.drop_duplicates(subset=['product_id', 'order_id'])
    pay_prod = p_items.groupby('product_id').agg(
        avg_payment_value=('total_payment', 'mean'),
        avg_num_payments=('num_payments', 'mean'),
        total_payment_sum=('total_payment', 'sum'),
    ).reset_index()

    features = grouped.merge(deliv_prod, on='product_id', how='left')
    features = features.merge(review_prod, on='product_id', how='left')
    features = features.merge(pay_prod, on='product_id', how='left')

    features.set_index('product_id', inplace=True)
    features = features.reindex(entity_ids)
    num_cols = features.select_dtypes(include=[np.number]).columns
    features[num_cols] = features[num_cols].fillna(0)
    return features
