import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    orders = db['orders']
    order_items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']

    seed_time = pd.Timestamp(seed_time)

    # История на момент seed_time: строка доступна, если время её события <= seed_time.
    # Заказы живут по order_purchase_timestamp, позиции и платежи по ts,
    # отзывы по review_creation_date (момент написания отзыва).
    orders_hist = orders[orders['order_purchase_timestamp'] <= seed_time]
    items_hist = order_items[order_items['ts'] <= seed_time].merge(
        orders_hist[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date']],
        on='order_id',
        how='inner'
    )

    # Доставка учитывается только если она уже произошла к seed_time.
    delivered = (
        items_hist['order_delivered_customer_date'].notna()
        & (items_hist['order_delivered_customer_date'] <= seed_time)
    )
    items_hist['delivery_days'] = np.where(
        delivered,
        (items_hist['order_delivered_customer_date'] - items_hist['order_purchase_timestamp']).dt.days,
        np.nan
    )

    # Базовые агрегаты по товарам
    agg = items_hist.groupby('product_id').agg(
        total_orders=('order_id', 'nunique'),
        total_quantity=('order_item_id', 'count'),
        total_sales=('price', 'sum'),
        avg_price=('price', 'mean'),
        std_price=('price', 'std'),
        total_freight=('freight_value', 'sum'),
        avg_freight=('freight_value', 'mean'),
        std_freight=('freight_value', 'std'),
        max_price=('price', 'max'),
        min_price=('price', 'min'),
        avg_delivery_days=('delivery_days', 'mean'),
        std_delivery_days=('delivery_days', 'std'),
    ).reset_index()

    # Давность последней и первой покупки
    last_purchase = items_hist.groupby('product_id')['order_purchase_timestamp'].max().reset_index()
    last_purchase['days_since_last'] = (seed_time - last_purchase['order_purchase_timestamp']).dt.days
    first_purchase = items_hist.groupby('product_id')['order_purchase_timestamp'].min().reset_index()
    first_purchase['days_since_first'] = (seed_time - first_purchase['order_purchase_timestamp']).dt.days
    agg = agg.merge(last_purchase[['product_id', 'days_since_last']], on='product_id', how='left')
    agg = agg.merge(first_purchase[['product_id', 'days_since_first']], on='product_id', how='left')

    # Оконные признаки (30, 60, 90 дней)
    for window in [30, 60, 90]:
        cutoff = seed_time - pd.Timedelta(days=window)
        window_items = items_hist[items_hist['order_purchase_timestamp'] >= cutoff]
        window_agg = window_items.groupby('product_id').agg(
            **{f'orders_last_{window}': ('order_id', 'nunique'),
               f'sales_last_{window}': ('price', 'sum')}
        ).reset_index()
        agg = agg.merge(window_agg, on='product_id', how='left')

    # Отзывы: только уже написанные к seed_time
    reviews_avail = reviews[reviews['review_creation_date'] <= seed_time]
    items_reviews = items_hist[['order_id', 'product_id']].merge(
        reviews_avail[['order_id', 'review_score', 'review_creation_date']],
        on='order_id',
        how='left'
    )
    reviews_agg = items_reviews.groupby('product_id').agg(
        review_count=('review_score', 'count'),
        avg_review_score=('review_score', 'mean'),
        std_review_score=('review_score', 'std'),
        pos_review_count=('review_score', lambda x: (x >= 4).sum())
    ).reset_index()
    reviews_agg['pos_review_ratio'] = (
        reviews_agg['pos_review_count'] / reviews_agg['review_count'].replace(0, np.nan)
    ).fillna(0)
    agg = agg.merge(reviews_agg, on='product_id', how='left')

    # Свежие отзывы за последние 90 дней
    recent_reviews = items_reviews[
        items_reviews['review_creation_date'] >= seed_time - pd.Timedelta(days=90)
    ]
    recent_agg = recent_reviews.groupby('product_id').agg(
        recent_review_count=('review_score', 'count'),
        recent_avg_review_score=('review_score', 'mean'),
    ).reset_index()
    agg = agg.merge(recent_agg, on='product_id', how='left')

    # Платежи: только проведённые к seed_time
    payments_hist = payments[payments['ts'] <= seed_time]
    pay_order = payments_hist.groupby('order_id', as_index=False).agg(
        order_pay_total=('payment_value', 'sum'),
        order_pay_installments=('payment_installments', 'max'),
    )
    items_pay = items_hist[['order_id', 'product_id']].merge(pay_order, on='order_id', how='left')
    pay_agg = items_pay.groupby('product_id').agg(
        avg_order_pay=('order_pay_total', 'mean'),
        avg_installments=('order_pay_installments', 'mean'),
    ).reset_index()
    agg = agg.merge(pay_agg, on='product_id', how='left')

    # Продавцы
    sellers_agg = items_hist.groupby('product_id').agg(
        unique_sellers=('seller_id', 'nunique')
    ).reset_index()
    agg = agg.merge(sellers_agg, on='product_id', how='left')

    # Интенсивность продаж с момента первого появления
    agg['sales_per_day_since_first'] = agg['total_sales'] / (agg['days_since_first'] + 1)
    agg['orders_per_day_since_first'] = agg['total_orders'] / (agg['days_since_first'] + 1)

    agg = agg.set_index('product_id')
    agg = agg.reindex(entity_ids)
    # Товар без истории: давность задают большой константой, прочие пропуски нулём
    for c in ['days_since_last', 'days_since_first']:
        agg[c] = agg[c].fillna(9999)
    agg = agg.fillna(0).astype(float)

    return agg
