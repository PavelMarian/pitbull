def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Импортируем pandas и numpy (они уже доступны, но для ясности)
    import pandas as pd
    import numpy as np

    # Извлекаем таблицы из словаря
    orders = db['orders'].copy()
    order_items = db['order_items'].copy()
    reviews = db['reviews'].copy()
    payments = db['payments'].copy()

    # Фильтрация по seed_time: используем только данные, доступные до момента предсказания
    orders = orders[orders['order_purchase_timestamp'] < seed_time].copy()

    # Поздно заполняемые поля заказа маскируем: событие считается доступным,
    # только если его метка времени уже наступила к моменту seed_time
    for col in ['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date']:
        orders.loc[orders[col] > seed_time, col] = pd.NaT

    # Присоединяем заказы к позициям заказа
    items = order_items.merge(
        orders[['order_id', 'order_purchase_timestamp',
                'order_approved_at', 'order_delivered_carrier_date',
                'order_delivered_customer_date', 'order_estimated_delivery_date']],
        on='order_id',
        how='inner'
    )

    # Временные разницы в днях: считаются только по уже наступившим событиям,
    # иначе остаётся NaN (доставка/одобрение ещё не произошли к seed_time)
    items['delivery_days'] = (items['order_delivered_customer_date'] - items['order_purchase_timestamp']).dt.days
    items['approval_days'] = (items['order_approved_at'] - items['order_purchase_timestamp']).dt.days
    items['estimate_diff'] = (items['order_delivered_customer_date'] - items['order_estimated_delivery_date']).dt.days

    # Факт доставки на момент seed_time: по дате доставки, а не по финальному статусу
    items['is_delivered'] = items['order_delivered_customer_date'].notna().astype(int)

    # Агрегация отзывов по заказам (до seed_time)
    reviews = reviews[reviews['review_creation_date'] < seed_time]
    if not reviews.empty:
        reviews['positive'] = (reviews['review_score'] >= 4).astype(int)
        review_agg = reviews.groupby('order_id').agg(
            review_mean=('review_score', 'mean'),
            review_count=('review_score', 'size'),
            pos_review_ratio=('positive', 'mean')
        )
        review_agg['has_review'] = (review_agg['review_count'] > 0).astype(int)
    else:
        review_agg = pd.DataFrame(columns=['review_mean', 'review_count', 'pos_review_ratio', 'has_review'])

    # Агрегация платежей по заказам (до seed_time)
    payments = payments[payments['ts'] < seed_time]
    if not payments.empty:
        payments['is_credit'] = (payments['payment_type'] == 'credit_card').astype(int)
        payment_agg = payments.groupby('order_id').agg(
            total_payment=('payment_value', 'sum'),
            payment_count=('payment_value', 'size'),
            avg_payment=('payment_value', 'mean'),
            max_payment=('payment_value', 'max'),
            min_payment=('payment_value', 'min'),
            total_installments=('payment_installments', 'sum'),
            avg_installments=('payment_installments', 'mean'),
            credit_ratio=('is_credit', 'mean')
        )
    else:
        payment_agg = pd.DataFrame(columns=['total_payment', 'payment_count', 'avg_payment',
                                            'max_payment', 'min_payment', 'total_installments',
                                            'avg_installments', 'credit_ratio'])

    # Присоединяем агрегаты к позициям
    items = items.merge(review_agg, on='order_id', how='left')
    items = items.merge(payment_agg, on='order_id', how='left')

    # Группируем по паре (product_id, order_id): один заказ с данным товаром
    order_product = items.groupby(['product_id', 'order_id']).agg(
        n_items=('order_item_id', 'size'),
        total_price=('price', 'sum'),
        total_freight=('freight_value', 'sum'),
        avg_price=('price', 'mean'),
        std_price=('price', 'std'),
        order_purchase_timestamp=('order_purchase_timestamp', 'first'),
        delivery_days=('delivery_days', 'first'),
        approval_days=('approval_days', 'first'),
        estimate_diff=('estimate_diff', 'first'),
        is_delivered=('is_delivered', 'first'),
        review_mean=('review_mean', 'first'),
        review_count=('review_count', 'first'),
        has_review=('has_review', 'first'),
        pos_review_ratio=('pos_review_ratio', 'first'),
        total_payment=('total_payment', 'first'),
        payment_count=('payment_count', 'first'),
        avg_payment=('avg_payment', 'first'),
        max_payment=('max_payment', 'first'),
        min_payment=('min_payment', 'first'),
        total_installments=('total_installments', 'first'),
        avg_installments=('avg_installments', 'first'),
        credit_ratio=('credit_ratio', 'first')
    ).reset_index()

    # Агрегация по product_id
    features = order_product.groupby('product_id').agg(
        n_orders=('order_id', 'size'),
        sum_total_price=('total_price', 'sum'),
        avg_total_price=('total_price', 'mean'),
        sum_total_freight=('total_freight', 'sum'),
        avg_total_freight=('total_freight', 'mean'),
        avg_n_items=('n_items', 'mean'),
        max_n_items=('n_items', 'max'),
        avg_avg_price=('avg_price', 'mean'),
        std_avg_price=('avg_price', 'std'),
        avg_std_price=('std_price', 'mean'),
        avg_delivery_days=('delivery_days', 'mean'),
        std_delivery_days=('delivery_days', 'std'),
        avg_approval_days=('approval_days', 'mean'),
        avg_estimate_diff=('estimate_diff', 'mean'),
        delivered_ratio=('is_delivered', 'mean'),
        avg_review_mean=('review_mean', 'mean'),
        sum_review_count=('review_count', 'sum'),
        avg_has_review=('has_review', 'mean'),
        avg_pos_review_ratio=('pos_review_ratio', 'mean'),
        avg_total_payment=('total_payment', 'mean'),
        sum_total_payment=('total_payment', 'sum'),
        avg_payment_count=('payment_count', 'mean'),
        avg_avg_payment=('avg_payment', 'mean'),
        max_max_payment=('max_payment', 'max'),
        avg_max_payment=('max_payment', 'mean'),
        avg_min_payment=('min_payment', 'mean'),
        avg_total_installments=('total_installments', 'mean'),
        avg_avg_installments=('avg_installments', 'mean'),
        avg_credit_ratio=('credit_ratio', 'mean')
    ).reset_index()

    # Признаки по продавцам: только по покупкам, совершённым до seed_time
    seller_agg = items.groupby('product_id').agg(
        n_sellers=('seller_id', 'nunique'),
        avg_seller_price=('price', 'mean')
    ).reset_index()
    features = features.merge(seller_agg, on='product_id', how='left')

    # Временные признаки: количество заказов за последние периоды, частота, давность последнего заказа
    times = order_product.groupby('product_id')['order_purchase_timestamp'].agg(
        last_order_ts='max',
        first_order_ts='min',
        total_orders='count'
    ).reset_index()

    times['days_since_last_order'] = (seed_time - times['last_order_ts']).dt.days
    times['days_since_first_order'] = (seed_time - times['first_order_ts']).dt.days
    times['order_span_days'] = (times['last_order_ts'] - times['first_order_ts']).dt.days
    # Частота заказов в месяц (защита от деления на ноль)
    times['order_freq_per_month'] = np.where(
        times['order_span_days'] > 0,
        times['total_orders'] / (times['order_span_days'] / 30.0),
        times['total_orders']  # если только один заказ, частота = количество (но лучше 0)
    )
    # Альтернатива: если span=0, то частота = total_orders (единичные случаи) — потом заменяем на 0
    times.loc[times['order_span_days'] == 0, 'order_freq_per_month'] = 0.0

    # Заказы за последние 30/60/90/180 дней
    for days in [30, 60, 90, 180]:
        threshold = seed_time - pd.Timedelta(days=days)
        recent = order_product[order_product['order_purchase_timestamp'] > threshold].groupby('product_id').size()
        times[f'recent_{days}_orders'] = recent.reindex(features['product_id']).fillna(0).astype(int)

    # Объединяем всё вместе
    features = features.merge(times.drop(columns=['last_order_ts', 'first_order_ts']), on='product_id', how='left')

    # Устанавливаем product_id в индекс
    features = features.set_index('product_id')

    # Реиндексация под все entity_ids и заполнение пропусков нулями
    features = features.reindex(entity_ids).fillna(0)

    return features
