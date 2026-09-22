def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    # Извлекаем таблицы
    oi = db['order_items']
    orders = db['orders']
    reviews = db['reviews']
    payments = db['payments']

    # Фильтруем отзывы по времени: отзыв доступен, только если он уже создан
    # (review_creation_date <= seed_time). Без этого фильтра в агрегаты попадают
    # отзывы, написанные после момента предсказания.
    reviews = reviews[reviews['review_creation_date'] <= seed_time]

    # Фильтруем order_items по целевым товарам и до seed_time
    oi_filtered = oi[oi['product_id'].isin(entity_ids) & (oi['ts'] < seed_time)].copy()

    # Если нет данных, сразу возвращаем DataFrame с нулями
    if len(oi_filtered) == 0:
        # Создаём пустой DataFrame с индексами entity_ids
        result = pd.DataFrame(index=entity_ids)
        # Добавляем заглушку, чтобы не было пустого DataFrame
        result['dummy'] = 0
        return result

    # Присоединяем orders для дат и статусов
    oi_filtered = oi_filtered.merge(
        orders[['order_id', 'order_status', 'order_purchase_timestamp',
                'order_delivered_customer_date', 'order_estimated_delivery_date']],
        on='order_id', how='left'
    )

    # Признаки времени
    oi_filtered['days_since_purchase'] = (seed_time - oi_filtered['ts']).dt.days
    # Дата доставки доступна только если она не пуста и не в будущем
    # относительно seed_time. Иначе delivery_time и on_time читают будущее.
    delivered = oi_filtered['order_delivered_customer_date']
    delivered_known = delivered.notna() & (delivered <= seed_time)
    oi_filtered['delivery_time'] = (
        (delivered - oi_filtered['order_purchase_timestamp']).dt.days
    ).where(delivered_known)
    oi_filtered['on_time'] = pd.Series(
        np.where(delivered_known & (delivered <= oi_filtered['order_estimated_delivery_date']), 1.0, np.nan),
        index=oi_filtered.index
    )
    oi_filtered['is_delivered'] = (delivered_known).astype(int)

    # Агрегаты по отзывам для заказа
    rev_agg = reviews.groupby('order_id').agg(
        review_mean=('review_score', 'mean'),
        review_count=('review_score', 'count')
    ).reset_index()
    oi_filtered = oi_filtered.merge(rev_agg, on='order_id', how='left')

    # Агрегаты по платежам для заказа (только платежи, уже существующие на
    # момент seed_time; ts платежа = момент покупки его заказа)
    payments = payments[payments['ts'] <= seed_time]
    pay_agg = payments.groupby('order_id').agg(
        payment_sum=('payment_value', 'sum'),
        payment_count=('payment_sequential', 'count'),
        payment_installments_mean=('payment_installments', 'mean'),
        payment_installments_max=('payment_installments', 'max'),
        payment_installments_gt1=('payment_installments', lambda x: (x > 1).mean())
    ).reset_index()
    oi_filtered = oi_filtered.merge(pay_agg, on='order_id', how='left')

    # Общая выручка позиции
    oi_filtered['revenue'] = oi_filtered['price'] + oi_filtered['freight_value']

    # Индикаторы заказов за последние N дней
    for window in [30, 90, 180]:
        oi_filtered[f'order_in_{window}d'] = (oi_filtered['days_since_purchase'] <= window).astype(int)

    # Основная агрегация по продукту
    agg_dict = {
        'order_count': ('order_id', 'nunique'),
        'item_count': ('order_id', 'count'),
        'total_revenue': ('revenue', 'sum'),
        'avg_price': ('price', 'mean'),
        'sum_price': ('price', 'sum'),
        'avg_freight': ('freight_value', 'mean'),
        'sum_freight': ('freight_value', 'sum'),
        'avg_review': ('review_mean', 'mean'),
        'total_reviews': ('review_count', 'sum'),
        'total_payment': ('payment_sum', 'sum'),
        'payment_transactions': ('payment_count', 'sum'),
        'avg_installments': ('payment_installments_mean', 'mean'),
        'max_installments': ('payment_installments_max', 'max'),
        'frac_installments_gt1': ('payment_installments_gt1', 'mean'),
        'avg_days_since': ('days_since_purchase', 'mean'),
        'min_days_since': ('days_since_purchase', 'min'),
        'max_days_since': ('days_since_purchase', 'max'),
        'frac_delivered': ('is_delivered', 'mean'),
        'avg_delivery_time': ('delivery_time', 'mean'),
        'frac_on_time': ('on_time', 'mean'),
    }

    prod_features = oi_filtered.groupby('product_id').agg(**agg_dict).reset_index()

    # Количество заказов за последние N дней
    for window in [30, 90, 180]:
        counts = oi_filtered.groupby('product_id')[f'order_in_{window}d'].sum().reset_index()
        counts.columns = ['product_id', f'orders_last_{window}d']
        prod_features = prod_features.merge(counts, on='product_id', how='left')

    # Характеристики продавца на основе всех его заказов до seed_time
    oi_all = oi[oi['ts'] < seed_time].copy()
    oi_all['revenue'] = oi_all['price'] + oi_all['freight_value']
    seller_oi = oi_all.groupby('seller_id').agg(
        seller_order_count=('order_id', 'nunique'),
        seller_item_count=('order_id', 'count'),
        seller_avg_price=('price', 'mean'),
        seller_avg_freight=('freight_value', 'mean'),
        seller_total_revenue=('revenue', 'sum'),
    ).reset_index()

    # Средний рейтинг продавца
    rev_agg_all = reviews.groupby('order_id').agg(rev_mean=('review_score', 'mean')).reset_index()
    oi_all_rev = oi_all.merge(rev_agg_all, on='order_id', how='left')
    seller_rev = oi_all_rev.groupby('seller_id').agg(
        seller_avg_review=('rev_mean', 'mean'),
        seller_review_count=('rev_mean', 'count')
    ).reset_index()
    seller_oi = seller_oi.merge(seller_rev, on='seller_id', how='left')

    # Основной продавец для каждого товара (самый частый)
    seller_mode = oi_filtered.groupby('product_id')['seller_id'].agg(
        lambda x: x.value_counts().idxmax()
    ).reset_index()
    seller_mode.columns = ['product_id', 'main_seller_id']

    prod_features = prod_features.merge(seller_mode, on='product_id', how='left')
    prod_features = prod_features.merge(seller_oi, left_on='main_seller_id', right_on='seller_id', how='left')
    prod_features.drop(columns=['seller_id', 'main_seller_id'], inplace=True)

    # Количество уникальных продавцов у товара
    seller_count = oi_filtered.groupby('product_id')['seller_id'].nunique().reset_index()
    seller_count.columns = ['product_id', 'unique_sellers']
    prod_features = prod_features.merge(seller_count, on='product_id', how='left')

    # Итоговый DataFrame с индексами entity_ids
    result = pd.DataFrame(index=entity_ids)
    prod_features = prod_features.set_index('product_id')
    result = result.join(prod_features, how='left')
    result = result.fillna(0)

    # Удаляем возможные нечисловые столбцы (если остались)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result = result[numeric_cols]

    return result