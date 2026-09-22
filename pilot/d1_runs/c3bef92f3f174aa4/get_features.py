def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Извлечение таблиц
    orders = db['orders']
    items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']  # пока не используем, но оставлено для расширения

    # Фильтруем позиции товаров по entity_ids
    items = items[items['product_id'].isin(entity_ids)]

    # Объединяем с заказами, чтобы получить дату покупки и даты доставки
    merged = items.merge(
        orders[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']],
        on='order_id',
        how='left'
    )

    # Оставляем только события строго до seed_time
    merged = merged[merged['order_purchase_timestamp'] < seed_time]

    # Создаём пустой каркас для результата
    features = pd.DataFrame(index=entity_ids)

    # Если после фильтрации нет данных — вернуть нулевые признаки
    if merged.empty:
        features['total_orders'] = 0.0
        features['total_items'] = 0.0
        features['total_sales'] = 0.0
        features['avg_price'] = 0.0
        features['avg_freight'] = 0.0
        features['avg_total_value'] = 0.0
        features['avg_delivery_days'] = 0.0
        features['delayed_count'] = 0.0
        features['avg_review_score'] = 0.0
        features['count_reviews'] = 0.0
        features['neg_review_count'] = 0.0
        return features

    # Дата доставки известна на момент seed_time только если доставка
    # уже произошла: доставленный ранее или не доставленный вовсе
    delivered_mask = merged['order_delivered_customer_date'].notna() & (
        merged['order_delivered_customer_date'] < seed_time
    )

    # Производные колонки
    merged['total_value'] = merged['price'] + merged['freight_value']
    merged['delivery_days'] = np.where(
        delivered_mask,
        (merged['order_delivered_customer_date'] - merged['order_purchase_timestamp']).dt.days,
        np.nan
    )
    merged['is_delayed'] = np.where(
        delivered_mask & (
            merged['order_delivered_customer_date'] > merged['order_estimated_delivery_date']
        ),
        1.0,
        0.0
    )

    # Основные агрегации по товару (без учёта отзывов)
    agg = merged.groupby('product_id').agg(
        total_orders=('order_id', 'nunique'),
        total_items=('order_id', 'count'),
        total_sales=('price', 'sum'),
        avg_price=('price', 'mean'),
        avg_freight=('freight_value', 'mean'),
        avg_total_value=('total_value', 'mean'),
        avg_delivery_days=('delivery_days', 'mean'),
        delayed_count=('is_delayed', 'sum'),
    ).reset_index()

    # Устанавливаем индекс product_id для будущего соединения
    agg.set_index('product_id', inplace=True)

    # --- Признаки из отзывов (рейтингов) ---
    # Отзыв известен на момент seed_time только если он уже создан
    reviews_sel = reviews[['order_id', 'review_score', 'review_creation_date']]
    reviews_sel = reviews_sel[reviews_sel['review_creation_date'] < seed_time][['order_id', 'review_score']]

    # Сопоставляем с текущими товарами (только заказы до seed_time)
    rev_merge = merged[['product_id', 'order_id']].merge(reviews_sel, on='order_id', how='left')
    rev_merge = rev_merge.dropna(subset=['review_score'])

    if rev_merge.empty:
        # Нет ни одного отзыва к товарам
        agg['avg_review_score'] = 0.0
        agg['count_reviews'] = 0.0
        agg['neg_review_count'] = 0.0
    else:
        rev_agg = rev_merge.groupby('product_id').agg(
            avg_review_score=('review_score', 'mean'),
            count_reviews=('review_score', 'count'),
            neg_review_count=('review_score', lambda x: (x <= 1).sum())
        )
        agg = agg.join(rev_agg, how='left')

    # Объединяем результат с каркасом и заполняем пропуски нулями
    features = features.join(agg, how='left')
    features = features.fillna(0.0)

    return features
