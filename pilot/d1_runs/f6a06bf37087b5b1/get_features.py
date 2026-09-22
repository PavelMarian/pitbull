def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    orders = db['orders']
    order_items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']

    # Фильтруем order_items по интересующим товарам
    items = order_items[order_items['product_id'].isin(entity_ids)].copy()

    # Присоединяем orders для получения времени покупки
    items = items.merge(
        orders[['order_id', 'order_purchase_timestamp']],
        on='order_id',
        how='left'
    )

    # Оставляем только заказы, сделанные до seed_time (включительно)
    items = items[items['order_purchase_timestamp'] <= seed_time]

    # Отзывы доступны только если они уже созданы на момент seed_time
    available_reviews = reviews[reviews['review_creation_date'] <= seed_time]

    # Агрегируем доступные отзывы по заказам (одна строка на заказ)
    review_agg = available_reviews.groupby('order_id').agg(
        avg_review_score=('review_score', 'mean'),
        review_count=('review_id', 'count')
    ).reset_index()
    review_agg = review_agg.drop_duplicates(subset='order_id')

    # Агрегируем платежи по заказам (одна строка на заказ)
    payment_agg = payments.groupby('order_id').agg(
        total_payment=('payment_value', 'sum'),
        payment_count=('payment_value', 'count'),
        avg_payment=('payment_value', 'mean')
    ).reset_index()
    payment_agg = payment_agg.drop_duplicates(subset='order_id')

    # Добавляем агрегированные данные к items
    items = items.merge(review_agg, on='order_id', how='left')
    items = items.merge(payment_agg, on='order_id', how='left')

    # Периоды для признаков
    periods = {
        'all': None,
        '30d': seed_time - pd.Timedelta(days=30),
        '60d': seed_time - pd.Timedelta(days=60),
        '90d': seed_time - pd.Timedelta(days=90)
    }

    feature_dfs = []

    for suffix, start_time in periods.items():
        if start_time is None:
            mask = pd.Series(True, index=items.index)
        else:
            mask = items['order_purchase_timestamp'] >= start_time

        sub = items[mask]

        # Агрегация по товарам
        agg = sub.groupby('product_id').agg(
            **{
                f'order_count_{suffix}': ('order_id', 'count'),
                f'total_price_{suffix}': ('price', 'sum'),
                f'avg_price_{suffix}': ('price', 'mean'),
                f'total_freight_{suffix}': ('freight_value', 'sum'),
                f'avg_freight_{suffix}': ('freight_value', 'mean'),
                f'avg_review_score_{suffix}': ('avg_review_score', 'mean'),
                f'review_count_{suffix}': ('review_count', 'sum'),
                f'total_payment_{suffix}': ('total_payment', 'sum'),
                f'avg_payment_{suffix}': ('avg_payment', 'mean'),
                f'payment_count_{suffix}': ('payment_count', 'sum'),
                f'unique_sellers_{suffix}': ('seller_id', 'nunique'),
                f'unique_orders_{suffix}': ('order_id', 'nunique'),
            }
        ).reset_index()
        agg.set_index('product_id', inplace=True)
        feature_dfs.append(agg)

    # Объединяем все признаки
    if feature_dfs:
        result = feature_dfs[0]
        for df in feature_dfs[1:]:
            result = result.join(df, how='outer')
    else:
        # Если нет ни одного периода (маловероятно), создаём пустой DataFrame
        result = pd.DataFrame(index=entity_ids)

    # Добавляем recency (дней с последнего заказа)
    if not items.empty:
        last_order = items.groupby('product_id')['order_purchase_timestamp'].max()
        recency = (seed_time - last_order).dt.days
    else:
        recency = pd.Series(dtype='float64')
    result['recency_days'] = recency

    # Добавляем age (дней с первого заказа)
    if not items.empty:
        first_order = items.groupby('product_id')['order_purchase_timestamp'].min()
        age = (seed_time - first_order).dt.days
    else:
        age = pd.Series(dtype='float64')
    result['age_days'] = age

    # Заполняем пропуски нулями
    result = result.fillna(0)

    # Реиндексируем по entity_ids
    result = result.reindex(entity_ids)
    result = result.fillna(0)

    return result
