def get_features(db: dict, entity_ids, seed_time):
    entity_ids = pd.unique(entity_ids)  # защита от дубликатов

    # Подготовка базовых данных
    orders = db['orders'][['order_id', 'customer_id']]
    items = db['order_items'][['order_id', 'product_id', 'seller_id', 'price', 'freight_value', 'ts']]
    reviews = db['reviews'][['order_id', 'review_score', 'review_creation_date']]

    df = items.merge(orders, on='order_id', how='left')

    # Только интересующие товары и заказы, созданные до seed_time
    df = df[df['product_id'].isin(entity_ids) & (df['ts'] < seed_time)]

    # Отзывы: учитываем только те, что уже существуют на момент seed_time
    reviews = reviews[reviews['review_creation_date'] <= seed_time]
    rv = df[['order_id', 'product_id']].merge(reviews, on='order_id', how='inner')

    # Пустой датасет (ни одного заказа среди entity_ids)
    if df.empty:
        features = pd.DataFrame(index=entity_ids)
        features['total_orders'] = 0
        features['total_units'] = 0
        features['total_revenue'] = 0.0
        features['total_freight'] = 0.0
        features['avg_price'] = 0.0
        features['avg_freight'] = 0.0
        features['unique_customers'] = 0
        features['unique_sellers'] = 0
        features['total_reviews'] = 0
        features['avg_review_score'] = 0.0
        features['days_since_last_order'] = -1
        features['orders_30d'] = 0
        features['units_30d'] = 0
        features['orders_60d'] = 0
        features['units_60d'] = 0
        features['orders_90d'] = 0
        features['units_90d'] = 0
        features['has_sales'] = 0
        features['repeat_rate'] = 0.0
        return features.reindex(entity_ids)

    # Базовые агрегации (по всей истории до seed_time)
    base_group = df.groupby('product_id').agg(
        total_orders=('order_id', 'nunique'),        # уникальных заказов
        total_units=('price', 'count'),              # количество позиций (единиц)
        total_revenue=('price', 'sum'),              # суммарная выручка
        total_freight=('freight_value', 'sum'),      # суммарная стоимость доставки
        avg_price=('price', 'mean'),
        avg_freight=('freight_value', 'mean'),
        unique_customers=('customer_id', 'nunique'),
        unique_sellers=('seller_id', 'nunique'),
    )

    # Агрегации по отзывам, доступным на момент seed_time
    review_group = rv.groupby('product_id').agg(
        total_reviews=('review_score', 'count'),     # количество непустых отзывов
        avg_review_score=('review_score', 'mean'),
    )

    # Дни с момента последнего заказа
    last_ts = df.groupby('product_id')['ts'].max()
    days_since = (seed_time - last_ts).dt.days

    # Агрегации за последние 30, 60, 90 дней
    window_features = {}
    for days in [30, 60, 90]:
        mask = df['ts'] >= seed_time - pd.Timedelta(days=days)
        sub = df[mask]
        if len(sub) == 0:
            orders_cnt = pd.Series(dtype=int)
            units_cnt = pd.Series(dtype=int)
        else:
            orders_cnt = sub.groupby('product_id')['order_id'].nunique().rename(f'orders_{days}d')
            units_cnt = sub.groupby('product_id')['price'].count().rename(f'units_{days}d')
        window_features[f'orders_{days}d'] = orders_cnt
        window_features[f'units_{days}d'] = units_cnt

    # Сбор итогового DataFrame
    features = pd.DataFrame(index=entity_ids)
    features = features.join(base_group)
    features = features.join(review_group)
    features = features.join(days_since.rename('days_since_last_order'))
    for col, series in window_features.items():
        features = features.join(series, how='left')

    # Заполнение пропусков
    cnt_cols = ['total_orders', 'total_units', 'total_revenue', 'total_freight',
                'unique_customers', 'unique_sellers', 'total_reviews',
                'orders_30d', 'units_30d', 'orders_60d', 'units_60d',
                'orders_90d', 'units_90d']
    features[cnt_cols] = features[cnt_cols].fillna(0)

    mean_cols = ['avg_price', 'avg_freight', 'avg_review_score']
    features[mean_cols] = features[mean_cols].fillna(0.0)

    features['days_since_last_order'] = features['days_since_last_order'].fillna(-1)

    # Дополнительные конструкционные признаки
    features['has_sales'] = (features['total_orders'] > 0).astype(int)
    features['repeat_rate'] = np.where(
        features['unique_customers'] > 0,
        features['total_orders'] / features['unique_customers'],
        0.0
    )

    # Сохранение порядка исходного запроса
    features = features.reindex(entity_ids)
    return features
