def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # import pandas as pd  # уже доступен, но можно импортировать не нужно
    order_items = db['order_items']
    orders = db['orders']      # не используется напрямую, но зарезервировано
    reviews = db['reviews']
    payments = db['payments']   # не используется, но зарезервировано

    # Преобразуем в список с сохранением порядка
    entity_list = list(entity_ids)

    # Инициализируем результирующий DataFrame
    features = pd.DataFrame(index=entity_list)

    # Список всех признаков (заранее, чтобы добавить их даже при пустых данных)
    feature_names = [
        'n_orders', 'n_items', 'total_sales', 'avg_price', 'std_price',
        'avg_freight', 'std_freight', 'n_sellers', 'days_since_first',
        'days_since_last', 'lifetime_days', 'has_sales',
        'avg_review_score', 'std_review_score',
        'n_orders_last_30', 'n_orders_last_60', 'n_orders_last_90'
    ]
    for f in feature_names:
        features[f] = 0.0  # заполняем нулями по умолчанию

    # Фильтруем позиции до seed_time
    mask = (order_items['product_id'].isin(entity_list)) & (order_items['ts'] <= seed_time)
    items = order_items[mask].copy()

    if not items.empty:
        # ---- Базовые агрегаты ----
        agg = items.groupby('product_id').agg(
            n_orders=('order_id', 'nunique'),
            n_items=('price', 'size'),
            total_sales=('price', 'sum'),
            avg_price=('price', 'mean'),
            std_price=('price', 'std'),
            avg_freight=('freight_value', 'mean'),
            std_freight=('freight_value', 'std'),
            n_sellers=('seller_id', 'nunique'),
            first_purchase=('ts', 'min'),
            last_purchase=('ts', 'max')
        ).reset_index()

        # Вычисляем временные признаки
        agg['days_since_first'] = (seed_time - agg['first_purchase']).dt.days
        agg['days_since_last'] = (seed_time - agg['last_purchase']).dt.days
        agg['lifetime_days'] = (agg['last_purchase'] - agg['first_purchase']).dt.days
        agg['has_sales'] = 1   # в этой группе есть продажи

        # Убираем временные колонки, оставляем только нужные
        agg = agg.set_index('product_id')
        base_cols = [
            'n_orders', 'n_items', 'total_sales', 'avg_price', 'std_price',
            'avg_freight', 'std_freight', 'n_sellers', 'days_since_first',
            'days_since_last', 'lifetime_days', 'has_sales'
        ]
        for col in base_cols:
            if col in features.columns:
                features.loc[agg.index, col] = agg[col].values

        # ---- Отзывы ----
        # Берём только отзывы, которые уже существуют на момент seed_time:
        # отзыв на заказ пишется позже покупки, поэтому фильтруем по дате создания.
        available_reviews = reviews[reviews['review_creation_date'] <= seed_time]
        uniq_order_items = items[['product_id', 'order_id']].drop_duplicates()
        uniq_order_items = uniq_order_items.merge(
            available_reviews[['order_id', 'review_score']], on='order_id', how='inner'
        )
        if not uniq_order_items.empty:
            rev_agg = uniq_order_items.groupby('product_id').agg(
                avg_review_score=('review_score', 'mean'),
                n_reviews=('review_score', 'count')
            )
            std_rev = uniq_order_items.groupby('product_id')['review_score'].std()
            rev_agg['std_review_score'] = std_rev

            for col in ['avg_review_score', 'n_reviews', 'std_review_score']:
                if col in features.columns:
                    features[col] = features[col].astype(float)
                    features.loc[rev_agg.index, col] = rev_agg[col]

        # ---- Агрегаты за последние периоды ----
        for period in [30, 60, 90]:
            cutoff = seed_time - pd.Timedelta(days=period)
            period_items = items[(items['ts'] >= cutoff) & (items['ts'] <= seed_time)]
            if not period_items.empty:
                order_counts = period_items.groupby('product_id')['order_id'].nunique()
                col = f'n_orders_last_{period}'
                # Изначально колонка уже существует
                features.loc[order_counts.index, col] = order_counts.values

    # Заполняем NaN нулями (если остались)
    features = features.fillna(0.0)

    # Реиндексация на исходный порядок entity_ids
    features = features.reindex(entity_list)

    # Обеспечим числовой тип
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce')
    features = features.fillna(0.0)

    return features