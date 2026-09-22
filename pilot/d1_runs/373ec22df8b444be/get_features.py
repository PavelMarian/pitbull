def get_features(db, entity_ids, seed_time):
    """
    Вычисляет признаки для каждого product_id на основе данных до seed_time.
    Возвращает DataFrame с индексами entity_ids (в исходном порядке).
    """
    # Распаковка баз данных
    orders = db['orders']
    items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']

    # Отбираем продажи целевых товаров, произошедшие не позднее seed_time
    items_use = items[items['product_id'].isin(entity_ids) & (items['ts'] <= seed_time)]

    # --- Агрегация по товарам, если есть история продаж ---
    if not items_use.empty:
        # Основные агрегированные метрики по каждой позиции товара
        agg = items_use.groupby('product_id').agg(
            sales_count=('order_id', 'count'),
            unique_orders=('order_id', 'nunique'),
            unique_sellers=('seller_id', 'nunique'),
            total_price=('price', 'sum'),
            avg_price=('price', 'mean'),
            std_price=('price', 'std'),
            min_price=('price', 'min'),
            max_price=('price', 'max'),
            total_freight=('freight_value', 'sum'),
            avg_freight=('freight_value', 'mean'),
            last_sale_ts=('ts', 'max')
        ).reset_index()

        # Продажи за последние 7 / 30 / 90 дней
        for period, suffix in [(7, '7d'), (30, '30d'), (90, '90d')]:
            mask = items_use['ts'] >= seed_time - pd.Timedelta(days=period)
            tmp = items_use[mask].groupby('product_id').agg(
                cnt=('order_id', 'count'),
                rev=('price', 'sum')
            ).reset_index()
            tmp.columns = ['product_id', f'sales_{suffix}', f'revenue_{suffix}']
            agg = agg.merge(tmp, on='product_id', how='left')

        # Отзывы на заказы, содержащие товар (по review_score).
        # Берём только отзывы, которые уже существуют на момент seed_time:
        # отзыв пишется позже покупки, и review_creation_date <= seed_time
        # гарантирует отсутствие чтения из будущего.
        reviews_avail = reviews[reviews['review_creation_date'] <= seed_time][['order_id', 'review_score']]
        rev = items_use[['order_id', 'product_id']].merge(
            reviews_avail,
            on='order_id',
            how='left'
        )
        rev = rev.dropna(subset=['review_score'])
        if not rev.empty:
            rev_agg = rev.groupby('product_id').agg(
                avg_review_score=('review_score', 'mean'),
                review_count=('review_score', 'count')
            ).reset_index()
        else:
            rev_agg = pd.DataFrame(columns=['product_id', 'avg_review_score', 'review_count'])

        agg = agg.merge(rev_agg, on='product_id', how='left')

    else:
        # Если история пуста – создаём пустой агрегат с нужными столбцами
        cols = ['product_id'] + [
            'sales_count', 'unique_orders', 'unique_sellers',
            'total_price', 'avg_price', 'std_price', 'min_price', 'max_price',
            'total_freight', 'avg_freight', 'sales_7d', 'revenue_7d',
            'sales_30d', 'revenue_30d', 'sales_90d', 'revenue_90d',
            'avg_review_score', 'review_count'
        ]
        agg = pd.DataFrame(columns=cols)

    # --- Сборка результирующего DataFrame ---
    # базовая таблица со всеми product_id
    res = pd.DataFrame({'product_id': entity_ids})

    # присоединяем агрегированные признаки
    if not agg.empty:
        res = res.merge(agg, on='product_id', how='left')

    # --- Признак "дней с последней продажи" ---
    if 'last_sale_ts' in res.columns:
        res['days_since_last_sale'] = (seed_time - res['last_sale_ts']).dt.days
        res['days_since_last_sale'] = res['days_since_last_sale'].fillna(999.0).astype(float)
        res = res.drop(columns=['last_sale_ts'])
    else:
        res['days_since_last_sale'] = 999.0

    # --- Заполнение пропусков числовых признаков нулями ---
    num_cols = [
        'sales_count', 'unique_orders', 'unique_sellers',
        'total_price', 'avg_price', 'std_price', 'min_price', 'max_price',
        'total_freight', 'avg_freight',
        'sales_7d', 'revenue_7d', 'sales_30d', 'revenue_30d',
        'sales_90d', 'revenue_90d',
        'avg_review_score', 'review_count'
    ]
    for col in num_cols:
        if col not in res.columns:
            res[col] = 0.0
        else:
            res[col] = res[col].fillna(0.0).astype(float)

    # Переиндексируем по entity_ids
    res = res.set_index('product_id')
    res = res.reindex(entity_ids)

    return res