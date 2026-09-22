import pandas as pd

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Извлекаем таблицы
    orders = db['orders']
    items = db['order_items']
    reviews = db['reviews']
    # payments = db['payments']  # можно использовать, но пока не добавляем

    # Убедимся, что entity_ids — список/массив
    if not isinstance(entity_ids, (list, pd.Series, np.ndarray)):
        entity_ids = list(entity_ids)

    # Фильтруем заказы товаров до seed_time
    mask = items['product_id'].isin(entity_ids) & (items['ts'] <= seed_time)
    items_f = items.loc[mask].copy()

    # Базовые признаки: если нет истории, заполняем нулями
    base_cols = [
        'sales_count', 'sales_last_30d', 'sales_last_90d', 'total_revenue',
        'total_freight', 'avg_price', 'avg_freight', 'days_since_last',
        'days_history', 'days_from_first', 'avg_inter_between',
        'review_count', 'avg_review_score'
    ]
    
    if items_f.empty:
        feats = pd.DataFrame(0.0, index=entity_ids, columns=base_cols)
        return feats

    # === 1. Общие агрегаты по товару ===
    agg = items_f.groupby('product_id').agg(
        sales_count=('order_id', 'count'),
        total_revenue=('price', 'sum'),
        total_freight=('freight_value', 'sum'),
        avg_price=('price', 'mean'),
        avg_freight=('freight_value', 'mean'),
        first_ts=('ts', 'min'),
        last_ts=('ts', 'max')
    ).reset_index()

    # Временные признаки
    agg['days_since_last'] = (seed_time - agg['last_ts']).dt.days
    agg['days_history'] = (agg['last_ts'] - agg['first_ts']).dt.days
    agg['days_from_first'] = (seed_time - agg['first_ts']).dt.days

    # Средний интервал между покупками
    agg['avg_inter_between'] = np.where(
        agg['sales_count'] > 1,
        agg['days_history'] / (agg['sales_count'] - 1),
        0.0
    )

    # === 2. Продажи за последние 30 и 90 дней ===
    thr30 = seed_time - pd.Timedelta(days=30)
    thr90 = seed_time - pd.Timedelta(days=90)

    sales_30 = items_f[items_f['ts'] >= thr30].groupby('product_id').size()
    sales_90 = items_f[items_f['ts'] >= thr90].groupby('product_id').size()

    agg = agg.merge(
        sales_30.rename('sales_last_30d'),
        left_on='product_id', right_index=True, how='left'
    )
    agg = agg.merge(
        sales_90.rename('sales_last_90d'),
        left_on='product_id', right_index=True, how='left'
    )
    agg['sales_last_30d'] = agg['sales_last_30d'].fillna(0).astype(int)
    agg['sales_last_90d'] = agg['sales_last_90d'].fillna(0).astype(int)

    # === 3. Отзывы по заказам с этим товаром ===
    # Берём только отзывы, которые уже существуют на момент seed_time:
    # отзыв появляется в базе позже покупки (дата создания отзыва).
    # Присоединяем отзывы к строкам товаров (по order_id)
    reviews_f = reviews.loc[reviews['review_creation_date'] <= seed_time,
                            ['order_id', 'review_score']]
    items_reviews = items_f.merge(
        reviews_f,
        on='order_id',
        how='left'
    )

    rev_agg = items_reviews.groupby('product_id').agg(
        review_count=('review_score', 'count'),
        avg_review_score=('review_score', 'mean')
    ).reset_index()

    agg = agg.merge(rev_agg, on='product_id', how='left')
    agg['review_count'] = agg['review_count'].fillna(0).astype(int)
    agg['avg_review_score'] = agg['avg_review_score'].fillna(0.0)

    # === 4. Финал — индексация по всем entity_ids ===
    feats = agg[['product_id'] + base_cols].set_index('product_id').reindex(entity_ids)
    feats[base_cols] = feats[base_cols].fillna(0.0)

    return feats