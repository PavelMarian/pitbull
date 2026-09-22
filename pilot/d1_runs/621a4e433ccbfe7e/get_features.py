def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Извлечение таблиц
    orders = db['orders']
    order_items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']

    # Приведение seed_time к типу Timestamp
    seed_time = pd.Timestamp(seed_time)

    # Фильтр заказов до seed_time и по нужным товарам
    work = order_items[(order_items['product_id'].isin(entity_ids)) & (order_items['ts'] < seed_time)]

    # Базовые признаки за всё время (без разбиения на окна)
    if not work.empty:
        # Уникальные заказы, число позиций, суммы
        total_grp = work.groupby('product_id').agg(
            orders_total=('order_id', 'nunique'),
            units_total=('order_item_id', 'count'),  # количество позиций, т.к. каждый ряд - отдельная позиция
            price_total=('price', 'sum'),
            freight_total=('freight_value', 'sum'),
            price_avg=('price', 'mean'),
            freight_avg=('freight_value', 'mean')
        ).reset_index()
    else:
        total_grp = pd.DataFrame({'product_id': [], 'orders_total': [], 'items_total': [], 
                                   'price_total': [], 'freight_total': [], 'price_avg': [], 'freight_avg': []})

    # Совместный df для отзывов.
    # Утечка: отзыв появляется позже покупки. Оставляем только отзывы,
    # которые уже существуют на момент seed_time (по review_creation_date).
    reviews_known = reviews[reviews['review_creation_date'] <= seed_time][['order_id', 'review_score']]

    if len(work) > 0:
        work_rev = work.merge(reviews_known, on='order_id', how='left')
        # средний рейтинг и количество отзывов
        rev_grp = work_rev.groupby('product_id').agg(
            review_avg=('review_score', 'mean'),
            review_cnt=('review_score', 'count')
        ).reset_index()
        # последняя дата продажи
        last_date = work.groupby('product_id')['ts'].max().reset_index()
        last_date.columns = ['product_id', 'last_sale']
        last_date['days_since_last'] = (seed_time - last_date['last_sale']).dt.days

        # Объединение в общую таблицу всех признаков
        feat = pd.merge(total_grp, rev_grp, on='product_id', how='left')
        feat = pd.merge(feat, last_date[['product_id', 'days_since_last']], on='product_id', how='left')
    else:
        feat = total_grp

    # Создаем результат DataFrame с индексом по entity_ids
    result = pd.DataFrame(index=pd.Index(entity_ids, name='product_id'))
    if len(feat) == 0:
        # Все признаки нулевые
        for col in ['orders_total','items_total','price_total','freight_total','price_avg','freight_avg',
                    'review_avg','review_cnt','days_since_last']:
            result[col] = 0.0
    else:
        feat = feat.set_index('product_id')
        for col in feat.columns:
            result[col] = feat[col].reindex(entity_ids).fillna(0)

    # Дополнительно: те же признаки, но разбитые по временным окнам (7, 30, 90 дней)
    windows = [7, 30, 90, 180, 365]
    for w in windows:
        cutoff = seed_time - pd.Timedelta(days=w)
        mask = work['ts'] >= cutoff
        work_w = work[mask]
        if len(work_w) > 0:
            grp_w = work_w.groupby('product_id').agg(
                orders_window=('order_id', 'nunique'),
                items_window=('order_item_id', 'count'),
                price_window=('price', 'sum'),
                freight_window=('freight_value', 'sum')
            ).reset_index()
            grp_w.columns = ['product_id', f'orders_{w}d', f'items_{w}d', f'price_{w}d', f'freight_{w}d']
            # присоединяем к result
            grp_w = grp_w.set_index('product_id')
            for col in grp_w.columns:
                result[col] = grp_w[col].reindex(entity_ids).fillna(0)
        else:
            # все нули
            for col in [f'orders_{w}d', f'items_{w}d', f'price_{w}d', f'freight_{w}d']:
                result[col] = 0

    # Отзывы за последние 90 дней
    cutoff_90 = seed_time - pd.Timedelta(days=90)
    work_90 = work[work['ts'] >= cutoff_90]
    if len(work_90) > 0:
        work_rev_90 = work_90.merge(reviews_known, on='order_id', how='left')
        rev_90_grp = work_rev_90.groupby('product_id').agg(
            review_avg_90d=('review_score', 'mean'),
            review_cnt_90d=('review_score', 'count')
        ).reset_index().set_index('product_id')
        for col in rev_90_grp.columns:
            result[col] = rev_90_grp[col].reindex(entity_ids).fillna(0)
    else:
        result['review_avg_90d'] = 0
        result['review_cnt_90d'] = 0

    return result