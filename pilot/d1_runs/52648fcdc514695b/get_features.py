def get_features(db, entity_ids, seed_time):
    orders = db['orders']
    items = db['order_items']
    reviews = db['reviews']

    # История продаж только до seed_time
    items = items[items['ts'] <= seed_time].copy()

    # Если нет ни одной продажи – простой DataFrame с нулями
    if items.empty:
        cols = [
            'n_items', 'n_orders', 'n_customers', 'sum_price', 'avg_price',
            'sum_freight', 'avg_freight', 'count_review', 'avg_review',
            'avg_delivery_days', 'recent_30', 'recent_60', 'recent_90',
            'lifetime_days', 'days_since_last', 'months_active'
        ]
        res = pd.DataFrame(0.0, index=list(entity_ids), columns=cols)
        return res.reindex(list(entity_ids)).astype(float)

    # customer_id из orders (известен на момент покупки)
    ord_sub = orders[['order_id', 'customer_id']]
    items = items.merge(ord_sub, on='order_id', how='left')

    # Отзывы известны только с момента review_creation_date
    rev = reviews[reviews['review_creation_date'] <= seed_time]
    rev_agg = rev.groupby('order_id')['review_score'].mean().reset_index()
    rev_agg.columns = ['order_id', 'review_mean']
    items = items.merge(rev_agg, on='order_id', how='left')

    # Доставка известна только если заказ уже доставлен к seed_time
    deliv = orders[['order_id', 'order_delivered_customer_date']].copy()
    deliv = deliv[(deliv['order_delivered_customer_date'].notna())
                  & (deliv['order_delivered_customer_date'] <= seed_time)]
    items = items.merge(deliv, on='order_id', how='left')
    items['delivery_days'] = (
        items['order_delivered_customer_date'] - items['ts']
    ).dt.days

    # Основные агрегаты по товару
    feat = items.groupby('product_id').agg(
        n_items=('order_id', 'count'),
        n_orders=('order_id', 'nunique'),
        n_customers=('customer_id', 'nunique'),
        sum_price=('price', 'sum'),
        avg_price=('price', 'mean'),
        sum_freight=('freight_value', 'sum'),
        avg_freight=('freight_value', 'mean'),
        count_review=('review_mean', 'count'),
        avg_review=('review_mean', 'mean'),
        avg_delivery_days=('delivery_days', 'mean')
    )

    # Окна 30/60/90 дней до seed_time
    for window in (30, 60, 90):
        start = seed_time - pd.Timedelta(days=window)
        mask = (items['ts'] >= start) & (items['ts'] <= seed_time)
        tmp = items[mask].groupby('product_id').size()
        feat[f'recent_{window}'] = tmp

    # Первая/последняя дата покупки, интервал, время с последней покупки
    first_last = items.groupby('product_id')['ts'].agg(['min', 'max'])
    feat['lifetime_days'] = (first_last['max'] - first_last['min']).dt.days
    feat['days_since_last'] = (seed_time - first_last['max']).dt.days

    # Количество активных месяцев
    items['ym'] = items['ts'].dt.strftime('%Y-%m')
    feat['months_active'] = items.groupby('product_id')['ym'].nunique()

    # Реиндексация по всем запрошенным товарам
    feat = feat.reindex(list(entity_ids))

    # Заполняем пропуски (товары без истории) нулями
    feat = feat.fillna(0.0).astype(float)
    return feat
