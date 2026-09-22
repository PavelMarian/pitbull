def get_features(db, entity_ids, seed_time):
    # заказы, совершённые строго до seed_time
    orders = db['orders'][['order_id', 'order_purchase_timestamp']].copy()
    orders = orders[orders['order_purchase_timestamp'] < seed_time]

    # позиции только по запрошенным товарам
    items = db['order_items'][['order_id', 'product_id', 'price', 'freight_value']].copy()
    items = items[items['product_id'].isin(entity_ids)]

    df = items.merge(orders, on='order_id', how='inner')

    # отзывы: учитываем только те, что уже существуют на момент seed_time
    reviews = db['reviews'][['order_id', 'review_score', 'review_creation_date']].copy()
    reviews = reviews[reviews['review_creation_date'] < seed_time]
    reviews_agg = reviews.groupby('order_id')['review_score'].agg(['mean', 'count']).reset_index()
    reviews_agg.columns = ['order_id', 'review_mean', 'review_count']
    df = df.merge(reviews_agg, on='order_id', how='left')

    # платежи: только те, что относятся к покупкам до seed_time
    pay = db['payments'][['order_id', 'payment_type', 'payment_value', 'payment_installments', 'ts']].copy()
    pay = pay[pay['ts'] < seed_time]
    pay_agg = pay.groupby('order_id').agg(
        total_pay=('payment_value', 'sum'),
        pay_count=('payment_value', 'count'),
        avg_install=('payment_installments', 'mean'),
        n_types=('payment_type', 'nunique')
    ).reset_index()
    df = df.merge(pay_agg, on='order_id', how='left')

    fill_cols = ['review_mean', 'review_count', 'total_pay', 'pay_count', 'avg_install', 'n_types']
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # сколько дней назад от seed_time совершена покупка
    df['days_ago'] = (seed_time - df['order_purchase_timestamp']).dt.days

    # --- общие признаки за всё доступное прошлое ---
    general = df.groupby('product_id').agg(
        num_positions=('price', 'count'),
        num_orders=('order_id', 'nunique'),
        revenue_total=('price', 'sum'),
        freight_total=('freight_value', 'sum'),
        avg_price=('price', 'mean'),
        std_price=('price', 'std'),
        min_price=('price', 'min'),
        max_price=('price', 'max'),
        avg_freight=('freight_value', 'mean'),
        avg_review=('review_mean', 'mean'),
        review_count=('review_count', 'sum'),
        avg_pay=('total_pay', 'mean'),
        pay_count=('pay_count', 'sum'),
        avg_install=('avg_install', 'mean'),
        n_pay_types=('n_types', 'max')
    ).reset_index()

    # --- признаки по временным окнам ---
    for period in [7, 30, 60, 90, 180]:
        sub = df[df['days_ago'] <= period]
        if sub.empty:
            continue
        sub_agg = sub.groupby('product_id').agg(
            cnt_orders=('order_id', 'nunique'),
            cnt_pos=('price', 'count'),
            revenue=('price', 'sum'),
            avg_price=('price', 'mean')
        ).rename(columns={
            'cnt_orders': f'orders_{period}',
            'cnt_pos': f'positions_{period}',
            'revenue': f'revenue_{period}',
            'avg_price': f'avg_price_{period}'
        }).reset_index()
        general = general.merge(sub_agg, on='product_id', how='left')

    # --- временные характеристики продаж ---
    ts = df.groupby('product_id')['order_purchase_timestamp'].agg(['min', 'max'])
    unique_days = df.groupby('product_id')['order_purchase_timestamp'].nunique().rename('unique_days')
    ts = ts.join(unique_days)

    general = general.set_index('product_id')

    general['days_since_first'] = (seed_time - ts['min']).dt.days
    general['days_since_last'] = (seed_time - ts['max']).dt.days
    general['active_days'] = (ts['max'] - ts['min']).dt.days
    general['unique_days'] = ts['unique_days']

    features = general.reindex(entity_ids, fill_value=0)
    features = features.fillna(0)

    return features
