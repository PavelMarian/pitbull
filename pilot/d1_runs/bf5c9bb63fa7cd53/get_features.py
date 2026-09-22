def get_features(db, entity_ids, seed_time):
    orders = db['orders']
    order_items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']

    seed_time = pd.Timestamp(seed_time)

    # Заказы, совершённые строго до seed_time
    orders_before = orders[orders['order_purchase_timestamp'] < seed_time].copy()

    # Состав заказов по товарам
    df = order_items.merge(
        orders_before[['order_id', 'customer_id', 'order_purchase_timestamp']],
        on='order_id', how='inner'
    )

    # Отзывы: доступны только те, что уже созданы к seed_time
    rev = reviews[reviews['review_creation_date'] <= seed_time]
    rev_agg = rev.groupby('order_id')['review_score'].agg(['mean', 'count']).reset_index()
    rev_agg.columns = ['order_id', 'avg_score', 'n_reviews']
    df = df.merge(rev_agg, on='order_id', how='left')

    # Платежи: только те, что относятся к покупкам до seed_time
    pay = payments[payments['ts'] <= seed_time]
    pay_agg = pay.groupby('order_id')['payment_value'].agg(['sum', 'mean', 'count']).reset_index()
    pay_agg.columns = ['order_id', 'total_payment', 'avg_payment', 'n_payments']
    install_agg = pay.groupby('order_id')['payment_installments'].mean().reset_index(name='avg_installments')
    pay_agg = pay_agg.merge(install_agg, on='order_id', how='left')
    df = df.merge(pay_agg, on='order_id', how='left')

    def aggregate(df_part):
        if df_part.empty:
            return pd.DataFrame(index=pd.Index([], name='product_id'))
        g = df_part.groupby('product_id')
        return g.agg(
            n_items=('order_item_id', 'size'),
            n_orders=('order_id', 'nunique'),
            n_customers=('customer_id', 'nunique'),
            n_sellers=('seller_id', 'nunique'),
            sum_price=('price', 'sum'),
            avg_price=('price', 'mean'),
            sum_freight=('freight_value', 'sum'),
            avg_freight=('freight_value', 'mean'),
            avg_rating=('avg_score', 'mean'),
            n_reviews=('n_reviews', 'sum'),
            avg_total_payment=('total_payment', 'mean'),
            avg_avg_payment=('avg_payment', 'mean'),
            avg_n_payments=('n_payments', 'mean'),
            avg_installments=('avg_installments', 'mean')
        )

    # Признаки на всей доступной истории
    feats = aggregate(df)

    # Recency и длина истории
    if not df.empty:
        last_order = df.groupby('product_id')['order_purchase_timestamp'].max()
        first_order = df.groupby('product_id')['order_purchase_timestamp'].min()
        feats['days_since_last_order'] = (seed_time - last_order).dt.days
        feats['order_span_days'] = (last_order - first_order).dt.days

    # Признаки по временным окнам до seed_time
    for wd in [30, 60, 90, 180]:
        start = seed_time - pd.Timedelta(days=wd)
        sub = df[df['order_purchase_timestamp'] >= start]
        if not sub.empty:
            f = aggregate(sub)
            f.columns = [f'{c}_{wd}d' for c in f.columns]
            feats = feats.join(f, how='left')

    # Реиндекс под список сущностей и заполнение пропусков
    feats = feats.reindex(entity_ids).fillna(0)

    for col in feats.columns:
        feats[col] = pd.to_numeric(feats[col], errors='coerce').fillna(0)

    return feats
