def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    base = pd.DataFrame(index=pd.Index(entity_ids, name='product_id'))

    # Purchase history strictly before seed_time
    items_all = db['order_items']
    items = items_all[
        items_all['product_id'].isin(entity_ids) & (items_all['ts'] < seed_time)
    ].copy()

    # Reviews are known at seed_time only if already created by then
    rev = db['reviews']
    rev = rev[rev['review_creation_date'] < seed_time]
    order_reviews = rev.groupby('order_id').agg(
        avg_review_score=('review_score', 'mean'),
        num_reviews=('review_score', 'count'),
        share_positive=('review_score', lambda x: (x >= 4).mean()),
    ).reset_index()

    # Delivery facts exist at seed_time only for orders delivered before it
    orders = db['orders'][
        ['order_id', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    ].copy()
    known_delivery = orders['order_delivered_customer_date'] < seed_time
    orders.loc[~known_delivery, 'order_delivered_customer_date'] = pd.NaT

    items = items.merge(orders, on='order_id', how='left')
    items = items.merge(order_reviews, on='order_id', how='left')

    if not items.empty:
        items['days_before_seed'] = (seed_time - items['ts']).dt.days
        items['window'] = pd.cut(
            items['days_before_seed'],
            bins=[-1, 90, 180, 360, np.inf],
            labels=['w1', 'w2', 'w3', 'older'],
        )
        items['delivery_delay_days'] = (
            items['order_delivered_customer_date'] - items['order_estimated_delivery_date']
        ).dt.days

    def window_agg(mask, suffix):
        if items.empty:
            return pd.DataFrame()
        f = items[mask]
        agg = f.groupby('product_id').agg(
            units_sold=('order_item_id', 'count'),
            num_orders=('order_id', 'nunique'),
            total_revenue=('price', 'sum'),
            total_freight=('freight_value', 'sum'),
            avg_price=('price', 'mean'),
            avg_freight=('freight_value', 'mean'),
            avg_delivery_delay=('delivery_delay_days', 'mean'),
            avg_review_score=('avg_review_score', 'mean'),
            num_reviews=('num_reviews', 'sum'),
            share_positive_reviews=('share_positive', 'mean'),
        )
        agg = agg.add_suffix(f'_{suffix}')
        return agg

    aggs = []
    if not items.empty:
        for suffix, label in [('w1', 'w1'), ('w2', 'w2'), ('w3', 'w3'), ('older', 'older')]:
            aggs.append(window_agg(items['window'] == label, suffix))
        prod_history = items.groupby('product_id').agg(
            total_units_history=('order_item_id', 'count'),
            total_revenue_history=('price', 'sum'),
            avg_review_score_history=('avg_review_score', 'mean'),
            total_reviews_history=('num_reviews', 'sum'),
        )
        last_ts = items.groupby('product_id')['ts'].max()
        prod_history['days_since_last_purchase'] = (seed_time - last_ts).dt.days
    else:
        prod_history = pd.DataFrame(
            columns=['total_units_history', 'total_revenue_history',
                     'avg_review_score_history', 'total_reviews_history',
                     'days_since_last_purchase'],
            index=base.index,
        )

    features = base.copy()
    for agg in aggs:
        features = features.join(agg, how='left')
    features = features.join(prod_history, how='left')

    # Ensure every expected column exists even when a window is empty
    window_cols = ['units_sold', 'num_orders', 'total_revenue', 'total_freight',
                   'avg_price', 'avg_freight', 'avg_delivery_delay',
                   'avg_review_score', 'num_reviews', 'share_positive_reviews']
    expected = [f'{c}_{w}' for w in ['w1', 'w2', 'w3', 'older'] for c in window_cols]
    expected += ['total_units_history', 'total_revenue_history',
                 'avg_review_score_history', 'total_reviews_history',
                 'days_since_last_purchase']
    for c in expected:
        if c not in features.columns:
            features[c] = np.nan

    features['has_history'] = (~features['total_units_history'].isna()).astype(int)
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    features[numeric_cols] = features[numeric_cols].fillna(0)
    features.loc[features['has_history'] == 0, 'days_since_last_purchase'] = 9999

    # Trend and ratio features
    features['units_growth_w1_w2'] = (
        features['units_sold_w1'] - features['units_sold_w2']) / (features['units_sold_w2'] + 1e-6)
    features['revenue_growth_w1_w2'] = (
        features['total_revenue_w1'] - features['total_revenue_w2']) / (features['total_revenue_w2'] + 1e-6)
    features['units_growth_w2_w3'] = (
        features['units_sold_w2'] - features['units_sold_w3']) / (features['units_sold_w3'] + 1e-6)
    features['aov_w1'] = features['total_revenue_w1'] / (features['num_orders_w1'] + 1e-6)
    features['aov_w2'] = features['total_revenue_w2'] / (features['num_orders_w2'] + 1e-6)
    features['aov_w3'] = features['total_revenue_w3'] / (features['num_orders_w3'] + 1e-6)
    features['freight_ratio_w1'] = features['total_freight_w1'] / (features['total_revenue_w1'] + 1e-6)
    features['freight_ratio_w2'] = features['total_freight_w2'] / (features['total_revenue_w2'] + 1e-6)

    features = features.reindex(entity_ids)
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    features[numeric_cols] = features[numeric_cols].fillna(0)
    return features
