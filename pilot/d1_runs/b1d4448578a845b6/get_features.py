def get_features(db, entity_ids, seed_time):
    entity_ids = list(entity_ids)
    items = db['order_items']
    orders = db['orders']
    reviews = db['reviews']
    payments = db['payments']

    feature_columns = [
        'num_orders', 'num_items', 'sum_price', 'sum_sales', 'mean_price',
        'std_price', 'min_price', 'max_price', 'sum_freight', 'mean_freight',
        'mean_delivery_days', 'avg_review_score', 'count_reviews',
        'positive_review_ratio', 'avg_installments', 'avg_payment_value',
        'credit_card_ratio', 'recency', 'sales_30', 'sales_90'
    ]

    # History of the requested products: purchases strictly before seed_time
    hist_mask = items['product_id'].isin(entity_ids) & (items['ts'] < seed_time)
    hist_items = items.loc[hist_mask].copy()

    # No history -> all NaNs for every product
    if hist_items.empty:
        return pd.DataFrame(columns=feature_columns, index=pd.Index(entity_ids, name='product_id'), dtype=float)

    # Orders of the historical items, purchased strictly before seed_time
    order_ids = hist_items['order_id'].unique()
    hist_orders = orders.loc[orders['order_id'].isin(order_ids) & (orders['order_purchase_timestamp'] < seed_time)].copy()
    hist_items = hist_items.loc[hist_items['order_id'].isin(hist_orders['order_id'])]

    # Delivery speed: only from orders already delivered before seed_time;
    # fallback: typical ratio of estimated delivery window, from pre-seed orders
    delivered = hist_orders.loc[
        hist_orders['order_delivered_customer_date'].notna()
        & (hist_orders['order_delivered_customer_date'] < seed_time)
    ]
    delivery_days = (delivered['order_delivered_customer_date'] - delivered['order_purchase_timestamp']).dt.total_seconds() / 86400.0
    delivery_days = delivery_days.loc[delivery_days > 0]
    est_window = (delivered['order_estimated_delivery_date'] - delivered['order_purchase_timestamp']).dt.total_seconds() / 86400.0
    est_window = est_window.loc[est_window > 0]
    typical_window = (hist_orders['order_estimated_delivery_date'] - hist_orders['order_purchase_timestamp']).dt.total_seconds() / 86400.0
    typical_window = typical_window.loc[typical_window > 0]
    if not delivery_days.empty:
        ratio = est_window.mean() / typical_window.mean() if not typical_window.empty else 1.0
        delivery_estimate = float(delivery_days.mean()) / ratio if ratio > 0 else float(delivery_days.mean())
        if not np.isfinite(delivery_estimate):
            delivery_estimate = np.nan
    else:
        delivery_estimate = np.nan
    hist_items['delivery_time_days'] = delivery_estimate

    # Reviews: only those already created before seed_time
    valid_order_ids = hist_orders['order_id']
    reviews_sub = reviews.loc[reviews['order_id'].isin(valid_order_ids) & (reviews['review_creation_date'] < seed_time)]
    if not reviews_sub.empty:
        review_agg = reviews_sub.groupby('order_id').agg(
            review_count=('review_score', 'count'),
            avg_review_score=('review_score', 'mean'),
            positive_review_ratio=('review_score', lambda x: (x >= 4).mean())
        ).reset_index()
    else:
        review_agg = pd.DataFrame({
            'order_id': valid_order_ids.iloc[:0],
            'review_count': np.nan,
            'avg_review_score': np.nan,
            'positive_review_ratio': np.nan
        })

    # Payments: only records already existing before seed_time
    payments_sub = payments.loc[payments['order_id'].isin(valid_order_ids) & (payments['ts'] < seed_time)]
    if not payments_sub.empty:
        payment_agg = payments_sub.groupby('order_id').agg(
            mean_payment=('payment_value', 'mean'),
            avg_inst=('payment_installments', 'mean'),
            credit_ratio=('payment_type', lambda x: (x == 'credit_card').mean())
        ).reset_index()
    else:
        payment_agg = pd.DataFrame({
            'order_id': valid_order_ids.iloc[:0],
            'mean_payment': np.nan,
            'avg_inst': np.nan,
            'credit_ratio': np.nan
        })

    hist_items = hist_items.merge(review_agg, on='order_id', how='left')
    hist_items = hist_items.merge(payment_agg, on='order_id', how='left')

    # Per-line totals and age in days (age is computed from ts, known at seed_time)
    hist_items['total'] = hist_items['price'] + hist_items['freight_value']
    hist_items['age'] = (seed_time - hist_items['ts']).dt.days

    grouped = hist_items.groupby('product_id').agg(
        num_orders=('order_id', 'nunique'),
        num_items=('product_id', 'count'),
        sum_price=('price', 'sum'),
        sum_sales=('total', 'sum'),
        mean_price=('price', 'mean'),
        std_price=('price', 'std'),
        min_price=('price', 'min'),
        max_price=('price', 'max'),
        sum_freight=('freight_value', 'sum'),
        mean_freight=('freight_value', 'mean'),
        mean_delivery_days=('delivery_time_days', 'mean'),
        avg_review_score=('avg_review_score', 'mean'),
        count_reviews=('review_count', 'sum'),
        positive_review_ratio=('positive_review_ratio', 'mean'),
        avg_installments=('avg_inst', 'mean'),
        avg_payment_value=('mean_payment', 'mean'),
        credit_card_ratio=('credit_ratio', 'mean'),
        last_purchase=('ts', 'max'),
        sales_30=('age', lambda x: (x <= 30).sum()),
        sales_90=('age', lambda x: (x <= 90).sum())
    ).reset_index()

    grouped['recency'] = (seed_time - grouped['last_purchase']).dt.days
    grouped.drop(columns=['last_purchase'], inplace=True)

    grouped = grouped.set_index('product_id').reindex(entity_ids)

    for c in feature_columns:
        if c not in grouped.columns:
            grouped[c] = np.nan

    return grouped[feature_columns]
