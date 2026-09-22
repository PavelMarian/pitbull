def get_features(db, entity_ids, seed_time):
    import pandas as pd
    import numpy as np

    seed_time = pd.Timestamp(seed_time)
    entity_ids = list(entity_ids)

    # ---------- 1. Prepare base table: orders + order_items ----------
    orders = db['orders']
    items = db['order_items']

    # Keep only orders strictly before seed_time
    orders_before = orders[orders['order_purchase_timestamp'] < seed_time]

    # Merge to get product-level information per order
    base = items.merge(
        orders_before[['order_id', 'order_purchase_timestamp']],
        on='order_id',
        how='inner'
    )

    # ---------- 2. Add per-order review statistics ----------
    # Reviews are written after the purchase, so only reviews that already
    # exist at seed_time may be used.
    reviews = db['reviews']
    reviews = reviews[reviews['review_creation_date'] < seed_time]
    if not reviews.empty:
        review_stats = reviews.groupby('order_id').agg(
            avg_review_score=('review_score', 'mean'),
            review_count=('review_score', 'count')
        ).reset_index()
        base = base.merge(review_stats, on='order_id', how='left')
    else:
        base['avg_review_score'] = np.nan
        base['review_count'] = 0

    # ---------- 3. Add per-order payment statistics ----------
    payments = db['payments']
    if not payments.empty:
        payment_stats = payments.groupby('order_id').agg(
            total_payment=('payment_value', 'sum'),
            payment_count=('payment_value', 'count'),
            avg_installments=('payment_installments', 'mean')
        ).reset_index()
        base = base.merge(payment_stats, on='order_id', how='left')
    else:
        base['total_payment'] = np.nan
        base['payment_count'] = 0
        base['avg_installments'] = np.nan

    # ---------- 4. Compute product-level features ----------
    if base.empty:
        # No historical data at all – return zeros with recency=9999
        feature_cols = [
            'order_count', 'total_items', 'total_price', 'avg_price',
            'total_freight', 'avg_freight', 'distinct_sellers', 'recency_days',
            'avg_review_score', 'review_count', 'total_payment', 'payment_count',
            'avg_installments',
            'order_count_last_30d', 'order_count_last_60d', 'order_count_last_90d',
            'total_price_last_30d', 'total_price_last_60d', 'total_price_last_90d'
        ]
        features = pd.DataFrame(0, index=entity_ids, columns=feature_cols)
        features['recency_days'] = 9999
        return features

    grouped = base.groupby('product_id')

    # Core aggregates
    features = pd.DataFrame({
        'order_count': grouped['order_id'].nunique(),
        'total_items': grouped.size(),
        'total_price': grouped['price'].sum(),
        'avg_price': grouped['price'].mean(),
        'total_freight': grouped['freight_value'].sum(),
        'avg_freight': grouped['freight_value'].mean(),
        'distinct_sellers': grouped['seller_id'].nunique(),
        'recency_days': (seed_time - grouped['order_purchase_timestamp'].max()).dt.days,
        'avg_review_score': grouped['avg_review_score'].mean(),
        'review_count': grouped['review_count'].sum(),
        'total_payment': grouped['total_payment'].sum(),
        'payment_count': grouped['payment_count'].sum(),
        'avg_installments': grouped['avg_installments'].mean(),
    })

    # Time‑window features (last 30/60/90 days before seed_time)
    for window in [30, 60, 90]:
        mask = (
            (base['order_purchase_timestamp'] >= seed_time - pd.Timedelta(days=window)) &
            (base['order_purchase_timestamp'] < seed_time)
        )
        temp_count = base[mask].groupby('product_id').size().rename(f'order_count_last_{window}d')
        temp_price = base[mask].groupby('product_id')['price'].sum().rename(f'total_price_last_{window}d')
        features = features.join(temp_count, how='left')
        features = features.join(temp_price, how='left')

    # Fill missing values (products without data in some windows)
    features = features.fillna(0)

    # Reindex to exactly the requested entity_ids
    features = features.reindex(entity_ids, fill_value=0)

    # For products with no orders, set recency to a large number
    features.loc[features['order_count'] == 0, 'recency_days'] = 9999

    # Ensure all columns are numeric (they already are)
    return features