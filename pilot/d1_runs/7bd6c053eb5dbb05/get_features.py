def get_features(db, entity_ids, seed_time):
    # Prepare base purchases data with timestamp
    items = db['order_items'][['order_id', 'product_id', 'seller_id', 'price', 'freight_value']]
    orders = db['orders'][['order_id', 'order_purchase_timestamp']]
    reviews = db['reviews'][['order_id', 'review_score', 'review_creation_date']]

    merged = items.merge(orders, on='order_id', how='left')
    # Keep only purchases before seed_time
    merged = merged[merged['order_purchase_timestamp'] < seed_time]
    if merged.empty:
        # Edge case: no history at all
        result = pd.DataFrame(index=entity_ids)
        # still add time‑based features
        sec = seed_time
        day_of_year = sec.dayofyear
        month = sec.month
        weekday = sec.weekday()
        result['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.0)
        result['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.0)
        result['month'] = month
        result['weekday'] = weekday
        return result.reindex(entity_ids)

    # Calculate days before seed
    merged['days_before'] = (seed_time - merged['order_purchase_timestamp']).dt.days

    # Only reviews that already exist at seed_time are visible:
    # a review is written after the purchase, often much later than seed_time.
    reviews_known = reviews[reviews['review_creation_date'] < seed_time]
    reviews_known = reviews_known[['order_id', 'review_score']]

    # Merge review scores (one or more per order)
    merged = merged.merge(reviews_known, on='order_id', how='left')

    # 1. All‑time features (no window restriction)
    all_time = merged.copy()
    all_agg = all_time.groupby('product_id').agg(
        count_orders_all=('order_id', 'nunique'),
        units_all=('price', 'count'),
        sum_price_all=('price', 'sum'),
        sum_freight_all=('freight_value', 'sum'),
        avg_price_all=('price', 'mean'),
        avg_freight_all=('freight_value', 'mean'),
        seller_count_all=('seller_id', 'nunique'),
        review_count_all=('review_score', 'count'),
        avg_review_all=('review_score', 'mean'),
        min_ts=('order_purchase_timestamp', 'min'),
        max_ts=('order_purchase_timestamp', 'max')
    ).reset_index()
    all_agg.set_index('product_id', inplace=True)

    # 2. Recency / frequency features from all‑time
    all_agg['days_since_first_all'] = (seed_time - all_agg['min_ts']).dt.days
    all_agg['days_since_last_all'] = (seed_time - all_agg['max_ts']).dt.days
    # Fill possible NaN (if a product appears only once, last=first)
    all_agg['days_since_first_all'] = all_agg['days_since_first_all'].fillna(0)
    all_agg['days_since_last_all'] = all_agg['days_since_last_all'].fillna(0)

    # 3. Window‑based features (30, 60, 90 days)
    windows = {
        '_30': 30,
        '_60': 60,
        '_90': 90
    }
    # We'll collect them into a list of Series/DataFrames
    win_features = []

    for suffix, days in windows.items():
        mask = merged['days_before'] <= days
        win = merged[mask]
        if len(win) == 0:
            # no data in this window → all zero defaults
            win_agg = pd.DataFrame(index=entity_ids)
            for col in ['count_orders', 'total_units', 'sum_price', 'sum_freight',
                        'avg_price', 'avg_freight', 'seller_count', 'review_count', 'avg_review']:
                win_agg[col + suffix] = 0.0
        else:
            win_agg = win.groupby('product_id').agg(
                count_orders=('order_id', 'nunique'),
                total_units=('price', 'count'),
                sum_price=('price', 'sum'),
                sum_freight=('freight_value', 'sum'),
                avg_price=('price', 'mean'),
                avg_freight=('freight_value', 'mean'),
                seller_count=('seller_id', 'nunique'),
                review_count=('review_score', 'count'),
                avg_review=('review_score', 'mean')
            )
            # rename columns
            win_agg.columns = [c + suffix for c in win_agg.columns]
            # if some products have no rows, they won't appear
        win_features.append(win_agg)

    # Combine all pieces:
    # Start from entity_ids as index
    result = pd.DataFrame(index=entity_ids)

    # All‑time features (with reindexing to entity_ids)
    for col in all_agg.columns:
        result[col] = all_agg[col].reindex(entity_ids).values

    # Window features: add them, default missing values to NaN (let classifier handle)
    for win_agg in win_features:
        # reindex and invert? win_agg might not contain all entity_ids, so reindex
        win_agg = win_agg.reindex(entity_ids)
        for col in win_agg.columns:
            result[col] = win_agg[col].values

    # Replace NaN to 0 for count/sum columns, keep avg as NaN
    # (optional: we'll just fill all NaN with 0 for sanity)
    result = result.fillna(0)

    # 4. Time‑based seasonality features
    # seed_time must be used as the anchor
    year_day = seed_time.dayofyear
    month = seed_time.month
    weekday = seed_time.weekday()  # Monday = 0

    result['sin_day'] = np.sin(2 * np.pi * year_day / 365.0)
    result['cos_day'] = np.cos(2 * np.pi * year_day / 365.0)
    result['month'] = month
    result['weekday'] = weekday

    # Ensure the index and order correspond to entity_ids
    return result.loc[entity_ids]
