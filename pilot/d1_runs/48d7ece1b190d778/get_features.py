def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    seed_time = pd.Timestamp(seed_time)
    if seed_time.tzinfo is not None:
        seed_time = seed_time.tz_localize(None)

    # Prepare main tables
    orders = db['orders'].copy()
    order_items = db['order_items'].copy()
    reviews = db['reviews'].copy()
    payments = db['payments'].copy()

    # Keep only orders placed before seed_time
    orders_before = orders[orders['order_purchase_timestamp'] <= seed_time][
        ['order_id', 'customer_id', 'order_estimated_delivery_date', 'order_delivered_customer_date']
    ]

    # Merge order_items with these orders
    items = order_items[['order_id', 'product_id', 'seller_id', 'price',
                         'freight_value', 'ts']].merge(
        orders_before, on='order_id', how='inner'
    )
    # Safety filter by ts
    items = items[items['ts'] <= seed_time].copy()

    # Create an empty result DataFrame, reindexed to entity_ids
    result = pd.DataFrame(index=pd.Index(entity_ids, name='product_id'))

    # ---- Per‑product aggregation ----
    grp = items.groupby('product_id')

    # 1. Item and order counts
    result['total_items'] = grp.size().reindex(entity_ids).fillna(0)
    result['total_orders'] = grp['order_id'].nunique().reindex(entity_ids).fillna(0)

    # 2. Price features
    result['sum_price'] = grp['price'].sum().reindex(entity_ids).fillna(0)
    result['mean_price'] = grp['price'].mean().reindex(entity_ids).fillna(0)
    result['min_price'] = grp['price'].min().reindex(entity_ids).fillna(0)
    result['max_price'] = grp['price'].max().reindex(entity_ids).fillna(0)
    result['std_price'] = grp['price'].std().reindex(entity_ids).fillna(0)

    # 3. Freight features
    result['sum_freight'] = grp['freight_value'].sum().reindex(entity_ids).fillna(0)
    result['mean_freight'] = grp['freight_value'].mean().reindex(entity_ids).fillna(0)
    result['std_freight'] = grp['freight_value'].std().reindex(entity_ids).fillna(0)

    # 4. Unique customers / sellers
    result['unique_customers'] = grp['customer_id'].nunique().reindex(entity_ids).fillna(0)
    result['unique_sellers'] = grp['seller_id'].nunique().reindex(entity_ids).fillna(0)

    # 5. Time‑related features
    first_purchase = grp['ts'].min()
    last_purchase = grp['ts'].max()

    result['days_since_first'] = (seed_time - first_purchase).dt.days.reindex(entity_ids).fillna(0)
    result['days_since_last'] = (seed_time - last_purchase).dt.days.reindex(entity_ids).fillna(0)
    span = (last_purchase - first_purchase).dt.days
    result['purchase_span_days'] = span.reindex(entity_ids).fillna(0)

    # 6. Average interval between distinct purchase dates
    # keep one row per (product, order) to get distinct purchase events
    unique_events = items[['product_id', 'order_id', 'ts']].drop_duplicates(
        subset=['product_id', 'order_id']
    ).sort_values(['product_id', 'ts'])
    unique_events['prev_ts'] = unique_events.groupby('product_id')['ts'].shift()
    unique_events['interval_days'] = (unique_events['ts'] - unique_events['prev_ts']).dt.days

    avg_interval = unique_events.groupby('product_id')['interval_days'].mean()
    min_interval = unique_events.groupby('product_id')['interval_days'].min()
    max_interval = unique_events.groupby('product_id')['interval_days'].max()
    n_distinct_dates = unique_events.groupby('product_id')['ts'].nunique()

    result['avg_purchase_interval'] = avg_interval.reindex(entity_ids).fillna(0)
    result['min_purchase_interval'] = min_interval.reindex(entity_ids).fillna(0)
    result['max_purchase_interval'] = max_interval.reindex(entity_ids).fillna(0)
    result['unique_purchase_dates'] = n_distinct_dates.reindex(entity_ids).fillna(0)

    # 7. Features from reviews (only for orders that occurred before seed_time).
    # A review is written after the purchase, so only reviews that already
    # exist at seed_time (review_creation_date <= seed_time) may be used.
    if len(items) > 0:
        reviews_before = reviews[reviews['review_creation_date'] <= seed_time][['order_id', 'review_score']]
        rev_merged = items[['order_id', 'product_id']].merge(
            reviews_before, on='order_id', how='inner'
        )
        if not rev_merged.empty:
            rev_grp = rev_merged.groupby('product_id')['review_score']
            result['avg_review_score'] = rev_grp.mean().reindex(entity_ids).fillna(0)
            result['count_reviews'] = rev_grp.count().reindex(entity_ids).fillna(0)
            result['std_review_score'] = rev_grp.std().reindex(entity_ids).fillna(0)
            result['min_review_score'] = rev_grp.min().reindex(entity_ids).fillna(0)
            result['max_review_score'] = rev_grp.max().reindex(entity_ids).fillna(0)
        else:
            for c in ['avg_review_score', 'count_reviews', 'std_review_score',
                      'min_review_score', 'max_review_score']:
                result[c] = 0
    else:
        for c in ['avg_review_score', 'count_reviews', 'std_review_score',
                  'min_review_score', 'max_review_score']:
            result[c] = 0

    # 8. Features from payments
    if len(items) > 0:
        pay_merged = items[['order_id', 'product_id']].merge(
            payments[['order_id', 'payment_installments', 'payment_type']],
            on='order_id', how='inner'
        )
        if not pay_merged.empty:
            result['total_payments'] = pay_merged.groupby('product_id').size().reindex(entity_ids).fillna(0)
            result['mean_installments'] = pay_merged.groupby('product_id')['payment_installments'].mean().reindex(entity_ids).fillna(0)
            # ratio of credit_card payments
            credit = pay_merged.copy()
            credit['is_credit'] = (credit['payment_type'] == 'credit_card').astype(float)
            result['credit_card_ratio'] = credit.groupby('product_id')['is_credit'].mean().reindex(entity_ids).fillna(0)
        else:
            result['total_payments'] = 0
            result['mean_installments'] = 0
            result['credit_card_ratio'] = 0
    else:
        result['total_payments'] = 0
        result['mean_installments'] = 0
        result['credit_card_ratio'] = 0

    # Ensure all columns are numeric
    result = result.astype(float)

    return result