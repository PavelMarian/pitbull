def get_features(db, entity_ids, seed_time):
    orders = db['orders']
    items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']

    # Только заказы, сделанные до seed_time
    orders_before = orders[orders['order_purchase_timestamp'] <= seed_time]

    # Только отзывы, уже существующие на момент seed_time
    reviews_before = reviews[reviews['review_creation_date'] <= seed_time]

    # Только платежи, уже существующие на момент seed_time
    payments_before = payments[payments['ts'] <= seed_time]

    # Объединяем позиции заказов с информацией о заказах
    merged = items.merge(
        orders_before[['order_id', 'customer_id', 'order_purchase_timestamp']],
        on='order_id',
        how='inner'
    )

    # --- Признаки из заказов ---
    if not merged.empty:
        group = merged.groupby('product_id')
        features = group.agg(
            total_orders=('order_id', 'nunique'),
            total_items=('order_id', 'count'),
            avg_price=('price', 'mean'),
            avg_freight=('freight_value', 'mean'),
            total_price=('price', 'sum'),
            total_freight=('freight_value', 'sum'),
            distinct_customers=('customer_id', 'nunique'),
            distinct_sellers=('seller_id', 'nunique'),
            first_order_date=('order_purchase_timestamp', 'min'),
            last_order_date=('order_purchase_timestamp', 'max')
        )
        features['recency_days'] = (seed_time - features['last_order_date']).dt.days
        features['tenure_days'] = (seed_time - features['first_order_date']).dt.days
        features = features.drop(columns=['first_order_date', 'last_order_date'])
    else:
        # Если нет данных – создаём пустой DataFrame с нужными колонками
        features = pd.DataFrame(
            columns=['total_orders', 'total_items', 'avg_price', 'avg_freight',
                     'total_price', 'total_freight', 'distinct_customers',
                     'distinct_sellers', 'recency_days', 'tenure_days']
        )

    # --- Признаки из отзывов ---
    if not merged.empty:
        merged_rev = merged.merge(reviews_before[['order_id', 'review_score']], on='order_id', how='left')
        rev_group = merged_rev.groupby('product_id')
        features['avg_review_score'] = rev_group['review_score'].mean()
        features['num_reviews'] = rev_group['review_score'].count()
    else:
        features['avg_review_score'] = np.nan
        features['num_reviews'] = 0

    # --- Признаки из платежей ---
    if not merged.empty:
        merged_pay = merged.merge(
            payments_before[['order_id', 'payment_type', 'payment_installments', 'payment_value']],
            on='order_id',
            how='left'
        )
        pay_group = merged_pay.groupby('product_id')
        features['avg_installments'] = pay_group['payment_installments'].mean()
        features['avg_payment_value'] = pay_group['payment_value'].mean()
        features['total_payment_value'] = pay_group['payment_value'].sum()
        merged_pay['is_cc'] = (merged_pay['payment_type'] == 'credit_card').astype(int)
        features['cc_ratio'] = merged_pay.groupby('product_id')['is_cc'].mean()
    else:
        features['avg_installments'] = np.nan
        features['avg_payment_value'] = np.nan
        features['total_payment_value'] = 0
        features['cc_ratio'] = 0

    # --- Признаки за последние N дней ---
    if not merged.empty:
        merged['days_diff'] = (seed_time - merged['order_purchase_timestamp']).dt.days
        for window in [7, 30, 90, 180]:
            mask = merged['days_diff'] <= window
            cnt = merged[mask].groupby('product_id')['order_id'].nunique()
            features[f'orders_last_{window}d'] = cnt
    else:
        for window in [7, 30, 90, 180]:
            features[f'orders_last_{window}d'] = 0

    # Средний интервал между заказами (в днях)
    if not merged.empty:
        features['avg_days_between_orders'] = features['tenure_days'] / features['total_orders'].replace(0, np.nan)
    else:
        features['avg_days_between_orders'] = np.nan

    # Заполняем пропуски нулями
    features = features.fillna(0)

    # Реиндексация под запрошенные product_id
    result = features.reindex(entity_ids, fill_value=0)

    return result