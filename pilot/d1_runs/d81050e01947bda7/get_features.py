def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    orders = db['orders'].copy()
    order_items = db['order_items'].copy()
    reviews = db['reviews'].copy()
    payments = db['payments'].copy()

    entity_ids = list(entity_ids)

    # Только заказы, момент покупки которых не позже seed_time
    orders = orders[orders['order_purchase_timestamp'] <= seed_time].copy()

    # Даты доставки известны только если доставка уже произошла к seed_time
    delivered = orders['order_delivered_customer_date'].notna() & (orders['order_delivered_customer_date'] <= seed_time)
    shipped = orders['order_delivered_carrier_date'].notna() & (orders['order_delivered_carrier_date'] <= seed_time)

    orders['delivery_time_days'] = np.where(
        delivered,
        (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days,
        np.nan,
    )
    orders['late'] = np.where(
        delivered,
        (orders['order_delivered_customer_date'] > orders['order_estimated_delivery_date']).astype(float),
        np.nan,
    )
    orders['is_delivered'] = delivered.astype(int)
    orders['is_shipped'] = shipped.astype(int)
    # Оценённая дата доставки известна на момент покупки, её можно сравнивать с seed_time
    orders['is_open_late'] = (
        (~delivered)
        & orders['order_estimated_delivery_date'].notna()
        & (orders['order_estimated_delivery_date'] < seed_time)
    ).astype(int)

    # Позиции только тех заказов, что куплены не позже seed_time
    order_items = order_items[order_items['product_id'].isin(entity_ids)]
    order_items = order_items[order_items['ts'] <= seed_time]

    basket_sizes = order_items.groupby('order_id')['order_item_id'].count().rename('basket_size')

    item_orders = order_items.merge(
        orders[['order_id', 'customer_id', 'order_purchase_timestamp',
                'delivery_time_days', 'late', 'is_delivered', 'is_shipped', 'is_open_late']],
        on='order_id',
        how='inner'
    )
    item_orders = item_orders.merge(basket_sizes, on='order_id', how='left')

    item_orders['days_since_purchase'] = (seed_time - item_orders['order_purchase_timestamp']).dt.days

    # --- Агрегации по всем доступным данным ---
    agg_item = item_orders.groupby('product_id').agg(
        count_orders=('order_id', 'nunique'),
        count_items=('order_id', 'count'),
        avg_price=('price', 'mean'),
        std_price=('price', 'std'),
        sum_price=('price', 'sum'),
        avg_freight=('freight_value', 'mean'),
        sum_freight=('freight_value', 'sum'),
        unique_sellers=('seller_id', 'nunique'),
        unique_customers=('customer_id', 'nunique'),
        avg_delivery_days=('delivery_time_days', 'mean'),
        late_rate=('late', 'mean'),
        delivered_orders=('is_delivered', 'sum'),
        shipped_orders=('is_shipped', 'sum'),
        open_late_orders=('is_open_late', 'sum'),
        avg_basket_size=('basket_size', 'mean'),
        first_order_ts=('order_purchase_timestamp', 'min'),
        last_order_ts=('order_purchase_timestamp', 'max'),
    ).reset_index()

    agg_item['std_price'] = agg_item['std_price'].fillna(0)
    agg_item['product_age_days'] = (seed_time - agg_item['first_order_ts']).dt.days
    agg_item['days_since_last_order'] = (seed_time - agg_item['last_order_ts']).dt.days
    agg_item = agg_item.drop(columns=['first_order_ts', 'last_order_ts'])

    # --- Агрегации за последние 30/60/90 дней ---
    for window in [30, 60, 90]:
        mask = item_orders['days_since_purchase'] <= window
        sub = item_orders[mask]
        if not sub.empty:
            agg_win = sub.groupby('product_id').agg(
                **{f'count_orders_{window}d': ('order_id', 'nunique'),
                   f'sum_price_{window}d': ('price', 'sum'),
                   f'unique_customers_{window}d': ('customer_id', 'nunique'),
                   f'avg_price_{window}d': ('price', 'mean')}
            ).reset_index()
            agg_item = agg_item.merge(agg_win, on='product_id', how='left')
        else:
            for col in [f'count_orders_{window}d', f'sum_price_{window}d',
                        f'unique_customers_{window}d', f'avg_price_{window}d']:
                agg_item[col] = np.nan

    # --- Платёжные признаки (позиции заказов, купленных не позже seed_time) ---
    order_ids = item_orders['order_id'].unique()
    payments_filtered = payments[payments['order_id'].isin(order_ids)]
    payments_filtered = payments_filtered[payments_filtered['ts'] <= seed_time]

    pay_agg = payments_filtered.groupby('order_id').agg(
        total_payment=('payment_value', 'sum'),
        avg_installments=('payment_installments', 'mean'),
        count_payments=('payment_value', 'count'),
    ).reset_index()

    pay_dummies = pd.get_dummies(payments_filtered['payment_type'], prefix='pay_type', dtype=float)
    pay_dummies['order_id'] = payments_filtered['order_id']
    pay_dummies_agg = pay_dummies.groupby('order_id').sum().reset_index()
    pay_agg = pay_agg.merge(pay_dummies_agg, on='order_id', how='left')

    product_orders = item_orders[['product_id', 'order_id']].drop_duplicates()
    product_pay = product_orders.merge(pay_agg, on='order_id', how='left')

    # Гарантируем наличие всех колонок типов оплаты, иначе named agg падает на усечённых базах
    for t in ['credit_card', 'boleto', 'voucher', 'debit_card']:
        col = 'pay_type_' + t
        if col not in product_pay.columns:
            product_pay[col] = np.nan

    agg_pay = product_pay.groupby('product_id').agg(
        avg_total_payment=('total_payment', 'mean'),
        avg_installments_order=('avg_installments', 'mean'),
        avg_count_payments=('count_payments', 'mean'),
        sum_total_payment=('total_payment', 'sum'),
        pay_type_credit_card_rate=('pay_type_credit_card', 'mean'),
        pay_type_boleto_rate=('pay_type_boleto', 'mean'),
        pay_type_voucher_rate=('pay_type_voucher', 'mean'),
        pay_type_debit_card_rate=('pay_type_debit_card', 'mean'),
    ).reset_index()

    # --- Отзывы: доступен только отзыв, уже созданный к seed_time ---
    reviews_filtered = reviews[reviews['order_id'].isin(order_ids)]
    reviews_filtered = reviews_filtered[reviews_filtered['review_creation_date'] <= seed_time]
    rev_agg = reviews_filtered.groupby('order_id').agg(
        avg_review_score=('review_score', 'mean'),
        review_count=('review_score', 'count'),
    ).reset_index()

    product_rev = product_orders.merge(rev_agg, on='order_id', how='left')
    agg_rev = product_rev.groupby('product_id').agg(
        avg_review_score=('avg_review_score', 'mean'),
        total_reviews=('review_count', 'sum'),
        review_count_per_order=('review_count', 'mean'),
    ).reset_index()

    # --- Объединение всех признаков ---
    features = agg_item.merge(agg_pay, on='product_id', how='left').merge(agg_rev, on='product_id', how='left')
    features = features.set_index('product_id')

    features = features.reindex(entity_ids)
    features = features.fillna(0)

    return features
