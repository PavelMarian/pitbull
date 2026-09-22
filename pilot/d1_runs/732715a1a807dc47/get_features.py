def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    orders = db['orders'].copy()
    order_items = db['order_items'].copy()
    reviews = db['reviews'].copy()
    payments = db['payments'].copy()

    seed_time = pd.to_datetime(seed_time)

    # Товары интересуют только в заказах, совершённых до seed_time
    items = order_items[order_items['product_id'].isin(entity_ids)]

    merged = items.merge(
        orders[['order_id', 'customer_id', 'order_purchase_timestamp', 'order_delivered_customer_date']],
        on='order_id',
        how='left'
    )
    merged = merged[merged['order_purchase_timestamp'] <= seed_time]

    features = pd.DataFrame(index=entity_ids)

    # --- Признаки по продажам ---
    order_counts = merged.groupby('product_id')['order_id'].nunique().rename('num_orders')
    item_counts = merged.groupby('product_id').size().rename('num_items')
    total_price = merged.groupby('product_id')['price'].sum().rename('total_price')
    total_freight = merged.groupby('product_id')['freight_value'].sum().rename('total_freight')
    avg_price = merged.groupby('product_id')['price'].mean().rename('avg_price')
    max_price = merged.groupby('product_id')['price'].max().rename('max_price')

    features = features.join(order_counts)
    features = features.join(item_counts)
    features = features.join(total_price)
    features = features.join(total_freight)
    features = features.join(avg_price)
    features = features.join(max_price)

    # --- Клиенты и продавцы
    customers = merged.groupby('product_id')['customer_id'].nunique().rename('num_customers')
    sellers = merged.groupby('product_id')['seller_id'].nunique().rename('num_sellers')
    features = features.join(customers)
    features = features.join(sellers)

    # --- Временные признаки (дней до seed_time)
    merged['days_since_purchase'] = (seed_time - merged['order_purchase_timestamp']).dt.days
    days_since_last = merged.groupby('product_id')['days_since_purchase'].min().rename('days_since_last_order')
    days_since_first = merged.groupby('product_id')['days_since_purchase'].max().rename('days_since_first_order')
    avg_days_since = merged.groupby('product_id')['days_since_purchase'].mean().rename('avg_days_since')

    features = features.join(days_since_last)
    features = features.join(days_since_first)
    features = features.join(avg_days_since)

    # --- Продажи за последние 30/60/90 дней
    for window in [30, 60, 90]:
        recent = merged[merged['days_since_purchase'] <= window]
        recent_count = recent.groupby('product_id')['order_id'].nunique().rename(f'num_orders_{window}d')
        features = features.join(recent_count)

    sum_30 = merged[merged['days_since_purchase'] <= 30].groupby('product_id')['price'].sum().rename('total_price_30d')
    features = features.join(sum_30)

    # --- Время доставки
    # Дата доставки известна только после того, как она наступила:
    # берём только заказы, доставленные до seed_time
    delivered_mask = merged['order_delivered_customer_date'].notna() & \
                     (merged['order_delivered_customer_date'] <= seed_time)
    delivered = merged[delivered_mask].copy()
    delivered['delivery_time'] = (delivered['order_delivered_customer_date'] - delivered['order_purchase_timestamp']).dt.days
    avg_delivery = delivered.groupby('product_id')['delivery_time'].mean().rename('avg_delivery_days')
    features = features.join(avg_delivery)

    # --- Отзывы
    # Отзыв появляется в момент review_creation_date, который позже покупки:
    # учитываем только отзывы, уже существующие на seed_time
    rev = reviews[reviews['review_creation_date'] <= seed_time][['order_id', 'review_score']]
    prod_orders = merged[['product_id', 'order_id']].drop_duplicates()
    merged_rev = prod_orders.merge(rev, on='order_id', how='left')
    avg_review = merged_rev.groupby('product_id')['review_score'].mean().rename('avg_review_score')
    num_reviews = merged_rev.groupby('product_id')['review_score'].count().rename('num_reviews')

    features = features.join(avg_review)
    features = features.join(num_reviews)

    # --- Платежи
    # Платёж существует не позже момента покупки заказа (ts = момент покупки):
    # фильтруем по ts <= seed_time
    pay = payments[payments['ts'] <= seed_time][['order_id', 'payment_type', 'payment_installments', 'payment_value']]
    merged_pay = prod_orders.merge(pay, on='order_id', how='left')
    avg_payment_value = merged_pay.groupby('product_id')['payment_value'].mean().rename('avg_payment_value')
    avg_installments = merged_pay.groupby('product_id')['payment_installments'].mean().rename('avg_payment_installments')
    total_payment_sum = merged_pay.groupby('product_id')['payment_value'].sum().rename('total_payment_sum')
    pay_count = merged_pay.groupby('product_id')['payment_value'].count().rename('payment_transactions_count')
    credit_ratio = merged_pay.assign(is_credit=(merged_pay['payment_type'] == 'credit_card').astype(int)) \
                           .groupby('product_id')['is_credit'].mean().rename('credit_card_ratio')

    features = features.join(avg_payment_value)
    features = features.join(avg_installments)
    features = features.join(total_payment_sum)
    features = features.join(pay_count)
    features = features.join(credit_ratio)

    # --- Индикатор наличия продаж
    has_sales = (order_counts > 0).astype(int).rename('has_sales')
    features = features.join(has_sales)

    features = features.fillna(0)

    features = features.reindex(entity_ids)

    return features
