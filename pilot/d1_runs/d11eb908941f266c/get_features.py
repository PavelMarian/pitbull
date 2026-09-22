def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    orders = db['orders']
    order_items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']
    
    # Заказы до seed_time
    orders_before = orders[orders['order_purchase_timestamp'] < seed_time]
    
    # Соединение order_items с orders
    merged = order_items.merge(
        orders_before[['order_id', 'customer_id', 'order_purchase_timestamp', 'order_status']],
        on='order_id',
        how='inner'
    )
    merged = merged[merged['product_id'].isin(entity_ids)]
    
    # Агрегация отзывов по заказам: только отзывы, уже существующие на seed_time
    reviews_before = reviews[reviews['review_creation_date'] < seed_time]
    reviews_agg = reviews_before.groupby('order_id').agg(
        review_score_mean=('review_score', 'mean'),
        review_count=('review_score', 'count')
    ).reset_index()
    merged = merged.merge(reviews_agg, on='order_id', how='left')
    
    # Агрегация платежей по заказам (включая тип)
    payments['is_credit_card'] = (payments['payment_type'] == 'credit_card').astype(int)
    payments_agg = payments.groupby('order_id').agg(
        payment_count=('payment_sequential', 'count'),
        payment_installments_mean=('payment_installments', 'mean'),
        payment_value_sum=('payment_value', 'sum'),
        credit_card_count=('is_credit_card', 'sum')
    ).reset_index()
    merged = merged.merge(payments_agg, on='order_id', how='left')
    
    # Дополнительные бинарные признаки
    merged['high_review'] = (merged['review_score_mean'] >= 4).astype(int)
    merged['is_delivered'] = (merged['order_status'] == 'delivered').astype(int)
    merged['days_before'] = (seed_time - merged['order_purchase_timestamp']).dt.days
    
    # Группировка по товарам
    grouped = merged.groupby('product_id')
    
    features = pd.DataFrame(index=entity_ids)
    
    # Основные агрегаты
    features['total_orders'] = grouped['order_id'].nunique()
    features['total_quantity'] = grouped.size()
    features['total_sales'] = grouped['price'].sum()
    features['avg_price'] = grouped['price'].mean()
    features['total_freight'] = grouped['freight_value'].sum()
    features['avg_freight'] = grouped['freight_value'].mean()
    features['unique_customers'] = grouped['customer_id'].nunique()
    features['avg_review_score'] = grouped['review_score_mean'].mean()
    features['review_count'] = grouped['review_count'].sum()
    features['high_review_ratio'] = grouped['high_review'].mean()
    features['delivered_ratio'] = grouped['is_delivered'].mean()
    features['payment_installments_mean'] = grouped['payment_installments_mean'].mean()
    features['payment_count_sum'] = grouped['payment_count'].sum()
    features['payment_value_sum'] = grouped['payment_value_sum'].sum()
    features['credit_card_count_sum'] = grouped['credit_card_count'].sum()
    
    # Производные признаки из платежей
    features['avg_payments_per_order'] = features['payment_count_sum'] / features['total_orders']
    features['credit_card_ratio'] = features['credit_card_count_sum'] / features['payment_count_sum']
    
    # Временные признаки
    features['first_order_date'] = grouped['order_purchase_timestamp'].min()
    features['last_order_date'] = grouped['order_purchase_timestamp'].max()
    features['days_since_last_order'] = (seed_time - features['last_order_date']).dt.days
    features['days_since_first_order'] = (seed_time - features['first_order_date']).dt.days
    
    # Средний интервал между заказами
    def avg_days_between(series):
        dates = series.sort_values().values
        if len(dates) <= 1:
            return np.nan
        diffs = np.diff(dates)
        return (diffs / np.timedelta64(1, 'D')).mean()
    features['avg_days_between_orders'] = grouped['order_purchase_timestamp'].apply(avg_days_between)
    
    # Признаки за последние окна
    for window in [30, 60, 90, 180]:
        mask = merged['days_before'] <= window
        temp = merged[mask].groupby('product_id')
        features[f'orders_last_{window}d'] = temp['order_id'].nunique()
        features[f'sales_last_{window}d'] = temp['price'].sum()
    
    # Заполнение пропусков (нет данных или деление на ноль)
    features = features.fillna(0)
    
    # Удаление нечисловых столбцов
    features = features.drop(columns=['first_order_date', 'last_order_date'])
    
    # Убедимся, что все индексы присутствуют и в нужном порядке
    features = features.reindex(entity_ids)
    
    return features