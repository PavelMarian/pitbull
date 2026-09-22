def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    if not isinstance(seed_time, pd.Timestamp):
        seed_time = pd.Timestamp(seed_time)
    
    # извлекаем таблицы
    orders = db['orders']
    order_items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']
    
    # --- 1. Исторические заказы до seed_time ---
    orders_hist = orders[orders['order_purchase_timestamp'] < seed_time].copy()
    
    if orders_hist.empty:
        # Если нет истории, все признаки будут нулевыми
        num_entities = len(entity_ids)
        features = pd.DataFrame(index=entity_ids)
        # добавим несколько нулевых колонок
        for col in ['num_orders', 'num_sales', 'sum_price', 'avg_price',
                    'median_price', 'sum_freight', 'avg_freight',
                    'num_sellers', 'num_customers', 'recency_days',
                    'orders_last_7_days', 'orders_last_30_days',
                    'orders_last_60_days', 'orders_last_90_days',
                    'orders_last_180_days', 'orders_last_365_days']:
            features[col] = 0
        # в этом случае вернем сразу с нулевыми фичами
        return features.reindex(entity_ids)
    
    # --- 2. Объединяем с order_items для получения продаж по товарам ---
    # используем inner join, т.к. нужны только товары с продажами до seed_time
    sale_events = order_items.merge(
        orders_hist[['order_id', 'customer_id', 'order_status',
                     'order_purchase_timestamp', 'order_delivered_customer_date']],
        on='order_id',
        how='inner'
    )
    
    if sale_events.empty:
        # аналогично, нет данных
        features = pd.DataFrame(index=entity_ids)
        for col in ['num_orders', 'num_sales', 'sum_price', 'avg_price',
                    'median_price', 'sum_freight', 'avg_freight',
                    'num_sellers', 'num_customers', 'recency_days',
                    'orders_last_7_days', 'orders_last_30_days',
                    'orders_last_60_days', 'orders_last_90_days',
                    'orders_last_180_days', 'orders_last_365_days']:
            features[col] = 0
        return features.reindex(entity_ids)
    
    # --- 3. Базовые агрегаты по продажам ---
    sales_agg = sale_events.groupby('product_id').agg(
        num_sales=('order_id', 'count'),
        num_orders=('order_id', 'nunique'),
        sum_price=('price', 'sum'),
        avg_price=('price', 'mean'),
        median_price=('price', 'median'),
        min_price=('price', 'min'),
        max_price=('price', 'max'),
        sum_freight=('freight_value', 'sum'),
        avg_freight=('freight_value', 'mean'),
        num_sellers=('seller_id', 'nunique'),
        num_customers=('customer_id', 'nunique'),
        last_sale_time=('order_purchase_timestamp', 'max'),
        first_sale_time=('order_purchase_timestamp', 'min'),
    ).reset_index()
    
    # recency и возраст товара в днях
    sales_agg['recency_days'] = (
        seed_time - sales_agg['last_sale_time']
    ).dt.days
    sales_agg['product_age_days'] = (
        seed_time - sales_agg['first_sale_time']
    ).dt.days
    
    # --- 4. Продажи за последние N дней ---
    for days in [7, 30, 60, 90, 180, 365]:
        start = seed_time - pd.Timedelta(days=days)
        recent = sale_events[sale_events['order_purchase_timestamp'] >= start]
        recent_agg = recent.groupby('product_id')['order_id'].nunique().reset_index()
        recent_agg.columns = ['product_id', f'orders_last_{days}_days']
        sales_agg = sales_agg.merge(recent_agg, on='product_id', how='left')
        sales_agg[f'orders_last_{days}_days'] = sales_agg[f'orders_last_{days}_days'].fillna(0).astype(int)
    
    # --- 5. Отзывы ---
    reviews_hist = reviews[reviews['review_creation_date'] < seed_time]
    if not reviews_hist.empty:
        # связываем отзывы с товарами через order_id
        reviews_sales = sale_events[['order_id', 'product_id']].drop_duplicates().merge(
            reviews_hist[['order_id', 'review_score']],
            on='order_id',
            how='inner'
        )
        if not reviews_sales.empty:
            reviews_agg = reviews_sales.groupby('product_id').agg(
                count_reviews=('review_score', 'count'),
                avg_review=('review_score', 'mean'),
                pos_reviews=('review_score', lambda x: (x >= 4).sum()),
            ).reset_index()
            reviews_agg['pos_review_ratio'] = (
                reviews_agg['pos_reviews'] / reviews_agg['count_reviews']
            )
            reviews_agg = reviews_agg.drop(columns=['pos_reviews'])
            sales_agg = sales_agg.merge(reviews_agg, on='product_id', how='left')
        else:
            sales_agg['count_reviews'] = 0
            sales_agg['avg_review'] = 0
            sales_agg['pos_review_ratio'] = 0
    else:
        sales_agg['count_reviews'] = 0
        sales_agg['avg_review'] = 0
        sales_agg['pos_review_ratio'] = 0
    
    # --- 6. Платежи ---
    payments_hist = payments[payments['ts'] < seed_time]
    if not payments_hist.empty:
        # агрегируем платежи по заказам
        pay_agg_order = payments_hist.groupby('order_id').agg(
            pay_total=('payment_value', 'sum'),
            pay_count=('payment_sequential', 'count'),
            avg_installments=('payment_installments', 'mean'),
            credit_card_payments=('payment_type', lambda x: (x == 'credit_card').sum()),
        ).reset_index()
        pay_agg_order['credit_card_ratio'] = (
            pay_agg_order['credit_card_payments'] / pay_agg_order['pay_count']
        )
        pay_agg_order = pay_agg_order.drop(columns=['credit_card_payments'])
        
        # уникальные пары заказ-товар
        order_product = sale_events[['order_id', 'product_id']].drop_duplicates()
        pay_product = order_product.merge(pay_agg_order, on='order_id', how='left')
        
        if not pay_product.empty:
            pay_agg = pay_product.groupby('product_id').agg(
                avg_pay_total=('pay_total', 'mean'),
                avg_pay_count=('pay_count', 'mean'),
                avg_installments=('avg_installments', 'mean'),
                avg_credit_card_ratio=('credit_card_ratio', 'mean'),
            ).reset_index()
            sales_agg = sales_agg.merge(pay_agg, on='product_id', how='left')
        else:
            for col in ['avg_pay_total', 'avg_pay_count', 'avg_installments', 'avg_credit_card_ratio']:
                sales_agg[col] = 0
    else:
        for col in ['avg_pay_total', 'avg_pay_count', 'avg_installments', 'avg_credit_card_ratio']:
            sales_agg[col] = 0
    
    # --- 7. Дополнительные признаки из заказов ---
    if 'order_delivered_customer_date' in sale_events.columns:
        # средняя длительность доставки (в днях): только по заказам, уже
        # доставленным до seed_time, иначе дата доставки известна из будущего
        delivery = sale_events.copy()
        delivered = delivery[delivery['order_delivered_customer_date'].notna()
                             & (delivery['order_delivered_customer_date'] <= seed_time)].copy()
        if not delivered.empty:
            delivered['delivery_days'] = (
                delivered['order_delivered_customer_date'] - delivered['order_purchase_timestamp']
            ).dt.days
            delivery_agg = delivered.groupby('product_id')['delivery_days'].mean().reset_index()
            delivery_agg.columns = ['product_id', 'avg_delivery_days']
            sales_agg = sales_agg.merge(delivery_agg, on='product_id', how='left')
        else:
            sales_agg['avg_delivery_days'] = 0
        # доля заказов, доставленных до seed_time (по факту)
        if 'order_delivered_customer_date' in sale_events.columns:
            sale_events['delivered_before_seed'] = (
                sale_events['order_delivered_customer_date'] <= seed_time
            ).astype(int)
            delivered_ratio = sale_events.groupby('product_id')['delivered_before_seed'].mean().reset_index()
            delivered_ratio.columns = ['product_id', 'delivered_ratio']
            sales_agg = sales_agg.merge(delivered_ratio, on='product_id', how='left')
        else:
            sales_agg['delivered_ratio'] = 0
        # доля отмененных заказов
        canceled_ratio = sale_events.groupby('product_id').apply(
            lambda x: (x['order_status'] == 'canceled').mean()
        ).reset_index()
        canceled_ratio.columns = ['product_id', 'canceled_ratio']
        sales_agg = sales_agg.merge(canceled_ratio, on='product_id', how='left')
    else:
        sales_agg['avg_delivery_days'] = 0
        sales_agg['delivered_ratio'] = 0
        sales_agg['canceled_ratio'] = 0
    
    # --- 8. Устанавливаем index = product_id ---
    sales_agg = sales_agg.set_index('product_id')
    
    # --- 9. Заполняем пропуски (для товаров с историей, но отсутствующих признаков) ---
    fill_cols = sales_agg.columns.drop(['last_sale_time', 'first_sale_time'], errors='ignore')
    sales_agg[fill_cols] = sales_agg[fill_cols].fillna(0)
    
    # --- 10. Удаляем временные столбцы last_sale_time, first_sale_time ---
    sales_agg = sales_agg.drop(columns=['last_sale_time', 'first_sale_time'], errors='ignore')
    
    # --- 11. Реиндексируем под entity_ids ---
    result = sales_agg.reindex(entity_ids)
    
    # --- 12. Для отсутствующих товаров заполняем нулями ---
    result = result.fillna(0)
    
    return result