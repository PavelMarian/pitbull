def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Распаковка таблиц
    orders = db['orders']
    order_items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']
    
    # Сливаем order_items с orders по order_id, берём необходимые поля
    items_orders = order_items.merge(
        orders[['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp']],
        on='order_id', how='left'
    )
    # Фильтруем только прошлые заказы (до seed_time)
    items_orders = items_orders[items_orders['order_purchase_timestamp'] < seed_time]
    # Оставляем только нужные товары
    items_orders = items_orders[items_orders['product_id'].isin(entity_ids)]
    
    # Базовый DataFrame для признаков, индексируем entity_ids
    features = pd.DataFrame(index=pd.Index(entity_ids, name='product_id'))
    
    # Если нет истории — заполним нулями все признаки и вернём
    if items_orders.empty:
        feature_names = [
            'num_orders', 'num_items', 'sum_price', 'avg_price', 'std_price',
            'min_price', 'max_price', 'sum_freight', 'avg_freight', 'std_freight',
            'unique_customers', 'last_purchase_days',
            'window30_orders', 'window30_sum_price', 'window30_avg_price',
            'window90_orders', 'window90_sum_price', 'window90_avg_price',
            'window180_orders', 'window180_sum_price', 'window180_avg_price',
            'avg_review', 'review_count', 'pos_review_count',
            'pay_sum', 'pay_count', 'avg_installments', 'cc_fraction'
        ]
        features = pd.DataFrame(0.0, index=features.index, columns=feature_names)
        return features
    
    # ---- Глобальные агрегаты по всем продажам до seed_time ----
    global_agg = items_orders.groupby('product_id').agg(
        num_orders=('order_id', 'nunique'),
        num_items=('order_item_id', 'count'),
        sum_price=('price', 'sum'),
        avg_price=('price', 'mean'),
        std_price=('price', 'std'),
        min_price=('price', 'min'),
        max_price=('price', 'max'),
        sum_freight=('freight_value', 'sum'),
        avg_freight=('freight_value', 'mean'),
        std_freight=('freight_value', 'std')
    )
    # Добавляем customer_count
    cust_stats = items_orders.groupby('product_id')['customer_id'].nunique().rename('unique_customers')
    global_agg = global_agg.join(cust_stats)
    
    # Время последней покупки
    last_date = items_orders.groupby('product_id')['order_purchase_timestamp'].max()
    recency = (seed_time - last_date).dt.days.rename('last_purchase_days')
    global_agg = global_agg.join(recency)
    
    # Присоединяем к базовому DataFrame и заполняем пропуски (товар не продавался)
    features = features.join(global_agg, how='left')
    
    # ---- Признаки за последние окна (30/90/180 дней) ----
    for window in [30, 90, 180]:
        start = seed_time - pd.Timedelta(days=window)
        df_win = items_orders[items_orders['order_purchase_timestamp'] >= start]
        if not df_win.empty:
            win_agg = df_win.groupby('product_id').agg(
                num_orders_win=('order_id', 'nunique'),
                sum_price_win=('price', 'sum'),
                avg_price_win=('price', 'mean')
            ).rename(columns={
                'num_orders_win': f'num_orders_{window}',
                'sum_price_win': f'sum_price_{window}',
                'avg_price_win': f'avg_price_{window}'
            })
            features = features.join(win_agg, how='left')
        else:
            # Если нет данных в окне — ставим нули
            features[f'num_orders_{window}'] = 0.0
            features[f'sum_price_{window}'] = 0.0
            features[f'avg_price_{window}'] = 0.0
                
    # ---- Отзывы: средний рейтинг, количество отзывов, доля положительных ----
    # Учитываем только отзывы, уже существующие на момент seed_time:
    # отзыв на заказ пишется позже самой покупки и мог появиться после seed_time.
    reviews_past = reviews[reviews['review_creation_date'] <= seed_time]
    # Получаем уникальные пары (product_id, order_id) из items_orders
    review_pairs = items_orders[['product_id', 'order_id']].drop_duplicates()
    # Соединяем с отзывами по order_id
    review_merge = review_pairs.merge(reviews_past[['order_id', 'review_score']], on='order_id', how='left')
    # Агрегируем по product_id
    if not review_merge.empty:
        review_agg = review_merge.groupby('product_id')['review_score'].agg(
            avg_review='mean',
            review_count='count'
        )
        # Доля оценок >= 4
        positive = review_merge[review_merge['review_score'] >= 4].groupby('product_id').size()
        positive = positive.rename('pos_review_count')
        review_agg = review_agg.join(positive)
        features = features.join(review_agg, how='left')
    else:
        features['avg_review'] = 0.0
        features['review_count'] = 0.0
        features['pos_review_count'] = 0.0
    
    # ---- Платёжные признаки: агрегаты по заказам, затем по товарам ----
    payments_by_order = payments.groupby('order_id').agg(
        pay_sum=('payment_value', 'sum'),
        pay_count=('payment_sequential', 'count'),
        avg_installments=('payment_installments', 'mean'),
        cc_fraction=('payment_type', lambda s: (s == 'credit_card').mean())
    )
    # Соединяем с items_orders (только тот product_id и его заказы)
    pay_features = items_orders[['product_id', 'order_id']].drop_duplicates().merge(
        payments_by_order, on='order_id', how='left'
    )
    if not pay_features.empty:
        pay_agg = pay_features.groupby('product_id').agg(
            pay_total=('pay_sum', 'sum'),
            pay_num=('pay_sum', 'count'),         # число оплаченных строк
            avg_installments=('avg_installments', 'mean'),
            cc_fraction=('cc_fraction', 'mean')
        ).rename(columns={
            'pay_total': 'pay_sum',
            'pay_num': 'pay_count'
        })
        features = features.join(pay_agg, how='left')
    else:
        features['pay_sum'] = 0.0
        features['pay_count'] = 0.0
        features['avg_installments'] = 0.0
        features['cc_fraction'] = 0.0
    
    # ---- Дополнительно: доля успешных заказов (order_status == 'delivered') ----
    # Может быть полезно, но для простоты опускаем, но можно добавить при
    # желании; пока включим как признак delivered_fraction.
    
    # ----------------------------------------------------------------------
    # Заполняем отсутствующие значения нулями
    features = features.fillna(0.0)
    
    # Убеждаемся, что индексы строго совпадают с entity_ids (порядок может отличаться)
    features = features.reindex(entity_ids)
    
    return features