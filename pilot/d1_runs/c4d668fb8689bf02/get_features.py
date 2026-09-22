def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Извлекаем таблицы
    orders = db['orders']
    order_items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']
    
    # Преобразуем entity_ids в список (на случай если это массив)
    entity_ids = list(entity_ids)
    
    # Если нет товаров, возвращаем пустой DataFrame
    if not entity_ids:
        return pd.DataFrame(index=entity_ids)
    
    # Фильтруем order_items по интересующим товарам
    items = order_items[order_items['product_id'].isin(entity_ids)].copy()
    
    # Если нет данных по товарам, создаем DataFrame с нулями
    if items.empty:
        return pd.DataFrame(index=entity_ids)
    
    # Присоединяем orders (только нужные колонки)
    orders_sub = orders[['order_id', 'order_purchase_timestamp',
                         'order_estimated_delivery_date',
                         'order_delivered_customer_date']].copy()
    df = items.merge(orders_sub, on='order_id', how='left')
    
    # Фильтруем по времени покупки <= seed_time (исторические данные)
    df = df[df['order_purchase_timestamp'] <= seed_time]
    
    # Если после фильтрации пусто, возвращаем нули
    if df.empty:
        return pd.DataFrame(index=entity_ids)
    
    # Добавляем агрегированные данные по отзывам (средний рейтинг на заказ).
    # Берём только отзывы, уже существующие на момент seed_time:
    # отзыв пишется позже покупки, отзыв с review_creation_date > seed_time
    # в момент предсказания ещё недоступен.
    reviews_known = reviews[reviews['review_creation_date'] <= seed_time]
    reviews_agg = reviews_known.groupby('order_id')['review_score'].mean().reset_index()
    reviews_agg.columns = ['order_id', 'avg_review_score']
    df = df.merge(reviews_agg, on='order_id', how='left')
    
    # Добавляем агрегированные данные по платежам (сумма, средний взнос, кол-во платежей)
    payments_agg = payments.groupby('order_id').agg(
        total_payment=('payment_value', 'sum'),
        num_payments=('payment_sequential', 'count'),
        avg_installments=('payment_installments', 'mean'),
        max_installments=('payment_installments', 'max')
    ).reset_index()
    df = df.merge(payments_agg, on='order_id', how='left')
    
    # Добавляем признак: была ли доставка вовремя (по оценке).
    # Учитываем только доставку, которая уже состоялась к seed_time:
    # order_delivered_customer_date заполняется после покупки, и если она
    # ещё пуста или в будущем, информация о доставке в момент seed_time
    # недоступна.
    if 'order_delivered_customer_date' in df.columns and 'order_estimated_delivery_date' in df.columns:
        delivered = (df['order_delivered_customer_date'].notna() &
                     (df['order_delivered_customer_date'] <= seed_time))
        estimated = df['order_estimated_delivery_date'].notna()
        df['delivered_on_time'] = 0
        df.loc[delivered & estimated, 'delivered_on_time'] = (
            df.loc[delivered & estimated, 'order_delivered_customer_date'] <=
            df.loc[delivered & estimated, 'order_estimated_delivery_date']
        ).astype(int)
        # Время доставки в днях (если есть)
        df['delivery_time_days'] = np.nan
        mask = delivered & estimated
        df.loc[mask, 'delivery_time_days'] = (
            (df.loc[mask, 'order_delivered_customer_date'] - df.loc[mask, 'order_purchase_timestamp']).dt.days
        )
    else:
        df['delivered_on_time'] = 0
        df['delivery_time_days'] = np.nan
    
    # Считаем количество дней от покупки до seed_time
    df['days_since_purchase'] = (seed_time - df['order_purchase_timestamp']).dt.days
    
    # Добавляем индикатор, был ли отзыв (если avg_review_score не NaN)
    df['has_review'] = df['avg_review_score'].notna().astype(int)
    
    # Заполняем пропуски в числовых признаках нулями
    df['avg_review_score'] = df['avg_review_score'].fillna(0)
    df['total_payment'] = df['total_payment'].fillna(0)
    df['num_payments'] = df['num_payments'].fillna(0)
    df['avg_installments'] = df['avg_installments'].fillna(0)
    df['max_installments'] = df['max_installments'].fillna(0)
    df['delivery_time_days'] = df['delivery_time_days'].fillna(0)
    
    # Базовые агрегаты по товарам (на уровне позиций)
    base_agg = df.groupby('product_id').agg(
        total_orders=('order_id', 'nunique'),
        total_items=('order_item_id', 'count'),
        total_sales=('price', 'sum'),
        total_freight=('freight_value', 'sum'),
        avg_price=('price', 'mean'),
        std_price=('price', 'std'),
        avg_freight=('freight_value', 'mean'),
        num_sellers=('seller_id', 'nunique'),
        avg_review_score=('avg_review_score', 'mean'),
        has_review_sum=('has_review', 'sum'),
        delivered_on_time_sum=('delivered_on_time', 'sum'),
        avg_delivery_time=('delivery_time_days', 'mean'),
        total_payment_sum=('total_payment', 'sum'),
        avg_payment=('total_payment', 'mean'),
        avg_installments=('avg_installments', 'mean'),
        max_installments=('max_installments', 'max'),
        last_purchase_days=('days_since_purchase', 'min')  # минимальное количество дней = самая свежая покупка
    ).reset_index()
    
    # Дополнительные признаки: доли отзывов, вовремя доставленных
    base_agg['review_rate'] = base_agg['has_review_sum'] / base_agg['total_items']
    base_agg['on_time_rate'] = base_agg['delivered_on_time_sum'] / base_agg['total_items']
    
    # Признаки для временных окон: за последние 7, 30, 90, 180 дней
    for window in [7, 30, 90, 180]:
        cutoff = seed_time - pd.Timedelta(days=window)
        mask = df['order_purchase_timestamp'] > cutoff
        if mask.any():
            window_df = df[mask].groupby('product_id').agg(
                **{f'orders_{window}d': ('order_id', 'nunique'),
                   f'items_{window}d': ('order_item_id', 'count'),
                   f'sales_{window}d': ('price', 'sum')}
            ).reset_index()
            base_agg = base_agg.merge(window_df, on='product_id', how='left')
        else:
            base_agg[f'orders_{window}d'] = 0
            base_agg[f'items_{window}d'] = 0
            base_agg[f'sales_{window}d'] = 0
    
    # Признаки по типам платежей (доли)
    # Сначала получим типы платежей для заказов с товаром
    payments_types = payments[['order_id', 'payment_type']].drop_duplicates()
    df_types = df[['order_id', 'product_id']].drop_duplicates().merge(payments_types, on='order_id', how='left')
    # Считаем количество каждого типа для товара
    type_counts = df_types.groupby(['product_id', 'payment_type']).size().unstack(fill_value=0)
    # Нормализуем на количество заказов
    if not type_counts.empty:
        type_counts = type_counts.div(base_agg.set_index('product_id')['total_orders'], axis=0)
        type_counts = type_counts.reset_index()
        base_agg = base_agg.merge(type_counts, on='product_id', how='left')
    else:
        for col in ['credit_card', 'boleto', 'voucher', 'debit_card']:
            base_agg[col] = 0
    
    # Заполняем пропуски нулями (для товаров без истории)
    base_agg = base_agg.fillna(0)
    
    # Устанавливаем индекс и реиндексируем под entity_ids
    result = base_agg.set_index('product_id')
    result = result.reindex(entity_ids, fill_value=0)
    
    return result