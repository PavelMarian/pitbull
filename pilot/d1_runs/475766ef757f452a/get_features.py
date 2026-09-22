def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]

    # Нормализуем список продуктов
    if isinstance(entity_ids, (pd.Series, list, np.ndarray)):
        prod_list = list(entity_ids)
    else:
        prod_list = [entity_ids]

    # Выборка заказов, содержащих нужные товары
    items = order_items[order_items['product_id'].isin(prod_list)]
    if items.empty:
        return pd.DataFrame({'has_sales': 0}, index=prod_list)

    # Присоединяем orders и оставляем только покупки, случившиеся до seed_time
    merged = items.merge(orders[['order_id', 'order_purchase_timestamp']],
                         on='order_id', how='left')
    merged = merged[merged['order_purchase_timestamp'] < seed_time]
    if merged.empty:
        return pd.DataFrame({'has_sales': 0}, index=prod_list)

    # Агрегируем по (product_id, order_id) — одна строка на заказ
    order_data = merged.groupby(['product_id', 'order_id']).agg(
        total_price=('price', 'sum'),
        total_freight=('freight_value', 'sum'),
        num_items=('order_item_id', 'count'),
        purchase_date=('order_purchase_timestamp', 'first')
    ).reset_index()

    # Отзывы: берём только те, что уже созданы к моменту seed_time
    rev = reviews[reviews['review_creation_date'] <= seed_time]
    reviews_agg = rev.groupby('order_id').agg(
        avg_review_score=('review_score', 'mean'),
        num_reviews=('order_id', 'count'),
        num_low_reviews=('review_score', lambda x: (x <= 2).sum())
    ).reset_index()

    # Платежи: оставляем только записи, доступные на момент seed_time
    pay = payments[payments['ts'] <= seed_time]
    payments_agg = pay.groupby('order_id').agg(
        avg_payment_value=('payment_value', 'mean'),
        num_payments=('payment_value', 'count'),
        sum_payment_value=('payment_value', 'sum'),
        max_installments=('payment_installments', 'max')
    ).reset_index()

    # Добавляем отзывы и платежи
    order_data = order_data.merge(reviews_agg, on='order_id', how='left')
    order_data = order_data.merge(payments_agg, on='order_id', how='left')

    # Заполняем пропуски
    fill_cols = ['avg_review_score', 'num_reviews', 'num_low_reviews',
                 'avg_payment_value', 'num_payments', 'sum_payment_value',
                 'max_installments']
    for col in fill_cols:
        if col in order_data.columns:
            order_data[col] = order_data[col].fillna(0)

    # Временные окна
    for days in [30, 60, 90, 180]:
        border = seed_time - pd.Timedelta(days=days)
        col_name = f'wd_{days}'
        order_data[col_name] = (order_data['purchase_date'] >= border).astype(int)

    # Собираем агрегаты по продуктам
    agg_dict = {
        'order_id': 'nunique',                    # всего заказов
        'total_price': ['sum', 'mean', 'std'],
        'total_freight': ['sum', 'mean'],
        'num_items': ['sum', 'mean'],
        'avg_review_score': ['mean', 'std'],
        'num_reviews': ['sum', 'mean'],
        'num_low_reviews': 'sum',
        'avg_payment_value': ['mean', 'std'],
        'num_payments': ['sum', 'mean'],
        'sum_payment_value': 'sum',
        'max_installments': 'max',
        'purchase_date': ['min', 'max']
    }
    for d in [30, 60, 90, 180]:
        agg_dict[f'wd_{d}'] = 'sum'

    grouped = order_data.groupby('product_id').agg(agg_dict)

    # Плоские имена столбцов
    grouped.columns = ['_'.join(c).strip('_') for c in grouped.columns.values]

    # Признаки, связанные с временем
    grouped['recency_days'] = (seed_time - grouped['purchase_date_max']).dt.days
    grouped['days_from_first_order'] = (
        grouped['purchase_date_max'] - grouped['purchase_date_min']
    ).dt.days

    # Удаляем служебные столбцы с датами
    grouped.drop(columns=[col for col in grouped.columns
                          if 'purchase_date_' in col], inplace=True)

    # Восстанавливаем все требуемые продукты
    features = grouped.reindex(prod_list).fillna(0)

    # Добавим бинарный признак наличия продаж
    features['has_sales'] = (features['order_id_nunique'] > 0).astype(int)

    return features
