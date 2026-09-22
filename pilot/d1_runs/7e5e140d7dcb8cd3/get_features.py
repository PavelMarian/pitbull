def get_features(db, entity_ids, seed_time):
    # Подготовка DataFrame для признаков
    features = pd.DataFrame(index=entity_ids)
    
    # Фильтрация заказов только до момента предсказания
    items = db['order_items']
    items = items[items['ts'] <= seed_time]
    
    # Отбираем только нужные товары
    items_target = items[items['product_id'].isin(entity_ids)]
    
    # ---- Признаки по продажам (order_items) ----
    agg_items = items_target.groupby('product_id').agg(
        sales_count=('order_id', 'count'),
        sum_price=('price', 'sum'),
        avg_price=('price', 'mean'),
        sum_freight=('freight_value', 'sum'),
        avg_freight=('freight_value', 'mean'),
        first_purchase_ts=('ts', 'min'),
        last_purchase_ts=('ts', 'max'),
        n_sellers=('seller_id', 'nunique')
    )
    
    features = features.join(agg_items, how='left')
    
    # Время с первой и последней покупки (в днях)
    features['days_since_first'] = (seed_time - features['first_purchase_ts']).dt.days.fillna(99999)
    features['days_since_last'] = (seed_time - features['last_purchase_ts']).dt.days.fillna(99999)
    
    # Удаляем временные столбцы
    features = features.drop(columns=['first_purchase_ts', 'last_purchase_ts'])
    
    # ---- Признаки по отзывам (reviews) ----
    # Берём только отзывы, уже существующие на момент seed_time.
    if not items_target.empty:
        reviews = db['reviews'][['order_id', 'review_score', 'review_creation_date']]
        reviews = reviews[reviews['review_creation_date'] <= seed_time]
        review_data = items_target[['order_id', 'product_id']].merge(
            reviews[['order_id', 'review_score']],
            on='order_id',
            how='left'
        )
        review_agg = review_data.groupby('product_id').agg(
            review_mean_score=('review_score', 'mean'),
            review_count=('review_score', 'count')
        )
        features = features.join(review_agg, how='left')
    else:
        features['review_mean_score'] = np.nan
        features['review_count'] = 0
    
    # Заполняем все пропуски нулями
    features = features.fillna(0)
    
    # Флаг: был ли товар продан ранее
    features['was_sold'] = (features['sales_count'] > 0).astype(int)
    
    # Гарантируем порядок строк в соответствии с переданным списком
    features = features.reindex(entity_ids)
    
    return features