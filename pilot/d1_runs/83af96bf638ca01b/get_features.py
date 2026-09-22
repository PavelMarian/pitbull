def get_features(db, entity_ids, seed_time):
    import pandas as pd

    ids = list(entity_ids)
    result = pd.DataFrame({'product_id': ids})

    items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']

    # Отбираем исторические заказы для данных товаров
    mask = items['product_id'].isin(ids) & (items['ts'] < seed_time)
    items_f = items[mask]

    # ===== 1. Базовые признаки по всем позициям =====
    if not items_f.empty:
        base = items_f.groupby('product_id').agg(
            num_items=('order_item_id', 'size'),
            num_orders=('order_id', 'nunique'),
            total_price=('price', 'sum'),
            total_freight=('freight_value', 'sum'),
            avg_price=('price', 'mean'),
            avg_freight=('freight_value', 'mean')
        ).reset_index()
    else:
        base = pd.DataFrame({'product_id': ids,
                            'num_items': 0,
                            'num_orders': 0,
                            'total_price': 0,
                            'total_freight': 0,
                            'avg_price': 0,
                            'avg_freight': 0})
    result = result.merge(base, on='product_id', how='left')
    result[['num_items','num_orders','total_price','total_freight','avg_price','avg_freight']] = \
        result[['num_items','num_orders','total_price','total_freight','avg_price','avg_freight']].fillna(0)

    # ===== 2. Признаки за несколько временных окон =====
    for win in [7, 30, 90]:
        cutoff = seed_time - pd.Timedelta(days=win)
        sub = items_f[items_f['ts'] >= cutoff]
        if not sub.empty:
            period = sub.groupby('product_id').agg(
                **{f'num_sales_{win}': ('order_item_id', 'size'),
                   f'num_orders_{win}': ('order_id', 'nunique'),
                   f'sum_price_{win}': ('price', 'sum')}
            ).reset_index()
        else:
            period = pd.DataFrame({
                'product_id': ids,
                f'num_sales_{win}': 0,
                f'num_orders_{win}': 0,
                f'sum_price_{win}': 0
            })
        result = result.merge(period, on='product_id', how='left')
        result[f'num_sales_{win}'] = result[f'num_sales_{win}'].fillna(0)
        result[f'num_orders_{win}'] = result[f'num_orders_{win}'].fillna(0)
        result[f'sum_price_{win}'] = result[f'sum_price_{win}'].fillna(0)

    # ===== 3. Признаки из отзывов =====
    if not items_f.empty:
        order_ids = items_f['order_id'].unique()
        rev = reviews[reviews['order_id'].isin(order_ids)]
    else:
        rev = pd.DataFrame()
    if not rev.empty:
        # Отзыв становится известен только в момент своего создания,
        # оставляем лишь те, что созданы до seed_time
        rev = rev[rev['review_creation_date'] < seed_time]
    if not rev.empty:
        rev_order = rev.groupby('order_id')['review_score'].agg(
            ['mean', 'size']
        ).rename(columns={'mean': 'avg_score', 'size': 'count_reviews'}).reset_index()
        # привязываем средний рейтинг к каждому заказу товара
        order_product = items_f[['product_id', 'order_id']].drop_duplicates()
        rev_merged = order_product.merge(rev_order, on='order_id', how='left')
        rev_feats = rev_merged.groupby('product_id').agg(
            avg_review_score=('avg_score', 'mean'),
            num_reviews=('count_reviews', 'sum')
        ).reset_index()
    else:
        rev_feats = pd.DataFrame({'product_id': ids,
                                 'avg_review_score': 0,
                                 'num_reviews': 0})
    result = result.merge(rev_feats, on='product_id', how='left')
    result['avg_review_score'] = result['avg_review_score'].fillna(0)
    result['num_reviews'] = result['num_reviews'].fillna(0)

    # ===== 4. Признаки из платежей =====
    if not items_f.empty:
        pay = payments[payments['order_id'].isin(items_f['order_id'].unique())]
    else:
        pay = pd.DataFrame()
    if not pay.empty:
        # агрегируем по каждому заказу
        pay_order = pay.groupby('order_id').agg(
            avg_installments=('payment_installments', 'mean'),
            avg_payment_value=('payment_value', 'mean'),
            num_payments=('payment_sequential', 'count'),
            # количество платежей с типом кредитная карта
            credit_card_count=('payment_type', lambda x: (x == 'credit_card').sum())
        ).reset_index()
        pay_order['share_credit_card'] = pay_order['credit_card_count'] / pay_order['num_payments']
        pay_order = pay_order[['order_id', 'avg_installments', 'avg_payment_value', 'num_payments', 'share_credit_card']]

        order_product = items_f[['product_id', 'order_id']].drop_duplicates()
        pay_merged = order_product.merge(pay_order, on='order_id', how='left')
        pay_feats = pay_merged.groupby('product_id').agg(
            avg_installments=('avg_installments', 'mean'),
            avg_payment_value=('avg_payment_value', 'mean'),
            num_payments=('num_payments', 'sum'),
            share_credit_card=('share_credit_card', 'mean')
        ).reset_index()
    else:
        pay_feats = pd.DataFrame({'product_id': ids,
                                 'avg_installments': 0,
                                 'avg_payment_value': 0,
                                 'num_payments': 0,
                                 'share_credit_card': 0})
    result = result.merge(pay_feats, on='product_id', how='left')
    result[['avg_installments', 'avg_payment_value', 'num_payments', 'share_credit_card']] = \
        result[['avg_installments', 'avg_payment_value', 'num_payments', 'share_credit_card']].fillna(0)

    # ===== 5. Дополнительные признаки =====
    if not items_f.empty:
        sellers = items_f.groupby('product_id')['seller_id'].nunique().rename('num_sellers').reset_index()
        max_ts = items_f.groupby('product_id')['ts'].max().reset_index()
        max_ts['days_since_last'] = (seed_time - max_ts['ts']).dt.days
        result = result.merge(sellers, on='product_id', how='left')
        result = result.merge(max_ts[['product_id', 'days_since_last']], on='product_id', how='left')
    else:
        result['num_sellers'] = 0
        result['days_since_last'] = 0
    result['num_sellers'] = result['num_sellers'].fillna(0)
    result['days_since_last'] = result['days_since_last'].fillna(0)

    # ===== Итог: индексирование =====
    result = result.set_index('product_id')
    result = result.reindex(ids)
    result = result.fillna(0)
    return result