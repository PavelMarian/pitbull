def get_features(db, entity_ids, seed_time):
    # вход: db - словарь с таблицами, entity_ids - list/array product_id, seed_time - Timestamp
    # выход: DataFrame с признаками, индексированный по entity_ids
    # Все признаки строятся только по информации, существующей на момент seed_time.

    orders = db['orders']
    order_items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']

    # Позиции покупок строго до seed_time (ts - момент покупки заказа)
    items = order_items[order_items['product_id'].isin(entity_ids)].copy()
    items = items[items['ts'] <= seed_time]

    items = items.merge(orders[['order_id', 'customer_id']], on='order_id', how='left')

    # Платежи учитываем только те, что относятся к покупкам до seed_time
    payments_ok = payments[payments['ts'] <= seed_time]
    pay_agg = payments_ok.groupby('order_id').agg(
        tot_payment=('payment_value', 'sum'),
        n_payments=('payment_sequential', 'count'),
        avg_installments=('payment_installments', 'mean'),
        n_payment_types=('payment_type', 'nunique')
    ).reset_index()
    items = items.merge(pay_agg, on='order_id', how='left')

    # Отзывы, уже существующие на момент seed_time
    rev_ok = reviews.loc[reviews['review_creation_date'] <= seed_time,
                         ['order_id', 'review_score']]

    windows = [7, 30, 90, 180, 365]
    rev_cols = ['rev_count', 'rev_mean', 'rev_high']
    time_cols = ['time_since_last_purchase', 'time_since_first_purchase']

    feature_frames = []

    seed_feats = pd.DataFrame({
        'seed_month': [seed_time.month] * len(entity_ids),
        'seed_quarter': [seed_time.quarter] * len(entity_ids),
        'seed_dayofweek': [seed_time.dayofweek] * len(entity_ids)
    }, index=entity_ids)
    feature_frames.append(seed_feats)

    for w in windows:
        start_date = seed_time - pd.Timedelta(days=w)
        win_items = items[items['ts'] >= start_date]

        if win_items.empty:
            agg = pd.DataFrame(index=pd.Index(entity_ids, name='product_id'))
        else:
            grp = win_items.groupby('product_id')
            agg = grp.agg(
                n_items=('order_id', 'count'),
                n_orders=('order_id', 'nunique'),
                n_customers=('customer_id', 'nunique'),
                total_price=('price', 'sum'),
                total_freight=('freight_value', 'sum'),
                avg_price=('price', 'mean'),
                avg_freight=('freight_value', 'mean'),
                total_payment=('tot_payment', 'sum'),
                avg_payment_per_order=('tot_payment', 'mean'),
                avg_installments=('avg_installments', 'mean'),
                n_payment_types=('n_payment_types', 'mean'),
                n_payments=('n_payments', 'sum')
            ).copy()
            last_ts = grp['ts'].max()
            first_ts = grp['ts'].min()
            agg['time_since_last_purchase'] = (seed_time - last_ts).dt.days
            agg['time_since_first_purchase'] = (seed_time - first_ts).dt.days

        for col in time_cols:
            if col not in agg.columns:
                agg[col] = np.nan

        # Рейтинговые признаки только по отзывам, доступным на seed_time
        if win_items.empty:
            rev_agg = pd.DataFrame(index=agg.index)
        else:
            order_ids_in_win = win_items['order_id'].unique()
            rv = rev_ok[rev_ok['order_id'].isin(order_ids_in_win)]
            product_order = win_items[['product_id', 'order_id']].drop_duplicates()
            rev_merged = product_order.merge(rv, on='order_id', how='inner')
            if rev_merged.empty:
                rev_agg = pd.DataFrame(index=agg.index)
            else:
                g = rev_merged.groupby('product_id')['review_score']
                rev_agg = pd.DataFrame({
                    'rev_count': g.count(),
                    'rev_mean': g.mean(),
                    'rev_high': g.apply(lambda x: (x >= 4).mean() * 100.0)
                })

        rev_agg = rev_agg.reindex(agg.index)
        for col in rev_cols:
            if col not in rev_agg.columns:
                rev_agg[col] = np.nan

        agg = agg.join(rev_agg[rev_cols], how='left')

        agg.columns = [f'win_{w}_{col}' for col in agg.columns]
        feature_frames.append(agg)

    result = pd.concat(feature_frames, axis=1)

    result = result.reindex(entity_ids, fill_value=0)
    result = result.fillna(0.0)

    return result
