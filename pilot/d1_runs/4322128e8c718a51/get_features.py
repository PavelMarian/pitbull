def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    product_ids = list(entity_ids)
    # базовый DataFrame с индексом из product_ids
    base = pd.DataFrame(index=product_ids)

    # отфильтрованные order_items
    oi = db['order_items']
    oi = oi[oi['product_id'].isin(product_ids) & (oi['ts'] <= seed_time)].copy()

    # если данных нет, всё равно пройдём по всем шагам (пустые агрегаты)
    if not oi.empty:
        # добавим customer_id из orders
        orders = db['orders'][['order_id', 'customer_id']]
        oi = oi.merge(orders, on='order_id', how='left')

    # ---- Основные агрегаты по товару ----
    if not oi.empty:
        grp = oi.groupby('product_id')
        total_orders = grp['order_id'].nunique()
        total_items = grp.size()
        total_revenue = grp['price'].sum()
        total_freight = grp['freight_value'].sum()
        avg_price = grp['price'].mean()
        std_price = grp['price'].std()
        min_price = grp['price'].min()
        max_price = grp['price'].max()
        avg_freight = grp['freight_value'].mean()
        std_freight = grp['freight_value'].std()
        unique_sellers = grp['seller_id'].nunique()
        unique_customers = grp['customer_id'].nunique()

        ts_min = grp['ts'].min()
        ts_max = grp['ts'].max()
        recency_days = (seed_time - ts_max).dt.days
        days_since_first = (seed_time - ts_min).dt.days
    else:
        # пустые серии
        empty_series = pd.Series(index=pd.Index(product_ids), dtype='float64')
        total_orders = empty_series.astype('int64')
        total_items = empty_series.astype('int64')
        total_revenue = empty_series.copy()
        total_freight = empty_series.copy()
        avg_price = empty_series.copy()
        std_price = empty_series.copy()
        min_price = empty_series.copy()
        max_price = empty_series.copy()
        avg_freight = empty_series.copy()
        std_freight = empty_series.copy()
        unique_sellers = empty_series.astype('int64')
        unique_customers = empty_series.astype('int64')
        recency_days = empty_series.copy()
        days_since_first = empty_series.copy()

    # ---- Признаки за последние 30/60/90 дней ----
    for days in [30, 60, 90]:
        mask = (oi['ts'] > seed_time - pd.Timedelta(days=days)) & (oi['ts'] <= seed_time)
        sub = oi[mask]
        if not sub.empty:
            orders_cnt = sub.groupby('product_id')['order_id'].nunique()
            items_cnt = sub.groupby('product_id').size()
            revenue = sub.groupby('product_id')['price'].sum()
        else:
            orders_cnt = pd.Series(index=product_ids, dtype='int64')
            items_cnt = pd.Series(index=product_ids, dtype='int64')
            revenue = pd.Series(index=product_ids, dtype='float64')
        base[f'orders_last_{days}'] = orders_cnt.reindex(product_ids, fill_value=0)
        base[f'items_last_{days}'] = items_cnt.reindex(product_ids, fill_value=0)
        base[f'revenue_last_{days}'] = revenue.reindex(product_ids, fill_value=0)

    # ---- Интервалы между заказами ----
    if not oi.empty:
        # даты заказов для каждого товара (уникальные order_id)
        order_dates = oi.groupby(['product_id', 'order_id'])['ts'].min().reset_index()
        order_dates = order_dates.sort_values(['product_id', 'ts'])
        order_dates['diff'] = order_dates.groupby('product_id')['ts'].diff().dt.days
        avg_interval = order_dates.groupby('product_id')['diff'].mean()
        std_interval = order_dates.groupby('product_id')['diff'].std()
    else:
        avg_interval = pd.Series(index=product_ids, dtype='float64')
        std_interval = pd.Series(index=product_ids, dtype='float64')

    # ---- Отзывы ----
    # Отзыв появляется в базе в момент review_creation_date (позже покупки),
    # поэтому берём только отзывы, уже существующие на seed_time.
    reviews = db['reviews'][['order_id', 'review_score', 'review_creation_date']]
    reviews = reviews[reviews['review_creation_date'] <= seed_time][['order_id', 'review_score']]
    if not oi.empty:
        oi_rev = oi[['product_id', 'order_id']].merge(reviews, on='order_id', how='inner')
    else:
        oi_rev = pd.DataFrame(columns=['product_id', 'review_score'])
    if not oi_rev.empty:
        rev_grp = oi_rev.groupby('product_id')['review_score']
        rev_mean = rev_grp.mean()
        rev_count = rev_grp.count()
        pos_cnt = oi_rev[oi_rev['review_score'] >= 4].groupby('product_id')['review_score'].count()
        positive_ratio = pos_cnt / rev_count
    else:
        rev_mean = pd.Series(index=product_ids, dtype='float64')
        rev_count = pd.Series(index=product_ids, dtype='int64')
        positive_ratio = pd.Series(index=product_ids, dtype='float64')

    # ---- Платежи ----
    payments = db['payments'][['order_id', 'payment_type', 'payment_installments', 'payment_value']]
    if not oi.empty:
        oi_pay = oi[['product_id', 'order_id']].merge(payments, on='order_id', how='inner')
    else:
        oi_pay = pd.DataFrame(columns=['product_id', 'payment_type', 'payment_installments', 'payment_value'])
    if not oi_pay.empty:
        pay_grp = oi_pay.groupby('product_id')
        avg_payment_value = pay_grp['payment_value'].mean()
        avg_installments = pay_grp['payment_installments'].mean()
        total_pay_count = pay_grp['payment_value'].count()
        credit_cnt = oi_pay[oi_pay['payment_type'] == 'credit_card'].groupby('product_id')['payment_value'].count()
        credit_ratio = credit_cnt / total_pay_count
    else:
        avg_payment_value = pd.Series(index=product_ids, dtype='float64')
        avg_installments = pd.Series(index=product_ids, dtype='float64')
        credit_ratio = pd.Series(index=product_ids, dtype='float64')

    # ---- Собираем всё в base ----
    base['total_orders'] = total_orders.reindex(product_ids).fillna(0)
    base['total_items'] = total_items.reindex(product_ids).fillna(0)
    base['total_revenue'] = total_revenue.reindex(product_ids).fillna(0)
    base['total_freight'] = total_freight.reindex(product_ids).fillna(0)
    base['avg_price'] = avg_price.reindex(product_ids).fillna(0)
    base['std_price'] = std_price.reindex(product_ids).fillna(0)
    base['min_price'] = min_price.reindex(product_ids).fillna(0)
    base['max_price'] = max_price.reindex(product_ids).fillna(0)
    base['avg_freight'] = avg_freight.reindex(product_ids).fillna(0)
    base['std_freight'] = std_freight.reindex(product_ids).fillna(0)
    base['unique_sellers'] = unique_sellers.reindex(product_ids).fillna(0)
    base['unique_customers'] = unique_customers.reindex(product_ids).fillna(0)

    # recency: если нет данных, ставим 9999
    base['recency_days'] = recency_days.reindex(product_ids).fillna(9999)
    base['days_since_first'] = days_since_first.reindex(product_ids).fillna(0)

    base['avg_interval'] = avg_interval.reindex(product_ids).fillna(0)
    base['std_interval'] = std_interval.reindex(product_ids).fillna(0)

    base['review_mean'] = rev_mean.reindex(product_ids).fillna(0)
    base['review_count'] = rev_count.reindex(product_ids).fillna(0)
    base['positive_ratio'] = positive_ratio.reindex(product_ids).fillna(0)

    base['avg_payment_value'] = avg_payment_value.reindex(product_ids).fillna(0)
    base['avg_installments'] = avg_installments.reindex(product_ids).fillna(0)
    base['credit_card_ratio'] = credit_ratio.reindex(product_ids).fillna(0)

    # Дополнительные признаки
    base['avg_items_per_order'] = base['total_items'] / base['total_orders'].replace(0, np.nan)
    base['avg_items_per_order'] = base['avg_items_per_order'].fillna(0)

    base['recent_90_ratio'] = base['orders_last_90'] / base['total_orders'].replace(0, np.nan)
    base['recent_90_ratio'] = base['recent_90_ratio'].fillna(0)

    # Переиндексация в исходном порядке
    base = base.reindex(product_ids)

    return base