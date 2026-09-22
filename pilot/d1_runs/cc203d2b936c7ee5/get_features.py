def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    orders = db['orders']
    order_items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']

    # 1. Отбираем позиции до seed_time
    items = order_items[order_items['product_id'].isin(entity_ids) & (order_items['ts'] < seed_time)]
    items = items.merge(orders[['order_id', 'customer_id']], on='order_id', how='left')

    # 2. Агрегации по заказам (только данные, существующие на seed_time)
    # Отзыв появляется позже покупки: берём лишь отзывы, созданные до seed_time.
    rev_avail = reviews[reviews['review_creation_date'] < seed_time]
    rev_agg = rev_avail.groupby('order_id').agg(
        mean_score=('review_score', 'mean'),
        cnt_review=('review_score', 'count')
    ).reset_index()
    pay_agg = payments.groupby('order_id').agg(
        sum_paid=('payment_value', 'sum'),
        cnt_payments=('payment_sequential', 'count'),
        mean_installments=('payment_installments', 'mean')
    ).reset_index()

    items = items.merge(rev_agg, on='order_id', how='left')
    items = items.merge(pay_agg, on='order_id', how='left')

    # 3. Базовая агрегация (вся доступная история)
    base = items.groupby('product_id').agg(
        cnt_orders=('order_id', 'nunique'),
        cnt_items=('order_item_id', 'count'),
        sum_price=('price', 'sum'),
        mean_price=('price', 'mean'),
        std_price=('price', 'std'),
        min_price=('price', 'min'),
        max_price=('price', 'max'),
        mean_freight=('freight_value', 'mean'),
        sum_freight=('freight_value', 'sum'),
        cnt_sellers=('seller_id', 'nunique'),
        cnt_customers=('customer_id', 'nunique'),
        mean_review_score=('mean_score', 'mean'),
        cnt_reviews=('cnt_review', 'sum'),
        sum_paid=('sum_paid', 'sum'),
        cnt_payments=('cnt_payments', 'sum'),
        mean_installments=('mean_installments', 'mean')
    ).reset_index()

    # 4. Оконные признаки за последние 14, 30, 60, 90, 180 дней
    for N in [14, 30, 60, 90, 180]:
        sub = items[items['ts'] > seed_time - pd.Timedelta(days=N)]
        ord_cnt = sub.groupby('product_id')['order_id'].nunique().reindex(base['product_id'], fill_value=0)
        rev_sum = sub.groupby('product_id')['price'].sum().reindex(base['product_id'], fill_value=0)
        base[f'orders_{N}d'] = ord_cnt.values
        base[f'revenue_{N}d'] = rev_sum.values

    # 5. Возраст продукта (в днях) от первой покупки
    first_ts = items.groupby('product_id')['ts'].min()
    base['days_since_first'] = (seed_time - first_ts.reindex(base['product_id'])).dt.days

    # 6. Приводим к полному списку экземпляров и убираем NaN
    base = base.set_index('product_id').reindex(entity_ids)
    base = base.fillna(0)

    return base