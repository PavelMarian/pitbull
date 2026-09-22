import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]
    
    # --- продажи до seed_time ---
    oi = order_items[order_items['ts'] < seed_time].copy()
    
    # базовые агрегаты по товарам
    agg = oi.groupby('product_id').agg(
        num_orders=('order_id', 'nunique'),
        num_items=('order_item_id', 'count'),
        total_price=('price', 'sum'),
        avg_price=('price', 'mean'),
        total_freight=('freight_value', 'sum'),
        avg_freight=('freight_value', 'mean'),
        unique_sellers=('seller_id', 'nunique')
    ).reset_index()
    
    # признаки за последние 7/30/60/90 дней
    for days in [7, 30, 60, 90]:
        mask = (oi['ts'] >= seed_time - pd.Timedelta(days=days))
        sub = oi[mask]
        if len(sub) > 0:
            tmp = sub.groupby('product_id').agg(
                num_items_last=('order_item_id', 'count'),
                total_price_last=('price', 'sum')
            )
            tmp.columns = [f'num_items_last_{days}', f'total_price_last_{days}']
            agg = agg.merge(tmp.reset_index(), on='product_id', how='left')
            
            tmp = sub.groupby('product_id')['order_id'].nunique()
            tmp = tmp.reset_index()
            tmp.columns = ['product_id', f'num_orders_last_{days}']
            agg = agg.merge(tmp, on='product_id', how='left')
        else:
            agg[f'num_items_last_{days}'] = 0
            agg[f'total_price_last_{days}'] = 0
            agg[f'num_orders_last_{days}'] = 0
    
    # отзывы (только созданные до seed_time)
    rev = reviews[reviews['review_creation_date'] < seed_time]
    oi_rev = oi[['order_id', 'product_id']].merge(rev[['order_id', 'review_score']], on='order_id', how='left')
    rev_agg = oi_rev.groupby('product_id').agg(
        rating_mean=('review_score', 'mean'),
        rating_count=('review_score', 'count'),
        rating_positive=('review_score', lambda x: (x >= 4).sum()),
        rating_negative=('review_score', lambda x: (x <= 2).sum())
    )
    rev_agg['rating_pos_share'] = rev_agg['rating_positive'] / rev_agg['rating_count'].replace(0, np.nan)
    rev_agg['rating_neg_share'] = rev_agg['rating_negative'] / rev_agg['rating_count'].replace(0, np.nan)
    rev_agg = rev_agg.drop(columns=['rating_positive', 'rating_negative']).reset_index()
    agg = agg.merge(rev_agg, on='product_id', how='left')
    
    # платежи (агрегируем до заказа, затем до товара)
    pay = payments[payments['ts'] < seed_time]
    pay_agg = pay.groupby('order_id').agg(
        pay_inst_avg=('payment_installments', 'mean'),
        pay_inst_max=('payment_installments', 'max'),
        pay_inst_min=('payment_installments', 'min'),
        pay_value_sum=('payment_value', 'sum'),
        pay_value_mean=('payment_value', 'mean')
    ).reset_index()
    
    oi_pay = oi[['order_id', 'product_id']].merge(pay_agg, on='order_id', how='left')
    pay_by_prod = oi_pay.groupby('product_id').agg(
        avg_inst_cnt=('pay_inst_avg', 'mean'),
        max_inst_cnt=('pay_inst_max', 'max'),
        min_inst_cnt=('pay_inst_min', 'min'),
        total_pay_sum=('pay_value_sum', 'sum'),
        avg_pay_value=('pay_value_mean', 'mean')
    ).reset_index()
    agg = agg.merge(pay_by_prod, on='product_id', how='left')
    
    # доставка: только заказы, доставленные ДО seed_time,
    # иначе подмешиваем информацию о будущих доставках
    ords = orders[orders['order_delivered_customer_date'].notna()][
        ['order_id', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    ]
    oi_del = oi.merge(ords, on='order_id', how='left')
    oi_del = oi_del[oi_del['order_delivered_customer_date'].notna()].copy()
    oi_del = oi_del[oi_del['order_delivered_customer_date'] <= seed_time]
    
    if len(oi_del) > 0:
        oi_del['delivery_time_days'] = (oi_del['order_delivered_customer_date'] - oi_del['ts']).dt.days
        oi_del['delay_days'] = (oi_del['order_delivered_customer_date'] - oi_del['order_estimated_delivery_date']).dt.days
        oi_del['is_late'] = (oi_del['delay_days'] > 0).astype(int)
        
        del_agg = oi_del.groupby('product_id').agg(
            avg_delivery_days=('delivery_time_days', 'mean'),
            avg_delay_days=('delay_days', 'mean'),
            late_share=('is_late', 'mean')
        ).reset_index()
    else:
        del_agg = pd.DataFrame(columns=['product_id', 'avg_delivery_days', 'avg_delay_days', 'late_share'])
    agg = agg.merge(del_agg, on='product_id', how='left')
    
    # recency: последняя и первая покупка
    last_purchase = oi.groupby('product_id')['ts'].max().reset_index()
    last_purchase['days_since_last_purchase'] = (seed_time - last_purchase['ts']).dt.days
    agg = agg.merge(last_purchase[['product_id', 'days_since_last_purchase']], on='product_id', how='left')
    
    first_purchase = oi.groupby('product_id')['ts'].min().reset_index()
    first_purchase['days_since_first_purchase'] = (seed_time - first_purchase['ts']).dt.days
    agg = agg.merge(first_purchase[['product_id', 'days_since_first_purchase']], on='product_id', how='left')
    
    # собираем итоговый DataFrame
    result = pd.DataFrame({'product_id': entity_ids})
    result = result.merge(agg, on='product_id', how='left').set_index('product_id')
    result = result.fillna(0)
    
    # индикатор наличия продаж
    result['has_sales'] = (result['num_orders'] > 0).astype(int)
    
    return result