import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Initialize index to ensure all input entities are included
    entity_index = pd.DataFrame(index=entity_ids)
    
    # Filter order_items to valid timestamps and target products
    oi = db['order_items']
    oi_valid = oi[(oi['ts'] < seed_time) & (oi['product_id'].isin(entity_ids))].copy()
    
    # Handle empty valid order_items case
    if oi_valid.empty:
        feature_cols = [
            'total_units_sold', 'total_orders_with_product', 'total_sellers',
            'total_revenue', 'total_freight', 'avg_price', 'avg_freight_per_unit',
            'time_since_last_sale_days', 'max_price', 'min_price', 'price_std',
            'count_delivered_orders', 'count_canceled_orders', 'count_processing_orders',
            'avg_delivery_time_days', 'count_late_deliveries',
            'total_reviews', 'has_reviews', 'avg_review_score',
            'count_5star', 'count_1star', 'share_5star', 'share_1star',
            'total_payment_value_orders', 'avg_payment_value_per_order',
            'count_credit_card_payments', 'count_boleto_payments',
            'count_installment_payments', 'avg_installments'
        ]
        entity_index[feature_cols] = 0
        entity_index['time_since_last_sale_days'] = 1000  # Large value for never-sold products
        return entity_index
    
    # Get relevant order IDs from valid order items
    relevant_order_ids = oi_valid['order_id'].unique()
    
    # Filter other tables to relevant orders and valid timestamps
    orders = db['orders']
    orders_valid = orders[
        (orders['order_id'].isin(relevant_order_ids)) & 
        (orders['order_purchase_timestamp'] < seed_time)
    ].copy()
    
    payments = db['payments']
    payments_valid = payments[
        (payments['order_id'].isin(relevant_order_ids)) & 
        (payments['ts'] < seed_time)
    ].copy()
    
    reviews = db['reviews']
    reviews_valid = reviews[
        (reviews['order_id'].isin(relevant_order_ids)) & 
        (reviews['review_creation_date'] < seed_time)
    ].copy()
    
    # --------------------------
    # Aggregate Order Items Features
    # --------------------------
    oi_agg = oi_valid.groupby('product_id').agg(
        total_units_sold=('order_item_id', 'count'),
        total_orders_with_product=('order_id', 'nunique'),
        total_sellers=('seller_id', 'nunique'),
        total_revenue=('price', 'sum'),
        total_freight=('freight_value', 'sum'),
        avg_price=('price', 'mean'),
        avg_freight_per_unit=('freight_value', 'mean'),
        last_sale_time=('ts', 'max'),
        max_price=('price', 'max'),
        min_price=('price', 'min'),
        price_std=('price', 'std')
    ).reset_index()
    
    # Calculate time since last sale in days
    oi_agg['time_since_last_sale_days'] = (seed_time - oi_agg['last_sale_time']).dt.days
    oi_agg.drop(columns=['last_sale_time'], inplace=True)
    
    # --------------------------
    # Aggregate Order Features
    # --------------------------
    oi_orders = pd.merge(
        oi_valid[['product_id', 'order_id']],
        orders_valid,
        on='order_id',
        how='left'
    )
    
    def order_aggregator(group):
        status = group['order_status']
        # Point-in-time: a delivery is visible at seed_time only if the
        # delivery already happened before seed_time. order_status and
        # order_delivered_customer_date for not-yet-delivered orders are
        # filled later, so they must not be used.
        delivered_mask = (
            (status == 'delivered')
            & group['order_delivered_customer_date'].notna()
            & (group['order_delivered_customer_date'] < seed_time)
        )
        delivered = group[delivered_mask]
        late_deliveries = delivered[
            delivered['order_delivered_customer_date'] > delivered['order_estimated_delivery_date']
        ]
        delivery_times = (
            delivered['order_delivered_customer_date'] - delivered['order_purchase_timestamp']
        ).dt.days.dropna()

        return pd.Series({
            'count_delivered_orders': int(delivered_mask.sum()),
            'count_canceled_orders': (status == 'canceled').sum(),
            'count_processing_orders': (status == 'processing').sum(),
            'avg_delivery_time_days': delivery_times.mean() if len(delivery_times) > 0 else 0,
            'count_late_deliveries': len(late_deliveries)
        })
    
    order_features = oi_orders.groupby('product_id').apply(order_aggregator).reset_index()
    
    # --------------------------
    # Aggregate Review Features
    # --------------------------
    oi_reviews = pd.merge(
        oi_valid[['product_id', 'order_id']],
        reviews_valid,
        on='order_id',
        how='left'
    )
    
    def review_aggregator(group):
        scores = group['review_score'].dropna()
        total = len(scores)
        count_5 = (scores == 5).sum()
        count_1 = (scores == 1).sum()
        
        return pd.Series({
            'total_reviews': total,
            'has_reviews': 1 if total > 0 else 0,
            'avg_review_score': scores.mean() if total > 0 else 0,
            'count_5star': count_5,
            'count_1star': count_1,
            'share_5star': count_5 / total if total > 0 else 0,
            'share_1star': count_1 / total if total > 0 else 0
        })
    
    review_features = oi_reviews.groupby('product_id').apply(review_aggregator).reset_index()
    
    # --------------------------
    # Aggregate Payment Features
    # --------------------------
    if not payments_valid.empty:
        payments_per_order = payments_valid.groupby('order_id').agg(
            total_order_payment=('payment_value', 'sum'),
            avg_installments_order=('payment_installments', 'mean'),
            count_credit_card=('payment_type', lambda x: (x == 'credit_card').sum()),
            count_boleto=('payment_type', lambda x: (x == 'boleto').sum()),
            count_installment_pay=('payment_installments', lambda x: (x > 1).sum())
        ).reset_index()
        
        oi_payments = pd.merge(
            oi_valid[['product_id', 'order_id']],
            payments_per_order,
            on='order_id',
            how='left'
        )
        
        payment_features = oi_payments.groupby('product_id').agg(
            total_payment_value_orders=('total_order_payment', 'sum'),
            avg_payment_value_per_order=('total_order_payment', 'mean'),
            count_credit_card_payments=('count_credit_card', 'sum'),
            count_boleto_payments=('count_boleto', 'sum'),
            count_installment_payments=('count_installment_pay', 'sum'),
            avg_installments=('avg_installments_order', 'mean')
        ).reset_index()
    else:
        payment_cols = [
            'product_id', 'total_payment_value_orders', 'avg_payment_value_per_order',
            'count_credit_card_payments', 'count_boleto_payments',
            'count_installment_payments', 'avg_installments'
        ]
        payment_features = pd.DataFrame(columns=payment_cols)
    
    # --------------------------
    # Merge All Features
    # --------------------------
    final_features = entity_index.copy()
    
    # Merge order items features
    final_features = final_features.merge(
        oi_agg,
        left_index=True,
        right_on='product_id',
        how='left'
    ).set_index('product_id')
    
    # Merge order features
    if not order_features.empty:
        order_features.set_index('product_id', inplace=True)
        final_features = final_features.join(order_features, how='left')
    
    # Merge review features
    if not review_features.empty:
        review_features.set_index('product_id', inplace=True)
        final_features = final_features.join(review_features, how='left')
    
    # Merge payment features
    if not payment_features.empty:
        payment_features.set_index('product_id', inplace=True)
        final_features = final_features.join(payment_features, how='left')
    
    # --------------------------
    # Clean Up and Finalize
    # --------------------------
    # Fill missing values
    fill_map = {'time_since_last_sale_days': 1000}
    for col in final_features.columns:
        if col not in fill_map:
            fill_map[col] = 0
    final_features = final_features.fillna(fill_map)
    
    # Convert integer columns to appropriate dtype
    int_columns = [
        'total_units_sold', 'total_orders_with_product', 'total_sellers',
        'count_delivered_orders', 'count_canceled_orders', 'count_processing_orders',
        'count_late_deliveries', 'total_reviews', 'has_reviews',
        'count_5star', 'count_1star', 'count_credit_card_payments',
        'count_boleto_payments', 'count_installment_payments', 'time_since_last_sale_days'
    ]
    for col in int_columns:
        if col in final_features.columns:
            final_features[col] = final_features[col].astype(int)
    
    # Reindex to ensure exact match of input entity IDs
    final_features = final_features.reindex(entity_ids)
    
    return final_features