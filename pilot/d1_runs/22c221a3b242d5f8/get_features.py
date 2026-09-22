import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Unpack database tables
    orders = db['orders']
    order_items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']
    
    # --------------------------
    # 1. Filter historical data (strictly before seed_time)
    # --------------------------
    # Historical order items (purchased before seed_time)
    hist_items = order_items[order_items['ts'] < seed_time].copy()
    # Historical order ids
    hist_order_ids = hist_items['order_id'].unique()
    # Only orders whose delivery already happened before seed_time:
    # order_delivered_customer_date is filled in only after the actual
    # delivery, so rows with a missing or future delivery date carry no
    # information that exists at seed_time and must be excluded
    hist_orders = orders[
        orders['order_id'].isin(hist_order_ids)
        & (orders['order_delivered_customer_date'] < seed_time)
    ].copy()
    # Historical reviews (created before seed_time)
    hist_reviews = reviews[reviews['review_creation_date'] < seed_time].copy()
    # Historical payments (recorded before seed_time, for historical orders)
    hist_payments = payments[
        (payments['ts'] < seed_time) & (payments['order_id'].isin(hist_order_ids))
    ].copy()
    
    # --------------------------
    # 2. Base DataFrame with all requested product IDs
    # --------------------------
    base = pd.DataFrame({'product_id': entity_ids})
    
    # --------------------------
    # 3. Aggregate core product sales metrics
    # --------------------------
    if not hist_items.empty:
        product_sales_stats = hist_items.groupby('product_id').agg(
            total_units_sold=('order_item_id', 'count'),
            total_revenue=('price', 'sum'),
            total_freight=('freight_value', 'sum'),
            avg_item_price=('price', 'mean'),
            avg_item_freight=('freight_value', 'mean'),
            first_sale_ts=('ts', 'min'),
            last_sale_ts=('ts', 'max')
        ).reset_index()
        
        # Compute time-based features relative to seed_time (convert to days)
        product_sales_stats['days_since_first_sale'] = (
            (seed_time - product_sales_stats['first_sale_ts']).dt.total_seconds() / (24 * 3600)
        ).clip(lower=0, upper=1000)  # Cap at 1000 days to avoid extreme outliers
        product_sales_stats['days_since_last_sale'] = (
            (seed_time - product_sales_stats['last_sale_ts']).dt.total_seconds() / (24 * 3600)
        ).clip(lower=0, upper=1000)
        
        # Drop timestamp columns (no longer needed)
        product_sales_stats = product_sales_stats.drop(columns=['first_sale_ts', 'last_sale_ts'])
    else:
        # Create empty DataFrame with required columns if no historical data
        product_sales_stats = pd.DataFrame(columns=[
            'product_id', 'total_units_sold', 'total_revenue', 'total_freight',
            'avg_item_price', 'avg_item_freight', 'days_since_first_sale', 'days_since_last_sale'
        ])
    
    # --------------------------
    # 4. Aggregate delivery performance metrics
    # --------------------------
    if not hist_orders.empty:
        # Calculate delivery delay (actual - estimated, in days)
        hist_orders['delivery_delay_days'] = (
            (hist_orders['order_delivered_customer_date'] - hist_orders['order_estimated_delivery_date']).dt.total_seconds() / (24 * 3600)
        )
        
        # Merge order items with delivery data
        order_delivery = pd.merge(
            hist_items[['order_id', 'product_id']],
            hist_orders[['order_id', 'delivery_delay_days']],
            on='order_id',
            how='left'
        )
        
        # Aggregate per product
        product_delivery_stats = order_delivery.groupby('product_id').agg(
            avg_delivery_delay=('delivery_delay_days', 'mean'),
            pct_on_time_deliveries=('delivery_delay_days', lambda x: (x <= 0).sum() / x.count() if x.count() > 0 else 0.5)
        ).reset_index()
    else:
        product_delivery_stats = pd.DataFrame(columns=[
            'product_id', 'avg_delivery_delay', 'pct_on_time_deliveries'
        ])
    
    # --------------------------
    # 5. Aggregate review metrics
    # --------------------------
    if not hist_reviews.empty:
        # Aggregate reviews per order first
        order_reviews = hist_reviews.groupby('order_id').agg(
            avg_review_score=('review_score', 'mean'),
            total_reviews=('review_id', 'count'),
            has_comment=('review_comment_message', lambda x: x.notna().any())
        ).reset_index()
        
        # Merge with order items to link to products
        order_item_reviews = pd.merge(
            hist_items[['order_id', 'product_id']],
            order_reviews,
            on='order_id',
            how='left'
        )
        
        # Aggregate per product
        product_review_stats = order_item_reviews.groupby('product_id').agg(
            avg_product_review=('avg_review_score', 'mean'),
            total_product_reviews=('total_reviews', 'sum'),
            pct_orders_with_comments=('has_comment', 'mean')
        ).reset_index()
    else:
        product_review_stats = pd.DataFrame(columns=[
            'product_id', 'avg_product_review', 'total_product_reviews', 'pct_orders_with_comments'
        ])
    
    # --------------------------
    # 6. Aggregate payment metrics
    # --------------------------
    if not hist_payments.empty:
        # Aggregate payments per order
        order_payments = hist_payments.groupby('order_id').agg(
            total_order_payment=('payment_value', 'sum'),
            avg_installments=('payment_installments', 'mean'),
            primary_payment_type=('payment_type', lambda x: x.mode()[0] if x.mode().size > 0 else 'unknown')
        ).reset_index()
        
        # Merge with order items to link to products
        order_item_payments = pd.merge(
            hist_items[['order_id', 'product_id']],
            order_payments,
            on='order_id',
            how='left'
        )
        
        # Numerical payment stats per product
        product_payment_stats = order_item_payments.groupby('product_id').agg(
            avg_order_value=('total_order_payment', 'mean'),
            avg_payment_installments=('avg_installments', 'mean')
        ).reset_index()
        
        # Payment type share features (one-hot encode then average for share)
        payment_type_dummies = pd.get_dummies(
            order_item_payments['primary_payment_type'],
            prefix='pct_payment_type'
        )
        payment_type_dummies['product_id'] = order_item_payments['product_id']
        product_payment_shares = payment_type_dummies.groupby('product_id').mean().reset_index()
    else:
        product_payment_stats = pd.DataFrame(columns=[
            'product_id', 'avg_order_value', 'avg_payment_installments'
        ])
        product_payment_shares = pd.DataFrame(columns=['product_id'])
    
    # --------------------------
    # 7. Merge all feature sets into final DataFrame
    # --------------------------
    features = pd.merge(base, product_sales_stats, on='product_id', how='left')
    features = pd.merge(features, product_delivery_stats, on='product_id', how='left')
    features = pd.merge(features, product_review_stats, on='product_id', how='left')
    features = pd.merge(features, product_payment_stats, on='product_id', how='left')
    features = pd.merge(features, product_payment_shares, on='product_id', how='left')
    
    # --------------------------
    # 8. Handle missing values & create derived features
    # --------------------------
    # Create binary indicator for historical sales
    features['has_historical_sales'] = (
        features['total_units_sold'].fillna(0) > 0
    ).astype(int)
    
    # Fill count/sum columns with 0 (no sales = 0)
    count_cols = [
        'total_units_sold', 'total_revenue', 'total_freight',
        'total_product_reviews'
    ]
    for col in count_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0)
    
    # Fill time columns with 1000 days (proxy for never sold)
    time_cols = ['days_since_first_sale', 'days_since_last_sale']
    for col in time_cols:
        if col in features.columns:
            features[col] = features[col].fillna(1000.0)
    
    # Fill remaining numeric columns with 0
    numeric_cols = [c for c in features.columns if c not in ['product_id', 'has_historical_sales']]
    features[numeric_cols] = features[numeric_cols].fillna(0)
    
    # Derived ratio features
    features['freight_revenue_ratio'] = np.where(
        features['total_revenue'] > 0,
        features['total_freight'] / features['total_revenue'],
        0.0
    )
    features['avg_items_per_order'] = np.where(
        features['has_historical_sales'] > 0,
        features['total_units_sold'] / features['has_historical_sales'],  # Wait, corrected: use total_orders proxy, actually total_units_sold / count of unique orders, but we don't have that, so use has_historical_sales (1 if any) to avoid division by zero, better:
        #  actually, we can compute number of unique orders per product from hist_items:
        # Let's add that now:
        0.0
    )
    # Corrected: compute number of unique orders per product properly
    if not hist_items.empty:
        product_order_counts = hist_items.groupby('product_id')[['order_id']].nunique().rename(columns={'order_id': 'total_unique_orders'})
        features = pd.merge(features, product_order_counts, left_on='product_id', right_index=True, how='left')
        features['total_unique_orders'] = features['total_unique_orders'].fillna(0)
    else:
        features['total_unique_orders'] = 0
    
    # Now correct derived features:
    features['avg_items_per_order'] = np.where(
        features['total_unique_orders'] > 0,
        features['total_units_sold'] / features['total_unique_orders'],
        0.0
    )
    features['review_rate'] = np.where(
        features['total_unique_orders'] > 0,
        features['total_product_reviews'] / features['total_unique_orders'],
        0.0
    )
    
    # --------------------------
    # 9. Final formatting: set index to match entity IDs
    # --------------------------
    features = features.set_index('product_id')
    # Ensure index matches input entity_ids (exact set, order preserved as input)
    features = features.reindex(entity_ids)
    
    # Drop any redundant columns (if any) and ensure all are numeric
    final_cols = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
    features = features[final_cols]
    
    return features