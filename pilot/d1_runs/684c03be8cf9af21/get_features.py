import pandas as pd
import numpy as np
def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Extract tables from database
    order_items_df = db['order_items']
    reviews_df = db['reviews']
    payments_df = db['payments']
    # Helper function to check if timestamp is in [window_start, seed_time)
    def get_window_flag(ts, window_start, seed):
        return (ts >= window_start) & (ts < seed)
    # Helper function to aggregate product data per time window
    def aggregate_product_data(data, suffix):
        agg_dict = {
            'order_id': 'nunique',
            'num_units': ['sum', 'mean', 'max'],
            'total_product_price': ['sum', 'mean', 'std'],
            'total_product_freight': ['sum', 'mean'],
            'total_product_cost': ['sum', 'mean', 'std'],
            'avg_unit_price': ['mean', 'std', 'min', 'max'],
            'avg_unit_freight': ['mean', 'std'],
            'avg_unit_cost': ['mean', 'std', 'min', 'max'],
            'has_review': ['sum', 'mean'],
            'avg_review_score': ['mean', 'std', 'min', 'max'],
            'total_comment_length': ['sum', 'mean'],
            'num_comments': ['sum', 'mean'],
            'total_order_payment': ['mean', 'std'],
            'num_payments': ['mean', 'sum'],
            'avg_installments': ['mean', 'std', 'max'],
        }
        grouped = data.groupby('product_id').agg(agg_dict)
        grouped.columns = [f'{col[0]}_{col[1]}_{suffix}' for col in grouped.columns]
        grouped = grouped.reset_index()
        return grouped
    # Step 1: Aggregate order items to (order_id, product_id) level
    order_items_product_agg = order_items_df.groupby(['order_id', 'product_id']).agg(
        num_units=('order_item_id', 'count'),
        total_product_price=('price', 'sum'),
        total_product_freight=('freight_value', 'sum'),
        avg_unit_price=('price', 'mean'),
        avg_unit_freight=('freight_value', 'mean'),
        order_purchase_ts=('ts', 'first')
    ).reset_index()
    # Filter to only include orders purchased before seed_time
    order_items_product_agg = order_items_product_agg[order_items_product_agg['order_purchase_ts'] < seed_time].copy()
    # Derived cost features (price and freight are known at purchase time)
    order_items_product_agg['total_product_cost'] = order_items_product_agg['total_product_price'] + order_items_product_agg['total_product_freight']
    order_items_product_agg['avg_unit_cost'] = order_items_product_agg['avg_unit_price'] + order_items_product_agg['avg_unit_freight']
    # Step 2: Process and merge review data
    # Only reviews that already exist at seed_time (creation date in the past)
    reviews_pre_seed = reviews_df[reviews_df['review_creation_date'] < seed_time].copy()
    reviews_pre_seed['has_comment'] = ~reviews_pre_seed['review_comment_message'].isna()
    reviews_pre_seed['comment_length'] = reviews_pre_seed['review_comment_message'].fillna('').str.len()
    # Aggregate reviews per order
    reviews_per_order = reviews_pre_seed.groupby('order_id').agg(
        num_reviews=('review_id', 'count'),
        avg_review_score=('review_score', 'mean'),
        total_comment_length=('comment_length', 'sum'),
        num_comments=('has_comment', 'sum')
    ).reset_index()
    # Keep only orders whose reviews already exist at seed_time
    order_product_with_reviews = pd.merge(
        order_items_product_agg,
        reviews_per_order,
        on='order_id',
        how='left'
    )
    order_product_with_reviews['has_review'] = (order_product_with_reviews['num_reviews'].fillna(0) > 0).astype(int)
    # Step 3: Process and merge payment data
    # Only payments of orders purchased before seed_time (ts is the purchase moment)
    payments_pre_seed = payments_df[payments_df['ts'] < seed_time].copy()
    # Aggregate payments per order
    payments_per_order = payments_pre_seed.groupby('order_id').agg(
        total_order_payment=('payment_value', 'sum'),
        num_payments=('payment_sequential', 'count'),
        avg_installments=('payment_installments', 'mean')
    ).reset_index()
    # Merge payments with order-product data
    order_product_full = pd.merge(
        order_product_with_reviews,
        payments_per_order,
        on='order_id',
        how='left'
    )
    # Step 4: Define time windows and assign flags
    window_30d = seed_time - pd.Timedelta(days=30)
    window_90d = seed_time - pd.Timedelta(days=90)
    window_180d = seed_time - pd.Timedelta(days=180)
    window_365d = seed_time - pd.Timedelta(days=365)
    order_product_full['in_30d'] = get_window_flag(order_product_full['order_purchase_ts'], window_30d, seed_time)
    order_product_full['in_90d'] = get_window_flag(order_product_full['order_purchase_ts'], window_90d, seed_time)
    order_product_full['in_180d'] = get_window_flag(order_product_full['order_purchase_ts'], window_180d, seed_time)
    order_product_full['in_365d'] = get_window_flag(order_product_full['order_purchase_ts'], window_365d, seed_time)
    # Split data into time windows
    data_30d = order_product_full[order_product_full['in_30d']].copy()
    data_90d = order_product_full[order_product_full['in_90d']].copy()
    data_180d = order_product_full[order_product_full['in_180d']].copy()
    data_365d = order_product_full[order_product_full['in_365d']].copy()
    data_all = order_product_full.copy()
    # Aggregate each window
    agg_30d = aggregate_product_data(data_30d, '30d')
    agg_90d = aggregate_product_data(data_90d, '90d')
    agg_180d = aggregate_product_data(data_180d, '180d')
    agg_365d = aggregate_product_data(data_365d, '365d')
    agg_all = aggregate_product_data(data_all, 'all')
    # Step 5: Compute additional recency/tenure/frequency features
    # Days since last order
    last_order_per_product = order_product_full.groupby('product_id')['order_purchase_ts'].max().reset_index()
    last_order_per_product['days_since_last_order'] = (
        seed_time - last_order_per_product['order_purchase_ts']
    ).dt.total_seconds() / (24 * 3600)
    last_order_per_product['days_since_last_order'] = last_order_per_product['days_since_last_order'].round(2)
    # Product tenure (days since first order)
    first_order_per_product = order_product_full.groupby('product_id')['order_purchase_ts'].min().reset_index()
    first_order_per_product['product_tenure_days'] = (
        seed_time - first_order_per_product['order_purchase_ts']
    ).dt.total_seconds() / (24 * 3600)
    first_order_per_product['product_tenure_days'] = first_order_per_product['product_tenure_days'].round(2)
    # Order frequency (orders per day)
    order_count_all = agg_all[['product_id', 'order_id_nunique_all']].copy()
    product_frequency = pd.merge(
        order_count_all,
        first_order_per_product[['product_id', 'product_tenure_days']],
        on='product_id',
        how='left'
    )
    product_frequency['orders_per_day'] = product_frequency['order_id_nunique_all'] / (product_frequency['product_tenure_days'] + 1e-6)
    product_frequency = product_frequency[['product_id', 'orders_per_day']]
    # Trend ratio features
    order_counts_trend = pd.merge(
        agg_30d[['product_id', 'order_id_nunique_30d']].rename(columns={'order_id_nunique_30d': 'orders_30d'}),
        agg_90d[['product_id', 'order_id_nunique_90d']].rename(columns={'order_id_nunique_90d': 'orders_90d'}),
        on='product_id',
        how='outer'
    )
    order_counts_trend = pd.merge(
        order_counts_trend,
        agg_180d[['product_id', 'order_id_nunique_180d']].rename(columns={'order_id_nunique_180d': 'orders_180d'}),
        on='product_id',
        how='outer'
    )
    order_counts_trend = order_counts_trend.fillna(0)
    order_counts_trend['ratio_30d_to_90d'] = order_counts_trend['orders_30d'] / (order_counts_trend['orders_90d'] + 1e-6)
    order_counts_trend['ratio_30d_to_180d'] = order_counts_trend['orders_30d'] / (order_counts_trend['orders_180d'] + 1e-6)
    order_counts_trend['ratio_90d_to_180d'] = (
        (order_counts_trend['orders_90d'] - order_counts_trend['orders_30d']) 
        / (order_counts_trend['orders_180d'] - order_counts_trend['orders_90d'] + 1e-6)
    )
    # Step 6: Merge all features into base dataframe
    base = pd.DataFrame({'product_id': entity_ids})
    dfs_to_merge = [
        agg_all, agg_365d, agg_180d, agg_90d, agg_30d,
        last_order_per_product[['product_id', 'days_since_last_order']],
        first_order_per_product[['product_id', 'product_tenure_days']],
        product_frequency,
        order_counts_trend
    ]
    features_df = base.copy()
    for df in dfs_to_merge:
        if not df.empty:
            features_df = pd.merge(features_df, df, on='product_id', how='left')
    # Add has_history indicator
    features_df['has_history'] = (~features_df['days_since_last_order'].isna()).astype(int)
    # Handle missing values
    all_cols = [c for c in features_df.columns if c != 'product_id']
    count_sum_cols = [
        c for c in all_cols 
        if ('_sum_' in c) or ('_count_' in c) or ('_nunique_' in c) or ('orders_' in c) or ('num_' in c)
    ]
    last_order_col = 'days_since_last_order'
    tenure_col = 'product_tenure_days'
    remaining_cols = [c for c in all_cols if c not in count_sum_cols and c not in [last_order_col, tenure_col]]
    features_df[count_sum_cols] = features_df[count_sum_cols].fillna(0)
    features_df[last_order_col] = features_df[last_order_col].fillna(9999)
    features_df[tenure_col] = features_df[tenure_col].fillna(0)
    features_df[remaining_cols] = features_df[remaining_cols].fillna(0)
    # Remove duplicates and reindex to match entity_ids
    features_df = features_df.drop_duplicates(subset='product_id')
    features_df = features_df.set_index('product_id')
    features_df = features_df.reindex(entity_ids)
    # Ensure all features are numeric
    features_df = features_df.astype(float)
    return features_df
