import pandas as pd
import numpy as np
def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Extract database tables
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]
    # Define time windows for historical aggregation (name, lookback duration)
    time_windows = [
        ("7d", pd.Timedelta(days=7)),
        ("30d", pd.Timedelta(days=30)),
        ("90d", pd.Timedelta(days=90)),
        ("all_time", pd.Timedelta(days=365 * 10)),  # Covers full dataset history
    ]
    valid_order_statuses = ["delivered", "shipped", "approved", "invoiced", "processing"]
    # Filter order items to valid, pre-seed orders, merge with order metadata
    order_items_valid = order_items[order_items["ts"] < seed_time].merge(
        orders[["order_id", "order_status", "order_delivered_customer_date", "order_estimated_delivery_date"]],
        on="order_id",
        how="left"
    )
    order_items_valid = order_items_valid[order_items_valid["order_status"].isin(valid_order_statuses)]
    # Delivery info exists only for orders already delivered by seed_time; later
    # delivery dates must stay unknown (NaN) at prediction time.
    delivered_before_seed = (
        order_items_valid["order_delivered_customer_date"].notna()
        & (order_items_valid["order_delivered_customer_date"] <= seed_time)
    )
    raw_delay = (
        order_items_valid["order_delivered_customer_date"] - order_items_valid["order_estimated_delivery_date"]
    ).dt.total_seconds() / (24 * 3600)
    order_items_valid["delivery_delay_days"] = np.where(delivered_before_seed, raw_delay, np.nan)
    features_list = []
    # Generate core order/item features for each time window
    for window_name, time_delta in time_windows:
        lower_bound = seed_time - time_delta
        window_data = order_items_valid[
            (order_items_valid["ts"] >= lower_bound) &
            (order_items_valid["ts"] < seed_time)
        ]
        # Aggregate metrics per product
        agg_funcs = {
            "order_id": ["nunique", "count"],
            "price": ["mean", "sum", "min", "max", "std"],
            "freight_value": ["mean", "sum"],
            "seller_id": ["nunique"],
            "delivery_delay_days": ["mean", "std", "count"],
        }
        window_feats = window_data.groupby("product_id").agg(agg_funcs)
        # Flatten multi-index columns and rename for readability
        window_feats.columns = [f"{col[0]}_{col[1]}" for col in window_feats.columns]
        rename_map = {
            "order_id_nunique": f"count_orders_{window_name}",
            "order_id_count": f"total_units_{window_name}",
            "price_mean": f"avg_price_{window_name}",
            "price_sum": f"total_revenue_{window_name}",
            "price_min": f"min_price_{window_name}",
            "price_max": f"max_price_{window_name}",
            "price_std": f"std_price_{window_name}",
            "freight_value_mean": f"avg_freight_{window_name}",
            "freight_value_sum": f"total_freight_{window_name}",
            "seller_id_nunique": f"unique_sellers_{window_name}",
            "delivery_delay_days_mean": f"avg_delivery_delay_days_{window_name}",
            "delivery_delay_days_std": f"std_delivery_delay_days_{window_name}",
            "delivery_delay_days_count": f"delivered_orders_{window_name}",
        }
        window_feats = window_feats.rename(columns=rename_map)
        features_list.append(window_feats)
    # Prepare and add review features
    reviews_valid = reviews[reviews["review_creation_date"] < seed_time]
    reviews_with_products = reviews_valid.merge(
        order_items_valid[["order_id", "product_id"]].drop_duplicates(),
        on="order_id",
        how="inner"
    )
    reviews_with_products["has_comment"] = (
        reviews_with_products["review_comment_title"].notna() |
        reviews_with_products["review_comment_message"].notna()
    )
    for window_name, time_delta in time_windows:
        lower_bound = seed_time - time_delta
        window_reviews = reviews_with_products[
            (reviews_with_products["review_creation_date"] >= lower_bound) &
            (reviews_with_products["review_creation_date"] < seed_time)
        ]
        review_agg = window_reviews.groupby("product_id").agg(
            avg_review_score=("review_score", "mean"),
            total_reviews=("review_score", "count"),
            std_review_score=("review_score", "std"),
            count_comments=("has_comment", "sum"),
            pct_comments=("has_comment", "mean")
        )
        review_agg.columns = [f"{col}_{window_name}" for col in review_agg.columns]
        features_list.append(review_agg)
    # Prepare and add payment features
    payments_valid = payments[payments["ts"] < seed_time]
    payments_with_products = payments_valid.merge(
        order_items_valid[["order_id", "product_id"]].drop_duplicates(),
        on="order_id",
        how="inner"
    )
    for window_name, time_delta in time_windows:
        lower_bound = seed_time - time_delta
        window_payments = payments_with_products[
            (payments_with_products["ts"] >= lower_bound) &
            (payments_with_products["ts"] < seed_time)
        ]
        # Aggregate payment metrics per order first to avoid double-counting
        order_payment_agg = window_payments.groupby("order_id").agg(
            total_order_payment=("payment_value", "sum"),
            avg_order_installments=("payment_installments", "mean")
        )
        order_payment_with_products = order_payment_agg.merge(
            order_items_valid[["order_id", "product_id"]].drop_duplicates(),
            on="order_id",
            how="inner"
        )
        # Core payment aggregates per product
        payment_agg = order_payment_with_products.groupby("product_id").agg(
            avg_order_value=("total_order_payment", "mean"),
            std_order_value=("total_order_payment", "std"),
            avg_installments=("avg_order_installments", "mean"),
            total_orders_paid=("order_id", "nunique")
        )
        # Payment type distribution
        payment_type_counts = window_payments.groupby(["product_id", "payment_type"]).size().unstack(fill_value=0)
        payment_type_counts.columns = [f"count_{pt}_{window_name}" for pt in payment_type_counts.columns]
        total_per_product = payment_type_counts.sum(axis=1)
        payment_type_pct = payment_type_counts.div(total_per_product + 1e-8, axis=0).fillna(0)
        payment_type_pct.columns = [f"pct_{pt}_{window_name}" for pt in payment_type_counts.columns]
        # Merge all payment features and rename
        payment_agg = pd.concat([payment_agg, payment_type_counts, payment_type_pct], axis=1)
        rename_payment = {
            "avg_order_value": f"avg_order_value_{window_name}",
            "std_order_value": f"std_order_value_{window_name}",
            "avg_installments": f"avg_installments_{window_name}",
            "total_orders_paid": f"total_orders_paid_{window_name}"
        }
        payment_agg = payment_agg.rename(columns=rename_payment)
        features_list.append(payment_agg)
    # Add recency and lifetime features
    product_timestamps = order_items_valid.groupby("product_id")["ts"].agg(["min", "max"])
    product_timestamps.columns = ["first_order_ts", "last_order_ts"]
    product_timestamps["days_since_first_order"] = (seed_time - product_timestamps["first_order_ts"]).dt.total_seconds() / (24 * 3600)
    product_timestamps["days_since_last_order"] = (seed_time - product_timestamps["last_order_ts"]).dt.total_seconds() / (24 * 3600)
    product_timestamps["product_lifetime_days"] = (product_timestamps["last_order_ts"] - product_timestamps["first_order_ts"]).dt.total_seconds() / (24 * 3600)
    features_list.append(product_timestamps[["days_since_first_order", "days_since_last_order", "product_lifetime_days"]])
    # Merge all features into a single DataFrame
    all_features = pd.concat(features_list, axis=1)
    # Reindex to include all requested entity IDs, fill missing values
    all_features = all_features.reindex(entity_ids)
    # Identify count-based vs numeric columns for targeted filling
    count_keywords = ['count', 'total', 'unique', 'delivered_orders', 'count_']
    count_cols = [col for col in all_features.columns if any(kw in col for kw in count_keywords)]
    numeric_cols = [col for col in all_features.columns if col not in count_cols]
    # Fill missing count data with 0, numeric data with 0 (safe for models to learn missing pattern)
    all_features[count_cols] = all_features[count_cols].fillna(0)
    all_features[numeric_cols] = all_features[numeric_cols].fillna(0)
    # Add history flag
    all_features["has_history"] = (all_features["count_orders_all_time"] > 0).astype(int)
    # Add recency ratio features (recent activity vs total history)
    for window in ["7d", "30d", "90d"]:
        # Avoid division by zero with small epsilon
        denom_orders = all_features["count_orders_all_time"] + 1e-8
        denom_units = all_features["total_units_all_time"] + 1e-8
        denom_rev = all_features["total_revenue_all_time"] + 1e-8
        denom_reviews = all_features["total_reviews_all_time"] + 1e-8
        
        all_features[f"order_ratio_{window}_all"] = all_features[f"count_orders_{window}"] / denom_orders
        all_features[f"unit_ratio_{window}_all"] = all_features[f"total_units_{window}"] / denom_units
        all_features[f"revenue_ratio_{window}_all"] = all_features[f"total_revenue_{window}"] / denom_rev
        all_features[f"review_ratio_{window}_all"] = all_features[f"total_reviews_{window}"] / denom_reviews
    return all_features