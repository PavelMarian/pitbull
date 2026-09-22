import pandas as pd
import numpy as np
def aggregate_window(df: pd.DataFrame, mask: pd.Series, suffix: str) -> pd.DataFrame:
    window_df = df[mask].copy()
    # Create unique order-product pairs for order-level metrics
    order_product = window_df[
        ["product_id", "order_id", "is_delivered", "is_cancelled", 
         "review_score", "delivery_early_days", "total_payment", 
         "avg_installments", "num_payments"]
    ].drop_duplicates()
    # Aggregate line-item level metrics
    line_item_agg = window_df.groupby("product_id").agg(
        num_items=("order_id", "count"),
        total_revenue=("item_total_value", "sum"),
        avg_price=("price", "mean"),
        med_price=("price", "median"),
        max_price=("price", "max"),
        min_price=("price", "min"),
        std_price=("price", "std"),
        avg_freight=("freight_value", "mean"),
        med_freight=("freight_value", "median"),
        max_freight=("freight_value", "max"),
        avg_item_total=("item_total_value", "mean"),
    ).reset_index()
    # Aggregate order-level metrics
    order_agg = order_product.groupby("product_id").agg(
        num_orders=("order_id", "nunique"),
        pct_delivered=("is_delivered", "mean"),
        num_delivered=("is_delivered", "sum"),
        pct_cancelled=("is_cancelled", "mean"),
        num_cancelled=("is_cancelled", "sum"),
        avg_review=("review_score", "mean"),
        num_reviews=("review_score", "count"),
        avg_delivery_early=("delivery_early_days", "mean"),
        avg_total_payment=("total_payment", "mean"),
        avg_installments=("avg_installments", "mean"),
        avg_num_payments=("num_payments", "mean"),
    ).reset_index()
    # Merge aggregates and add items-per-order ratio
    agg = pd.merge(line_item_agg, order_agg, on="product_id", how="outer")
    agg["avg_items_per_order"] = agg["num_items"] / agg["num_orders"]
    # Add suffix to feature names
    agg.columns = [
        "product_id" if col == "product_id" else f"{col}{suffix}"
        for col in agg.columns
    ]
    return agg
def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Extract input DataFrames
    orders_df = db["orders"]
    order_items_df = db["order_items"]
    reviews_df = db["reviews"]
    payments_df = db["payments"]
    # Filter historical order items (no data after seed_time)
    historical_items = order_items_df[order_items_df["ts"] <= seed_time].copy()
    # Merge with orders to get status and delivery timestamps
    items_with_orders = pd.merge(
        historical_items,
        orders_df[["order_id", "order_status", "order_delivered_customer_date", "order_estimated_delivery_date"]],
        on="order_id",
        how="left"
    )
    # Merge with reviews to get review scores, only reviews already created by seed_time
    reviews_available = reviews_df[reviews_df["review_creation_date"] <= seed_time]
    items_with_reviews = pd.merge(
        items_with_orders,
        reviews_available[["order_id", "review_score"]],
        on="order_id",
        how="left"
    )
    # Aggregate payment metrics per order
    payments_agg = payments_df.groupby("order_id").agg(
        total_payment=("payment_value", "sum"),
        num_payments=("payment_sequential", "count"),
        avg_installments=("payment_installments", "mean"),
    ).reset_index()
    # Merge payment aggregates with item data
    items_full = pd.merge(
        items_with_reviews,
        payments_agg,
        on="order_id",
        how="left"
    )
    # Derive additional features; delivered date used only if it already happened by seed_time
    items_full["item_total_value"] = items_full["price"] + items_full["freight_value"]
    delivered_ok = items_full["order_delivered_customer_date"].notna() & (
        items_full["order_delivered_customer_date"] <= seed_time
    )
    items_full["delivery_early_days"] = np.where(
        delivered_ok,
        (
            items_full["order_estimated_delivery_date"] - items_full["order_delivered_customer_date"]
        ).dt.total_seconds() / (24 * 3600),
        np.nan,
    )
    items_full["is_delivered"] = (
        delivered_ok & (items_full["order_status"] == "delivered")
    ).astype(int)
    items_full["is_cancelled"] = (items_full["order_status"] == "canceled").astype(int)
    items_full["days_since_purchase"] = (seed_time - items_full["ts"]).dt.total_seconds() / (24 * 3600)
    # Create time window masks
    masks = {
        "_7d": items_full["days_since_purchase"] <= 7,
        "_30d": items_full["days_since_purchase"] <= 30,
        "_90d": items_full["days_since_purchase"] <= 90,
        "_180d": items_full["days_since_purchase"] <= 180,
        "_all": pd.Series(True, index=items_full.index),
    }
    # Generate windowed aggregate features
    agg_dfs = [aggregate_window(items_full, mask, suffix) for suffix, mask in masks.items()]
    # Generate global features
    last_purchase = historical_items.groupby("product_id")["ts"].max().reset_index()
    last_purchase.columns = ["product_id", "last_order_ts"]
    last_purchase["days_since_last_purchase"] = (
        seed_time - last_purchase["last_order_ts"]
    ).dt.total_seconds() / (24 * 3600)
    first_purchase = historical_items.groupby("product_id")["ts"].min().reset_index()
    first_purchase.columns = ["product_id", "first_order_ts"]
    first_purchase["days_on_sale"] = (
        seed_time - first_purchase["first_order_ts"]
    ).dt.total_seconds() / (24 * 3600)
    sellers_per_product = historical_items.groupby("product_id")["seller_id"].nunique().reset_index()
    sellers_per_product.columns = ["product_id", "num_unique_sellers"]
    # Combine global features
    global_features = last_purchase.merge(first_purchase, on="product_id", how="outer")
    global_features = global_features.merge(sellers_per_product, on="product_id", how="outer")
    # Create base DataFrame with all input entity IDs
    base = pd.DataFrame({"product_id": entity_ids})
    # Merge all features into base
    features = pd.merge(base, global_features, on="product_id", how="left")
    for agg_df in agg_dfs:
        features = pd.merge(features, agg_df, on="product_id", how="left")
    # Add cold-start flag
    features["has_any_sales"] = (~features["days_since_last_purchase"].isna()).astype(int)
    # Drop non-numeric datetime columns
    features = features.drop(columns=["last_order_ts", "first_order_ts"], errors="ignore")
    # Fill missing values for global features
    features["days_since_last_purchase"] = features["days_since_last_purchase"].fillna(9999)
    features["days_on_sale"] = features["days_on_sale"].fillna(-1)
    features["num_unique_sellers"] = features["num_unique_sellers"].fillna(0)
    # Separate column types for filling
    count_cols = [col for col in features.columns if col.startswith("num_") or col.startswith("total_")]
    non_stat_cols = {"product_id", "days_since_last_purchase", "days_on_sale", "num_unique_sellers", "has_any_sales"}
    stat_cols = [col for col in features.columns if col not in non_stat_cols and col not in count_cols]
    # Fill missing values
    features[count_cols] = features[count_cols].fillna(0)
    features[stat_cols] = features[stat_cols].fillna(-1)
    # Set index to entity IDs and reindex to match input order
    features = features.set_index("product_id")
    features = features.reindex(entity_ids)
    return features