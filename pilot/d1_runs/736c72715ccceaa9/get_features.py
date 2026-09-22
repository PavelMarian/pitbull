import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Define lookback windows
    lookback_90 = seed_time - pd.Timedelta(days=90)
    lookback_180 = seed_time - pd.Timedelta(days=180)
    
    # Extract table references
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    
    # Filter relevant order items
    relevant_items = order_items[order_items["product_id"].isin(entity_ids)]
    # Merge with orders to get purchase timestamps and status
    items_full = relevant_items.merge(
        orders[["order_id", "order_purchase_timestamp", "order_status"]],
        on="order_id",
        how="left"
    )
    # Filter to items purchased before seed time
    items_historic = items_full[items_full["order_purchase_timestamp"] <= seed_time]
    
    # 90-day lookback features
    items_90 = items_historic[items_historic["order_purchase_timestamp"] > lookback_90]
    agg_90 = items_90.groupby("product_id").agg(
        units_sold_90=("order_item_id", "count"),
        revenue_90=("price", "sum"),
        avg_price_90=("price", "mean"),
        total_freight_90=("freight_value", "sum"),
        unique_orders_90=("order_id", "nunique")
    ).reset_index()
    
    # 180-day lookback features
    items_180 = items_historic[items_historic["order_purchase_timestamp"] > lookback_180]
    agg_180 = items_180.groupby("product_id").agg(
        units_sold_180=("order_item_id", "count"),
        revenue_180=("price", "sum"),
        avg_price_180=("price", "mean"),
        total_freight_180=("freight_value", "sum"),
        unique_orders_180=("order_id", "nunique")
    ).reset_index()
    
    # Recency feature
    last_purchase = items_historic.groupby("product_id")["order_purchase_timestamp"].max().reset_index()
    last_purchase["recency_days"] = (seed_time - last_purchase["order_purchase_timestamp"]).dt.days
    recency = last_purchase[["product_id", "recency_days"]]
    
    # Review features: only reviews that already exist at seed time
    reviews_available = reviews[reviews["review_creation_date"] <= seed_time]
    items_with_reviews = items_historic.merge(
        reviews_available[["order_id", "review_score"]],
        on="order_id",
        how="left"
    )
    review_agg = items_with_reviews.groupby("product_id").agg(
        avg_review=("review_score", "mean"),
        total_reviews=("review_score", "count")
    ).reset_index()
    
    # Create base dataframe with all entity IDs
    final_features = pd.DataFrame({"product_id": entity_ids})
    
    # Merge all feature groups
    for df in [agg_90, agg_180, recency, review_agg]:
        final_features = final_features.merge(df, on="product_id", how="left")
    
    # Set index to product ID (matches entity_ids)
    final_features = final_features.set_index("product_id")
    
    # Fill missing values for products with no history
    fill_zero_cols = [
        "units_sold_90", "revenue_90", "avg_price_90", "total_freight_90", "unique_orders_90",
        "units_sold_180", "revenue_180", "avg_price_180", "total_freight_180", "unique_orders_180",
        "avg_review", "total_reviews"
    ]
    for col in fill_zero_cols:
        final_features[col] = final_features[col].fillna(0)
    
    # Fill recency with a large value (no recent sales)
    final_features["recency_days"] = final_features["recency_days"].fillna(365)
    
    # Reindex to ensure exact entity_ids order (requirement compliance)
    final_features = final_features.reindex(entity_ids)
    
    return final_features