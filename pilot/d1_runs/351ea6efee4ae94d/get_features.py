import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Extract tables from database
    orders_df = db["orders"]
    order_items_df = db["order_items"]
    reviews_df = db["reviews"]
    payments_df = db["payments"]
    
    # Filter historical order items for target products and time
    oi_hist = order_items_df[
        (order_items_df["product_id"].isin(entity_ids)) &
        (order_items_df["ts"] <= seed_time)
    ].copy()
    
    # Initialize empty features DataFrame for edge case of no historical data
    if oi_hist.empty:
        result = pd.DataFrame(index=entity_ids)
        # Add placeholder columns with default values
        default_features = [
            "recency_days", "total_items_sold", "total_orders",
            "total_revenue", "total_freight", "avg_item_price",
            "days_since_first_purchase", "orders_per_day",
            "avg_review_score", "total_reviews", "frac_5star",
            "frac_1star", "days_since_last_review",
            "avg_payment_installments", "total_payment_value",
            "frac_credit_card", "frac_delivered", "frac_canceled",
            "avg_delivery_days", "frac_late_deliveries"
        ]
        for col in default_features:
            result[col] = 0
        result["has_historical_data"] = 0
        # Ensure all columns are numeric
        for col in result.columns:
            result[col] = pd.to_numeric(result[col])
        return result
    
    # Get unique order IDs from historical order items
    order_ids_hist = oi_hist["order_id"].unique()
    
    # Filter other tables to relevant orders and time
    orders_hist = orders_df[
        (orders_df["order_id"].isin(order_ids_hist)) &
        (orders_df["order_purchase_timestamp"] <= seed_time)
    ].copy()
    
    reviews_hist = reviews_df[
        (reviews_df["order_id"].isin(order_ids_hist)) &
        (reviews_df["review_creation_date"] <= seed_time)
    ].copy()
    
    payments_hist = payments_df[
        (payments_df["order_id"].isin(order_ids_hist)) &
        (payments_df["ts"] <= seed_time)
    ].copy()
    
    # Create order-product mapping to link other tables to product_id
    order_product = oi_hist[["order_id", "product_id"]].drop_duplicates()
    
    # --------------------
    # 1. Order Items Features
    # --------------------
    oi_grouped = oi_hist.groupby("product_id")
    oi_features = pd.DataFrame()
    
    last_purchase = oi_grouped["ts"].max()
    oi_features["recency_days"] = (seed_time - last_purchase).dt.days
    
    oi_features["total_items_sold"] = oi_grouped.size()
    oi_features["total_orders"] = oi_grouped["order_id"].nunique()
    oi_features["total_revenue"] = oi_grouped["price"].sum()
    oi_features["total_freight"] = oi_grouped["freight_value"].sum()
    oi_features["avg_item_price"] = oi_grouped["price"].mean()
    
    first_purchase = oi_grouped["ts"].min()
    oi_features["days_since_first_purchase"] = (seed_time - first_purchase).dt.days
    oi_features["orders_per_day"] = oi_features["total_orders"] / (oi_features["days_since_first_purchase"] + 1e-6)
    
    # --------------------
    # 2. Reviews Features
    # --------------------
    reviews_product = pd.merge(reviews_hist, order_product, on="order_id", how="inner")
    reviews_grouped = reviews_product.groupby("product_id")
    reviews_features = pd.DataFrame()
    
    reviews_features["avg_review_score"] = reviews_grouped["review_score"].mean()
    reviews_features["total_reviews"] = reviews_grouped.size()
    reviews_features["frac_5star"] = reviews_grouped["review_score"].apply(lambda x: (x == 5).sum() / len(x) if len(x) > 0 else 0.0)
    reviews_features["frac_1star"] = reviews_grouped["review_score"].apply(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0.0)
    
    last_review = reviews_grouped["review_creation_date"].max()
    reviews_features["days_since_last_review"] = (seed_time - last_review).dt.days
    
    # --------------------
    # 3. Payments Features
    # --------------------
    payments_product = pd.merge(payments_hist, order_product, on="order_id", how="inner")
    payments_product["is_credit_card"] = (payments_product["payment_type"] == "credit_card").astype(int)
    payments_grouped = payments_product.groupby("product_id")
    payments_features = pd.DataFrame()
    
    payments_features["avg_payment_installments"] = payments_grouped["payment_installments"].mean()
    payments_features["total_payment_value"] = payments_grouped["payment_value"].sum()
    payments_features["frac_credit_card"] = payments_grouped["is_credit_card"].mean()
    
    # --------------------
    # 4. Orders Features
    # --------------------
    # Point-in-time discipline: order_status and delivery dates evolve after
    # purchase, so only facts already fixed by seed_time may be used.
    orders_product = pd.merge(orders_hist, order_product, on="order_id", how="inner")
    delivered_mask = (
        orders_product["order_delivered_customer_date"].notna()
        & (orders_product["order_delivered_customer_date"] <= seed_time)
    )
    orders_product["is_delivered"] = delivered_mask.astype(int)
    # A cancellation is known at seed_time only if the order never got delivered
    # and its status is already terminal.
    is_cancel_status = orders_product["order_status"].isin(["canceled", "unavailable"])
    orders_product["is_canceled"] = (is_cancel_status & ~delivered_mask).astype(int)

    orders_grouped = orders_product.groupby("product_id")
    orders_features = pd.DataFrame()

    orders_features["frac_delivered"] = orders_grouped["is_delivered"].mean()
    orders_features["frac_canceled"] = orders_grouped["is_canceled"].mean()

    # Delivery speed and lateness: computed only over orders that were already
    # delivered strictly before seed_time, so both dates are in the past.
    delivered_rows = orders_product[delivered_mask].copy()
    delivered_rows["delivery_days"] = (delivered_rows["order_delivered_customer_date"] - delivered_rows["order_purchase_timestamp"]).dt.days
    delivered_rows["is_late"] = (delivered_rows["order_delivered_customer_date"] > delivered_rows["order_estimated_delivery_date"]).astype(int)

    delivered_grouped = delivered_rows.groupby("product_id")
    orders_features["avg_delivery_days"] = delivered_grouped["delivery_days"].mean()
    orders_features["frac_late_deliveries"] = delivered_grouped["is_late"].mean()
    
    # --------------------
    # Combine All Features
    # --------------------
    all_features = pd.concat(
        [oi_features, reviews_features, payments_features, orders_features],
        axis=1,
        join="outer"
    )
    
    # Create result DataFrame indexed by target entity_ids
    result = pd.DataFrame(index=entity_ids)
    result = result.join(all_features, how="left")
    
    # Add indicator for historical data presence
    result["has_historical_data"] = (~result["recency_days"].isna()).astype(int)
    
    # Fill missing values with defaults
    fill_defaults = {
        "recency_days": 730,
        "total_items_sold": 0,
        "total_orders": 0,
        "total_revenue": 0,
        "total_freight": 0,
        "avg_item_price": 0,
        "days_since_first_purchase": 0,
        "orders_per_day": 0,
        "avg_review_score": 0,
        "total_reviews": 0,
        "frac_5star": 0.0,
        "frac_1star": 0.0,
        "days_since_last_review": 730,
        "avg_payment_installments": 0,
        "total_payment_value": 0,
        "frac_credit_card": 0.0,
        "frac_delivered": 0.0,
        "frac_canceled": 0.0,
        "avg_delivery_days": 0,
        "frac_late_deliveries": 0.0
    }
    result = result.fillna(fill_defaults)
    
    # Ensure all columns are numeric
    for col in result.columns:
        result[col] = pd.to_numeric(result[col])
    
    return result