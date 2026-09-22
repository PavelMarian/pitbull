def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Extract tables from the database
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]
    
    # Filter order items to only those before seed_time (historical data)
    oi_hist = order_items[order_items["ts"] < seed_time].copy()
    
    # Merge order items with orders to get customer and order status info
    oi_orders = pd.merge(
        oi_hist,
        orders[["order_id", "customer_id", "order_status"]],
        on="order_id",
        how="left"
    ).dropna(subset=["order_id", "customer_id"])  # Drop invalid rows (should be none)
    
    # Define time windows relative to seed_time
    win_30 = seed_time - pd.Timedelta(days=30)
    win_90 = seed_time - pd.Timedelta(days=90)
    win_180 = seed_time - pd.Timedelta(days=180)
    
    # ------------------------------
    # ALL-TIME HISTORICAL FEATURES
    # ------------------------------
    prod_all = oi_orders.groupby("product_id").agg(
        # Count metrics
        total_units=("order_id", "count"),
        total_orders=("order_id", "nunique"),
        total_customers=("customer_id", "nunique"),
        # Price metrics
        avg_price=("price", "mean"),
        median_price=("price", "median"),
        max_price=("price", "max"),
        min_price=("price", "min"),
        std_price=("price", "std"),
        total_revenue=("price", "sum"),
        # Freight metrics
        avg_freight=("freight_value", "mean"),
        total_freight=("freight_value", "sum"),
        # Time metrics
        last_purchase_ts=("ts", "max"),
        first_purchase_ts=("ts", "min"),
    ).reset_index()
    
    # Compute time-since features
    prod_all["days_since_last_purchase"] = (seed_time - prod_all["last_purchase_ts"]).dt.total_seconds() / (24 * 3600)
    prod_all["days_since_first_purchase"] = (seed_time - prod_all["first_purchase_ts"]).dt.total_seconds() / (24 * 3600)
    prod_all["avg_units_per_order"] = prod_all["total_units"] / prod_all["total_orders"].replace(0, np.nan)
    
    # ------------------------------
    # WINDOWED HISTORICAL FEATURES
    # ------------------------------
    # Last 30 days
    oi_30 = oi_orders[oi_orders["ts"] >= win_30].copy()
    prod_30 = oi_30.groupby("product_id").agg(
        total_units_30d=("order_id", "count"),
        total_orders_30d=("order_id", "nunique"),
        total_revenue_30d=("price", "sum"),
    ).reset_index()
    
    # Last 90 days
    oi_90 = oi_orders[oi_orders["ts"] >= win_90].copy()
    prod_90 = oi_90.groupby("product_id").agg(
        total_units_90d=("order_id", "count"),
        total_orders_90d=("order_id", "nunique"),
        total_customers_90d=("customer_id", "nunique"),
        total_revenue_90d=("price", "sum"),
        avg_price_90d=("price", "mean"),
        last_purchase_90d=("ts", "max"),
    ).reset_index()
    prod_90["days_since_last_purchase_90d"] = (seed_time - prod_90["last_purchase_90d"]).dt.total_seconds() / (24 * 3600)
    
    # Last 180 days
    oi_180 = oi_orders[oi_orders["ts"] >= win_180].copy()
    prod_180 = oi_180.groupby("product_id").agg(
        total_units_180d=("order_id", "count"),
        total_orders_180d=("order_id", "nunique"),
        total_revenue_180d=("price", "sum"),
    ).reset_index()
    
    # ------------------------------
    # ORDER STATUS FEATURES
    # ------------------------------
    order_status = oi_orders.groupby(["product_id", "order_status"]).agg(
        status_count=("order_id", "nunique")
    ).reset_index()
    order_status_pivot = order_status.pivot(
        index="product_id",
        columns="order_status",
        values="status_count"
    ).fillna(0).reset_index()
    order_status_pivot.columns = [
        f"order_status_{col.lower().replace(' ', '_')}" if col != "product_id" else col
        for col in order_status_pivot.columns
    ]
    
    # ------------------------------
    # REVIEW FEATURES
    # ------------------------------
    # Only reviews that already exist at seed_time: a review is written
    # after the purchase, so without this filter review scores leak the future.
    reviews_avail = reviews[reviews["review_creation_date"] <= seed_time]
    oi_reviews = pd.merge(
        oi_orders[["product_id", "order_id"]],
        reviews_avail[["order_id", "review_score", "review_comment_title", "review_comment_message"]],
        on="order_id",
        how="left"
    )
    oi_reviews["has_comment"] = (
        oi_reviews["review_comment_title"].notna() | oi_reviews["review_comment_message"].notna()
    ).astype(int)
    review_features = oi_reviews.groupby("product_id").agg(
        total_reviews=("review_score", "count"),
        avg_review_score=("review_score", "mean"),
        min_review_score=("review_score", "min"),
        max_review_score=("review_score", "max"),
        std_review_score=("review_score", "std"),
        frac_reviews_with_comment=("has_comment", "mean"),
    ).reset_index()
    
    # ------------------------------
    # PAYMENT FEATURES
    # ------------------------------
    oi_payments = pd.merge(
        oi_orders[["product_id", "order_id"]],
        payments[["order_id", "payment_type", "payment_installments", "payment_value"]],
        on="order_id",
        how="left"
    )
    # Aggregated payment features
    payment_features = oi_payments.groupby("product_id").agg(
        total_payment_value=("payment_value", "sum"),
        avg_payment_value_per_item=("payment_value", "mean"),
        avg_installments=("payment_installments", "mean"),
        max_installments=("payment_installments", "max"),
        num_unique_payment_types=("payment_type", "nunique"),
    ).reset_index()
    # Payment type dummy features
    oi_payments = pd.concat([
        oi_payments,
        pd.get_dummies(oi_payments["payment_type"], prefix="pay_type")
    ], axis=1)
    payment_type_features = oi_payments.groupby("product_id").agg(
        **{f"pay_{col}_count": (col, "sum") for col in oi_payments.columns if col.startswith("pay_type_")},
        **{f"pay_{col}_frac": (col, "mean") for col in oi_payments.columns if col.startswith("pay_type_")},
    ).reset_index()
    
    # ------------------------------
    # MERGE ALL FEATURES
    # ------------------------------
    all_features = prod_all.copy()
    all_features = pd.merge(all_features, prod_30, on="product_id", how="outer")
    all_features = pd.merge(all_features, prod_90, on="product_id", how="outer")
    all_features = pd.merge(all_features, prod_180, on="product_id", how="outer")
    all_features = pd.merge(all_features, order_status_pivot, on="product_id", how="outer")
    all_features = pd.merge(all_features, review_features, on="product_id", how="outer")
    all_features = pd.merge(all_features, payment_features, on="product_id", how="outer")
    all_features = pd.merge(all_features, payment_type_features, on="product_id", how="outer")
    
    # ------------------------------
    # CREATE ADDITIONAL DERIVED FEATURES
    # ------------------------------
    # Binary flag for any sales history
    all_features["has_any_sales"] = (~all_features["total_units"].isna()).astype(int)
    
    # Growth ratio features
    all_features["ratio_30d_to_90d_units"] = all_features["total_units_30d"] / all_features["total_units_90d"].replace(0, np.nan)
    all_features["ratio_90d_to_180d_units"] = all_features["total_units_90d"] / all_features["total_units_180d"].replace(0, np.nan)
    all_features["ratio_30d_to_all_units"] = all_features["total_units_30d"] / all_features["total_units"].replace(0, np.nan)
    all_features["ratio_90d_to_all_units"] = all_features["total_units_90d"] / all_features["total_units"].replace(0, np.nan)
    
    # Average order interval
    all_features["avg_days_between_orders"] = np.where(
        all_features["total_orders"] > 1,
        (all_features["days_since_first_purchase"] - all_features["days_since_last_purchase"]) / (all_features["total_orders"] - 1),
        0
    )
    
    # ------------------------------
    # CLEAN AND REINDEX TO INPUT ENTITIES
    # ------------------------------
    # Drop non-numeric datetime columns
    datetime_cols = ["last_purchase_ts", "first_purchase_ts", "last_purchase_90d"]
    all_features = all_features.drop(columns=[col for col in datetime_cols if col in all_features.columns], errors="ignore")
    
    # Set product_id as index and reindex to input entity_ids
    all_features = all_features.set_index("product_id")
    final_features = all_features.reindex(entity_ids)
    
    # Identify feature groups for consistent NaN filling
    count_cols = [col for col in final_features.columns if (
        col.startswith("total_") or col.startswith("num_") or 
        (col.startswith("pay_") and col.endswith("_count")) or
        col.startswith("order_status_")
    )]
    avg_cols = [col for col in final_features.columns if (
        col.startswith("avg_") or col.startswith("median_") or
        col.startswith("max_") or col.startswith("min_") or
        col.startswith("std_") or col.endswith("_frac")
    )]
    time_cols = [
        "days_since_last_purchase", "days_since_first_purchase",
        "days_since_last_purchase_90d"
    ]
    ratio_cols = [col for col in final_features.columns if col.startswith("ratio_")]
    
    # Fill NaNs for all feature groups
    final_features["has_any_sales"] = final_features["has_any_sales"].fillna(0)
    final_features[count_cols] = final_features[count_cols].fillna(0)
    final_features[avg_cols] = final_features[avg_cols].fillna(0)
    final_features[time_cols] = final_features[time_cols].fillna(10000)  # Arbitrary large value for "never purchased"
    final_features[ratio_cols] = final_features[ratio_cols].fillna(0)
    final_features["avg_units_per_order"] = final_features["avg_units_per_order"].fillna(0)
    final_features["avg_days_between_orders"] = final_features["avg_days_between_orders"].fillna(0)
    
    # Final cleanup of any remaining invalid values
    final_features = final_features.fillna(0)
    final_features = final_features.replace([np.inf, -np.inf], 0)
    
    return final_features