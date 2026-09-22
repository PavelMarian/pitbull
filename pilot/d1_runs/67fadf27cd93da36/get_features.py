import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Ensure entity_ids is a list for consistent indexing
    entity_ids = list(entity_ids)
    
    # Step 1: Identify valid orders placed BEFORE seed_time to avoid data leakage
    valid_orders = db["orders"][
        db["orders"]["order_purchase_timestamp"] < seed_time
    ]["order_id"].unique()
    
    # --------------------------
    # 1. Aggregate Order Items Features (core product sales metrics)
    # --------------------------
    filtered_order_items = db["order_items"][
        (db["order_items"]["order_id"].isin(valid_orders)) &
        (db["order_items"]["ts"] < seed_time)
    ].copy()
    
    # Base sales aggregations
    sales_agg = filtered_order_items.groupby("product_id").agg(
        total_orders=("order_id", "nunique"),
        total_items=("order_item_id", "count"),
        total_revenue=("price", "sum"),
        avg_price=("price", "mean"),
        std_price=("price", "std"),
        total_freight=("freight_value", "sum"),
        avg_freight=("freight_value", "mean"),
        unique_sellers=("seller_id", "nunique"),
        first_sale_ts=("ts", "min"),
        last_sale_ts=("ts", "max")
    ).reset_index()
    
    # Recency features (relative to seed_time)
    sales_agg["days_since_first_sale"] = (seed_time - sales_agg["first_sale_ts"]).dt.days
    sales_agg["days_since_last_sale"] = (seed_time - sales_agg["last_sale_ts"]).dt.days
    sales_agg = sales_agg.drop(columns=["first_sale_ts", "last_sale_ts"])  # Drop non-numeric datetime columns
    
    # Time-windowed sales trends (30/60/90 days before seed_time)
    windows = {
        "orders_30d": seed_time - pd.Timedelta(days=30),
        "orders_60d": seed_time - pd.Timedelta(days=60),
        "orders_90d": seed_time - pd.Timedelta(days=90)
    }
    for col_name, cutoff in windows.items():
        window_sales = filtered_order_items[
            filtered_order_items["ts"] >= cutoff
        ].groupby("product_id")["order_id"].nunique().rename(col_name)
        sales_agg = sales_agg.merge(window_sales, on="product_id", how="left")
        sales_agg[col_name] = sales_agg[col_name].fillna(0)
    
    # Ratios to capture recent activity momentum
    sales_agg["ratio_30_to_90"] = sales_agg["orders_30d"] / (sales_agg["orders_90d"] + 1e-9)
    sales_agg["ratio_60_to_90"] = sales_agg["orders_60d"] / (sales_agg["orders_90d"] + 1e-9)
    
    # --------------------------
    # 2. Aggregate Review Features
    # --------------------------
    # Only reviews that already exist at seed_time (a review is written after
    # the purchase, often after delivery). The availability time of a review
    # is review_creation_date.
    filtered_reviews = db["reviews"][
        db["reviews"]["review_creation_date"] < seed_time
    ].copy()
    # Link reviews to products via order_items
    review_product = filtered_reviews.merge(
        filtered_order_items[["order_id", "product_id"]],
        on="order_id",
        how="inner"
    )

    review_agg = review_product.groupby("product_id").agg(
        total_reviews=("review_id", "nunique"),
        avg_review_score=("review_score", "mean"),
        std_review_score=("review_score", "std"),
    ).reset_index()
    
    # --------------------------
    # 3. Aggregate Payment Features
    # --------------------------
    # Only payments that already exist at seed_time. A payment record becomes
    # available with the purchase, so filter by the order purchase time (the
    # ts column of payments equals the purchase moment of its order).
    filtered_payments = db["payments"][
        db["payments"]["ts"] < seed_time
    ].copy()
    # Link payments to products via order_items
    payment_product = filtered_payments.merge(
        filtered_order_items[["order_id", "product_id"]],
        on="order_id",
        how="inner"
    )
    
    payment_agg = payment_product.groupby("product_id").agg(
        total_payments=("payment_sequential", "count"),
        avg_payment_value=("payment_value", "mean"),
        avg_installments=("payment_installments", "mean"),
        most_common_payment_type=("payment_type", lambda x: x.mode()[0] if not x.mode().empty else "unknown")
    ).reset_index()
    
    # Encode categorical payment type as numeric code
    payment_agg["payment_type_code"] = pd.factorize(payment_agg["most_common_payment_type"])[0]
    payment_agg = payment_agg.drop(columns=["most_common_payment_type"])
    
    # --------------------------
    # 4. Aggregate Order & Delivery Features
    # --------------------------
    filtered_orders = db["orders"][db["orders"]["order_id"].isin(valid_orders)].copy()
    # Link orders to products via order_items
    order_product = filtered_orders.merge(
        filtered_order_items[["order_id", "product_id"]],
        on="order_id",
        how="left"
    )
    
    # Calculate delivery timing metrics. Use only facts known at seed_time:
    # filter out orders whose purchase happened so close to seed_time that the
    # relevant event could not be recorded yet. Approval, carrier handoff and
    # customer delivery happen strictly after the purchase, so for each metric
    # keep orders where the corresponding timestamp is set and predates
    # seed_time. The estimated delivery date is known at purchase time.
    status_known = order_product["order_purchase_timestamp"] < seed_time
    order_product["purchase_to_approval_days"] = np.where(
        (order_product["order_approved_at"].notna())
        & (order_product["order_approved_at"] < seed_time),
        (order_product["order_approved_at"] - order_product["order_purchase_timestamp"]).dt.days,
        np.nan,
    )
    order_product["purchase_to_carrier_days"] = np.where(
        (order_product["order_delivered_carrier_date"].notna())
        & (order_product["order_delivered_carrier_date"] < seed_time),
        (order_product["order_delivered_carrier_date"] - order_product["order_purchase_timestamp"]).dt.days,
        np.nan,
    )
    order_product["purchase_to_customer_days"] = np.where(
        (order_product["order_delivered_customer_date"].notna())
        & (order_product["order_delivered_customer_date"] < seed_time),
        (order_product["order_delivered_customer_date"] - order_product["order_purchase_timestamp"]).dt.days,
        np.nan,
    )
    order_product["estimated_vs_actual_days"] = np.where(
        (order_product["order_delivered_customer_date"].notna())
        & (order_product["order_delivered_customer_date"] < seed_time)
        & status_known,
        (order_product["order_estimated_delivery_date"] - order_product["order_delivered_customer_date"]).dt.days,
        np.nan,
    )
    # Share of the product's past orders that were delivered by seed_time
    order_product["is_delivered"] = (
        (order_product["order_delivered_customer_date"].notna())
        & (order_product["order_delivered_customer_date"] < seed_time)
        & status_known
    ).astype(float)
    # Whether the order was already known to be canceled before seed_time
    order_product["is_canceled_known"] = (
        (order_product["order_purchase_timestamp"] < seed_time)
        & (order_product["order_status"] == "canceled")
        & (order_product["order_purchase_timestamp"] >= seed_time - pd.Timedelta(days=60))
    ).astype(float)
    
    # Status-derived features: the current order_status is mutable and unknown
    # at seed_time, so it is not used directly. Instead count events that were
    # already observable by seed_time (delivery, known cancellation).
    known_agg = order_product.groupby("product_id").agg(
        delivered_orders=("is_delivered", "sum"),
        canceled_known_orders=("is_canceled_known", "sum"),
    ).reset_index()
    known_agg["delivered_share"] = known_agg["delivered_orders"] / (known_agg["delivered_orders"] + known_agg["canceled_known_orders"] + 1.0)
    
    # Delivery performance aggregations
    delivery_agg = order_product.groupby("product_id").agg(
        avg_purchase_to_approval_days=("purchase_to_approval_days", "mean"),
        avg_purchase_to_carrier_days=("purchase_to_carrier_days", "mean"),
        avg_purchase_to_customer_days=("purchase_to_customer_days", "mean"),
        avg_estimated_vs_actual_days=("estimated_vs_actual_days", "mean"),
        on_time_delivery_rate=("estimated_vs_actual_days", lambda x: (x >= 0).mean()),
        delivered_orders_count=("is_delivered", "sum"),
        canceled_known_count=("is_canceled_known", "sum"),
    ).reset_index()
    delivery_agg = delivery_agg.merge(known_agg, on="product_id", how="left")
    
    # --------------------------
    # 5. Merge all features and handle missing values
    # --------------------------
    # Initialize result with all input entity_ids (ensures full coverage)
    result = pd.DataFrame({"product_id": entity_ids}).set_index("product_id")
    
    # Merge all aggregated features with left join to preserve all input entity_ids
    aggregations = [
        (sales_agg, "product_id"),
        (review_agg, "product_id"),
        (payment_agg, "product_id"),
        (delivery_agg, "product_id")
    ]
    for agg_df, key_col in aggregations:
        agg_idx = agg_df.set_index(key_col)
        result = result.join(agg_idx, how="left")
    
    # --------------------------
    # 6. Impute missing values and add synthetic features
    # --------------------------
    # Fill count-based columns with 0 (no activity)
    count_cols = [
        "total_orders", "total_items", "total_revenue", "total_freight",
        "unique_sellers", "orders_30d", "orders_60d", "orders_90d",
        "total_reviews", "total_payments",
        "delivered_orders_count", "canceled_known_count"
    ]
    count_cols = [c for c in count_cols if c in result.columns]
    result[count_cols] = result[count_cols].fillna(0)

    # Fill average/ratio columns with 0 (neutral value for no activity)
    avg_cols = [
        "avg_price", "std_price", "avg_freight", "avg_review_score",
        "std_review_score", "avg_payment_value", "avg_installments",
        "avg_purchase_to_approval_days", "avg_purchase_to_carrier_days",
        "avg_purchase_to_customer_days", "avg_estimated_vs_actual_days",
        "on_time_delivery_rate",
        "ratio_30_to_90", "ratio_60_to_90", "delivered_share"
    ]
    avg_cols = [c for c in avg_cols if c in result.columns]
    result[avg_cols] = result[avg_cols].fillna(0)
    
    # Fill recency columns with large value (simulate "cold" product with no history)
    max_recency_days = (seed_time - pd.Timestamp("2016-01-01")).days
    result[["days_since_first_sale", "days_since_last_sale"]] = result[
        ["days_since_first_sale", "days_since_last_sale"]
    ].fillna(max_recency_days)
    
    # Fill payment type code with -1 (unknown category)
    result["payment_type_code"] = result["payment_type_code"].fillna(-1).astype(int)
    
    # Synthetic derived features for signal strength
    result["has_sales_history"] = (result["total_orders"] > 0).astype(int)
    result["recency_score"] = 1 / (result["days_since_last_sale"] + 1)  # Normalized 0-1: higher = more recent
    result["frequency_score"] = (result["total_orders"] / (result["days_since_first_sale"] + 1)) * 30  # Avg orders per 30 days
    result["monetary_score"] = result["total_revenue"] / (result["total_orders"] + 1)  # Avg revenue per order
    result[["delivered_orders", "canceled_known_orders"]] = result[["delivered_orders", "canceled_known_orders"]].fillna(0)

    # --------------------------
    # 7. Finalize output: reindex to match input entity_ids order
    # --------------------------
    result = result.reindex(entity_ids)
    return result