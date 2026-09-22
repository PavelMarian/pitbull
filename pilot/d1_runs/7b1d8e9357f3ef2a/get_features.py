import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time: pd.Timestamp) -> pd.DataFrame:
    # Extract tables from database
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]
    
    # Step 1: Filter all data to strictly historical (no leakage beyond seed_time)
    # Filter order_items first to get valid order_ids (purchased on/before seed_time)
    filtered_order_items = order_items[order_items["ts"] <= seed_time].copy()
    valid_order_ids = filtered_order_items["order_id"].unique()
    
    # Filter other tables to valid orders AND historical timestamps
    filtered_orders = orders[orders["order_id"].isin(valid_order_ids)].copy()
    filtered_reviews = reviews[
        reviews["order_id"].isin(valid_order_ids) & 
        (reviews["review_creation_date"] <= seed_time)
    ].copy()
    filtered_payments = payments[
        payments["order_id"].isin(valid_order_ids) & 
        (payments["ts"] <= seed_time)
    ].copy()
    
    # Initialize output DataFrame indexed by input entity_ids
    output = pd.DataFrame(index=pd.Index(entity_ids, name="product_id"))
    
    # -------------------------------------------------------------------------
    # 1. Core Order Items Features (sales volume, pricing, timing)
    # -------------------------------------------------------------------------
    # Aggregate core stats from order_items
    order_items_agg = filtered_order_items.groupby("product_id").agg(
        total_units_sold=("order_item_id", "count"),
        total_orders=("order_id", "nunique"),
        total_sellers=("seller_id", "nunique"),
        total_revenue=("price", "sum"),
        total_freight=("freight_value", "sum"),
        avg_price=("price", "mean"),
        avg_freight=("freight_value", "mean"),
        first_sale_ts=("ts", "min"),
        last_sale_ts=("ts", "max")
    ).reset_index()
    
    # Compute time-based features
    order_items_agg["days_since_first_sale"] = (
        (seed_time - order_items_agg["first_sale_ts"]).dt.total_seconds() / (24 * 3600)
    )
    order_items_agg["days_since_last_sale"] = (
        (seed_time - order_items_agg["last_sale_ts"]).dt.total_seconds() / (24 * 3600)
    )
    order_items_agg["sales_frequency"] = (
        order_items_agg["total_units_sold"] / order_items_agg["days_since_first_sale"].replace(0, np.nan)
    )
    
    # Helper to get rolling window sales counts
    def get_window_sales(window_days):
        cutoff = seed_time - pd.Timedelta(days=window_days)
        return filtered_order_items[
            filtered_order_items["ts"] >= cutoff
        ].groupby("product_id")["order_item_id"].count().rename(f"units_sold_last_{window_days}d")
    
    # Compute rolling window features
    for window in [30, 90, 180, 365]:
        order_items_agg = order_items_agg.merge(get_window_sales(window), on="product_id", how="left")
    
    # Compute average days between consecutive sales
    def compute_avg_sale_interval(group):
        if len(group) < 2:
            return np.nan
        sorted_ts = group["ts"].sort_values()
        intervals = sorted_ts.diff().dropna()
        return intervals.dt.total_seconds().mean() / (24 * 3600)
    
    avg_sale_intervals = filtered_order_items.groupby("product_id").apply(
        compute_avg_sale_interval
    ).rename("avg_days_between_sales").reset_index()
    order_items_agg = order_items_agg.merge(avg_sale_intervals, on="product_id", how="left")
    
    # Merge order_items features to output
    output = output.join(order_items_agg.set_index("product_id"), how="left")
    
    # Fill NaN values for products with no sales history
    fill_core = {
        "total_units_sold": 0, "total_orders": 0, "total_sellers": 0,
        "total_revenue": 0.0, "total_freight": 0.0, "avg_price": 0.0,
        "avg_freight": 0.0, "days_since_first_sale": 730.0,
        "days_since_last_sale": 730.0, "sales_frequency": 0.0,
        "units_sold_last_30d": 0, "units_sold_last_90d": 0,
        "units_sold_last_180d": 0, "units_sold_last_365d": 0,
        "avg_days_between_sales": 730.0
    }
    output.fillna(value=fill_core, inplace=True)
    output.drop(columns=["first_sale_ts", "last_sale_ts"], inplace=True, errors="ignore")
    
    # -------------------------------------------------------------------------
    # 2. Order Status & Delivery Features
    # -------------------------------------------------------------------------
    # Merge order items with orders data
    order_items_orders = filtered_order_items.merge(filtered_orders, on="order_id", how="left")
    
    # Order status counts and ratios
    status_counts = order_items_orders.groupby(["product_id", "order_status"])["order_id"].nunique().unstack(fill_value=0)
    status_ratios = status_counts.div(status_counts.sum(axis=1), axis=0).fillna(0)
    status_counts.columns = [f"order_count_{s}" for s in status_counts.columns]
    status_ratios.columns = [f"order_ratio_{s}" for s in status_ratios.columns]
    order_status_features = pd.concat([status_counts, status_ratios], axis=1).reset_index()
    
    # Delivery performance for orders that were already delivered by seed_time.
    # A delivery date later than seed_time does not exist yet at prediction time.
    delivered_mask = (
        filtered_orders["order_delivered_customer_date"].notna()
        & (filtered_orders["order_delivered_customer_date"] <= seed_time)
    )
    delivered_orders = filtered_orders[delivered_mask][
        ["order_id", "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
    ].copy()
    delivered_orders["delivery_days"] = (
        (delivered_orders["order_delivered_customer_date"] - delivered_orders["order_purchase_timestamp"]).dt.total_seconds() / (24 * 3600)
    )
    delivered_orders["is_late"] = (
        delivered_orders["order_delivered_customer_date"] > delivered_orders["order_estimated_delivery_date"]
    )
    delivered = filtered_order_items[["order_id", "product_id"]].merge(
        delivered_orders.drop(columns=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]),
        on="order_id", how="inner"
    )
    if len(delivered) > 0:
        delivery_agg = delivered.groupby("product_id").agg(
            avg_delivery_days=("delivery_days", "mean"),
            pct_late_deliveries=("is_late", "mean")
        ).reset_index()
    else:
        delivery_agg = pd.DataFrame(columns=["product_id", "avg_delivery_days", "pct_late_deliveries"])
    
    order_features = order_status_features.merge(delivery_agg, on="product_id", how="left")
    output = output.join(order_features.set_index("product_id"), how="left")
    
    # Global defaults for delivery features
    global_delivery_days = 14.0
    global_late_pct = 0.1
    if len(delivered_orders) > 0:
        global_delivery_days = delivered_orders["delivery_days"].mean()
        global_late_pct = delivered_orders["is_late"].mean()
    
    fill_order = {}
    for col in output.columns:
        if col.startswith("order_count_") or col.startswith("order_ratio_"):
            fill_order[col] = 0
    fill_order["avg_delivery_days"] = global_delivery_days if not np.isnan(global_delivery_days) else 14.0
    fill_order["pct_late_deliveries"] = global_late_pct if not np.isnan(global_late_pct) else 0.1
    output.fillna(value=fill_order, inplace=True)
    
    # -------------------------------------------------------------------------
    # 3. Review Features
    # -------------------------------------------------------------------------
    # Merge order items with reviews
    order_items_reviews = filtered_order_items.merge(filtered_reviews, on="order_id", how="left")
    
    # Core review aggregates
    review_agg = order_items_reviews.groupby("product_id").agg(
        total_reviews=("review_id", "count"),
        avg_review_score=("review_score", "mean"),
        min_review_score=("review_score", "min"),
        max_review_score=("review_score", "max")
    ).reset_index()
    
    # 5-star and 1-star review ratios
    five_star = order_items_reviews[order_items_reviews["review_score"] == 5].groupby("product_id")["review_id"].count().rename("five_star")
    one_star = order_items_reviews[order_items_reviews["review_score"] == 1].groupby("product_id")["review_id"].count().rename("one_star")
    review_agg = review_agg.merge(five_star, on="product_id", how="left").merge(one_star, on="product_id", how="left")
    review_agg["five_star_ratio"] = review_agg["five_star"] / review_agg["total_reviews"].replace(0, np.nan)
    review_agg["one_star_ratio"] = review_agg["one_star"] / review_agg["total_reviews"].replace(0, np.nan)
    review_agg.drop(columns=["five_star", "one_star"], inplace=True)
    
    output = output.join(review_agg.set_index("product_id"), how="left")
    
    # Global defaults for review features
    global_avg_rev = 3.0
    global_min_rev = 1.0
    global_max_rev = 5.0
    if len(filtered_reviews) > 0:
        global_avg_rev = filtered_reviews["review_score"].mean()
        global_min_rev = filtered_reviews["review_score"].min()
        global_max_rev = filtered_reviews["review_score"].max()
    
    fill_review = {
        "total_reviews": 0, "avg_review_score": global_avg_rev,
        "min_review_score": global_min_rev, "max_review_score": global_max_rev,
        "five_star_ratio": 0.0, "one_star_ratio": 0.0
    }
    output.fillna(value=fill_review, inplace=True)
    
    # -------------------------------------------------------------------------
    # 4. Payment Features
    # -------------------------------------------------------------------------
    # Aggregate payments per order first
    order_payments = filtered_payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        avg_installments=("payment_installments", "mean"),
        num_payments=("payment_sequential", "count")
    ).reset_index()
    
    # Pivot payment type counts per order
    if len(filtered_payments) > 0:
        payment_types = filtered_payments.pivot_table(
            index="order_id", columns="payment_type", values="payment_sequential",
            aggfunc="count", fill_value=0
        )
        payment_types.columns = [f"pay_type_{c}" for c in payment_types.columns]
        order_payments = order_payments.merge(payment_types, on="order_id", how="left").fillna(0)
    else:
        # No payments: add dummy columns to avoid errors
        order_payments["pay_type_unknown"] = 0
    
    # Merge with order items, then aggregate per product
    order_items_pay = filtered_order_items.merge(order_payments, on="order_id", how="left")
    pay_agg = order_items_pay.groupby("product_id").agg(
        avg_total_payment=("total_payment_value", "mean"),
        avg_installments=("avg_installments", "mean"),
        avg_payments_per_order=("num_payments", "mean")
    ).reset_index()
    
    # Compute payment type ratios per product
    pay_type_cols = [c for c in order_payments.columns if c.startswith("pay_type_")]
    pay_type_totals = order_items_pay.groupby("product_id")[pay_type_cols].sum().reset_index()
    total_pay_counts = pay_type_totals[pay_type_cols].sum(axis=1)
    for col in pay_type_cols:
        pay_type_totals[f"{col}_ratio"] = pay_type_totals[col] / total_pay_counts.replace(0, np.nan)
    pay_type_totals.drop(columns=pay_type_cols, inplace=True)
    pay_agg = pay_agg.merge(pay_type_totals, on="product_id", how="left")
    
    output = output.join(pay_agg.set_index("product_id"), how="left")
    
    # Global defaults for payment features
    global_avg_pay = 0.0
    global_avg_inst = 1.0
    global_avg_pay_count = 1.0
    if len(order_payments) > 0:
        global_avg_pay = order_payments["total_payment_value"].mean()
        global_avg_inst = order_payments["avg_installments"].mean()
        global_avg_pay_count = order_payments["num_payments"].mean()
    
    fill_pay = {
        "avg_total_payment": global_avg_pay,
        "avg_installments": global_avg_inst,
        "avg_payments_per_order": global_avg_pay_count
    }
    for col in output.columns:
        if col.startswith("pay_type_") and col.endswith("_ratio"):
            fill_pay[col] = 0.0
    output.fillna(value=fill_pay, inplace=True)
    
    # -------------------------------------------------------------------------
    # Final cleanup: ensure numeric types, reindex to input entity_ids
    # -------------------------------------------------------------------------
    # Convert all integer columns to float64 for consistent dtype handling
    for col in output.columns:
        if pd.api.types.is_integer_dtype(output[col]):
            output[col] = output[col].astype(np.float64)
    
    # Strictly reindex to input entity_ids (guarantees correct order and coverage)
    output = output.reindex(entity_ids)
    
    return output