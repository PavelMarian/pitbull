import pandas as pd
import numpy as np
def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Create base dataframe with input entity IDs as index
    base = pd.DataFrame(index=entity_ids)
    base.index.name = "product_id"
    # Define time windows for historical aggregation
    win_30 = seed_time - pd.Timedelta(days=30)
    win_90 = seed_time - pd.Timedelta(days=90)
    win_180 = seed_time - pd.Timedelta(days=180)
    # --------------------------
    # 1. Order Items Aggregations
    # --------------------------
    hist_items = db["order_items"][db["order_items"]["ts"] < seed_time].copy()
    order_agg = pd.DataFrame()
    if not hist_items.empty:
        # Precompute window flags
        hist_items = hist_items.assign(
            in_30d=(hist_items["ts"] >= win_30).astype(int),
            in_90d=(hist_items["ts"] >= win_90).astype(int),
            in_180d=(hist_items["ts"] >= win_180).astype(int)
        )
        # Group by product_id
        order_agg = hist_items.groupby("product_id").agg(
            total_units_alltime=("order_item_id", "count"),
            total_units_180d=("in_180d", "sum"),
            total_units_90d=("in_90d", "sum"),
            total_units_30d=("in_30d", "sum"),
            total_revenue_alltime=("price", "sum"),
            total_revenue_90d=("price", lambda x: (x * hist_items.loc[x.index, "in_90d"]).sum()),
            avg_price_alltime=("price", "mean"),
            avg_freight_90d=("freight_value", lambda x: x[hist_items.loc[x.index, "in_90d"] == 1].mean()),
            unique_sellers_alltime=("seller_id", "nunique"),
            last_sale_ts=("ts", "max")
        ).reset_index()
        # Derive days since last sale
        order_agg["days_since_last_sale"] = (seed_time - order_agg["last_sale_ts"]).dt.days
        order_agg = order_agg.drop(columns=["last_sale_ts"])
        order_agg = order_agg.set_index("product_id")
    # --------------------------
    # 2. Product-Order Links (shared for downstream aggregations)
    # --------------------------
    product_order_links = pd.DataFrame()
    if not hist_items.empty:
        product_order_links = hist_items[["product_id", "order_id"]].drop_duplicates()
    # --------------------------
    # 3. Review Aggregations
    # --------------------------
    review_agg = pd.DataFrame()
    if not product_order_links.empty:
        hist_reviews = db["reviews"][db["reviews"]["review_creation_date"] < seed_time].copy()
        if not hist_reviews.empty:
            review_product = hist_reviews.merge(product_order_links, on="order_id", how="inner")
            if not review_product.empty:
                review_agg = review_product.groupby("product_id").agg(
                    avg_review_score=("review_score", "mean"),
                    total_reviews=("review_id", "count"),
                    last_review_ts=("review_creation_date", "max")
                ).reset_index()
                review_agg["days_since_last_review"] = (seed_time - review_agg["last_review_ts"]).dt.days
                review_agg = review_agg.drop(columns=["last_review_ts"])
                review_agg = review_agg.set_index("product_id")
    # --------------------------
    # 4. Order Status (Cancellation) Aggregations
    # --------------------------
    order_status_agg = pd.DataFrame()
    if not product_order_links.empty:
        order_status = db["orders"][["order_id", "order_status"]].copy()
        product_order_status = product_order_links.merge(order_status, on="order_id", how="left")
        order_status_agg = product_order_status.groupby("product_id").agg(
            total_orders_alltime=("order_id", "nunique"),
            canceled_orders=("order_status", lambda x: (x == "canceled").sum())
        ).reset_index()
        order_status_agg["canceled_rate"] = order_status_agg["canceled_orders"] / order_status_agg["total_orders_alltime"].replace(0, np.nan)
        order_status_agg = order_status_agg.drop(columns=["canceled_orders"])
        order_status_agg = order_status_agg.set_index("product_id")
    # --------------------------
    # 5. Delivery Performance Aggregations
    # --------------------------
    delivery_agg = pd.DataFrame()
    if not product_order_links.empty:
        delivery_data = db["orders"][["order_id", "order_delivered_customer_date", "order_estimated_delivery_date"]].copy()
        product_delivery = product_order_links.merge(delivery_data, on="order_id", how="left")
        # Only orders actually delivered on or before seed_time are usable:
        # the delivered date is filled in after purchase, so a later date is
        # information from the future at seed_time.
        delivered_mask = product_delivery["order_delivered_customer_date"].notna() & (
            product_delivery["order_delivered_customer_date"] <= seed_time
        )
        product_delivery = product_delivery[delivered_mask].copy()
        # Calculate delivery delay (days late/early)
        def calc_delay(row):
            if pd.notna(row["order_delivered_customer_date"]) and pd.notna(row["order_estimated_delivery_date"]):
                return (row["order_delivered_customer_date"] - row["order_estimated_delivery_date"]).days
            return np.nan
        product_delivery["delivery_delay_days"] = product_delivery.apply(calc_delay, axis=1)
        delivery_agg = product_delivery.groupby("product_id").agg(
            avg_delivery_delay=("delivery_delay_days", "mean"),
            total_delivered_orders=("delivery_delay_days", "count")
        ).reset_index()
        delivery_agg = delivery_agg.set_index("product_id")
    # --------------------------
    # 6. Payment Aggregations
    # --------------------------
    pay_agg = pd.DataFrame()
    if not product_order_links.empty:
        hist_payments = db["payments"][db["payments"]["ts"] < seed_time].copy()
        if not hist_payments.empty:
            # Aggregate per order first
            order_pay_agg = hist_payments.groupby("order_id").agg(
                total_order_payment=("payment_value", "sum"),
                avg_installments=("payment_installments", "mean")
            ).reset_index()
            # Link to products
            product_pay = product_order_links.merge(order_pay_agg, on="order_id", how="left")
            pay_agg = product_pay.groupby("product_id").agg(
                avg_total_order_payment=("total_order_payment", "mean"),
                avg_payment_installments=("avg_installments", "mean"),
                total_orders_with_payment=("total_order_payment", "count")
            ).reset_index()
            pay_agg = pay_agg.set_index("product_id")
    # --------------------------
    # 7. Sales Activity Aggregations
    # --------------------------
    activity_agg = pd.DataFrame()
    if not hist_items.empty:
        sale_dates = hist_items.groupby("product_id")["ts"].agg(["min", "max"]).reset_index()
        sale_dates.columns = ["product_id", "first_sale_ts", "last_sale_ts"]
        sale_dates["days_active"] = (sale_dates["last_sale_ts"] - sale_dates["first_sale_ts"]).dt.days
        total_units = hist_items.groupby("product_id").size().reset_index(name="total_units")
        activity_agg = sale_dates.merge(total_units, on="product_id", how="inner")
        activity_agg["sales_per_week"] = activity_agg.apply(
            lambda r: r["total_units"] / (r["days_active"] / 7) if r["days_active"] > 0 else 0,
            axis=1
        )
        activity_agg = activity_agg.drop(columns=["first_sale_ts", "last_sale_ts", "total_units"])
        activity_agg = activity_agg.set_index("product_id")
    # --------------------------
    # Merge all features into base dataframe
    # --------------------------
    result = base.copy()
    for agg_df in [order_agg, review_agg, order_status_agg, delivery_agg, pay_agg, activity_agg]:
        if not agg_df.empty:
            result = result.join(agg_df, how="left")
    # --------------------------
    # Fill missing values with consistent defaults
    # --------------------------
    fill_values = {
        # Count columns (fill 0 for no data)
        "total_units_alltime": 0,
        "total_units_180d": 0,
        "total_units_90d": 0,
        "total_units_30d": 0,
        "total_revenue_alltime": 0.0,
        "total_revenue_90d": 0.0,
        "unique_sellers_alltime": 0,
        "total_reviews": 0,
        "total_orders_alltime": 0,
        "total_delivered_orders": 0,
        "total_orders_with_payment": 0,
        "days_active": 0,
        # Average/rate columns (fill 0 for no data, acting as "no signal" value)
        "avg_price_alltime": 0.0,
        "avg_freight_90d": 0.0,
        "canceled_rate": 0.0,
        "avg_delivery_delay": 0.0,
        "avg_total_order_payment": 0.0,
        "avg_payment_installments": 0.0,
        "sales_per_week": 0.0,
        "avg_review_score": 0.0,
        # Days since last activity (fill large value to indicate no prior activity)
        "days_since_last_sale": 1000,
        "days_since_last_review": 1000
    }
    # Apply fill values, then final safety fill for any remaining NaNs
    result = result.fillna(value=fill_values)
    result = result.fillna(0)
    # --------------------------
    # Enforce consistent column types
    # --------------------------
    int_cols = [
        "total_units_alltime", "total_units_180d", "total_units_90d",
        "total_units_30d", "unique_sellers_alltime", "total_reviews",
        "total_orders_alltime", "total_delivered_orders", "total_orders_with_payment",
        "days_active", "days_since_last_sale", "days_since_last_review"
    ]
    for col in int_cols:
        if col in result.columns:
            result[col] = result[col].astype(np.int64)
    # Cast all other columns to float64
    for col in result.columns:
        if col not in int_cols:
            result[col] = result[col].astype(np.float64)
    # Ensure final index matches input entity IDs exactly (handles duplicates, ordering)
    result = result.reindex(entity_ids)
    return result