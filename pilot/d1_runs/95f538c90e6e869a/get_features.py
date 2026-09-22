def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Convert seed time to Timestamp
    seed_ts = pd.Timestamp(seed_time)
    
    # Step 1: Filter valid (non-canceled) orders and historical order items
    valid_orders = db["orders"][~db["orders"]["order_status"].isin(["canceled", "unavailable"])]
    valid_order_ids = valid_orders["order_id"].unique()
    oi_hist = db["order_items"][
        db["order_items"]["order_id"].isin(valid_order_ids) & 
        (db["order_items"]["ts"] <= seed_ts)
    ].copy()
    
    # Step 2: Generate windowed RFM features
    time_windows = [
        ("7d", 7),
        ("30d", 30),
        ("90d", 90),
        ("180d", 180),
        ("365d", 365),
        ("alltime", 100000)
    ]
    window_dfs = []
    
    for name, days in time_windows:
        cutoff = seed_ts - pd.Timedelta(days=days)
        oi_window = oi_hist[oi_hist["ts"] >= cutoff].copy()
        
        if oi_window.empty:
            # Create empty DataFrame with expected columns
            cols = [
                "product_id",
                f"{name}_order_count",
                f"{name}_unit_count",
                f"{name}_total_revenue",
                f"{name}_avg_price",
                f"{name}_avg_freight",
                f"{name}_max_price",
                f"{name}_min_price",
                f"{name}_avg_order_size"
            ]
            window_df = pd.DataFrame(columns=cols)
        else:
            # Calculate total item value (price + freight)
            oi_window["total_item_value"] = oi_window["price"] + oi_window["freight_value"]
            # Aggregate per product
            agg_df = oi_window.groupby("product_id").agg(
                order_count=("order_id", "nunique"),
                unit_count=("order_item_id", "count"),
                avg_price=("price", "mean"),
                max_price=("price", "max"),
                min_price=("price", "min"),
                avg_freight=("freight_value", "mean"),
                total_revenue=("total_item_value", "sum")
            ).reset_index()
            # Calculate average units per order
            agg_df["avg_order_size"] = agg_df["unit_count"] / agg_df["order_count"].replace(0, np.nan)
            # Rename columns with window prefix
            rename_map = {
                "order_count": f"{name}_order_count",
                "unit_count": f"{name}_unit_count",
                "avg_price": f"{name}_avg_price",
                "max_price": f"{name}_max_price",
                "min_price": f"{name}_min_price",
                "avg_freight": f"{name}_avg_freight",
                "total_revenue": f"{name}_total_revenue",
                "avg_order_size": f"{name}_avg_order_size"
            }
            window_df = agg_df.rename(columns=rename_map)
        window_dfs.append(window_df)
    
    # Merge all windowed features into one DataFrame
    if not window_dfs:
        rfm_features = pd.DataFrame(columns=["product_id"])
    else:
        rfm_features = window_dfs[0]
        for df in window_dfs[1:]:
            rfm_features = pd.merge(rfm_features, df, on="product_id", how="outer")
        rfm_features = rfm_features.fillna(0)
    
    # Add trend features (recent vs older window ratios)
    rfm_features["trend_30d_90d_orders"] = rfm_features["30d_order_count"] / rfm_features["90d_order_count"].replace(0, np.nan)
    rfm_features["trend_7d_30d_orders"] = rfm_features["7d_order_count"] / rfm_features["30d_order_count"].replace(0, np.nan)
    rfm_features["trend_30d_90d_revenue"] = rfm_features["30d_total_revenue"] / rfm_features["90d_total_revenue"].replace(0, np.nan)
    rfm_features["trend_7d_30d_revenue"] = rfm_features["7d_total_revenue"] / rfm_features["30d_total_revenue"].replace(0, np.nan)
    # Fill NaNs and infinities
    rfm_features = rfm_features.fillna(0).replace([np.inf, -np.inf], 1)
    
    # Create base DataFrame with all input product IDs
    base = pd.DataFrame(index=entity_ids)
    base.index.name = "product_id"
    base_reset = base.reset_index()
    
    # Step 3: Order delivery features
    # Use only orders whose delivery already happened by seed_time:
    # order_delivered_customer_date is filled after the delivery, so values
    # later than seed_time are future information.
    order_product_link = oi_hist[["order_id", "product_id"]].drop_duplicates()
    order_info = db["orders"][[
        "order_id", "order_purchase_timestamp",
        "order_delivered_customer_date", "order_estimated_delivery_date"
    ]]
    product_order_info = pd.merge(order_product_link, order_info, on="order_id", how="left")
    delivered_mask = (
        product_order_info["order_delivered_customer_date"].notna()
        & (product_order_info["order_delivered_customer_date"] <= seed_ts)
    )
    product_order_info.loc[~delivered_mask, "order_delivered_customer_date"] = pd.NaT
    # Calculate delivery metrics (NaN for orders not yet delivered)
    product_order_info["delivery_time"] = (
        product_order_info["order_delivered_customer_date"] - product_order_info["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 3600)
    product_order_info["is_late"] = np.where(
        delivered_mask,
        (
            product_order_info["order_delivered_customer_date"]
            > product_order_info["order_estimated_delivery_date"]
        ).astype(float),
        np.nan,
    )
    # Aggregate per product
    order_features = product_order_info.groupby("product_id").agg(
        avg_delivery_time=("delivery_time", "mean"),
        std_delivery_time=("delivery_time", "std"),
        late_delivery_rate=("is_late", "mean"),
        last_order_date=("order_purchase_timestamp", "max")
    ).reset_index()
    # Calculate recency (days since last order)
    order_features["recency_days"] = (
        seed_ts - order_features["last_order_date"]
    ).dt.total_seconds() / (24 * 3600)
    order_features = order_features.drop(columns=["last_order_date"])
    # Add has_delivered_orders flag
    order_features["has_delivered_orders"] = (~order_features["avg_delivery_time"].isna()).astype(int)
    # Fill NaNs
    order_features = order_features.fillna({
        "avg_delivery_time": 0,
        "std_delivery_time": 0,
        "late_delivery_rate": 0,
        "recency_days": 9999
    })
    
    # Step 4: Payment features
    # Aggregate payments per order first
    order_payments = db["payments"].groupby("order_id").agg(
        total_order_payment=("payment_value", "sum"),
        avg_payment_installments=("payment_installments", "mean"),
        num_payment_methods=("payment_sequential", "nunique")
    ).reset_index()
    # Link to products
    product_payment_info = pd.merge(order_product_link, order_payments, on="order_id", how="left")
    # Aggregate per product
    payment_features = product_payment_info.groupby("product_id").agg(
        avg_total_order_value=("total_order_payment", "mean"),
        std_total_order_value=("total_order_payment", "std"),
        avg_installments=("avg_payment_installments", "mean"),
        avg_num_payment_methods=("num_payment_methods", "mean")
    ).reset_index()
    # Add has_payments flag
    payment_features["has_payments"] = (~payment_features["avg_total_order_value"].isna()).astype(int)
    # Fill NaNs
    payment_features = payment_features.fillna(0)
    
    # Step 5: Review features
    # Reviews are written after the purchase; use only reviews already
    # created by seed_time (review_creation_date is the creation moment).
    order_reviews = db["reviews"][["order_id", "review_score", "review_creation_date"]]
    order_reviews = order_reviews[
        order_reviews["review_creation_date"].notna()
        & (order_reviews["review_creation_date"] <= seed_ts)
    ]
    product_review_info = pd.merge(order_product_link, order_reviews, on="order_id", how="left")
    # Create binary star columns for share calculations
    product_review_info["is_five_star"] = (product_review_info["review_score"] == 5).astype(float)
    product_review_info["is_one_star"] = (product_review_info["review_score"] == 1).astype(float)
    # Aggregate per product
    review_features = product_review_info.groupby("product_id").agg(
        avg_review_score=("review_score", "mean"),
        std_review_score=("review_score", "std"),
        num_reviews=("review_score", "count"),
        last_review_date=("review_creation_date", "max"),
        five_star_share=("is_five_star", "mean"),
        one_star_share=("is_one_star", "mean")
    ).reset_index()
    # Calculate review recency
    review_features["review_recency_days"] = (
        seed_ts - review_features["last_review_date"]
    ).dt.total_seconds() / (24 * 3600)
    review_features = review_features.drop(columns=["last_review_date"])
    # Add has_reviews flag
    review_features["has_reviews"] = (~review_features["avg_review_score"].isna()).astype(int)
    # Fill NaNs
    review_features = review_features.fillna({
        "avg_review_score": 0,
        "std_review_score": 0,
        "num_reviews": 0,
        "five_star_share": 0,
        "one_star_share": 0,
        "review_recency_days": 9999
    })
    
    # Merge all feature groups into final DataFrame
    all_features = pd.merge(base_reset, rfm_features, on="product_id", how="left")
    all_features = pd.merge(all_features, order_features, on="product_id", how="left")
    all_features = pd.merge(all_features, payment_features, on="product_id", how="left")
    all_features = pd.merge(all_features, review_features, on="product_id", how="left")
    
    # Set product_id as index
    all_features = all_features.set_index("product_id")
    
    # Create fill dictionary for remaining NaNs (products with no history)
    fill_dict = {}
    # RFM and trend features
    for col in rfm_features.columns.drop("product_id"):
        fill_dict[col] = 0
    # Order features
    fill_dict["recency_days"] = 9999
    fill_dict["avg_delivery_time"] = 0
    fill_dict["std_delivery_time"] = 0
    fill_dict["late_delivery_rate"] = 0
    fill_dict["has_delivered_orders"] = 0
    # Payment features
    fill_dict["avg_total_order_value"] = 0
    fill_dict["std_total_order_value"] = 0
    fill_dict["avg_installments"] = 0
    fill_dict["avg_num_payment_methods"] = 0
    fill_dict["has_payments"] = 0
    # Review features
    fill_dict["avg_review_score"] = 0
    fill_dict["std_review_score"] = 0
    fill_dict["num_reviews"] = 0
    fill_dict["five_star_share"] = 0
    fill_dict["one_star_share"] = 0
    fill_dict["review_recency_days"] = 9999
    fill_dict["has_reviews"] = 0
    
    # Fill all remaining NaNs
    all_features = all_features.fillna(fill_dict)
    
    # Reindex to match input entity_ids exactly (ensures correct order and all entities present)
    all_features = all_features.reindex(entity_ids)
    
    return all_features