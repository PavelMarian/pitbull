import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Unpack database tables
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]

    # ----------------------
    # Preprocess Payments: aggregate per order
    # ----------------------
    # Aggregate payment numerical metrics per order
    payments_agg = payments.groupby("order_id").agg(
        total_order_payment=("payment_value", "sum"),
        avg_payment_installments=("payment_installments", "mean"),
        num_payment_sequentials=("payment_sequential", "count")
    ).reset_index()

    # Create payment type percentage features
    payment_type_dummies = pd.get_dummies(payments[["order_id", "payment_type"]], columns=["payment_type"])
    payment_type_dummies = payment_type_dummies.groupby("order_id").sum().reset_index()

    # Merge payment metrics
    payments_agg = payments_agg.merge(payment_type_dummies, on="order_id", how="left")
    # Normalize payment type counts to percentages
    payment_type_cols = [c for c in payment_type_dummies.columns if c.startswith("payment_type_")]
    for col in payment_type_cols:
        payments_agg[f"{col}_pct"] = payments_agg[col] / payments_agg["num_payment_sequentials"]
    # Drop raw count columns
    payments_agg = payments_agg.drop(columns=payment_type_cols + ["num_payment_sequentials"])

    # ----------------------
    # Preprocess Reviews: aggregate per order
    # ----------------------
    # Point-in-time filter: a review exists at seed_time only if it was
    # created on or before seed_time. Reviews written later must not be used.
    reviews_known = reviews[reviews["review_creation_date"] <= seed_time].copy()

    # Merge with order timestamps to compute review delay
    reviews_with_order = reviews_known.merge(
        orders[["order_id", "order_purchase_timestamp"]],
        on="order_id",
        how="left"
    )
    reviews_with_order["days_to_review_creation"] = (
        reviews_with_order["review_creation_date"] - reviews_with_order["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 3600)  # Convert to days

    # Aggregate review metrics per order
    reviews_agg = reviews_with_order.groupby("order_id").agg(
        avg_order_review_score=("review_score", "mean"),
        total_reviews_per_order=("review_id", "count"),
        pct_reviews_with_title=("review_comment_title", lambda x: x.notna().mean()),
        pct_reviews_with_message=("review_comment_message", lambda x: x.notna().mean()),
        avg_days_to_review=("days_to_review_creation", "mean")
    ).reset_index()

    # ----------------------
    # Create base dataset with all order-item details
    # ----------------------
    # Merge order-items with order metadata
    order_items_orders = order_items.merge(
        orders[["order_id", "customer_id", "order_status",
                "order_purchase_timestamp", "order_delivered_carrier_date",
                "order_delivered_customer_date", "order_estimated_delivery_date"]],
        on="order_id",
        how="inner"
    )

    # Filter to orders strictly before seed time
    order_items_orders = order_items_orders[order_items_orders["order_purchase_timestamp"] < seed_time].copy()

    # Merge with payment and review data
    order_items_full = order_items_orders.merge(payments_agg, on="order_id", how="left")
    order_items_full = order_items_full.merge(reviews_agg, on="order_id", how="left")

    # ----------------------
    # Engineer derived delivery and status features
    # ----------------------
    # Point-in-time masks: a delivery date is usable at seed_time only if it
    # is already filled and does not lie in the future relative to seed_time.
    carrier_known = (
        order_items_full["order_delivered_carrier_date"].notna()
        & (order_items_full["order_delivered_carrier_date"] <= seed_time)
    )
    customer_known = (
        order_items_full["order_delivered_customer_date"].notna()
        & (order_items_full["order_delivered_customer_date"] <= seed_time)
    )

    delta_purchase_to_carrier = (
        order_items_full["order_delivered_carrier_date"] - order_items_full["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 3600)
    delta_carrier_to_customer = (
        order_items_full["order_delivered_customer_date"] - order_items_full["order_delivered_carrier_date"]
    ).dt.total_seconds() / (24 * 3600)
    delta_delivery_delay = (
        order_items_full["order_delivered_customer_date"] - order_items_full["order_estimated_delivery_date"]
    ).dt.total_seconds() / (24 * 3600)

    order_items_full["days_to_carrier"] = np.where(carrier_known, delta_purchase_to_carrier, np.nan)
    order_items_full["days_carrier_to_customer"] = np.where(carrier_known & customer_known, delta_carrier_to_customer, np.nan)
    order_items_full["delivery_delay_days"] = np.where(customer_known, delta_delivery_delay, np.nan)
    order_items_full["is_late"] = np.where(customer_known, (delta_delivery_delay > 0).astype(float), np.nan)

    # Binary status flags
    order_items_full["is_canceled"] = (order_items_full["order_status"] == "canceled").astype(int)
    order_items_full["is_delivered"] = (order_items_full["order_status"] == "delivered").astype(int)

    # ----------------------
    # Define lookback windows
    # ----------------------
    lookback_windows = [
        ("7d", pd.Timedelta(days=7)),
        ("14d", pd.Timedelta(days=14)),
        ("30d", pd.Timedelta(days=30)),
        ("90d", pd.Timedelta(days=90)),
        ("180d", pd.Timedelta(days=180)),
        ("all", None)  # All historical data before seed_time
    ]
    window_names = [win[0] for win in lookback_windows]

    # ----------------------
    # Initialize output with full entity list
    # ----------------------
    base_features = pd.DataFrame(index=entity_ids)
    base_features.index.name = "product_id"

    # ----------------------
    # Compute features per lookback window
    # ----------------------
    for window_name, window_delta in lookback_windows:
        # Filter to current window
        if window_delta is not None:
            start_time = seed_time - window_delta
            window_data = order_items_full[
                (order_items_full["order_purchase_timestamp"] >= start_time) &
                (order_items_full["order_purchase_timestamp"] < seed_time)
            ].copy()
        else:
            window_data = order_items_full.copy()

        # Skip if no data in window (will be filled later)
        if window_data.empty:
            continue

        # ----------------------
        # Line-item level features (per product)
        # ----------------------
        line_item_feats = window_data.groupby("product_id").agg(
            total_units=("order_item_id", "count"),
            unique_orders=("order_id", "nunique"),
            unique_sellers=("seller_id", "nunique"),
            unique_customers=("customer_id", "nunique"),
            total_revenue=("price", "sum"),
            total_freight=("freight_value", "sum"),
            avg_unit_price=("price", "mean"),
            avg_unit_freight=("freight_value", "mean"),
            last_sale_time=("order_purchase_timestamp", "max"),
            first_sale_time=("order_purchase_timestamp", "min")
        ).reset_index()

        # Time-based features
        line_item_feats["days_since_last_sale"] = (
            seed_time - line_item_feats["last_sale_time"]
        ).dt.total_seconds() / (24 * 3600)
        line_item_feats["product_age_days"] = (
            seed_time - line_item_feats["first_sale_time"]
        ).dt.total_seconds() / (24 * 3600)
        line_item_feats = line_item_feats.drop(columns=["last_sale_time", "first_sale_time"])

        # ----------------------
        # Order-level features (per product, unique orders)
        # ----------------------
        unique_product_orders = window_data[
            ["product_id", "order_id", "is_canceled", "is_delivered", "is_late",
             "days_to_carrier", "days_carrier_to_customer", "delivery_delay_days",
             "total_order_payment", "avg_payment_installments",
             "payment_type_credit_card_pct", "payment_type_boleto_pct",
             "payment_type_voucher_pct", "payment_type_debit_card_pct",
             "avg_order_review_score", "total_reviews_per_order",
             "pct_reviews_with_title", "pct_reviews_with_message", "avg_days_to_review"]
        ].drop_duplicates(subset=["product_id", "order_id"])

        order_level_feats = unique_product_orders.groupby("product_id").agg(
            pct_canceled_orders=("is_canceled", "mean"),
            pct_delivered_orders=("is_delivered", "mean"),
            pct_late_deliveries=("is_late", "mean"),
            avg_days_to_carrier=("days_to_carrier", "mean"),
            avg_days_carrier_to_customer=("days_carrier_to_customer", "mean"),
            avg_delivery_delay=("delivery_delay_days", "mean"),
            avg_total_order_payment=("total_order_payment", "mean"),
            avg_payment_installments=("avg_payment_installments", "mean"),
            avg_pct_credit_card=("payment_type_credit_card_pct", "mean"),
            avg_pct_boleto=("payment_type_boleto_pct", "mean"),
            avg_pct_voucher=("payment_type_voucher_pct", "mean"),
            avg_pct_debit_card=("payment_type_debit_card_pct", "mean"),
            avg_review_score=("avg_order_review_score", "mean"),
            total_reviews=("total_reviews_per_order", "sum"),
            avg_pct_reviews_with_titles=("pct_reviews_with_title", "mean"),
            avg_pct_reviews_with_messages=("pct_reviews_with_message", "mean"),
            avg_days_to_review=("avg_days_to_review", "mean")
        ).reset_index()

        # ----------------------
        # Merge window features
        # ----------------------
        window_features = line_item_feats.merge(order_level_feats, on="product_id", how="outer")
        # Rename columns with window suffix
        rename_map = {col: f"{col}_{window_name}" for col in window_features.columns if col != "product_id"}
        window_features = window_features.rename(columns=rename_map)
        window_features = window_features.set_index("product_id")

        # Join to base features
        base_features = base_features.join(window_features, how="left")

    # ----------------------
    # Create indicator features for data availability
    # ----------------------
    for win in window_names:
        sales_col = f"total_units_{win}"
        review_col = f"total_reviews_{win}"
        base_features[f"has_sales_{win}"] = (base_features[sales_col] > 0).astype(int) if sales_col in base_features.columns else 0
        base_features[f"has_reviews_{win}"] = (base_features[review_col] > 0).astype(int) if review_col in base_features.columns else 0

    # ----------------------
    # Handle missing values
    # ----------------------
    # Indicator columns: fill NaNs with 0
    indicator_cols = [c for c in base_features.columns if c.startswith("has_")]
    base_features[indicator_cols] = base_features[indicator_cols].fillna(0).astype(int)

    # Count/sum columns: fill NaNs with 0
    count_sum_keys = ["total_units", "unique_orders", "unique_sellers", "unique_customers",
                      "total_revenue", "total_freight", "total_reviews"]
    count_sum_cols = [c for c in base_features.columns if any(k in c for k in count_sum_keys)]
    base_features[count_sum_cols] = base_features[count_sum_cols].fillna(0)

    # Days since last sale: fill with large value (1e5 days = ~273 years)
    days_since_cols = [c for c in base_features.columns if "days_since_last_sale" in c]
    base_features[days_since_cols] = base_features[days_since_cols].fillna(1e5)

    # Product age: fill with 0 (no prior sales)
    age_cols = [c for c in base_features.columns if "product_age_days" in c]
    base_features[age_cols] = base_features[age_cols].fillna(0)

    # All remaining columns: fill with 0
    remaining_cols = [c for c in base_features.columns
                      if c not in count_sum_cols + days_since_cols + age_cols + indicator_cols]
    base_features[remaining_cols] = base_features[remaining_cols].fillna(0)

    # ----------------------
    # Finalize output to match required entity list
    # ----------------------
    base_features = base_features.reindex(entity_ids)
    base_features.index.name = "product_id"

    return base_features
