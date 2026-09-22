import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Historical window: 1 year prior to seed time
    start_hist = seed_time - pd.Timedelta(days=365)

    # Filter order items to historical window (purchase moment ts < seed_time)
    oi_hist = db["order_items"].query(
        "ts >= @start_hist and ts < @seed_time"
    ).copy()

    # Merge with orders for delivery data
    oi_orders = oi_hist.merge(
        db["orders"][[
            "order_id",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]],
        on="order_id",
        how="left"
    )

    # Delivery info is known at seed_time only if the customer was already
    # delivered by then; otherwise treat as not-yet-delivered (extreme delay).
    delivered_mask = (
        oi_orders["order_delivered_customer_date"].notna()
        & (oi_orders["order_delivered_customer_date"] <= seed_time)
    )
    delivery_delay_days = (
        oi_orders["order_delivered_customer_date"]
        - oi_orders["order_estimated_delivery_date"]
    ).dt.total_seconds() / (24 * 3600)
    oi_orders["delivery_delay_days"] = delivery_delay_days.where(
        delivered_mask, 999.0
    )
    # Orders not yet delivered (or without estimate) count as extreme delay
    oi_orders["delivery_delay_days"] = oi_orders["delivery_delay_days"].fillna(999.0)
    oi_orders["frac_delivered"] = delivered_mask.astype(np.float64)

    # Merge with reviews written BEFORE seed_time only
    # (a review is created after the purchase, often after seed_time)
    reviews_hist = db["reviews"].query(
        "review_creation_date < @seed_time"
    )[["order_id", "review_score"]]
    oi_reviews = oi_orders.merge(
        reviews_hist,
        on="order_id",
        how="left"
    )
    # Fill missing reviews with neutral score
    oi_reviews["review_score"] = oi_reviews["review_score"].fillna(3.0)

    # Aggregate payment data per order (only payments known by seed_time)
    payments_hist = db["payments"].query("ts < @seed_time")
    payments_agg = payments_hist.groupby("order_id").agg(
        total_payment=("payment_value", "sum"),
        avg_installments=("payment_installments", "mean"),
        frac_credit_card=("payment_type", lambda x: (x == "credit_card").mean())
    ).reset_index()

    # Merge payment data into order item dataset
    oi_full = oi_reviews.merge(
        payments_agg,
        on="order_id",
        how="left"
    )

    # Aggregate features per product_id
    product_features = oi_full.groupby("product_id").agg(
        # Sales volume features
        total_units_sold=("order_item_id", "count"),
        total_revenue=("price", "sum"),
        total_freight=("freight_value", "sum"),
        # Price metrics
        avg_price=("price", "mean"),
        std_price=("price", "std"),
        # Delivery performance
        avg_delivery_delay=("delivery_delay_days", "mean"),
        max_delivery_delay=("delivery_delay_days", "max"),
        frac_delivered=("frac_delivered", "mean"),
        # Review quality
        avg_review_score=("review_score", "mean"),
        min_review_score=("review_score", "min"),
        frac_5star_reviews=("review_score", lambda x: (x == 5).mean()),
        frac_1star_reviews=("review_score", lambda x: (x == 1).mean()),
        # Payment behavior
        avg_order_total=("total_payment", "mean"),
        avg_payment_installments=("avg_installments", "mean"),
        frac_credit_card_payments=("frac_credit_card", "mean")
    ).reset_index()

    # Fill std_price for single-unit products (0 variance)
    product_features["std_price"] = product_features["std_price"].fillna(0.0)

    # Create base DataFrame for all requested entity IDs
    result = pd.DataFrame({"product_id": entity_ids}).merge(
        product_features,
        on="product_id",
        how="left"
    )

    # Fill missing values for products with no history
    # Review metrics: fill with neutral values
    review_cols = [
        "avg_review_score",
        "min_review_score",
        "frac_5star_reviews",
        "frac_1star_reviews"
    ]
    result[review_cols] = result[review_cols].fillna(3.0)

    # Delivery metrics: fill with on-time assumption
    result[["avg_delivery_delay", "max_delivery_delay"]] = result[
        ["avg_delivery_delay", "max_delivery_delay"]
    ].fillna(0.0)
    result["frac_delivered"] = result["frac_delivered"].fillna(0.0)

    # All other features: fill with 0
    result = result.fillna(0.0)

    # Set index to entity IDs and ensure correct order
    result = result.set_index("product_id").reindex(entity_ids)

    # Ensure all columns are numeric
    result = result.astype(np.float64)

    return result
