import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Extract tables
    orders = db["orders"].copy()
    order_items = db["order_items"].copy()
    reviews = db["reviews"].copy()
    payments = db["payments"].copy()
    
    # Filter to target products for faster processing
    oi_target = order_items[order_items["product_id"].isin(entity_ids)].copy()
    # Merge with orders to get status and timestamps
    oi_orders = oi_target.merge(
        orders[["order_id", "order_status", "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]],
        on="order_id",
        how="left"
    )
    
    # Split into historical (all before seed) and recent 90-day windows
    oi_hist = oi_orders[oi_orders["order_purchase_timestamp"] < seed_time].copy()
    window_90 = seed_time - pd.Timedelta(days=90)
    oi_90 = oi_hist[oi_hist["order_purchase_timestamp"] >= window_90].copy()
    
    # 1. Historical aggregate features
    agg_hist = oi_hist.groupby("product_id").agg(
        total_units_sold=pd.NamedAgg(column="order_item_id", aggfunc="count"),
        total_revenue=pd.NamedAgg(column="price", aggfunc="sum"),
        total_freight_cost=pd.NamedAgg(column="freight_value", aggfunc="sum"),
        avg_price=pd.NamedAgg(column="price", aggfunc="mean"),
        avg_freight=pd.NamedAgg(column="freight_value", aggfunc="mean"),
        first_sale_date=pd.NamedAgg(column="order_purchase_timestamp", aggfunc="min"),
        last_sale_date=pd.NamedAgg(column="order_purchase_timestamp", aggfunc="max"),
        num_unique_sellers=pd.NamedAgg(column="seller_id", aggfunc="nunique")
    ).reset_index()
    # Calculate time-based features
    agg_hist["days_since_last_sale"] = (seed_time - agg_hist["last_sale_date"]).dt.days
    agg_hist["days_since_first_sale"] = (seed_time - agg_hist["first_sale_date"]).dt.days
    agg_hist = agg_hist.drop(columns=["first_sale_date", "last_sale_date"])
    
    # 2. Recent 90-day features
    agg_90 = oi_90.groupby("product_id").agg(
        units_sold_90d=pd.NamedAgg(column="order_item_id", aggfunc="count"),
        revenue_90d=pd.NamedAgg(column="price", aggfunc="sum"),
        avg_price_90d=pd.NamedAgg(column="price", aggfunc="mean"),
        num_orders_90d=pd.NamedAgg(column="order_id", aggfunc="nunique")
    ).reset_index()
    
    # 3. Delivery performance features
    # Only deliveries that already happened by seed_time are observable:
    # order_delivered_customer_date is filled in after the purchase, so it can
    # lie in the future even for orders purchased before seed_time.
    oi_delivered = oi_hist[
        oi_hist["order_delivered_customer_date"].notna()
        & (oi_hist["order_delivered_customer_date"] < seed_time)
    ].copy()
    oi_delivered["delivery_delay_days"] = (oi_delivered["order_delivered_customer_date"] - oi_delivered["order_estimated_delivery_date"]).dt.days
    agg_delivery = oi_delivered.groupby("product_id").agg(
        avg_delivery_delay=pd.NamedAgg(column="delivery_delay_days", aggfunc="mean"),
        num_delivered=pd.NamedAgg(column="order_id", aggfunc="count")
    ).reset_index()
    
    # 4. Review features
    rev_hist = reviews[reviews["review_creation_date"] < seed_time].copy()
    order_product = oi_hist[["order_id", "product_id"]].drop_duplicates()
    rev_product = rev_hist.merge(order_product, on="order_id", how="inner")
    agg_review = rev_product.groupby("product_id").agg(
        avg_review_score=pd.NamedAgg(column="review_score", aggfunc="mean"),
        num_reviews=pd.NamedAgg(column="review_id", aggfunc="count"),
        pct_low_reviews=pd.NamedAgg(column="review_score", aggfunc=lambda x: (x <= 2).sum() / len(x) if len(x) > 0 else 0.0)
    ).reset_index()
    
    # 5. Payment features
    pay_hist = payments[payments["ts"] < seed_time].copy()
    order_pay = pay_hist.merge(oi_hist[["order_id", "product_id"]].drop_duplicates(), on="order_id", how="inner")
    agg_pay = order_pay.groupby("product_id").agg(
        avg_payment_value=pd.NamedAgg(column="payment_value", aggfunc="mean"),
        avg_installments=pd.NamedAgg(column="payment_installments", aggfunc="mean"),
        pct_credit_card=pd.NamedAgg(column="payment_type", aggfunc=lambda x: (x == "credit_card").sum() / len(x) if len(x) > 0 else 0.0)
    ).reset_index()
    
    # Combine all features
    base = pd.DataFrame({"product_id": entity_ids})
    features = (
        base
        .merge(agg_hist, on="product_id", how="left")
        .merge(agg_90, on="product_id", how="left")
        .merge(agg_delivery, on="product_id", how="left")
        .merge(agg_review, on="product_id", how="left")
        .merge(agg_pay, on="product_id", how="left")
    )
    
    # Impute missing values
    fill_0 = ["total_units_sold", "total_revenue", "total_freight_cost", "units_sold_90d", "revenue_90d", "num_orders_90d", "num_delivered", "num_reviews"]
    fill_neg1 = ["days_since_last_sale", "days_since_first_sale"]
    fill_05 = ["pct_low_reviews", "pct_credit_card"]
    fill_0num = ["avg_price", "avg_freight", "avg_price_90d", "avg_delivery_delay", "avg_review_score", "avg_payment_value", "avg_installments", "num_unique_sellers"]
    
    features[fill_0] = features[fill_0].fillna(0).astype(np.int64)
    features[fill_neg1] = features[fill_neg1].fillna(-1).astype(np.int64)
    features[fill_05] = features[fill_05].fillna(0.5).astype(np.float64)
    features[fill_0num] = features[fill_0num].fillna(0.0).astype(np.float64)
    
    # Additional engineered features
    features["has_any_sales"] = (features["total_units_sold"] > 0).astype(np.int8)
    features["units_90d_share"] = np.where(
        features["total_units_sold"] == 0,
        0.0,
        features["units_sold_90d"] / features["total_units_sold"]
    ).astype(np.float64)
    features["sales_per_seller"] = np.where(
        features["num_unique_sellers"] == 0,
        0.0,
        features["total_units_sold"] / features["num_unique_sellers"]
    ).astype(np.float64)
    
    # Finalize index
    features = features.set_index("product_id").reindex(entity_ids)
    # Ensure all columns are numeric
    features = features.select_dtypes(include=[np.number])
    return features