import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Define time windows
    lookback_30d = seed_time - pd.Timedelta(days=30)
    lookback_90d = seed_time - pd.Timedelta(days=90)
    lookback_180d = seed_time - pd.Timedelta(days=180)

    # Rename timestamp column for clarity
    order_items = db["order_items"].rename(columns={"ts": "order_purchase_ts"})
    # Filter items to relevant products and valid timestamps
    filtered_items = order_items[
        order_items["product_id"].isin(entity_ids) &
        (order_items["order_purchase_ts"] < seed_time)
    ].copy()

    # Aggregate lifetime product sales metrics
    product_base = filtered_items.groupby("product_id").agg(
        total_sales_ever=("order_item_id", "count"),
        total_revenue_ever=("price", "sum"),
        avg_price_ever=("price", "mean"),
        avg_freight_ever=("freight_value", "mean"),
        last_sale_days=("order_purchase_ts", lambda x: (seed_time - x.max()).days if x.notna().any() else np.inf)
    ).reset_index()

    # 30-day window features
    win30 = filtered_items[filtered_items["order_purchase_ts"] >= lookback_30d]
    win30_agg = win30.groupby("product_id").agg(
        sales_30d=("order_item_id", "count"),
        revenue_30d=("price", "sum"),
        avg_price_30d=("price", "mean"),
        uniq_orders_30d=("order_id", "nunique")
    ).reset_index()

    # 90-day window features
    win90 = filtered_items[filtered_items["order_purchase_ts"] >= lookback_90d]
    win90_agg = win90.groupby("product_id").agg(
        sales_90d=("order_item_id", "count"),
        revenue_90d=("price", "sum"),
        avg_price_90d=("price", "mean"),
        uniq_orders_90d=("order_id", "nunique")
    ).reset_index()

    # 180-day window features
    win180 = filtered_items[filtered_items["order_purchase_ts"] >= lookback_180d]
    win180_agg = win180.groupby("product_id").agg(
        sales_180d=("order_item_id", "count"),
        revenue_180d=("price", "sum"),
        avg_price_180d=("price", "mean"),
        uniq_orders_180d=("order_id", "nunique")
    ).reset_index()

    # Seller concentration features
    seller_agg = filtered_items.groupby(["product_id", "seller_id"]).agg(
        seller_sales=("order_item_id", "count")
    ).reset_index()
    seller_conc = seller_agg.groupby("product_id").agg(
        total_sellers=("seller_id", "nunique"),
        top_seller_share=("seller_sales", lambda x: x.max() / x.sum() if x.sum() > 0 else 0)
    ).reset_index()

    # Review features: only reviews that already exist at seed_time
    reviews = db["reviews"]
    reviews_past = reviews[reviews["review_creation_date"] < seed_time]
    items_with_reviews = filtered_items.merge(
        reviews_past[["order_id", "review_score", "review_comment_message"]],
        on="order_id",
        how="left"
    )
    items_with_reviews["has_comment"] = items_with_reviews["review_comment_message"].notna().astype(int)
    review_feats = items_with_reviews.groupby("product_id").agg(
        avg_review_score=("review_score", "mean"),
        total_reviews=("review_score", "count"),
        share_commented_reviews=("has_comment", "mean")
    ).reset_index()

    # Payment features
    payments_order = db["payments"].groupby("order_id").agg(
        total_payment=("payment_value", "sum"),
        avg_installments=("payment_installments", "mean"),
        has_card=("payment_type", lambda x: int(("credit_card" in x.values) or ("boleto" in x.values)))
    ).reset_index()
    items_with_pay = filtered_items.merge(payments_order, on="order_id", how="left")
    pay_feats = items_with_pay.groupby("product_id").agg(
        avg_order_value=("total_payment", "mean"),
        avg_installments=("avg_installments", "mean"),
        share_card_payments=("has_card", "mean")
    ).reset_index()

    # Build base DataFrame with all required product IDs
    base = pd.DataFrame({"product_id": entity_ids})

    # Merge all feature sets
    features = (
        base
        .merge(product_base, on="product_id", how="left")
        .merge(win30_agg, on="product_id", how="left")
        .merge(win90_agg, on="product_id", how="left")
        .merge(win180_agg, on="product_id", how="left")
        .merge(seller_conc, on="product_id", how="left")
        .merge(review_feats, on="product_id", how="left")
        .merge(pay_feats, on="product_id", how="left")
    )

    # Fill missing values with safe defaults
    fill_map = {
        "total_sales_ever": 0,
        "total_revenue_ever": 0,
        "avg_price_ever": 0,
        "avg_freight_ever": 0,
        "last_sale_days": 9999,
        "sales_30d": 0,
        "revenue_30d": 0,
        "avg_price_30d": 0,
        "uniq_orders_30d": 0,
        "sales_90d": 0,
        "revenue_90d": 0,
        "avg_price_90d": 0,
        "uniq_orders_90d": 0,
        "sales_180d": 0,
        "revenue_180d": 0,
        "avg_price_180d": 0,
        "uniq_orders_180d": 0,
        "total_sellers": 0,
        "top_seller_share": 0,
        "avg_review_score": 3,
        "total_reviews": 0,
        "share_commented_reviews": 0,
        "avg_order_value": 0,
        "avg_installments": 0,
        "share_card_payments": 0
    }
    features = features.fillna(value=fill_map)

    # Reindex to input entity IDs, set as index
    features = features.set_index("product_id").reindex(entity_ids)
    # Ensure all columns are numeric
    features = features.astype(np.float64)

    return features