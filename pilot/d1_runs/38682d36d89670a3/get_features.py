import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]
    
    seed_ts = pd.to_datetime(seed_time)
    
    features = pd.DataFrame(index=entity_ids)
    features.index.name = "product_id"
    
    historical_order_items = order_items[
        (order_items["product_id"].isin(entity_ids)) & 
        (order_items["ts"] < seed_ts)
    ].copy()
    
    historical_items = pd.merge(
        historical_order_items,
        orders[["order_id", "order_status", "order_delivered_customer_date", "order_estimated_delivery_date"]],
        on="order_id",
        how="left"
    )
    
    historical_items["total_revenue"] = historical_items["price"] + historical_items["freight_value"]
    total_revenue = historical_items.groupby("product_id")["total_revenue"].sum().rename("total_revenue")
    features = features.join(total_revenue, how="left").fillna(0)
    
    total_units = historical_items.groupby("product_id").size().rename("total_units")
    features = features.join(total_units, how="left").fillna(0)
    
    avg_price = historical_items.groupby("product_id")["price"].mean().rename("avg_price")
    features = features.join(avg_price, how="left").fillna(0)
    
    avg_freight = historical_items.groupby("product_id")["freight_value"].mean().rename("avg_freight")
    features = features.join(avg_freight, how="left").fillna(0)
    
    unique_orders = historical_items.groupby("product_id")["order_id"].nunique().rename("unique_orders")
    features = features.join(unique_orders, how="left").fillna(0)
    
    unique_sellers = historical_items.groupby("product_id")["seller_id"].nunique().rename("unique_sellers")
    features = features.join(unique_sellers, how="left").fillna(0)
    
    delivered_items = historical_items[
        (historical_items["order_status"] == "delivered")
        & (historical_items["order_delivered_customer_date"] < seed_ts)
    ]
    delivered_count = delivered_items.groupby("product_id").size().rename("delivered_units")
    features = features.join(delivered_count, how="left").fillna(0)
    features["delivered_share"] = features["delivered_units"] / features["total_units"].replace(0, 1)
    features = features.drop(columns=["delivered_units"])

    known_delivery = historical_items[
        historical_items["order_delivered_customer_date"] < seed_ts
    ].copy()
    known_delivery["delivery_difference_days"] = (
        known_delivery["order_delivered_customer_date"] - known_delivery["order_estimated_delivery_date"]
    ).dt.total_seconds() / (24 * 3600)
    avg_delivery_diff = known_delivery.groupby("product_id")["delivery_difference_days"].mean().rename("avg_delivery_diff_days")
    features = features.join(avg_delivery_diff, how="left").fillna(0)
    
    last_sale = historical_items.groupby("product_id")["ts"].max().rename("last_sale_ts")
    features = features.join(last_sale, how="left")
    features["days_since_last_sale"] = (seed_ts - features["last_sale_ts"]).dt.total_seconds() / (24 * 3600)
    features["days_since_last_sale"] = features["days_since_last_sale"].fillna(730)
    features = features.drop(columns=["last_sale_ts"])
    
    relevant_order_ids = order_items[order_items["product_id"].isin(entity_ids)]["order_id"].unique()
    relevant_reviews = pd.merge(
        reviews[reviews["order_id"].isin(relevant_order_ids)],
        order_items[["order_id", "product_id"]],
        on="order_id",
        how="left"
    )
    historical_reviews = relevant_reviews[relevant_reviews["review_creation_date"] < seed_ts]
    
    avg_review_score = historical_reviews.groupby("product_id")["review_score"].mean().rename("avg_review_score")
    features = features.join(avg_review_score, how="left").fillna(0)
    
    review_count = historical_reviews.groupby("product_id").size().rename("review_count")
    features = features.join(review_count, how="left").fillna(0)
    
    high_score_reviews = historical_reviews[historical_reviews["review_score"] == 5]
    high_score_count = high_score_reviews.groupby("product_id").size().rename("high_score_reviews")
    features = features.join(high_score_count, how="left").fillna(0)
    features["high_score_share"] = features["high_score_reviews"] / features["review_count"].replace(0, 1)
    features = features.drop(columns=["high_score_reviews"])
    
    relevant_payments = pd.merge(
        payments[payments["order_id"].isin(relevant_order_ids)],
        order_items[["order_id", "product_id"]],
        on="order_id",
        how="left"
    )
    historical_payments = relevant_payments[relevant_payments["ts"] < seed_ts]
    
    payments_per_order = historical_payments.groupby(["product_id", "order_id"])["payment_sequential"].max().rename("payments_count")
    avg_payments_per_order = payments_per_order.groupby("product_id").mean().rename("avg_payments_per_order")
    features = features.join(avg_payments_per_order, how="left").fillna(0)
    
    card_payments = historical_payments[historical_payments["payment_type"] == "credit_card"]
    total_payment_value = historical_payments.groupby("product_id")["payment_value"].sum().rename("total_payment_value")
    card_payment_value = card_payments.groupby("product_id")["payment_value"].sum().rename("card_payment_value")
    features = features.join(total_payment_value, how="left").fillna(0)
    features = features.join(card_payment_value, how="left").fillna(0)
    features["card_payment_share"] = features["card_payment_value"] / features["total_payment_value"].replace(0, 1)
    features = features.drop(columns=["total_payment_value", "card_payment_value"])
    
    avg_installments = historical_payments.groupby("product_id")["payment_installments"].mean().rename("avg_installments")
    features = features.join(avg_installments, how="left").fillna(0)
    
    features = features.reindex(entity_ids)
    features.index.name = "product_id"
    
    return features