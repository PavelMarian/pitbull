import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Normalize inputs
    entity_ids = list(entity_ids)
    seed_time = pd.Timestamp(seed_time)

    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]

    # Keep only order items of requested products that were already purchased
    # strictly before seed_time (ts = purchase moment of the order).
    mask_items = order_items["product_id"].isin(entity_ids) & (order_items["ts"] < seed_time)
    items = order_items.loc[mask_items].copy()

    # If no items, return a zero matrix (with the same columns as the non-empty case)
    feature_cols = [
        "num_orders", "num_items", "total_price", "total_freight",
        "mean_price", "max_price", "min_price", "std_price",
        "nunique_seller", "nunique_customer", "mean_freight",
        "mean_review", "review_count", "mean_payment_total",
        "mean_payment_installments", "mean_payment_count",
        "days_since_last", "product_age_days", "sales_per_month",
        "review_rate", "mean_delivery_delay", "has_sales"
    ]
    if items.empty:
        df = pd.DataFrame(0.0, index=pd.Index(entity_ids, name="product_id"), columns=feature_cols)
        return df

    # Orders info: customer id is known at purchase time; delivery dates only
    # for orders already delivered before seed_time.
    orders_sub = orders[["order_id", "customer_id",
                         "order_delivered_customer_date", "order_estimated_delivery_date"]]
    items = items.merge(orders_sub, how="left", on="order_id")

    # Payments: only payment rows already registered before seed_time.
    pay_past = payments[payments["ts"] <= seed_time]
    pay_agg = pay_past.groupby("order_id").agg(
        pay_count=("payment_sequential", "count"),
        pay_total=("payment_value", "sum"),
        pay_install_avg=("payment_installments", "mean")
    ).reset_index()

    # Reviews: a review exists for seed_time only if it was created before it.
    rev_past = reviews[reviews["review_creation_date"] <= seed_time]
    rev_agg = rev_past.groupby("order_id").agg(
        rev_count=("review_score", "count"),
        rev_mean=("review_score", "mean")
    ).reset_index()

    # Join the aggregates into the items frame
    items = items.merge(pay_agg, on="order_id", how="left")
    items = items.merge(rev_agg, on="order_id", how="left")

    # Group by product_id for the final features
    grouped = items.groupby("product_id")

    # Build base feature aggregates
    feature_agg = pd.DataFrame({
        "num_orders": grouped["order_id"].nunique(),
        "num_items": grouped["price"].count(),
        "total_price": grouped["price"].sum(),
        "total_freight": grouped["freight_value"].sum(),
        "mean_price": grouped["price"].mean(),
        "max_price": grouped["price"].max(),
        "min_price": grouped["price"].min(),
        "std_price": grouped["price"].std(),
        "nunique_seller": grouped["seller_id"].nunique(),
        "nunique_customer": grouped["customer_id"].nunique(),
        "mean_freight": grouped["freight_value"].mean(),
        "mean_review": grouped["rev_mean"].mean(),
        "review_count": grouped["rev_count"].sum(),
        "mean_payment_total": grouped["pay_total"].mean(),
        "mean_payment_installments": grouped["pay_install_avg"].mean(),
        "mean_payment_count": grouped["pay_count"].mean(),
        "first_ts": grouped["ts"].min(),
        "last_ts": grouped["ts"].max()
    })

    # Compute time-based features
    feature_agg["days_since_last"] = (seed_time - feature_agg["last_ts"]).dt.days
    feature_agg["product_age_days"] = (seed_time - feature_agg["first_ts"]).dt.days
    # sales per month: avoid division by zero
    age_positive = feature_agg["product_age_days"].clip(lower=1)
    feature_agg["sales_per_month"] = feature_agg["num_orders"] * 30 / age_positive
    # review ratio
    feature_agg["review_rate"] = feature_agg["review_count"] / feature_agg["num_orders"].replace(0, np.nan)

    # Delivery delay only from orders physically delivered before seed_time;
    # the estimate is known from the purchase moment, so it is safe to use.
    delivered_mask = (
        items["order_delivered_customer_date"].notna()
        & (items["order_delivered_customer_date"] <= seed_time)
    )
    if delivered_mask.any():
        delivered = items.loc[delivered_mask].copy()
        delivered["delivery_delay"] = (
            delivered["order_delivered_customer_date"] - delivered["order_estimated_delivery_date"]
        ).dt.days
        delay_mean = delivered.groupby("product_id")["delivery_delay"].mean()
    else:
        delay_mean = pd.Series(dtype=float)
    feature_agg = feature_agg.join(delay_mean.rename("mean_delivery_delay"))

    # Flag for products that had sales
    feature_agg["has_sales"] = 1

    # Drop internal columns, fill NaN with 0
    feature_agg = feature_agg.drop(columns=["first_ts", "last_ts"], errors="ignore")
    feature_agg = feature_agg.fillna(0)

    # Reindex under entity_ids
    feature_agg = feature_agg.reindex(entity_ids, fill_value=0)
    return feature_agg
