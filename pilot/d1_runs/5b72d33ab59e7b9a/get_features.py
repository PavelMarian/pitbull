import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Unpack tables
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]
    
    # Base dataframe with requested product IDs
    base = pd.DataFrame({"product_id": entity_ids})
    seed_month = seed_time.month
    seed_quarter = (seed_month - 1) // 3 + 1
    
    # 1. Preprocess historical order items (all data before seed_time)
    hist_order_items = order_items[order_items["ts"] < seed_time].copy()
    if not hist_order_items.empty:
        # Add time window flags
        window_days_list = [7, 30, 90, 180, 365]
        for wd in window_days_list:
            cutoff = seed_time - pd.Timedelta(days=wd)
            hist_order_items[f"in_last_{wd}d"] = hist_order_items["ts"] >= cutoff
        
        # 1a. Recency feature
        last_sale = hist_order_items.groupby("product_id")["ts"].max().reset_index(name="last_sale_ts")
        last_sale["recency_days"] = (seed_time - last_sale["last_sale_ts"]).dt.days
        last_sale = last_sale.drop(columns=["last_sale_ts"])
        base = base.merge(last_sale, on="product_id", how="left")
        
        # 1b. All-time order items aggregates
        all_time_agg = hist_order_items.groupby("product_id").agg(
            total_orders=("order_id", "nunique"),
            total_units_sold=("order_id", "count"),
            total_revenue=("price", "sum"),
            avg_price=("price", "mean"),
            std_price=("price", "std"),
            total_freight=("freight_value", "sum"),
            avg_freight=("freight_value", "mean"),
            std_freight=("freight_value", "std")
        ).reset_index()
        all_time_agg["avg_items_per_order"] = all_time_agg["total_units_sold"] / all_time_agg["total_orders"].replace(0, np.nan)
        base = base.merge(all_time_agg, on="product_id", how="left")
        
        # 1c. Windowed order items aggregates
        for wd in window_days_list:
            window_col = f"in_last_{wd}d"
            if window_col in hist_order_items.columns:
                window_data = hist_order_items[hist_order_items[window_col]]
                if not window_data.empty:
                    w_agg = window_data.groupby("product_id").agg(
                        **{
                            f"orders_last_{wd}d": ("order_id", "nunique"),
                            f"units_sold_last_{wd}d": ("order_id", "count"),
                            f"revenue_last_{wd}d": ("price", "sum"),
                            f"avg_price_last_{wd}d": ("price", "mean"),
                            f"freight_last_{wd}d": ("freight_value", "sum")
                        }
                    ).reset_index()
                    base = base.merge(w_agg, on="product_id", how="left")
        
        # 2. Order-level features (delivery, status)
        # Only orders already purchased before seed_time; delivery info is
        # only used for orders actually delivered before seed_time, since
        # delivery dates and order_status get filled in after the purchase.
        hist_orders = orders[orders["order_purchase_timestamp"] < seed_time]
        hist_orders_items = hist_order_items.merge(hist_orders, on="order_id", how="left")
        if not hist_orders_items.empty:
            delivered_mask = (
                hist_orders_items["order_delivered_customer_date"].notna()
                & (hist_orders_items["order_delivered_customer_date"] <= seed_time)
            )
            hist_orders_items["is_delivered"] = delivered_mask.astype(int)

            share_delivered = hist_orders_items.groupby("product_id").agg(
                share_delivered_orders=("is_delivered", "mean")
            ).reset_index()
            base = base.merge(share_delivered, on="product_id", how="left")

            delivered_items = hist_orders_items[delivered_mask].copy()
            if not delivered_items.empty:
                delivered_items["delivery_delay_days"] = (delivered_items["order_delivered_customer_date"] - delivered_items["order_estimated_delivery_date"]).dt.days
                delivered_items["delivery_time_days"] = (delivered_items["order_delivered_customer_date"] - delivered_items["order_purchase_timestamp"]).dt.days

                order_features = delivered_items.groupby("product_id").agg(
                    avg_delivery_delay=("delivery_delay_days", "mean"),
                    std_delivery_delay=("delivery_delay_days", "std"),
                    avg_delivery_time=("delivery_time_days", "mean"),
                    std_delivery_time=("delivery_time_days", "std")
                ).reset_index()
                base = base.merge(order_features, on="product_id", how="left")
        
        # 3. Review features
        hist_reviews = reviews[reviews["review_creation_date"] < seed_time].copy()
        if not hist_reviews.empty:
            # Link reviews to products via order_items
            reviews_with_product = hist_reviews.merge(
                hist_order_items[["order_id", "product_id"]].drop_duplicates(),
                on="order_id",
                how="inner"
            )
            if not reviews_with_product.empty:
                reviews_with_product["has_comment"] = (
                    ~reviews_with_product["review_comment_title"].isna() | 
                    ~reviews_with_product["review_comment_message"].isna()
                ).astype(int)
                reviews_with_product = reviews_with_product.merge(
                    orders[["order_id", "order_purchase_timestamp"]],
                    on="order_id",
                    how="left"
                )
                reviews_with_product["time_to_review_days"] = (
                    reviews_with_product["review_creation_date"] - reviews_with_product["order_purchase_timestamp"]
                ).dt.days
                
                review_features = reviews_with_product.groupby("product_id").agg(
                    avg_review_score=("review_score", "mean"),
                    total_reviews=("review_id", "count"),
                    share_5star_reviews=("review_score", lambda x: (x == 5).mean()),
                    share_1star_reviews=("review_score", lambda x: (x == 1).mean()),
                    share_commented_reviews=("has_comment", "mean"),
                    avg_days_to_review=("time_to_review_days", "mean")
                ).reset_index()
                base = base.merge(review_features, on="product_id", how="left")
        
        # 4. Payment features
        hist_payments = payments[payments["ts"] < seed_time].copy()
        if not hist_payments.empty:
            # Aggregate per order first
            order_payment_agg = hist_payments.groupby("order_id").agg(
                total_order_payment=("payment_value", "sum"),
                num_payment_methods=("payment_type", "nunique"),
                avg_installments=("payment_installments", "mean"),
                share_credit_card=("payment_type", lambda x: (x == "credit_card").mean()),
                share_boleto=("payment_type", lambda x: (x == "boleto").mean())
            ).reset_index()
            # Link to products
            product_payments = hist_order_items[["order_id", "product_id"]].drop_duplicates().merge(
                order_payment_agg,
                on="order_id",
                how="inner"
            )
            if not product_payments.empty:
                payment_features = product_payments.groupby("product_id").agg(
                    avg_order_payment=("total_order_payment", "mean"),
                    avg_payment_methods_per_order=("num_payment_methods", "mean"),
                    avg_installments_per_order=("avg_installments", "mean"),
                    share_orders_credit_card=("share_credit_card", "mean"),
                    share_orders_boleto=("share_boleto", "mean")
                ).reset_index()
                base = base.merge(payment_features, on="product_id", how="left")
    
    # 5. Add global/seed time features and clean up
    # Recency fillna for products with no history
    base["recency_days"] = base["recency_days"].fillna(9999)
    # Has history flag
    base["has_history"] = (base["total_units_sold"].fillna(0) > 0).astype(int)
    # Seed time features
    base["seed_month"] = seed_month
    base["seed_quarter"] = seed_quarter
    # Fill remaining NaNs with 0
    base = base.fillna(0)
    # Set index and reindex to input entity_ids
    base = base.set_index("product_id").reindex(entity_ids)
    
    return base