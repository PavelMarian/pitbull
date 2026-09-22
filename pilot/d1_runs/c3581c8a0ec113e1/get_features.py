def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    if not isinstance(seed_time, pd.Timestamp):
        seed_time = pd.Timestamp(seed_time)

    ids = pd.Index(entity_ids)
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]

    items = order_items[order_items["product_id"].isin(ids)]
    if items.empty:
        return pd.DataFrame(0, index=ids, columns=["dummy"])

    order_cols = [
        "order_id",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    ]
    df = items.merge(orders[order_cols], on="order_id", how="left")
    df = df[df["order_purchase_timestamp"] < seed_time]

    feats = pd.DataFrame(index=ids)

    if not df.empty:
        g = df.groupby("product_id")

        base = pd.DataFrame({
            "total_items": g["product_id"].size(),
            "total_orders": g["order_id"].nunique(),
            "sum_price": g["price"].sum(),
            "mean_price": g["price"].mean(),
            "median_price": g["price"].median(),
            "sum_freight": g["freight_value"].sum(),
            "mean_freight": g["freight_value"].mean(),
            "unique_sellers": g["seller_id"].nunique(),
            "first_ts": g["order_purchase_timestamp"].min(),
            "last_ts": g["order_purchase_timestamp"].max(),
        })
        feats = feats.join(base)

        feats["history_days"] = (feats["last_ts"] - feats["first_ts"]).dt.days
        feats["days_since_last"] = (seed_time - feats["last_ts"]).dt.days
        feats["items_per_order"] = feats["total_items"] / feats["total_orders"].replace(0, pd.NA)
        feats["avg_purchase_gap"] = feats["history_days"] / (feats["total_orders"] - 1).replace(0, pd.NA)

        for days in (30, 60, 90):
            recent = (
                df[df["order_purchase_timestamp"] >= seed_time - pd.Timedelta(days=days)]
                .groupby("product_id")
                .size()
                .rename(f"recent_{days}")
            )
            feats = feats.join(recent)

        # Estimated delivery is known at purchase time, safe for all rows.
        df = df.copy()
        df["est_delivery_days"] = (
            (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400
        )
        dg_est = df.groupby("product_id")["est_delivery_days"].mean().rename("avg_est_delivery_days")
        feats = feats.join(dg_est)

        # Actual delivery facts exist only for orders delivered before seed_time.
        delivered = df[
            df["order_delivered_customer_date"].notna()
            & (df["order_delivered_customer_date"] <= seed_time)
        ].copy()
        if not delivered.empty:
            delivered["delivery_days"] = (
                (delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]).dt.total_seconds() / 86400
            )
            delivered["late_delivery"] = (
                delivered["delivery_days"] > delivered["est_delivery_days"]
            ).astype(float)

            dg = delivered.groupby("product_id")["delivery_days"].mean().rename("avg_delivery_days")
            feats = feats.join(dg)
            dg_late = delivered.groupby("product_id")["late_delivery"].mean().rename("late_delivery_rate")
            feats = feats.join(dg_late)

        # Reviews: only those already created before seed_time.
        rev = reviews[reviews["review_creation_date"] < seed_time][["order_id", "review_score"]]
        review_part = df[["product_id", "order_id"]].drop_duplicates().merge(
            rev, on="order_id", how="inner"
        )
        if not review_part.empty:
            rg = review_part.groupby("product_id")
            review_feats = pd.DataFrame({
                "review_cnt": rg["review_score"].count(),
                "review_avg": rg["review_score"].mean(),
                "review_pos_rate": (
                    review_part[review_part["review_score"] >= 4]
                    .groupby("product_id")["review_score"].count() / rg["review_score"].count().clip(lower=1)
                ),
                "review_neg_rate": (
                    review_part[review_part["review_score"] <= 2]
                    .groupby("product_id")["review_score"].count() / rg["review_score"].count().clip(lower=1)
                ),
            })
            feats = feats.join(review_feats)

        # Payments: aggregate per order, restricted to payments known at seed_time.
        pay_known = payments[payments["ts"] < seed_time]
        pay_ord = pay_known.groupby("order_id").agg(
            pay_total=("payment_value", "sum"),
            pay_avg=("payment_value", "mean"),
            pay_count=("payment_value", "size"),
            pay_avg_installments=("payment_installments", "mean"),
            pay_credit_rate=("payment_type", lambda x: (x == "credit_card").mean()),
        )

        pay_prod = df[["product_id", "order_id"]].drop_duplicates().merge(
            pay_ord, on="order_id", how="left"
        )
        if not pay_prod.empty and pay_prod["pay_total"].notna().any():
            pg = pay_prod.groupby("product_id")
            pay_feats = pd.DataFrame({
                "pay_total": pg["pay_total"].sum(),
                "pay_avg": pg["pay_avg"].mean(),
                "pay_count": pg["pay_count"].mean(),
                "pay_avg_installments": pg["pay_avg_installments"].mean(),
                "pay_credit_rate": pg["pay_credit_rate"].mean(),
            })
            feats = feats.join(pay_feats)

    feats = feats.reindex(ids)
    feats = feats.drop(columns=[c for c in ["first_ts", "last_ts"] if c in feats.columns], errors="ignore")
    feats = feats.fillna(0)
    feats = feats.astype(float)
    return feats
