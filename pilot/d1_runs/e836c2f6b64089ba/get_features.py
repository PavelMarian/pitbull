import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    """
    Строит матрицу признаков для каждого product_id на основе истории
    заказов до seed_time. Используются таблицы orders, order_items, reviews.
    """
    # extract tables
    orders = db['orders']
    items = db['order_items']
    reviews = db['reviews']

    # convert ids to list
    ids = list(entity_ids)

    # filter order_items to only products of interest
    items_filt = items[items['product_id'].isin(ids)].copy()

    # merge with orders to get customer and purchase timestamp
    orders_sub = orders[['order_id', 'customer_id', 'order_purchase_timestamp']]
    df = items_filt.merge(orders_sub, on='order_id', how='left')

    # keep only orders made before seed_time
    df = df[df['order_purchase_timestamp'] < seed_time]

    # keep only reviews that already exist at seed_time
    # (a review is written after the purchase, so later reviews must be dropped)
    rev = reviews[reviews['review_creation_date'] < seed_time]
    rev_agg = rev.groupby('order_id', as_index=False)['review_score'].mean()
    df = df.merge(rev_agg, on='order_id', how='left')

    # compute days since order
    df['days_since'] = (seed_time - df['order_purchase_timestamp']).dt.days

    # windows for recent activity
    for w in [30, 60, 90]:
        df[f'last{w}'] = (df['days_since'] <= w).astype(int)
        df[f'revenue_last{w}'] = df['price'] * df[f'last{w}']

    # define aggregations
    agg_dict = {
        'price': ['count', 'sum', 'mean', 'min', 'max'],
        'freight_value': ['sum', 'mean'],
        'order_id': ['nunique'],
        'customer_id': ['nunique'],
        'last30': ['sum'],
        'last60': ['sum'],
        'last90': ['sum'],
        'revenue_last30': ['sum'],
        'revenue_last60': ['sum'],
        'revenue_last90': ['sum'],
        'review_score': ['mean', 'count'],
        'days_since': ['min', 'max']
    }

    # group by product
    feat = df.groupby('product_id').agg(agg_dict)
    # flatten MultiIndex columns
    feat.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in feat.columns]

    # reindex to ensure all product ids are present
    result = feat.reindex(ids)
    result = result.fillna(0)

    # name the index
    result.index.name = 'product_id'
    return result