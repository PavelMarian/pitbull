import pandas as pd
import numpy as np

def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    """
    Функция строит матрицу признаков для заданных товаров (product_id)
    на момент seed_time (т.е. используя только информацию, доступную до seed_time).
    Возвращает DataFrame с индексами = entity_ids и числовыми признаками.
    """
    orders = db['orders']
    order_items = db['order_items']
    reviews = db['reviews']
    payments = db['payments']

    # Соединяем позиции с заказами, получаем дату и покупателя
    merged = order_items.merge(
        orders[['order_id', 'customer_id', 'order_purchase_timestamp']],
        on='order_id'
    )
    # Оставляем только заказы, совершенные до seed_time
    past = merged[merged['order_purchase_timestamp'] < seed_time]

    # ---- Базовые агрегаты по продажам ----
    base_agg = past.groupby('product_id').agg(
        order_count=('order_id', 'count'),
        price_sum=('price', 'sum'),
        price_avg=('price', 'mean'),
        price_max=('price', 'max'),
        freight_sum=('freight_value', 'sum'),
        freight_avg=('freight_value', 'mean'),
        unique_buyers=('customer_id', 'nunique'),
        last_purchase_date=('order_purchase_timestamp', 'max'),
    ).reset_index()

    # ---- Агрегаты по отзывам (через заказы) ----
    # Все заказы с товаром, привязываем отзывы.
    # Учитываем только отзывы, уже существующие на момент seed_time:
    # отзыв появляется позже покупки, доступность определяем по review_creation_date.
    available_reviews = reviews[reviews['review_creation_date'] < seed_time]
    past_reviews = past[['product_id', 'order_id']].merge(
        available_reviews[['order_id', 'review_score']],
        on='order_id',
        how='left'
    )
    rev_agg = past_reviews.groupby('product_id').agg(
        review_mean=('review_score', 'mean'),
        review_count=('review_score', 'count'),
    ).reset_index()

    # ---- Агрегаты по платежам ----
    past_payments = past[['product_id', 'order_id']].merge(
        payments[['order_id', 'payment_value', 'payment_installments']],
        on='order_id',
        how='left'
    )
    pay_agg = past_payments.groupby('product_id').agg(
        payment_sum=('payment_value', 'sum'),
        payment_avg=('payment_value', 'mean'),
        installment_mean=('payment_installments', 'mean'),
        payment_count=('payment_value', 'count'),  # количество записей платежей
    ).reset_index()

    # ---- Объединяем все признаки ----
    feats = base_agg.merge(rev_agg, on='product_id', how='left')
    feats = feats.merge(pay_agg, on='product_id', how='left')

    # ---- Признак: сколько дней прошло с последней покупки ----
    feats['days_since_last_purchase'] = (seed_time - feats['last_purchase_date']).dt.days

    # ---- Признаки: количество продаж за последние 30/60/90/180 дней ----
    # Вспомогательная функция, чтобы не дублировать код
    def count_last_n_days(days):
        threshold = seed_time - pd.Timedelta(days=days)
        mask = past['order_purchase_timestamp'] >= threshold
        recent = past[mask]
        return recent.groupby('product_id').size().reset_index(name=f'count_last_{days}_days')

    for days in (30, 60, 90, 180):
        cnt = count_last_n_days(days)
        feats = feats.merge(cnt, on='product_id', how='left')

    # ---- Средний интервал между покупками (в днях) ----
    # Для каждого товара посчитаем количество заказов и разброс дат
    # Интервал = (макс - мин) / (число покупок - 1), если покупок>1
    purchase_stats = past.groupby('product_id').agg(
        min_date=('order_purchase_timestamp', 'min'),
        max_date=('order_purchase_timestamp', 'max'),
        order_count2=('order_id', 'count')
    ).reset_index()
    purchase_stats['interval_mean'] = (
        (purchase_stats['max_date'] - purchase_stats['min_date']).dt.days
        / (purchase_stats['order_count2'] - 1)
    )
    purchase_stats.loc[purchase_stats['order_count2'] <= 1, 'interval_mean'] = np.nan
    feats = feats.merge(
        purchase_stats[['product_id', 'interval_mean']],
        on='product_id',
        how='left'
    )

    # ---- Дополнительные признаки: доля оплаты CREDIT_CARD, содержащая товар ----
    past_pay_type = past[['product_id', 'order_id']].merge(
        payments[['order_id', 'payment_type']],
        on='order_id',
        how='left'
    )
    pay_type_agg = past_pay_type.groupby('product_id').apply(
        lambda x: (x['payment_type'] == 'credit_card').mean(),
        include_groups=False
    ).reset_index(name='credit_card_share')
    feats = feats.merge(pay_type_agg, on='product_id', how='left')

    # ---- Признак: количество уникальных продавцов для товара ----
    seller_agg = past.groupby('product_id')['seller_id'].nunique().reset_index(name='unique_sellers')
    feats = feats.merge(seller_agg, on='product_id', how='left')

    # ---- Удаляем нечисловые вспомогательные колонки (datetime) ----
    feats.drop(columns=['last_purchase_date', 'min_date', 'max_date'], inplace=True, errors='ignore')

    # ---- Заменяем пропуски на 0 ----
    feats = feats.fillna(0)

    # ---- Реиндексируем под entity_ids ----
    feats.set_index('product_id', inplace=True)
    feats = feats.reindex(entity_ids, fill_value=0)

    # Гарантируем, что все колонки числовые (преобразование floats даёт float64)
    return feats[feats.columns] # все являются числами по контексту