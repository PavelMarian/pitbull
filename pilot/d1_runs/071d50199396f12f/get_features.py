def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame:
    # Извлекаем таблицы
    orders = db["orders"]
    order_items = db["order_items"]
    reviews = db["reviews"]
    payments = db["payments"]
    
    # Приводим seed_time к datetime, если необходимо
    seed_time = pd.Timestamp(seed_time)
    
    # Объединяем order_items с orders для получения времени покупки
    df = order_items.merge(
        orders[["order_id", "order_purchase_timestamp"]],
        on="order_id",
        how="left"
    )
    # Фильтруем только заказы до seed_time
    df = df[df["order_purchase_timestamp"] < seed_time]
    
    # Если нет данных вообще, возвращаем DataFrame с нулями
    if df.empty:
        result = pd.DataFrame(index=entity_ids)
        # Заполним все признаки нулями (количество признаков будет добавлено позже)
        # Пока вернём пустой DF, но добавим колонки позже? Лучше сразу создать с нужными колонками.
        # Но мы не знаем количество колонок. Сделаем заглушку: вернём DF с одним столбцом 0.
        # Однако функция должна вернуть матрицу признаков, поэтому создадим фиктивные колонки.
        # Для простоты создадим несколько колонок с нулями.
        # Но лучше вернуть DF с нулевым числом колонок? Нет, нужно числовые столбцы.
        # Поэтому создадим стандартный набор признаков и заполним нулями.
        # Определим список признаков заранее.
        # Но чтобы не дублировать, создадим функцию, которая генерирует признаки.
        # Пока вернём пустой DF, но позже добавим обработку.
        pass
    
    # Агрегируем payments по order_id
    payments_agg = payments.groupby("order_id").agg(
        payment_sum=("payment_value", "sum"),
        payment_installments_mean=("payment_installments", "mean"),
        payment_count=("payment_sequential", "count")
    ).reset_index()
    
    # Агрегируем reviews по order_id: учитываем только отзывы, которые уже
    # существуют на момент seed_time (отзыв создаётся позже покупки)
    reviews = reviews[reviews["review_creation_date"] < seed_time]
    reviews_agg = reviews.groupby("order_id").agg(
        review_score_mean=("review_score", "mean"),
        review_count=("review_id", "count")
    ).reset_index()
    
    # Присоединяем агрегаты к df
    df = df.merge(payments_agg, on="order_id", how="left")
    df = df.merge(reviews_agg, on="order_id", how="left")
    
    # Вычисляем количество дней до seed_time
    df["days_before"] = (seed_time - df["order_purchase_timestamp"]).dt.days
    
    # Оставляем только нужные product_id
    df = df[df["product_id"].isin(entity_ids)]
    
    # Если после фильтрации нет данных, создадим пустой DF с нужными индексами
    if df.empty:
        # Создадим DataFrame с индексами entity_ids и нулевыми значениями для всех признаков
        # Определим список признаков (мы его построим ниже)
        # Но чтобы не усложнять, вернём DF с фиксированным набором признаков, которые мы будем использовать.
        # Лучше создать функцию, которая строит признаки, и при пустом df вернуть нули.
        # Мы реализуем это в конце.
        pass
    
    # Функция для вычисления признаков за окно (в днях)
    def window_features(data, window_days):
        # Фильтруем по окну
        if window_days is None:
            mask = data["days_before"] >= 0
        else:
            mask = (data["days_before"] < window_days) & (data["days_before"] >= 0)
        sub = data[mask]
        if sub.empty:
            # Возвращаем DataFrame с одним столбцом? Лучше вернуть пустой DF с индексами product_id
            # Но мы потом объединим, поэтому вернём DF с индексами всех product_id и нулями
            # Создадим временный DF с нулями для всех entity_ids
            idx = pd.Index(entity_ids, name="product_id")
            return pd.DataFrame({
                f"count_{window_days}d": 0,
                f"sum_price_{window_days}d": 0,
                f"avg_price_{window_days}d": 0,
                f"sum_freight_{window_days}d": 0,
                f"avg_freight_{window_days}d": 0,
                f"sum_payment_{window_days}d": 0,
                f"avg_payment_{window_days}d": 0,
                f"avg_installments_{window_days}d": 0,
                f"avg_review_score_{window_days}d": 0,
                f"positive_review_ratio_{window_days}d": 0
            }, index=idx)
        grouped = sub.groupby("product_id").agg(
            count=("order_id", "nunique"),  # количество уникальных заказов
            sum_price=("price", "sum"),
            avg_price=("price", "mean"),
            sum_freight=("freight_value", "sum"),
            avg_freight=("freight_value", "mean"),
            sum_payment=("payment_sum", "sum"),
            avg_payment=("payment_sum", "mean"),
            avg_installments=("payment_installments_mean", "mean"),
            avg_review_score=("review_score_mean", "mean"),
            # Доля положительных отзывов (score >= 4)
            positive_review_ratio=("review_score_mean", lambda x: (x >= 4).mean() if len(x) > 0 else 0)
        ).reset_index()
        # Переименуем столбцы с суффиксом окна
        grouped.columns = ["product_id"] + [f"{col}_{window_days}d" for col in grouped.columns[1:]]
        return grouped.set_index("product_id")
    
    # Вычислим признаки для окон 7, 30, 90 и всех (None)
    windows = [7, 30, 90, None]  # None означает все время
    frames = []
    for w in windows:
        fw = window_features(df, w)
        frames.append(fw)
    
    # Объединяем все окна
    features = pd.concat(frames, axis=1)
    
    # Вычислим тренд: count_30d - count_prev_30d (окно 30-60 дней назад)
    # Сначала посчитаем для предыдущего окна (30-60)
    prev_30 = window_features(df, 60)  # это даст count_60d и т.д., но нам нужны именно за 30-60
    # Лучше сделать отдельно: фильтруем 30 <= days_before < 60
    mask_prev = (df["days_before"] >= 30) & (df["days_before"] < 60)
    sub_prev = df[mask_prev]
    if sub_prev.empty:
        count_prev_30 = pd.Series(0, index=pd.Index(entity_ids, name="product_id"))
        sum_price_prev_30 = pd.Series(0, index=pd.Index(entity_ids, name="product_id"))
    else:
        count_prev_30 = sub_prev.groupby("product_id").size()
        sum_price_prev_30 = sub_prev.groupby("product_id")["price"].sum()
    # Приводим к общему индексу
    count_prev_30 = count_prev_30.reindex(entity_ids, fill_value=0)
    sum_price_prev_30 = sum_price_prev_30.reindex(entity_ids, fill_value=0)
    
    # Добавим тренд в features
    features["count_trend_30d"] = features["count_30d"].fillna(0) - count_prev_30
    features["sum_price_trend_30d"] = features["sum_price_30d"].fillna(0) - sum_price_prev_30
    
    # Вычислим возраст и давность последней продажи
    # Возраст: дней с первой продажи
    first_sale = df.groupby("product_id")["order_purchase_timestamp"].min()
    last_sale = df.groupby("product_id")["order_purchase_timestamp"].max()
    age_days = (seed_time - first_sale).dt.days
    recency_days = (seed_time - last_sale).dt.days
    # Заполняем отсутствующих (нет продаж) нулями? Лучше большим числом, например, 9999
    age_days = age_days.reindex(entity_ids, fill_value=0)
    recency_days = recency_days.reindex(entity_ids, fill_value=0)
    features["age_days"] = age_days
    features["recency_days"] = recency_days
    
    # Добавим общее количество заказов за всё время (уже есть count_None_d? У нас окно None дало count_None_d, переименуем)
    # Приведём имена в порядок: заменим "None" на "all"
    features.columns = [col.replace("None", "all") for col in features.columns]
    
    # Заполним NaN нулями (могут возникнуть из-за отсутствия данных в некоторых окнах)
    features = features.fillna(0)
    
    # Реиндексируем по entity_ids
    features = features.reindex(entity_ids)
    
    # Убедимся, что все столбцы числовые (уже)
    return features