
def prefilter_items(data, take_n_popular=5000):
    """Предфильтрация товаров"""

    # 1. Удаление товаров, со средней ценой < 1$
    data = data[data['sales_value'] >= 1.]

    # 2. Удаление товаров со соедней ценой > 30$
    data = data[data['sales_value'] <= 30.]

    # 3. Придумайте свой фильтр
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data_train = data_train[~data_train['item_id'].isin(top_notpopular)]

    # 4. Выбор топ-N самых популярных товаров (N = take_n_popular)
    popularity = data_train.groupby('item_id')['quantity'].count().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_N = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    data_train = data_train[data_train['item_id'].isin(top_N)]
    items = data_train['item_id'].unique().tolist()

    return items