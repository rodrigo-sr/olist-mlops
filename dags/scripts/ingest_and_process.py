import pandas as pd
import numpy as np

df_order_items = pd.read_csv('/opt/airflow/data/raw/olist_order_items_dataset.csv')
df_orders = pd.read_csv('/opt/airflow/data/raw/olist_orders_dataset.csv')

df_orders_items_merged = pd.merge(df_orders, df_order_items, on='order_id', how='inner')

print("Merged DataFrame:\n"
      , df_orders_items_merged.head())
print(df_orders_items_merged.columns)

df_orders_delivered = df_orders_items_merged[df_orders_items_merged['order_status'] == 'delivered'].copy()

print(df_orders_delivered.head())

# Cast order_purchase_timestamp to datetime
df_orders_delivered['order_purchase_timestamp'] = pd.to_datetime(df_orders_delivered['order_purchase_timestamp'])

# Criar uma coluna para a semana do ano (year_week)
df_orders_delivered['year_week'] = df_orders_delivered['order_purchase_timestamp'].dt.strftime('%Y-%U')

#group by product_id and year_week to get total sales per week
df_grouped = df_orders_delivered.groupby(['product_id', 'year_week']).agg(
    sales_qtd=('order_item_id', 'count'),
    price_mean=('price', 'mean')
).reset_index()

df_grouped.to_parquet('/opt/airflow/data/processed/df_sales_weekly.parquet', index=False)

# Fix lost weeks

df_grouped = pd.read_parquet('/opt/airflow/data/processed/df_sales_weekly.parquet')
print(df_grouped.head())

df_grouped['product_id'] = df_grouped['product_id'].astype('category')

unique_products = df_grouped['product_id'].unique()
unique_weeks = df_grouped['year_week'].unique()

full_index = pd.MultiIndex.from_product(
    [unique_products, unique_weeks],
    names=['product_id', 'year_week']
)

df_grouped_filled = df_grouped.set_index(['product_id', 'year_week']).reindex(full_index).reset_index()


df_grouped_filled['sales_qtd'] = df_grouped_filled['sales_qtd'].fillna(0).astype(int)
df_grouped_filled['price_mean'] = df_grouped_filled.groupby('product_id', observed=True)['price_mean'].ffill()
df_grouped_filled['price_mean'] = df_grouped_filled['price_mean'].fillna(0)

# Lag features
df_grouped_filled = df_grouped_filled.sort_values(by=['product_id', 'year_week'])

grouper = df_grouped_filled.groupby('product_id', observed=True)

## Sales quantity lags
df_grouped_filled['sales_qtd_lag_1'] = grouper['sales_qtd'].shift(1)
df_grouped_filled['sales_qtd_lag_2'] = grouper['sales_qtd'].shift(2)
df_grouped_filled['sales_qtd_lag_3'] = grouper['sales_qtd'].shift(3)

## Price mean lags
df_grouped_filled['price_mean_lag_1'] = grouper['price_mean'].shift(1)
df_grouped_filled['price_mean_lag_2'] = grouper['price_mean'].shift(2)
df_grouped_filled['price_mean_lag_3'] = grouper['price_mean'].shift(3)  

# Rolling mean features
## Sales quantity rolling means
df_grouped_filled['sales_qtd_roll_mean_2'] = grouper['sales_qtd'].transform(lambda x: x.shift(1).rolling(window=2, min_periods=1).mean())
df_grouped_filled['sales_qtd_roll_mean_3'] = grouper['sales_qtd'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
df_grouped_filled['sales_qtd_roll_mean_4'] = grouper['sales_qtd'].transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())

## Price mean rolling means
df_grouped_filled['price_mean_roll_mean_2'] = grouper['price_mean'].transform(lambda x: x.shift(1).rolling(window=2, min_periods=1).mean())
df_grouped_filled['price_mean_roll_mean_3'] = grouper['price_mean'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
df_grouped_filled['price_mean_roll_mean_4'] = grouper['price_mean'].transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())

# Rolling sum features
## Sales quantity rolling sums
df_grouped_filled['sales_qtd_roll_sum_2'] = grouper['sales_qtd'].transform(lambda x: x.shift(1).rolling(window=2, min_periods=1).sum())
df_grouped_filled['sales_qtd_roll_sum_3'] = grouper['sales_qtd'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).sum())
df_grouped_filled['sales_qtd_roll_sum_4'] = grouper['sales_qtd'].transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).sum())

## Price mean rolling sums
df_grouped_filled['price_mean_roll_sum_2'] = grouper['price_mean'].transform(lambda x: x.shift(1).rolling(window=2, min_periods=1).sum())
df_grouped_filled['price_mean_roll_sum_3'] = grouper['price_mean'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).sum())
df_grouped_filled['price_mean_roll_sum_4'] = grouper['price_mean'].transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).sum())

# Target de Ruptura (Target Engineering)
df_grouped_target = df_grouped_filled.copy()
df_grouped_target['target_stockout'] = np.where((df_grouped_target['sales_qtd'] == 0) &
                                                (df_grouped_target['sales_qtd_roll_mean_4'] > 0), 1, 0)

# Remove NaN from features created by lagging
df_grouped_target = df_grouped_target.dropna(subset=['sales_qtd_lag_1', 'sales_qtd_lag_2', 'sales_qtd_lag_3',
                                                     'price_mean_lag_1', 'price_mean_lag_2', 'price_mean_lag_3'])

print(f"Quantidade de linhas para o modelo: {df_grouped_target.shape[0]}")
print(df_grouped_target[df_grouped_target['sales_qtd_roll_mean_4'] > 0][['product_id', 'year_week', 'sales_qtd', 'sales_qtd_roll_mean_4', 'target_stockout']].tail(10))

df_grouped_target.to_parquet('/opt/airflow/data/processed/df_features_v1.parquet', index=False)