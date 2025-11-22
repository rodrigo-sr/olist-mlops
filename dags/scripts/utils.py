import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook

# Cast object columns of pandas to dadetime format
def process_date_columns(df, date_columns):
    df_processed = df.copy()
    
    for col in date_columns:
        df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
    
    return df_processed

# Check if the combination of specified columns is unique
def check_unique_combination(df, columns):
    unique_combinations = df[columns].duplicated().sum()
    
    return unique_combinations == 0

# Postgres connection hook
def check_if_already_processed(table_name, year, month):

    pg_hook = PostgresHook(postgres_conn_id='olist_postgres') 
    
    sql = """
        SELECT 1 FROM control_files_processed 
        WHERE table_name = %s AND reference_year = %s AND reference_month = %s
    """
    
    records = pg_hook.get_records(sql, parameters=(table_name, year, str(month).zfill(2)))
    
    # Se a lista não estiver vazia, significa que já processou
    return len(records) > 0

def log_processed_file(table_name, year, month):
    """
    Insere o registro de sucesso na tabela de controle.
    """
    pg_hook = PostgresHook(postgres_conn_id='olist_postgres')
    
    sql = """
        INSERT INTO control_files_processed (table_name, reference_year, reference_month, status)
        VALUES (%s, %s, %s, 'SUCCESS')
        ON CONFLICT (table_name, reference_year, reference_month) 
        DO UPDATE SET processed_at = CURRENT_TIMESTAMP;
    """
    
    pg_hook.run(sql, parameters=(table_name, year, str(month).zfill(2)))
    
def get_execution_date(context_obj):
    execution_date = context_obj['logical_date']
    
    return execution_date.strftime('%Y'), int(execution_date.strftime('%m')) 