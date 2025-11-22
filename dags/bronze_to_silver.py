from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.hooks.base import BaseHook
#boto3
import boto3
from botocore.exceptions import ClientError
from botocore.client import Config

from datetime import datetime
import pandas as pd
import s3fs
import json

#utils
from scripts.utils import (
    process_date_columns, check_unique_combination, check_if_already_processed, log_processed_file,
    get_execution_date
    )

# Configuração das tabelas
TABLES_CONFIG = {
    'orders': {
        'date_cols': [
            'order_purchase_timestamp', 'order_approved_at',
            'order_delivered_carrier_date', 'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ],
        'pk_cols': ['order_id']
    },
    'items': {
        'date_cols': ['shipping_limit_date'],
        'pk_cols': ['order_id', 'order_item_id']
    },
    'payments': {
        'date_cols': [],
        'pk_cols': ['order_id', 'payment_sequential']
    },
    'reviews': {
        'date_cols': ['review_creation_date', 'review_answer_timestamp'],
        'pk_cols': ['review_id', 'order_id']
    }
}

# Configuração de argumentos default
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2017, 1, 1),
    'retries': 2,
}

# Função para verificar se o arquivo existe
def check_file_existance(table_name, **context):
    # execution_date = context['logical_date']
    # year = execution_date.strftime('%Y')
    # month = execution_date.strftime('%m')
    year, month = get_execution_date(context)
    
    # Conexão ao S3 MinIO
    connection = BaseHook.get_connection('aws_default')
    minio_user = connection.login
    minio_pass = connection.password
    
    minio_endpoint = connection.extra_dejson.get('endpoint_url')
    if not minio_endpoint:
        minio_endpoint = "http://minio:9000"
    
    s3_client = boto3.client(
        's3',
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_user,
        aws_secret_access_key=minio_pass,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    
    bucket_name = 'landing'
    file_path = f'{table_name}/year={year}/month={month}/data.csv'
    
    print(f'Check existance of: s3://{bucket_name}/{file_path}')
    
    try:
        s3_client.head_object(Bucket=bucket_name, Key=file_path)
        print("File founded!")
        return f'process_table_{table_name}'
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print("File Not Found! Skip processing...")
            return f'file_not_found_{table_name}'
        else:
            print(f"Error checking file existence: {e}")
            raise e
    


# Função para processar um arquivo
def process_table(table_name, **context):
    # execution_date = context['logical_date']
    # year = execution_date.strftime('%Y')
    # month = execution_date.strftime('%m')
    year, month = get_execution_date(context)
    
    print(f'Processing data for Year: {year}, Month: {month}')
    
    # Conexão ao S3
    connection = BaseHook.get_connection('aws_default')
    
    minio_endpoint = connection.extra_dejson.get('endpoint_url')
    minio_user = connection.login
    minio_pass = connection.password
    
    if not minio_endpoint:
        print("⚠️ Aviso: Endpoint não encontrado na conexão. Usando padrão Docker.")
        minio_endpoint = "http://minio:9000"
    
    fs_options = {
        "key": minio_user,
        "secret": minio_pass,
        "client_kwargs": {'endpoint_url': minio_endpoint}
    }
    
    configs = TABLES_CONFIG.get(table_name, {})
    
    print(f'Connecting to MinIO at {minio_endpoint} with user {minio_user}')
    
    # Caminhos
    source_path = f's3://landing/{table_name}/year={year}/month={month}/data.csv'
    dest_path = f's3://processing/{table_name}/year={year}/month={month}/data.parquet'
    
    if check_if_already_processed(table_name, year, month):
        print(f"SKIPPING: Table {table_name} for {year}-{month} already processed successfully.")
        return
    
    try:
        
        print(f'Reading data from {source_path}')
        
        df = pd.read_csv(source_path, storage_options=fs_options)

        date_columns = configs.get('date_cols', [])
        pk_columns = configs.get('pk_cols', [])

        df_processed = df.copy()
        
        if date_columns:
            df_processed = process_date_columns(df, date_columns)
            
            
        has_unique_pk = check_unique_combination(df_processed, pk_columns)
        
        if not has_unique_pk:
            raise ValueError(f"Primary key constraint violated for table {table_name} on columns {pk_columns}")
        
        print(f'Writing processed data to {dest_path}')
        df_processed.to_parquet(dest_path, index=False, storage_options=fs_options)
        
        print(f'Logging processed file for table {table_name}, Year: {year}, Month: {month}')
        log_processed_file(table_name, year, month)
      
    except Exception as e:
        print(f'Error processing table {table_name}: {e}')
        raise e
    
def log_missing_file(table_name, **context):
    year, month = get_execution_date(context)
    print(f'File for table {table_name} not found on bucket!')
        
with DAG(
    dag_id='bronze_to_silver_v1',
    default_args=default_args,
    schedule_interval='@monthly',
    catchup=False,
    tags=['etl', 'minio', 's3', 'olist'],
    max_active_runs=1,
    max_active_tasks=2
) as dag:
    
    # Tarefas
    for table_name in TABLES_CONFIG.keys():
        
        # Verifica se o arquivo existe e decide qual caminho seguir
        branch_task = BranchPythonOperator(
            task_id=f'check_file_{table_name}',
            python_callable=check_file_existance,
            op_kwargs={'table_name': table_name},
            provide_context=True
        )
        
        # Realiza o processamento da tabela
        process_task = PythonOperator(
            task_id = f'process_table_{table_name}',
            python_callable=process_table,
            op_kwargs={'table_name': table_name},
            provide_context=True
        )
        
        # Irá informar que o arquivo não foi encontrado no bucket
        skip_task = PythonOperator(
            task_id=f'file_not_found_{table_name}',
            python_callable=log_missing_file,
            op_kwargs={'table_name':table_name},
            provide_context=True
        )
        
        # Task para somente unir o processamento até o final da DAG
        join_task = EmptyOperator(
            task_id=f'join_{table_name}',
            trigger_rule='none_failed_min_one_success'
        )
        
        branch_task >> [process_task, skip_task]
        process_task >> join_task
        skip_task >> join_task
        
                
                
                
            
            