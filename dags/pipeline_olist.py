from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

# Definição do DAG
with DAG(
    dag_id='olist_ml_pipeline_v1',
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['mlops', 'olist']
) as dag:
    
    # Ingestão
    ingest_task = BashOperator(
        task_id='ingest_and_aggregate',
        bash_command='python3 /opt/airflow/dags/scripts/ingest_and_process.py'
    )
    
    # Treinamento do modelo
    train_task = BashOperator(
        task_id='training_model',
        bash_command='python3 /opt/airflow/dags/scripts/train_model.py'
    )
    
    # Orquestação das tasks
    ingest_task >> train_task