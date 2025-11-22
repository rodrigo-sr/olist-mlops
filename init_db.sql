CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO airflow;

CREATE TABLE IF NOT EXISTS public.control_files_processed (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    reference_year CHAR(4) NOT NULL,
    reference_month CHAR(2) NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'SUCCESS',
    
    -- Constraint para garantir que não haja duplicidade
    CONSTRAINT unique_processing UNIQUE (table_name, reference_year, reference_month)
);