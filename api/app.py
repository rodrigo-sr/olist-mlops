import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Incialização da API
app = FastAPI(title="Olist Stockout Predictor", version="1.0")

# Configuração para conectar ao MLflow
mlflow.set_tracking_uri("http://mlops_mlflow:5000")

# Selecionado o modelo via RUN_ID
RUN_ID = "c652eedf8da842baa6b15dd9c86d638f" 
LOGGED_MODEL = f"runs:/{RUN_ID}/random_forest_baseline"

# Carregar o modelo treinado
print(f"Carregando modelo do MLflow: {LOGGED_MODEL}...")
try:
    model = mlflow.pyfunc.load_model(LOGGED_MODEL)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    model = None

# Definindo o schema de entrado com Pydantic
class ProductHistory(BaseModel):
    sales_qtd_lag_1: float
    sales_qtd_lag_2: float
    sales_qtd_lag_3: float
    price_mean_lag_1: float
    price_mean_lag_2: float
    price_mean_lag_3: float
    sales_qtd_roll_mean_2: float
    sales_qtd_roll_mean_3: float
    sales_qtd_roll_mean_4: float
    price_mean_roll_mean_2: float
    price_mean_roll_mean_3: float
    price_mean_roll_mean_4: float
    sales_qtd_roll_sum_2: float
    sales_qtd_roll_sum_3: float
    sales_qtd_roll_sum_4: float
    price_mean_roll_sum_2: float
    price_mean_roll_sum_3: float
    price_mean_roll_sum_4: float

# Endpoint para predição
@app.post("/predict", tags=["Inference"])
async def predict(data: List[ProductHistory]):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")
    
    # Converter a lista de objetos Pydantic para DataFrame
    # O modelo espera um DataFrame com as colunas certas
    input_df = pd.DataFrame([item.dict() for item in data])
    
    try:
        # Fazer a predição
        prediction = model.predict(input_df)
        
        # Retornar o resultado (0 ou 1)
        return {"stockout_prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na inferência: {str(e)}")

# Rota de Health Check (Para saber se a API está viva)
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}