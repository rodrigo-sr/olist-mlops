import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix

# Confiiguração do ambiente MLflow
print("Configurando o MLflow...")
mlflow.set_tracking_uri("http://mlops_mlflow:5000/")
mlflow.set_experiment("olist_stockout_prediction")

print("Iniciando o treinamento do modelo...")
# Função para ajsutar o modelo
def train():
    
    print("Lendo os dados...")
    df = pd.read_parquet('/opt/airflow/data/processed/df_features_v1.parquet')
    
    # Separando features e target
    features = ['sales_qtd_lag_1', 'sales_qtd_lag_2', 'sales_qtd_lag_3',
                'price_mean_lag_1', 'price_mean_lag_2', 'price_mean_lag_3',
                'sales_qtd_roll_mean_2', 'sales_qtd_roll_mean_3', 'sales_qtd_roll_mean_4',
                'price_mean_roll_mean_2', 'price_mean_roll_mean_3', 'price_mean_roll_mean_4',
                'sales_qtd_roll_sum_2', 'sales_qtd_roll_sum_3', 'sales_qtd_roll_sum_4',
                'price_mean_roll_sum_2', 'price_mean_roll_sum_3', 'price_mean_roll_sum_4']
    target = 'target_stockout'
    
    X = df[features].astype(float)
    y = df[target]
    
    # Realizar split temporal (train/test)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    with mlflow.start_run():
        
        #hiperparâmetros
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 14
        }
        
        # Inferir assinatura
        signature = mlflow.models.signature.infer_signature(X_train, y_test)
        
        # Exemplo input
        input_example = X_train.head()
        
        # Log dos hiperparâmetros
        mlflow.log_params(params)
        
        # Ajuste do modelo
        print("Ajustando o modelo...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Previsões e avaliação
        y_pred = model.predict(X_test)
        
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
        print(f"Confusion Matrix:\n{cm}")
        
        # Log das Métricas
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        
        # Log do modelo - artefato
        mlflow.sklearn.log_model(model, "random_forest_baseline",
                                 signature=signature,
                                 input_example=input_example)
        
        print("Modelo treinado e registrado com sucesso no MLflow.")
        
        
if __name__ == "__main__":
    train()
        