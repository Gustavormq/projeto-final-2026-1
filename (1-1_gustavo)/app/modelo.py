import joblib
import pandas as pd

modelo = joblib.load("modelo_churn.pkl")
colunas_modelo = joblib.load("colunas_modelo.pkl")


def prever_probabilidade(dados_cliente: dict) -> float:
    """Recebe os dados de um cliente e retorna a probabilidade de churn (0 a 1)."""
    X_novo = pd.DataFrame([dados_cliente])
    X_novo = X_novo.reindex(columns=colunas_modelo, fill_value=0)
    probabilidade = modelo.predict_proba(X_novo)[:, 1][0]
    return float(probabilidade)