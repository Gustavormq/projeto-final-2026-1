from fastapi.staticfiles import StaticFiles


from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel
from app.agente import gerar_analise

app = FastAPI(title="Agente de Previsão de Churn")


class ClienteInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract_One_year: int = 0
    Contract_Two_year: int = 0


@app.post("/prever")
def prever(cliente: ClienteInput):
    dados = cliente.model_dump()
    # Pydantic não aceita "-" no nome, então ajustamos de volta pro nome real da coluna
    dados["Contract_One year"] = dados.pop("Contract_One_year")
    dados["Contract_Two year"] = dados.pop("Contract_Two_year")
    return gerar_analise(dados)

app.mount("/", StaticFiles(directory="static", html=True), name="static")