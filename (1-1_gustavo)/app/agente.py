import os
import requests
from app.modelo import prever_probabilidade

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def _montar_prompt(dados_cliente: dict, probabilidade: float) -> tuple[str, str]:
    prob_formatada = f"{probabilidade:.1%}"

    if probabilidade >= 0.5:
        nivel_risco = "ALTO risco de cancelamento"
        instrucao = "Explique quais fatores aumentam o risco deste cliente e sugira uma acao de retencao urgente."
    elif probabilidade >= 0.3:
        nivel_risco = "risco MODERADO de cancelamento"
        instrucao = "Explique os fatores de atencao e sugira uma acao preventiva."
    else:
        nivel_risco = "BAIXO risco de cancelamento"
        instrucao = "Explique por que este cliente parece estavel e sugira uma acao de manutencao do relacionamento (nao de retencao urgente)."

    prompt = f"""Responda SEMPRE em portugues do Brasil, independentemente do idioma dos dados.

Voce e um assistente de retencao de clientes. Um modelo de Machine Learning
calculou que um cliente tem {prob_formatada} de probabilidade de churn - classificado como {nivel_risco}.

Dados do cliente:
- Tempo de contrato (tenure): {dados_cliente.get('tenure', 'nao informado')} meses
- Cobranca mensal: R${dados_cliente.get('MonthlyCharges', 'nao informado')}
- Total gasto ate agora: R${dados_cliente.get('TotalCharges', 'nao informado')}

{instrucao}

Nao invente informacoes que nao foram fornecidas. Seja coerente com o nivel de risco informado.
Lembre-se: sua resposta deve ser inteiramente em portugues do Brasil."""

    return prompt, nivel_risco


def _chamar_llm_local(prompt: str) -> str:
    resposta = requests.post(
        LM_STUDIO_URL,
        json={"messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
        timeout=30,
    )
    resposta.raise_for_status()
    return resposta.json()["choices"][0]["message"]["content"]


def _chamar_llm_cloud(prompt: str) -> str:
    resposta = requests.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resposta.raise_for_status()
    return resposta.json()["content"][0]["text"]


def gerar_analise(dados_cliente: dict) -> dict:
    """Orquestra o agente: chama o modelo, monta o prompt, chama o LLM (local ou cloud)."""
    probabilidade = prever_probabilidade(dados_cliente)
    prompt, nivel_risco = _montar_prompt(dados_cliente, probabilidade)

    try:
        if LLM_PROVIDER == "cloud":
            explicacao = _chamar_llm_cloud(prompt)
        else:
            explicacao = _chamar_llm_local(prompt)
    except requests.exceptions.RequestException:
        explicacao = (
            "Nao foi possivel gerar a explicacao detalhada no momento "
            "(servico de IA indisponivel). Consulte a probabilidade calculada acima."
        )

    return {
        "probabilidade_churn": round(probabilidade, 4),
        "nivel_risco": nivel_risco,
        "explicacao": explicacao,
        "llm_provider": LLM_PROVIDER,
    }