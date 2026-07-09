# Agente de Previsão de Churn com Explicação por IA

**Aplicação:** https://projeto-final-2026-1-production.up.railway.app
**Repositório:** https://github.com/Gustavormq/projeto-final-2026-1/tree/main/(1-1_gustavo)
**Equipe:** [Gustavo da Rocha]
**Trilha:** Trilha 1 — Predição tabular, projeto 1.1 (Previsão de Churn)

---

## 1. Definição do problema

### Que dor é essa e por que importa?

Empresas de assinatura/serviço recorrente perdem receita quando clientes cancelam (churn) sem que o time de retenção tenha tempo de agir. Identificar clientes em risco *antes* do cancelamento permite ações preventivas (desconto, contato humano, ajuste de plano) que custam muito menos do que reconquistar um cliente perdido.

### Stakeholders

- **Time de retenção/CS:** usa a previsão para priorizar quais clientes contatar.
- **Cliente final:** afetado indiretamente — o objetivo é que ações de retenção sejam relevantes, não intrusivas.

### Métrica de sucesso

- **De negócio:** proporção de clientes de alto risco identificados corretamente por unidade de esforço do time de retenção (permite priorizar contatos, em vez de abordar a base inteira).
- **Técnica:** recall e F1-score na classe positiva (churn=1), com PR-AUC como métrica complementar — **acurácia foi descartada** como métrica principal por ser enganosa em dataset desbalanceado (ver seção de avaliação).

---

## 2. Como o sistema é montado

### Diagrama de arquitetura

```
Usuário (navegador)
      │
      ▼
Frontend (HTML/JS estático, servido pela própria API)
      │  POST /prever { tenure, MonthlyCharges, TotalCharges, Contract_* }
      ▼
API (FastAPI)
      │
      ├──► Modelo estatístico (Random Forest, scikit-learn)
      │        → probabilidade de churn (0 a 1)
      │
      └──► Agente (LLM)
               ├─ Modo "local": LM Studio (Llama 3 8B Instruct, quantizado)
               └─ Modo "cloud": Anthropic API (Claude)
               → explicação em linguagem natural + sugestão de ação
      │
      ▼
Resposta estruturada (JSON):
{ probabilidade_churn, nivel_risco, explicacao, llm_provider }
```

### Agent/model exploration

Testei três configurações de modelo estatístico antes de decidir:

| Modelo | Recall (churn) | Precision (churn) | F1 |
|---|---|---|---|
| Regressão Logística (padrão) | 0.57 | 0.65 | 0.61 |
| Regressão Logística (class_weight=balanced) | 0.80 | 0.49 | 0.61 |
| Random Forest (balanced, threshold 0.5) | 0.50 | 0.63 | 0.56 |
| **Random Forest (balanced, threshold 0.3) — escolhido** | **0.73** | 0.51 | 0.60 |

Decisão: Random Forest com threshold ajustado para 0.3 (em vez do padrão 0.5), priorizando recall sobre precision — o custo de negócio de não identificar um cliente em risco é maior que o custo de uma ação de retenção desnecessária em um falso positivo. O Random Forest também foi preferido pela feature importance nativa, usada para embasar a explicação gerada pelo agente.

Para o LLM, explorei dois provedores:
- **LM Studio (Llama 3 8B Instruct, quantizado Q4)** rodando localmente — sem custo, mas dependente da máquina do usuário estar ligada e com o servidor ativo.
- **Anthropic API (Claude)** — sem limitação de disponibilidade, mas com custo por token; não há camada gratuita permanente para uso em produção.

### Deployment

- Empacotado via Docker (`Dockerfile` + `docker-compose.yml`), sobe com `docker compose up`.
- API hospedada em Railway
- Frontend servido como arquivo estático pela própria API FastAPI (`StaticFiles`), eliminando a necessidade de um serviço separado.
- **Limitação documentada:** a versão hospedada na nuvem roda em modo `LLM_PROVIDER=local`, que aponta para um servidor LM Studio que só existe na máquina de desenvolvimento. Em produção na nuvem, essa chamada falha por design, e o guardrail de fallback assume — a API continua respondendo com a probabilidade calculada e uma mensagem informando que a explicação detalhada está indisponível. A demonstração completa (com explicação do LLM) foi gravada localmente, onde o LM Studio está ativo.

### CI/CD

"Não implementei CI/CD automatizado neste ciclo por restrição de tempo; o build e testes foram feitos manualmente antes de cada entrega."

---

## 3. Descrição do agente

### Modelo base e ferramentas

- **Modelo estatístico:** Random Forest (scikit-learn), 200 árvores, `class_weight="balanced"`, threshold de decisão ajustado para 0.3.
- **LLM:** Llama 3 8B Instruct (quantizado, via LM Studio) em desenvolvimento/demo; Anthropic Claude como alternativa documentada para produção.
- **Ferramenta do agente:** o LLM não decide a probabilidade — ele recebe o resultado do modelo estatístico como fato dado e apenas interpreta/comunica. Essa separação (ferramenta determinística + raciocínio em linguagem natural) é o que caracteriza o sistema como agente, não apenas um classificador.

### Dados e contexto

- **Dataset:** Telco Customer Churn (IBM), via Kaggle — 7.043 clientes, 21 colunas originais.
- **Licença:** dataset público, uso educacional.
- **Preparo:**
  - Remoção de 11 linhas (0,16% do total) com `TotalCharges` inválido (valor em branco disfarçado de string).
  - One-hot encoding em variáveis categóricas (`drop_first=True`), expandindo para 30 features.
  - Padronização (`StandardScaler`) para o modelo de Regressão Logística (não aplicada ao Random Forest, que não é sensível à escala).
  - Split treino/teste 80/20, estratificado pela variável alvo.

### Guardrails

**Entrada:**
- Validação de schema via Pydantic (FastAPI): tipos e campos obrigatórios verificados automaticamente antes do código de negócio rodar.

**Saída:**
- **Coerência de risco:** identifiquei que o LLM, quando não recebia instrução explícita sobre o nível de risco, por vezes construía narrativas de "sinais de insatisfação" mesmo para clientes de baixo risco — invertendo o sentido do resultado do modelo. Corrigido determinando o enquadramento (alto/moderado/baixo risco) em código Python *antes* de montar o prompt, removendo do LLM a decisão de interpretar a direção do risco.
- **Idioma:** o LLM ocasionalmente respondia em inglês mesmo com prompt em português. Corrigido com instrução explícita de idioma, reforçada no início e no fim do prompt.
- **Fallback:** chamadas ao LLM são protegidas por `try/except` com timeout de 30s. Em caso de falha (serviço indisponível, timeout), a API retorna a probabilidade calculada pelo modelo estatístico (sempre confiável, independente do LLM) e uma mensagem informativa, nunca um erro técnico cru.

### Iterações de prompt e design

1. Prompt inicial: pedia apenas "explique por que o cliente pode estar em risco" — gerava explicações plausíveis mas incoerentes com a probabilidade real (viés de sempre narrar risco).
2. V2: adicionado enquadramento condicional por faixa de probabilidade (alto/moderado/baixo), decidido em código antes do prompt.
3. V3: adicionada instrução explícita de idioma após observar respostas em inglês.
4. V4: adicionado fallback estruturado e suporte a dois provedores de LLM (local/cloud) via variável de ambiente, sem duplicar lógica de negócio.

**O que não funcionou:** a primeira tentativa de usar apenas `class_weight="balanced"` no Random Forest teve efeito bem mais fraco que na Regressão Logística (recall 0.50 vs. 0.80) — decidi não insistir em ajustar mais parâmetros do Random Forest e usar threshold tuning diretamente, que se mostrou mais previsível e fácil de justificar.

---

## 4. Avaliação do sistema

### Performance

- **Dataset de teste:** 1.407 clientes (20% do dataset, split estratificado).
- **Métricas do modelo final (Random Forest, threshold 0.3):** recall 0.73, precision 0.51, F1 0.60 na classe churn.
- **Linha de base:** um classificador que sempre prevê "não cancela" atinge 73,5% de acurácia sem identificar nenhum cliente em risco — usado como referência para justificar por que acurácia não é a métrica adequada aqui.
- **Feature importance (top 3):** `TotalCharges` (0.18), `tenure` (0.16), `MonthlyCharges` (0.15) — indicando que clientes novos e com maior gasto mensal concentram o maior risco.

Latência ponta a ponta: média de 11,89s (medições: 13,09s / 12,97s / 9,61s), utilizando LLM local (LM Studio, Llama 3 8B). A maior parte do tempo é consumida pela geração de texto do LLM — o modelo estatístico (Random Forest) responde em milissegundos. Essa latência é uma limitação conhecida de LLMs locais rodando em GPU de consumidor; a versão com Anthropic API tende a ser significativamente mais rápida, ao custo de depender de um provedor pago.

### UX

A interface permite que o usuário insira dados do cliente e receba, em segundos: a probabilidade numérica, uma classificação de risco (alto/moderado/baixo) com destaque visual (cor), a explicação em linguagem natural e uma sugestão de ação concreta. Quando o LLM está indisponível, o sistema informa isso claramente em vez de travar ou mostrar erro técnico.

### Casos extremos testados

- Cliente com tempo de contrato 0 (novo).
- Cliente com contrato de longo prazo e alto gasto acumulado (baixo risco esperado).
- Indisponibilidade do serviço de LLM (fallback acionado com sucesso).

---

## 5. Demonstração

https://youtu.be/wo9p_HljWs8

---

## 6. Reflexão sobre o que aprendemos

**O que funcionou bem:** a separação entre modelo estatístico (decisão numérica confiável) e LLM (comunicação) evitou que o sistema dependesse do LLM para a parte mais crítica (a própria previsão). Isso tornou o sistema resiliente mesmo quando o LLM falha.

**O que não funcionou como planejado:** o LLM local (LM Studio) demonstrou comportamento instável entre sessões (descarregava o modelo da memória após inatividade, mesmo com o servidor HTTP ainda ativo), o que reforçou a importância prática do guardrail de fallback.

**Próximos passos com mais tempo:**
- Explicabilidade por cliente individual via SHAP (hoje usamos apenas feature importance agregada do modelo).
- Testes de robustez contra prompt injection na entrada.
- Pipeline de CI/CD automatizado.
- Avaliação sistemática de custo/latência comparando LM Studio local vs. Anthropic API em volume.

---

## 7. Impactos e ética

**Quem pode ser prejudicado por um erro do sistema?** Falsos negativos (cliente em risco não identificado) resultam em perda de receita sem chance de ação preventiva — custo para a empresa, não diretamente para o cliente. Falsos positivos podem gerar contato de retenção desnecessário/inoportuno para um cliente que não pretendia cancelar.

**Risco de viés:** o dataset não inclui variáveis demográficas sensíveis (raça, gênero explícito como variável causal relevante), mas `gender` está entre as features usadas — vale investigar se essa variável contribui desproporcionalmente para a decisão em algum subgrupo. [Análise de fairness por gênero: o modelo apresenta recall de 76% para clientes do gênero masculino (n=726) contra 70% para o feminino (n=681) no conjunto de teste — uma diferença de 6 pontos percentuais. Isso significa que, proporcionalmente, mais clientes em risco real do gênero feminino deixam de ser identificados pelo sistema. Embora a diferença não seja extrema, ela indica um viés mensurável que merece atenção: em um cenário de produção, recomendo investigar se essa disparidade decorre de padrões reais nos dados (ex: diferenças de comportamento de churn entre os grupos) ou de um artefato do modelo, e considerar técnicas de mitigação (reponderação por grupo, ajuste de threshold por subgrupo) antes de decisões de retenção serem tomadas com base nesse sistema.]

**Privacidade:** os dados usados são de um dataset público anonimizado; em uso real, dados de clientes (cobrança, tempo de contrato) são sensíveis e exigiriam tratamento conforme LGPD.

**Mitigação:** o sistema não toma decisões automáticas sobre o cliente (como cancelamento ou aumento de cobrança) — ele apenas informa um risco e sugere ação para um humano do time de retenção decidir.

---

## 8. Referências

- Dataset: Telco Customer Churn (IBM), via Kaggle — https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- scikit-learn (modelo estatístico)
- FastAPI (API)
- LM Studio + Llama 3 8B Instruct (LLM local)
- Anthropic API / Claude (LLM cloud, alternativa documentada)
