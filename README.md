# Analista Virtual de BI — Sample Superstore

Sistema multiagente de inteligência de negócios construído em **Python puro**, **DuckDB**, **Streamlit** e **API do Gemini** (camada gratuita, custo total de **US$ 0**). O sistema recebe perguntas de negócio em português, consulta uma camada semântica estruturada, gera e valida consultas SQL em modo somente leitura e devolve respostas executivas com transparência total do código e dos dados retornados.

---

## Arquitetura do Sistema

O fluxo é dividido em **três agentes com papéis distintos** (`src/agents.py`), coordenados em sequência pelo orquestrador (`src/orchestrator.py`):

```mermaid
flowchart LR
    U[Pergunta do Usuário] --> A1[1. Orquestrador]
    A1 -->|get_schema / get_metric| SL[(semantic_layer.yaml)]
    A1 -->|Fora de escopo| R1[Recusa Segura sem SQL]
    A1 -->|Ambígua| R2[Pedido de Esclarecimento]
    A1 -->|Dados| A2[2. Agente SQL]
    A2 -->|validate_sql / run_sql<br/>até 3 tentativas| DB[(DuckDB<br/>Somente Leitura)]
    DB --> A3[3. Agente Analista]
    A3 --> F[Resposta Executiva + SQL + Tabela CSV]