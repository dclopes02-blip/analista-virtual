# Registro de Uso de IA no Desenvolvimento

Este documento registra de forma transparente como ferramentas de inteligência artificial generativa (Gemini) foram utilizadas como apoio técnico ao longo do desenvolvimento do projeto **Analista Virtual de BI**.

## 1. Papel da IA vs. Decisões Humanas

A IA atuou como apoio iterativo na estruturação do código, revisão de regras de negócio, elaboração de testes e redação técnica. Todas as execuções, validações no banco de dados, escolhas arquiteturais, restrições de orçamento (custo zero) e auditorias manuais de saída foram conduzidas localmente no ambiente de desenvolvimento.

## 2. Uso por Etapa do Projeto

- **Etapas 1 e 2 (Dados e Camada Semântica):** apoio na definição do esquema estrela em DuckDB (`src/load_data.py`), remoção de dados pessoais (`Customer Name`), tratamento de registros inconsistentes e redação do contrato de métricas e traduções em `src/semantic_layer.yaml`. Todos os testes de integridade referencial e contagem de linhas foram executados e conferidos localmente.
- **Etapa 3 (Ferramentas e Arquitetura Multiagente):** apoio na implementação das travas determinísticas em `src/tools.py` (somente `SELECT`, bloqueio de `sales * quantity`, validação de chaves substitutas e `LIMIT 100`) e na evolução incremental de 1 agente protótipo para os 3 agentes obrigatórios (`src/agents.py` e `src/orchestrator.py`).
  - *Decisão humana crítica:* após esbarrar no limite diário de 20 requisições do modelo `gemini-3.6-flash` na camada gratuita e na exigência de pré-pagamento de R$ 180,00 no faturamento em nuvem, foi feita a análise das cotas reais no painel do Google AI Studio e a migração controlada para `gemini-3.5-flash-lite` (500 requisições/dia) com `gemini-3.1-flash-lite` como reserva, mantendo o projeto em **US$ 0**.
- **Etapa 4 (Avaliação Objetiva):** apoio na construção do conjunto de avaliação congelado (`eval/golden_set.csv`) e do avaliador automatizado (`eval/run_eval.py`). Após a 1ª rodada (90% de acerto), os dois desvios encontrados (#15 e #17) foram analisados criticamente e corrigidos por meio de regras gerais na camada semântica, elevando o resultado para 100% na 2ª rodada. Além da métrica automática, foi realizada auditoria manual das respostas textuais geradas pelo Agente Analista para identificar pontos sutis de redação e limitações documentadas em `eval/analise_erros.md`.
- **Etapa 5 (Interface e Documentação):** apoio na criação da interface interativa em Streamlit (`src/app.py`), com campo livre de perguntas, atalhos de demonstração e exibição transparente do SQL e da tabela exportável em CSV, além da consolidação do `README.md`.

## 3. Cuidados de Qualidade e Integridade

- O arquivo `eval/golden_set.csv` permaneceu congelado durante todo o desenvolvimento e não foi utilizado como exemplo nos comandos dos agentes.
- Nenhum dado sensível ou chave de API foi inserido no código-fonte ou versionado no repositório (`.env` mantido fora do controle de versão).