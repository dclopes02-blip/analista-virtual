# Análise de Erros e Evolução da Avaliação

## 1. Metodologia e configuração dos testes

A avaliação objetiva foi conduzida sobre o conjunto de referência congelado (`eval/golden_set.csv`), composto por **20 perguntas inéditas** elaboradas antes da escrita dos comandos dos agentes, distribuídas em:

- **7 perguntas fáceis:** agregações e filtros diretos em uma tabela;
- **7 perguntas médias:** junções entre tabela fato e dimensões com filtros temporais, tradução de categorias e métricas compostas;
- **4 perguntas difíceis:** variação percentual entre anos, participação no total via função de janela e comparação contra média via expressão de tabela comum (CTE);
- **2 perguntas armadilha:** 1 pergunta fora de escopo (recusa esperada) e 1 pergunta ambígua (pedido de esclarecimento esperado).

Ambas as rodadas foram executadas em Python puro por meio do script `eval/run_eval.py`, utilizando `gemini-3.5-flash-lite` como modelo principal e `gemini-3.1-flash-lite` como modelo reserva, com intervalo de 12 segundos entre perguntas para respeitar o limite de requisições por minuto da camada gratuita.

---

## 2. Comparativo quantitativo (Antes vs. Depois das melhorias)

| Métrica | 1ª Rodada (Antes) | 2ª Rodada (Depois) | Variação |
| :--- | :---: | :---: | :---: |
| **Taxa de acerto geral** | 18/20 (90,0%) | 20/20 (100,0%) | +10,0 p.p. |
| — Nível fácil | 7/7 (100,0%) | 7/7 (100,0%) | 0,0 p.p. |
| — Nível médio | 7/7 (100,0%) | 7/7 (100,0%) | 0,0 p.p. |
| — Nível difícil | 2/4 (50,0%) | 4/4 (100,0%) | +50,0 p.p. |
| — Nível armadilha | 2/2 (100,0%) | 2/2 (100,0%) | 0,0 p.p. |
| **SQL válido na 1ª tentativa** | 18/18 (100,0%) | 18/18 (100,0%) | 0,0 p.p. |
| **Média de tentativas de correção** | 0,00 | 0,00 | 0,00 |
| **Tempo médio de resposta** | 13,32 s | 12,10 s | -1,22 s (-9,2%) |
| **Consumo total de tokens (20 perguntas)** | 255.277 | 243.643 | -11.634 (-4,6%) |
| **Consumo médio por pergunta** | 12.764 tokens | 12.182 tokens | -582 (-4,6%) |

Os arquivos brutos de ambas as execuções foram preservados em `eval/resultados_antes.csv` e `eval/resultados.csv`.

---

## 3. Diagnóstico das divergências da 1ª rodada e ajustes aplicados

Na 1ª rodada, não houve erros de sintaxe SQL, nomes de colunas inexistentes, junções inválidas nem falhas nas perguntas armadilha. As duas divergências ocorreram exclusivamente em perguntas de nível difícil e estavam ligadas a **convenções de apresentação analítica**, não a erros matemáticos:

### Caso #15 — Ausência de valores absolutos de base
- **Pergunta:** *Quais 5 subcategorias mais cresceram percentualmente em faturamento de 2016 para 2017?*
- **Diagnóstico:** O Agente SQL acertou as 5 subcategorias e os percentuais exatos (`Appliances 64,78%`, `Art 48,69%`, `Binders 46,50%`, `Accessories 43,08%`, `Labels 36,57%`), mas retornou apenas a coluna percentual, omitindo os valores absolutos de faturamento de 2016 e 2017. Sem a escala absoluta, o Agente Analista perde a capacidade de contextualizar o peso financeiro de cada subcategoria para o gestor.

### Caso #17 — Inversão de convenção de sinal em variação percentual
- **Pergunta:** *Quais 3 estados tiveram a maior queda percentual de lucro de 2016 para 2017, considerando apenas estados com lucro positivo em 2016?*
- **Diagnóstico:** O Agente SQL identificou corretamente os 3 estados (`North Carolina`, `Wyoming` e `Maine`) e os lucros exatos de 2016 e 2017, porém expressou a coluna `queda_percentual` como valor positivo (`2050,82` e `100,0`) em vez de manter a convenção contábil e analítica com sinal (`-2050,82%` e `-100,0%`).

### Melhorias implementadas (sem alterar o conjunto de referência)
1. **Camada semântica (`src/semantic_layer.yaml`):** inclusão de duas regras gerais de negócio padronizando (a) o uso da fórmula com sinal `ROUND(100.0 * (valor_atual - valor_anterior) / NULLIF(valor_anterior, 0), 2)` em análises de variação temporal e (b) o retorno obrigatório dos valores absolutos de base junto a percentuais de variação ou participação.
2. **Instrução do Agente SQL (`src/agents.py`):** reforço das mesmas duas boas práticas analíticas na instrução de sistema do agente.
3. **Redução de redundância nas ferramentas (`src/tools.py`):** remoção do bloco duplicado de `regras_de_negocio` na resposta de `get_metric()`, uma vez que `get_schema()` já entrega integralmente essas regras no mesmo turno do Orquestrador.

Na 2ª rodada, ambas as consultas passaram a atender ao padrão esperado sem causar regressão em nenhuma das outras 18 perguntas.

---

## 4. Auditoria qualitativa das respostas textuais

Como o script `eval/run_eval.py` verifica automaticamente a exatidão tabular dos dados retornados pelo SQL e a classificação das armadilhas, realizou-se uma auditoria manual das respostas textuais geradas nas perguntas complexas (#15 a #18) e armadilhas (#19 e #20).

- **Pontos positivos confirmados:** ausência de alucinações numéricas ou causais, uso consistente da moeda `US$`, ressalvas temporais adequadas e perguntas de acompanhamento pertinentes ao negócio. Na #16, o Agente Analista consolidou corretamente a soma das três categorias (`US$ 733.215,26`).
- **Ajustes finos identificados na leitura textual:**
  1. Na pergunta **#17**, o texto combinou substantivos de queda com o sinal negativo (*"queda de -2.050,82%"* e *"redução de -100,0%"*), criando redundância de sinal; a redação ideal seria *"variação de -2.050,82%"* ou *"queda de 2.050,82%"*.
  2. Na pergunta **#19**, a recusa justificou corretamente a ausência do ano de `2018` e de dados de devolução, mas omitiu mencionar que os nomes dos clientes também foram removidos da base por privacidade.

---

## 5. Limitações conhecidas e análise crítica

1. **Influência da instabilidade externa no tempo médio:** a redução de 1,22 s no tempo médio entre a 1ª e a 2ª rodada deve-se majoritariamente à menor incidência de erros temporários da API (4 erros `503` com 16 s de espera acumulada na 1ª rodada contra apenas 1 erro `429` com 4 s de espera na 2ª), e não apenas à economia de tokens.
2. **Custo de tokens em perguntas simples:** mesmo após a redução de 4,6%, consultas diretas (como contagem de clientes) ainda consomem cerca de 12 mil tokens por enviarem a camada semântica integral em cada execução. Em bases corporativas com dezenas de tabelas, seria necessário adotar busca seletiva de metadados e armazenamento temporário de contexto.
3. **Generalização e risco de sobreajuste:** embora as regras inseridas sejam boas práticas genéricas de inteligência de negócios, elas foram formuladas após a inspeção da 1ª rodada. Uma validação complementar com novas perguntas inéditas de variação temporal fortaleceria ainda mais a prova de generalização.
4. **Laço de autocorreção não exercitado na bateria:** como o Agente SQL atingiu 100% de acerto na 1ª tentativa em todas as 36 consultas das duas rodadas, o mecanismo de autocorreção (até 3 tentativas) não foi acionado durante a avaliação automatizada.
5. **Critério de desempate na pergunta #17:** estados sem movimentação em 2017 empatam em `-100,0%` de variação de lucro. Sem um critério secundário explícito na ordenação (como `profit_2016 DESC` ou `state ASC`), a seleção entre estados empatados depende da ordem interna de leitura do banco.
6. **Avaliação textual ainda manual:** o avaliador automático audita a tabela retornada e a classificação, mas não verifica automaticamente nuances linguísticas da resposta final do Agente Analista.

---

## 6. Próximos passos de evolução (trabalhos futuros)

- Incluir no comando do Agente Analista uma regra de estilo proibindo dupla negação em quedas percentuais e reforçar no Orquestrador a listagem de todos os motivos de recusa quando houver múltiplas restrições violadas.
- Implementar recuperação seletiva da camada semântica por tabela/métrica para reduzir o consumo de tokens em perguntas simples.
- Criar testes automatizados específicos de falha controlada no Agente SQL para validar empiricamente a recuperação na 2ª e na 3ª tentativa.
- Adicionar um avaliador automático secundário para verificar aderência de moeda (`US$`), ausência de explicações causais fora dos dados e clareza textual.