import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.tools import get_metric, get_schema, run_sql, validate_sql

load_dotenv()

PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite")


def get_gemini_client() -> genai.Client:
    """Cria o cliente oficial da API do Gemini a partir da variável de ambiente."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "cole_sua_chave_aqui":
        raise ValueError("Configure uma GEMINI_API_KEY válida no arquivo .env.")
    return genai.Client(api_key=api_key)


# --- Adaptadores das ferramentas ---

def tool_get_schema(table_name: str = "") -> str:
    """
    Consulta a camada semântica para obter contexto geral, tabelas, colunas,
    relacionamentos, tradução de valores categóricos e regras de negócio.
    Passe string vazia para obter o esquema completo.
    """
    return get_schema(table_name.strip() or None)


def tool_get_metric(metric_name: str = "") -> str:
    """
    Consulta a fórmula SQL oficial e os sinônimos de uma métrica de negócio na
    camada semântica. Passe string vazia para listar todas as métricas.
    """
    return get_metric(metric_name.strip() or None)


def tool_validate_sql(sql: str) -> dict[str, Any]:
    """
    Valida uma consulta SQL antes da execução (somente SELECT, tabelas válidas,
    regras de negócio, LIMIT padrão e EXPLAIN no DuckDB).
    """
    return validate_sql(sql)


def tool_run_sql(sql: str) -> dict[str, Any]:
    """
    Executa uma consulta SELECT validada no DuckDB em modo somente leitura e
    retorna as colunas, linhas e o SQL exato executado.
    """
    return run_sql(sql)


ORCHESTRATOR_TOOLS = [tool_get_schema, tool_get_metric]
SQL_AGENT_TOOLS = [tool_validate_sql, tool_run_sql]

ALL_TOOL_MAP = {
    fn.__name__: fn for fn in (ORCHESTRATOR_TOOLS + SQL_AGENT_TOOLS)
}


def call_gemini_with_fallback(
    client: genai.Client,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
) -> tuple[Any, str]:
    """
    Chama o Gemini com retentativa para erros temporários (503/429 por minuto)
    e troca imediata para o modelo reserva em caso de cota diária esgotada.
    """
    modelos = [PRIMARY_MODEL]
    if FALLBACK_MODEL and FALLBACK_MODEL != PRIMARY_MODEL:
        modelos.append(FALLBACK_MODEL)

    ultimo_erro: Exception | None = None

    for modelo in modelos:
        for tentativa in range(1, 3):
            try:
                response = client.models.generate_content(
                    model=modelo,
                    contents=contents,
                    config=config,
                )
                return response, modelo
            except APIError as e:
                ultimo_erro = e
                code = getattr(e, "code", None)
                msg = str(e)
                # Se a cota diária acabou, pula direto para o modelo reserva
                if code == 429 and ("PerDay" in msg or "per day" in msg.lower()):
                    print(f"[Aviso] Cota diária esgotada em {modelo}. Alternando modelo...")
                    break
                if code in (429, 500, 503, 504):
                    espera = 4 * tentativa
                    print(
                        f"[Aviso] Modelo {modelo} retornou {code} "
                        f"(tentativa {tentativa}/2). Aguardando {espera}s..."
                    )
                    time.sleep(espera)
                else:
                    raise

    raise RuntimeError(f"Falha ao chamar os modelos Gemini ({modelos}): {ultimo_erro}")


def _extract_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return 0, 0
    return (
        getattr(usage, "prompt_token_count", 0) or 0,
        getattr(usage, "candidates_token_count", 0) or 0,
    )


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extrai um objeto JSON da resposta textual do modelo com tolerância a markdown."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


# ============================================================================
# AGENTE 1: ORQUESTRADOR
# ============================================================================

ORCHESTRATOR_PROMPT = """
Você é o Agente Orquestrador de um Analista Virtual de BI sobre uma base de varejo
(Superstore) em DuckDB.

Suas responsabilidades exclusivas:
1. Consultar a camada semântica chamando `tool_get_schema` (com string vazia para
   obter o esquema completo) e `tool_get_metric` (com string vazia para consultar
   todas as métricas oficiais) JUNTAS no mesmo turno ANTES de tomar qualquer decisão.
2. Classificar a pergunta do usuário em exatamente uma das 3 categorias:
   - "dados": pergunta clara e respondível com as tabelas e período disponíveis (2014–2017).
   - "fora_de_escopo": pergunta sobre dados inexistentes na base (anos fora de 2014–2017,
     outros países, estoque, devoluções, cancelamentos, fornecedores, marketing, nomes de
     clientes ou assuntos não relacionados à base).
   - "ambigua": pergunta genuinamente subespecificada em que falta um critério essencial
     para escrever o SQL (ex.: pedir "melhor" ou "pior" produto/região sem definir se o
     critério é faturamento, lucro, margem ou volume). Não classifique como ambígua se a
     pergunta já deixar a métrica e o recorte claros (ex.: se não especificar ano, considere
     todo o período 2014–2017).
3. Responder OBRIGATORIAMENTE apenas com um JSON válido, sem texto fora do JSON, no formato:
{
  "classificacao": "dados" | "fora_de_escopo" | "ambigua",
  "motivo": "explicação breve da classificação",
  "mensagem_usuario": "Se fora_de_escopo, recuse com educação explicando o que há na base e sugira uma pergunta válida. Se ambigua, peça esclarecimento listando as opções possíveis de métrica/critério. Se dados, deixe string vazia.",
  "plano_consulta": "Se dados, descreva objetivamente quais tabelas, JOINs, métricas oficiais, traduções de valores em inglês e filtros de período/negócio o Agente SQL deve usar. Caso contrário, deixe string vazia."
}
""".strip()


def run_orchestrator_agent(
    pergunta: str, client: genai.Client | None = None
) -> dict[str, Any]:
    """Agente 1: classifica a pergunta e coleta contexto na camada semântica."""
    client = client or get_gemini_client()
    config = types.GenerateContentConfig(
        system_instruction=ORCHESTRATOR_PROMPT,
        tools=ORCHESTRATOR_TOOLS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.0,
    )

    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=pergunta)])
    ]

    ferramentas_chamadas: list[str] = []
    contexto_ferramentas: list[str] = []
    tokens_in, tokens_out = 0, 0
    modelo_usado = PRIMARY_MODEL
    texto_final = ""

    for _ in range(5):
        response, modelo_usado = call_gemini_with_fallback(client, contents, config)
        tin, tout = _extract_tokens(response)
        tokens_in += tin
        tokens_out += tout

        if not response.function_calls:
            texto_final = (response.text or "").strip()
            break

        if response.candidates and response.candidates[0].content:
            contents.append(response.candidates[0].content)

        tool_parts: list[types.Part] = []
        for call in response.function_calls:
            nome = call.name
            args = dict(call.args) if call.args else {}
            ferramentas_chamadas.append(nome)
            fn = ALL_TOOL_MAP.get(nome)
            resultado = fn(**args) if fn else f"Ferramenta não permitida: {nome}"

            contexto_ferramentas.append(f"### {nome}({args})\n{resultado}")
            tool_parts.append(
                types.Part.from_function_response(
                    name=nome, response={"resultado": resultado}
                )
            )

        contents.append(types.Content(role="user", parts=tool_parts))

    parsed = _parse_json_response(texto_final)
    classificacao = str(parsed.get("classificacao", "")).strip().lower()
    if classificacao not in {"dados", "fora_de_escopo", "ambigua"}:
        classificacao = "dados" if parsed.get("plano_consulta") else "fora_de_escopo"

    return {
        "classificacao": classificacao,
        "motivo": parsed.get("motivo", ""),
        "mensagem_usuario": parsed.get("mensagem_usuario", texto_final),
        "plano_consulta": parsed.get("plano_consulta", ""),
        "contexto_semantico_bruto": "\n\n".join(contexto_ferramentas),
        "ferramentas_chamadas": ferramentas_chamadas,
        "modelo_usado": modelo_usado,
        "tokens_entrada": tokens_in,
        "tokens_saida": tokens_out,
    }


# ============================================================================
# AGENTE 2: AGENTE SQL
# ============================================================================

SQL_AGENT_PROMPT = """
Você é o Agente SQL de um Analista Virtual de BI especializado em DuckDB.

Suas responsabilidades exclusivas:
1. Escrever UMA ÚNICA consulta SQL (somente SELECT ou WITH ... SELECT) que responda
   diretamente e exatamente à pergunta do usuário, usando o contexto da camada semântica
   fornecido pelo Orquestrador.
2. Seguir rigorosamente todas as regras da camada semântica:
   - Usar apenas as tabelas e colunas existentes.
   - Fazer JOINs usando sempre product_sk e location_sk (nunca product_id ou postal_code).
   - Nunca calcular sales * quantity (sales já é o valor total da linha).
   - Usar COUNT(DISTINCT order_id) para contar pedidos.
   - Usar valores de filtro exatamente em inglês conforme mapeado na camada semântica.
   - Usar discount em formato decimal (ex.: 0.20 para 20%).
   - Arredondar métricas monetárias e percentuais com ROUND(..., 2).
   - Em análises de variação percentual entre períodos (crescimento ou queda), manter
     sempre o sinal padrão: ROUND(100.0 * (valor_atual - valor_anterior) / NULLIF(valor_anterior, 0), 2)
     (positivo para aumento, negativo para queda).
   - Em consultas de variação ou participação percentual, retornar também os valores
     absolutos de base junto ao percentual para dar contexto de escala ao gestor.
3. Fluxo obrigatório de ferramentas:
   - Passo 1: chame `tool_validate_sql` com a sua consulta.
   - Passo 2: somente se a validação retornar `valido: true`, chame `tool_run_sql`.
   - Se `tool_validate_sql` ou `tool_run_sql` retornar erro, corrija o SQL e tente
     novamente (você tem no máximo 3 tentativas no total).
   - Não execute consultas extras além do necessário para responder à pergunta.
4. Após `tool_run_sql` retornar `ok: true`, encerre respondendo brevemente "OK".
""".strip()


def run_sql_agent(
    pergunta: str,
    plano_consulta: str,
    contexto_semantico_bruto: str,
    max_tentativas: int = 3,
    client: genai.Client | None = None,
) -> dict[str, Any]:
    """Agente 2: gera, valida e executa SQL em modo somente leitura (até 3 tentativas)."""
    client = client or get_gemini_client()
    config = types.GenerateContentConfig(
        system_instruction=SQL_AGENT_PROMPT,
        tools=SQL_AGENT_TOOLS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.0,
    )

    user_prompt = (
        f"Pergunta do usuário:\n{pergunta}\n\n"
        f"Plano indicado pelo Orquestrador:\n{plano_consulta}\n\n"
        f"Contexto oficial da camada semântica:\n{contexto_semantico_bruto}"
    )

    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
    ]

    ferramentas_chamadas: list[str] = []
    tokens_in, tokens_out = 0, 0
    modelo_usado = PRIMARY_MODEL

    falhas = 0
    primeira_tentativa_avaliada = False
    sql_valido_primeira_tentativa = False

    execucao_ok = False
    sql_executado: str | None = None
    colunas: list[str] = []
    linhas: list[dict[str, Any]] = []
    ultimo_erro: str | None = None

    # Cada tentativa pode consumir 2 passos (validate_sql + run_sql) + margem
    max_steps = max_tentativas * 3 + 2

    for _ in range(max_steps):
        response, modelo_usado = call_gemini_with_fallback(client, contents, config)
        tin, tout = _extract_tokens(response)
        tokens_in += tin
        tokens_out += tout

        if not response.function_calls:
            break

        if response.candidates and response.candidates[0].content:
            contents.append(response.candidates[0].content)

        tool_parts: list[types.Part] = []
        falhou_no_turno = False
        sucesso_com_dados = False

        for call in response.function_calls:
            nome = call.name
            args = dict(call.args) if call.args else {}
            ferramentas_chamadas.append(nome)

            fn = ALL_TOOL_MAP.get(nome)
            if fn is None or fn not in SQL_AGENT_TOOLS:
                resultado: dict[str, Any] = {
                    "ok": False,
                    "erro": f"Ferramenta não permitida para o Agente SQL: {nome}",
                }
            else:
                resultado = fn(**args)

            if nome == "tool_validate_sql":
                if resultado.get("valido"):
                    if not primeira_tentativa_avaliada:
                        sql_valido_primeira_tentativa = True
                        primeira_tentativa_avaliada = True
                else:
                    falhou_no_turno = True
                    ultimo_erro = resultado.get("erro", "Erro de validação SQL.")
                    if not primeira_tentativa_avaliada:
                        sql_valido_primeira_tentativa = False
                        primeira_tentativa_avaliada = True

            elif nome == "tool_run_sql":
                if resultado.get("ok"):
                    if not primeira_tentativa_avaliada:
                        sql_valido_primeira_tentativa = True
                        primeira_tentativa_avaliada = True
                    execucao_ok = True
                    sql_executado = resultado.get("sql_executado")
                    colunas = resultado.get("colunas", [])
                    linhas = resultado.get("linhas", [])
                    ultimo_erro = None
                    if resultado.get("total_linhas", 0) > 0:
                        sucesso_com_dados = True
                else:
                    falhou_no_turno = True
                    ultimo_erro = resultado.get("erro", "Erro de execução SQL.")
                    sql_executado = resultado.get("sql_executado")
                    if not primeira_tentativa_avaliada:
                        sql_valido_primeira_tentativa = False
                        primeira_tentativa_avaliada = True

            tool_parts.append(
                types.Part.from_function_response(name=nome, response=resultado)
            )

        if falhou_no_turno:
            falhas += 1

        if sucesso_com_dados:
            break

        if falhas >= max_tentativas:
            break

        contents.append(types.Content(role="user", parts=tool_parts))

    tentativas_sql = min(falhas + (1 if execucao_ok else 0), max_tentativas)
    tentativas_correcao = falhas if execucao_ok else max(0, tentativas_sql - 1)

    return {
        "ok": execucao_ok,
        "sql_executado": sql_executado,
        "colunas": colunas,
        "linhas": linhas,
        "total_linhas": len(linhas),
        "erro": ultimo_erro if not execucao_ok else None,
        "tentativas_sql": tentativas_sql,
        "tentativas_correcao": tentativas_correcao,
        "sql_valido_primeira_tentativa": sql_valido_primeira_tentativa,
        "ferramentas_chamadas": ferramentas_chamadas,
        "modelo_usado": modelo_usado,
        "tokens_entrada": tokens_in,
        "tokens_saida": tokens_out,
    }


# ============================================================================
# AGENTE 3: AGENTE ANALISTA
# ============================================================================

ANALYST_AGENT_PROMPT = """
Você é o Agente Analista de Negócios de um Analista Virtual de BI.

Você recebe: (a) a pergunta do gestor, (b) a consulta SQL executada e (c) os dados exatos
retornados do DuckDB. Sua tarefa é transformar esses dados em uma resposta executiva clara
em português.

Regras rigorosas:
1. Responda diretamente à pergunta usando os números exatos retornados.
2. Formate todos os valores monetários em dólares americanos com o prefixo `US$`
   (ex.: US$ 733.215,26) e percentuais com `%`.
3. Destaque comparações, destaques, rankings ou tendências presentes NOS DADOS RETORNADOS.
4. NUNCA invente causas, explicações externas ou números que não estejam presentes no
   resultado da consulta (por exemplo, não diga que uma variação foi causada por descontos
   ou por outra dimensão se essa coluna não foi consultada).
5. Inclua ressalvas objetivas quando relevante (ex.: se o resultado veio vazio, se o limite
   de 100 linhas foi atingido, ou o recorte temporal aplicado).
6. Finalize com exatamente 1 sugestão curta e útil de próxima pergunta de negócio que possa
   ser respondida com esta mesma base (2014–2017).
7. Não inclua o bloco de código SQL na sua resposta textual, pois a interface já o exibe
   separadamente para o usuário.
""".strip()


def run_analyst_agent(
    pergunta: str,
    sql_executado: str,
    colunas: list[str],
    linhas: list[dict[str, Any]],
    client: genai.Client | None = None,
) -> dict[str, Any]:
    """Agente 3: transforma o resultado da consulta em uma resposta de negócio."""
    client = client or get_gemini_client()
    config = types.GenerateContentConfig(
        system_instruction=ANALYST_AGENT_PROMPT,
        temperature=0.2,
    )

    amostra_linhas = linhas[:50]
    nota_truncamento = (
        f"(Exibindo as primeiras 50 de {len(linhas)} linhas retornadas.)"
        if len(linhas) > 50
        else f"(Total de {len(linhas)} linha(s) retornada(s).)"
    )

    prompt_analise = (
        f"Pergunta do gestor:\n{pergunta}\n\n"
        f"SQL executado:\n{sql_executado}\n\n"
        f"Colunas: {colunas}\n"
        f"Resultado {nota_truncamento}:\n"
        f"{json.dumps(amostra_linhas, ensure_ascii=False, indent=2)}"
    )

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt_analise)])
    ]
    response, modelo_usado = call_gemini_with_fallback(client, contents, config)
    tin, tout = _extract_tokens(response)

    return {
        "resposta": (response.text or "").strip(),
        "modelo_usado": modelo_usado,
        "tokens_entrada": tin,
        "tokens_saida": tout,
    }