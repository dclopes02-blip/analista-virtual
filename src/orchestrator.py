import time
from typing import Any

from src.agents import (
    get_gemini_client,
    run_analyst_agent,
    run_orchestrator_agent,
    run_sql_agent,
)


def answer_question(pergunta: str, max_tentativas_sql: int = 3) -> dict[str, Any]:
    """
    Executa o fluxo completo do Analista Virtual:
    Pergunta -> Orquestrador -> Agente SQL (até 3 tentativas) -> Agente Analista
    """
    inicio = time.perf_counter()
    client = get_gemini_client()

    tokens_in = 0
    tokens_out = 0
    modelos_usados: list[str] = []
    etapas_executadas: list[str] = []

    # 1. Agente Orquestrador
    etapas_executadas.append("orquestrador")
    orq = run_orchestrator_agent(pergunta, client=client)
    tokens_in += orq["tokens_entrada"]
    tokens_out += orq["tokens_saida"]
    modelos_usados.append(orq["modelo_usado"])

    classificacao = orq["classificacao"]

    # Desvio seguro para perguntas fora de escopo ou ambíguas
    if classificacao in {"fora_de_escopo", "ambigua"}:
        tempo_s = round(time.perf_counter() - inicio, 2)
        return {
            "sucesso": True,
            "classificacao": classificacao,
            "motivo": orq["motivo"],
            "pergunta": pergunta,
            "resposta": orq["mensagem_usuario"],
            "sql_executado": None,
            "colunas": [],
            "dados": [],
            "tentativas_sql": 0,
            "tentativas_correcao": 0,
            "sql_valido_primeira_tentativa": None,
            "etapas_executadas": etapas_executadas,
            "ferramentas_chamadas": orq["ferramentas_chamadas"],
            "modelos_usados": list(dict.fromkeys(modelos_usados)),
            "tempo_segundos": tempo_s,
            "tokens_entrada": tokens_in,
            "tokens_saida": tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "erro": None,
        }

    # 2. Agente SQL (validação + execução somente leitura, até 3 tentativas)
    etapas_executadas.append("agente_sql")
    sql_res = run_sql_agent(
        pergunta=pergunta,
        plano_consulta=orq["plano_consulta"],
        contexto_semantico_bruto=orq["contexto_semantico_bruto"],
        max_tentativas=max_tentativas_sql,
        client=client,
    )
    tokens_in += sql_res["tokens_entrada"]
    tokens_out += sql_res["tokens_saida"]
    modelos_usados.append(sql_res["modelo_usado"])

    todas_ferramentas = orq["ferramentas_chamadas"] + sql_res["ferramentas_chamadas"]

    if not sql_res["ok"]:
        tempo_s = round(time.perf_counter() - inicio, 2)
        return {
            "sucesso": False,
            "classificacao": classificacao,
            "motivo": orq["motivo"],
            "pergunta": pergunta,
            "resposta": (
                "Não foi possível executar uma consulta SQL válida após "
                f"{sql_res['tentativas_sql']} tentativa(s). "
                f"Último erro registrado: {sql_res['erro']}"
            ),
            "sql_executado": sql_res["sql_executado"],
            "colunas": [],
            "dados": [],
            "tentativas_sql": sql_res["tentativas_sql"],
            "tentativas_correcao": sql_res["tentativas_correcao"],
            "sql_valido_primeira_tentativa": sql_res["sql_valido_primeira_tentativa"],
            "etapas_executadas": etapas_executadas,
            "ferramentas_chamadas": todas_ferramentas,
            "modelos_usados": list(dict.fromkeys(modelos_usados)),
            "tempo_segundos": tempo_s,
            "tokens_entrada": tokens_in,
            "tokens_saida": tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "erro": sql_res["erro"],
        }

    # 3. Agente Analista (síntese de negócio)
    etapas_executadas.append("agente_analista")
    analista = run_analyst_agent(
        pergunta=pergunta,
        sql_executado=sql_res["sql_executado"],
        colunas=sql_res["colunas"],
        linhas=sql_res["linhas"],
        client=client,
    )
    tokens_in += analista["tokens_entrada"]
    tokens_out += analista["tokens_saida"]
    modelos_usados.append(analista["modelo_usado"])

    tempo_s = round(time.perf_counter() - inicio, 2)
    return {
        "sucesso": True,
        "classificacao": classificacao,
        "motivo": orq["motivo"],
        "pergunta": pergunta,
        "resposta": analista["resposta"],
        "sql_executado": sql_res["sql_executado"],
        "colunas": sql_res["colunas"],
        "dados": sql_res["linhas"],
        "tentativas_sql": sql_res["tentativas_sql"],
        "tentativas_correcao": sql_res["tentativas_correcao"],
        "sql_valido_primeira_tentativa": sql_res["sql_valido_primeira_tentativa"],
        "etapas_executadas": etapas_executadas,
        "ferramentas_chamadas": todas_ferramentas,
        "modelos_usados": list(dict.fromkeys(modelos_usados)),
        "tempo_segundos": tempo_s,
        "tokens_entrada": tokens_in,
        "tokens_saida": tokens_out,
        "tokens_total": tokens_in + tokens_out,
        "erro": None,
    }


if __name__ == "__main__":
    # Ensaio do Checkpoint 2: 3 perguntas do Checkpoint 1 + 1 fora de escopo + 1 ambígua
    perguntas_demo = [
        "Qual foi o faturamento total e o lucro total da operação em 2017?",
        "Qual é o tíquete médio por pedido em cada região dos Estados Unidos?",
        "Como a margem de lucro varia entre vendas sem desconto, com desconto moderado (até 20%) e com desconto alto (acima de 20%)?",
        "Qual é o estoque atual de cadeiras no depósito?",
        "Qual foi a pior região da empresa em 2017?",
    ]

    for idx, pergunta in enumerate(perguntas_demo, start=1):
        print("=" * 85)
        print(f"DEMO {idx}: {pergunta}")
        res = answer_question(pergunta)
        print(
            f"Classificação: {res['classificacao']} | Etapas: {res['etapas_executadas']} | "
            f"SQL 1ª tent.: {res['sql_valido_primeira_tentativa']} | "
            f"Correções: {res['tentativas_correcao']} | "
            f"Tempo: {res['tempo_segundos']}s | Tokens: {res['tokens_total']}"
        )
        print(f"SQL executado:\n{res['sql_executado']}")
        print(f"\nResposta:\n{res['resposta']}\n")