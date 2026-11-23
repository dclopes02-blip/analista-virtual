import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.orchestrator import answer_question

GOLDEN_SET_PATH = BASE_DIR / "eval" / "golden_set.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "eval" / "resultados.csv"
PAUSA_ENTRE_PERGUNTAS_S = 12


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _row_matches(expected_row: dict[str, Any], actual_row: dict[str, Any]) -> bool:
    """
    Verifica se todos os valores esperados (textos e números com tolerância de
    arredondamento) estão presentes na linha retornada, independentemente do
    nome dado às colunas pelo Agente SQL.
    """
    actual_strs = [
        str(v).strip().lower()
        for v in actual_row.values()
        if isinstance(v, str)
    ]
    actual_nums = [float(v) for v in actual_row.values() if _is_number(v)]

    used_str_idx: set[int] = set()
    used_num_idx: set[int] = set()

    for exp_val in expected_row.values():
        if exp_val is None:
            continue

        if isinstance(exp_val, str):
            alvo = exp_val.strip().lower()
            idx_match = next(
                (i for i, s in enumerate(actual_strs) if i not in used_str_idx and s == alvo),
                None,
            )
            if idx_match is None:
                return False
            used_str_idx.add(idx_match)

        elif _is_number(exp_val):
            alvo_num = float(exp_val)
            tol = max(0.05, abs(alvo_num) * 0.001)
            idx_match = next(
                (
                    i
                    for i, n in enumerate(actual_nums)
                    if i not in used_num_idx and abs(n - alvo_num) <= tol
                ),
                None,
            )
            if idx_match is None:
                return False
            used_num_idx.add(idx_match)

    return True


def avaliar_caso(
    nivel: str, resposta_esperada: str, resultado: dict[str, Any]
) -> tuple[bool, str]:
    """Avalia automaticamente se a saída do sistema bate com o gabarito."""
    classificacao = resultado.get("classificacao")
    sql_exec = resultado.get("sql_executado")

    # Casos de armadilha (recusa fora de escopo ou pedido de esclarecimento)
    if nivel == "armadilha":
        if resposta_esperada.startswith("RECUSA:"):
            if classificacao == "fora_de_escopo" and not sql_exec:
                return True, "Recusa correta (fora_de_escopo sem SQL)"
            return (
                False,
                f"Esperado fora_de_escopo sem SQL; obtido classificacao='{classificacao}', sql={'sim' if sql_exec else 'nao'}",
            )

        if resposta_esperada.startswith("ESCLARECIMENTO:"):
            if classificacao == "ambigua" and not sql_exec:
                return True, "Esclarecimento correto (ambigua sem SQL)"
            return (
                False,
                f"Esperado ambigua sem SQL; obtido classificacao='{classificacao}', sql={'sim' if sql_exec else 'nao'}",
            )

    # Casos de consulta de dados (fácil, médio, difícil)
    if classificacao != "dados":
        return False, f"Classificacao incorreta: '{classificacao}' (esperado 'dados')"

    if not resultado.get("sucesso"):
        return False, f"Falha na execucao SQL: {resultado.get('erro')}"

    actual_rows = resultado.get("dados") or []
    try:
        expected_rows = json.loads(resposta_esperada)
    except json.JSONDecodeError:
        return False, "Erro ao ler JSON da resposta esperada no golden_set.csv"

    if len(actual_rows) != len(expected_rows):
        return (
            False,
            f"Quantidade de linhas diferente (esperado {len(expected_rows)}, obtido {len(actual_rows)})",
        )

    used_actual: set[int] = set()
    for exp_row in expected_rows:
        match_idx = next(
            (
                i
                for i, act_row in enumerate(actual_rows)
                if i not in used_actual and _row_matches(exp_row, act_row)
            ),
            None,
        )
        if match_idx is None:
            return False, f"Valores divergentes para linha esperada: {exp_row}"
        used_actual.add(match_idx)

    return True, "OK"


def imprimir_resumo(registros: list[dict[str, Any]]) -> None:
    total = len(registros)
    acertos = sum(1 for r in registros if r["acertou"])
    taxa_geral = (100.0 * acertos / total) if total else 0.0

    casos_dados = [r for r in registros if r["nivel"] != "armadilha"]
    sql_1a_ok = sum(1 for r in casos_dados if r["sql_valido_1a_tentativa"] is True)
    taxa_sql_1a = (100.0 * sql_1a_ok / len(casos_dados)) if casos_dados else 0.0

    media_correcoes = (
        sum(r["tentativas_correcao"] for r in casos_dados) / len(casos_dados)
        if casos_dados
        else 0.0
    )
    tempo_medio = sum(r["tempo_segundos"] for r in registros) / total if total else 0.0
    tokens_total = sum(r["tokens_total"] for r in registros)
    tokens_medio = tokens_total / total if total else 0.0

    print("\n" + "=" * 85)
    print("RESUMO DA AVALIACAO (GOLDEN SET)")
    print("=" * 85)
    print(f"Taxa de acerto geral           : {acertos}/{total} ({taxa_geral:.1f}%)")

    for nivel in ("facil", "medio", "dificil", "armadilha"):
        sub = [r for r in registros if r["nivel"] == nivel]
        sub_ok = sum(1 for r in sub if r["acertou"])
        pct = (100.0 * sub_ok / len(sub)) if sub else 0.0
        print(f"  - Nivel {nivel:10s}           : {sub_ok}/{len(sub)} ({pct:.1f}%)")

    print(f"SQL valido na 1a tentativa     : {sql_1a_ok}/{len(casos_dados)} ({taxa_sql_1a:.1f}%)")
    print(f"Media de tentativas de correcao: {media_correcoes:.2f}")
    print(f"Tempo medio de resposta        : {tempo_medio:.2f}s")
    print(f"Consumo de tokens              : {tokens_total:,} total ({tokens_medio:,.0f} media/pergunta)")
    print("=" * 85)

    falhas = [r for r in registros if not r["acertou"]]
    if falhas:
        print("\nCASOS COM DIVERGENCIA / ERRO PARA ANALISE:")
        for r in falhas:
            print(f"- [#{r['id']:02d} | {r['nivel']}] {r['pergunta']}")
            print(f"    Motivo   : {r['motivo_avaliacao']}")
            print(f"    Esperado : {r['resposta_esperada']}")
            print(f"    Obtido   : {r['dados_obtidos'] or r['classificacao']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa avaliacao no golden_set.csv")
    parser.add_argument(
        "--saida",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Caminho do CSV de resultados (padrao: eval/resultados.csv)",
    )
    parser.add_argument(
        "--pausa",
        type=int,
        default=PAUSA_ENTRE_PERGUNTAS_S,
        help="Pausa em segundos entre perguntas para respeitar limite RPM (padrao: 12s)",
    )
    args = parser.parse_args()

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        casos = list(csv.DictReader(f))

    fieldnames = [
        "id",
        "nivel",
        "pergunta",
        "acertou",
        "motivo_avaliacao",
        "classificacao",
        "sql_valido_1a_tentativa",
        "tentativas_sql",
        "tentativas_correcao",
        "tempo_segundos",
        "tokens_total",
        "modelos_usados",
        "sql_referencia",
        "sql_gerado",
        "resposta_esperada",
        "dados_obtidos",
        "resposta_agente",
    ]

    registros: list[dict[str, Any]] = []
    args.saida.parent.mkdir(parents=True, exist_ok=True)

    with open(args.saida, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        total = len(casos)
        for idx, caso in enumerate(casos, start=1):
            pergunta = caso["pergunta"]
            nivel = caso["nivel"]
            resposta_esperada = caso["resposta_esperada"]

            print(f"[{idx:02d}/{total} | {nivel:9s}] {pergunta}")
            try:
                res = answer_question(pergunta)
                acertou, motivo = avaliar_caso(nivel, resposta_esperada, res)
            except Exception as e:
                res = {
                    "classificacao": "erro_execucao",
                    "sql_valido_primeira_tentativa": False,
                    "tentativas_sql": 0,
                    "tentativas_correcao": 0,
                    "tempo_segundos": 0.0,
                    "tokens_total": 0,
                    "modelos_usados": [],
                    "sql_executado": None,
                    "dados": [],
                    "resposta": f"EXCECAO: {e}",
                }
                acertou, motivo = False, f"Excecao na execucao: {e}"

            status = "ACERTO" if acertou else "ERRO  "
            print(
                f"   -> [{status}] {motivo} | "
                f"SQL 1a: {res['sql_valido_primeira_tentativa']} | "
                f"Tempo: {res['tempo_segundos']}s | Tokens: {res['tokens_total']}"
            )

            registro = {
                "id": idx,
                "nivel": nivel,
                "pergunta": pergunta,
                "acertou": acertou,
                "motivo_avaliacao": motivo,
                "classificacao": res["classificacao"],
                "sql_valido_1a_tentativa": res["sql_valido_primeira_tentativa"],
                "tentativas_sql": res["tentativas_sql"],
                "tentativas_correcao": res["tentativas_correcao"],
                "tempo_segundos": res["tempo_segundos"],
                "tokens_total": res["tokens_total"],
                "modelos_usados": ",".join(res.get("modelos_usados") or []),
                "sql_referencia": caso["sql_referencia"],
                "sql_gerado": res.get("sql_executado") or "",
                "resposta_esperada": resposta_esperada,
                "dados_obtidos": json.dumps(res.get("dados") or [], ensure_ascii=False),
                "resposta_agente": (res.get("resposta") or "").replace("\n", " "),
            }
            registros.append(registro)
            writer.writerow(registro)
            f_out.flush()

            if idx < total and args.pausa > 0:
                time.sleep(args.pausa)

    imprimir_resumo(registros)
    print(f"\nResultados salvos em: {args.saida}")


if __name__ == "__main__":
    main()