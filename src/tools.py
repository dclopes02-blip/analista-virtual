from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import re
import threading
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "processed" / "superstore.duckdb"
SEMANTIC_LAYER_PATH = BASE_DIR / "src" / "semantic_layer.yaml"

ALLOWED_TABLES = {"fato_vendas", "dim_clientes", "dim_produtos", "dim_localizacao"}
DEFAULT_LIMIT = 100
DEFAULT_TIMEOUT_SECONDS = 10

FORBIDDEN_AST_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Command,
)


def _load_semantic_layer() -> dict[str, Any]:
    with open(SEMANTIC_LAYER_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clean_sql(sql: str) -> str:
    """Remove blocos markdown (```sql ... ```) e espaços extras."""
    cleaned = sql.strip()
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _serialize_value(val: Any) -> Any:
    """Converte tipos de retorno do DuckDB para tipos nativos serializáveis em JSON."""
    if isinstance(val, (date, datetime)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return float(val)
    return val


def get_schema(table_name: str | None = None) -> str:
    """
    Retorna a documentação da camada semântica: contexto geral, tabelas,
    colunas, relacionamentos, valores categóricos e regras de negócio.
    Se table_name for informado, filtra apenas a tabela solicitada.
    """
    sl = _load_semantic_layer()
    tabelas = sl.get("tabelas", {})

    if table_name:
        key = table_name.strip().lower()
        if key not in tabelas:
            disponiveis = ", ".join(tabelas.keys())
            return (
                f"Tabela '{table_name}' não encontrada na camada semântica. "
                f"Tabelas disponíveis: {disponiveis}."
            )
        tabelas_selecionadas = {key: tabelas[key]}
    else:
        tabelas_selecionadas = tabelas

    payload = {
        "contexto_geral": sl.get("contexto_geral", {}),
        "tabelas": tabelas_selecionadas,
        "relacionamentos": sl.get("relacionamentos", []),
        "valores_categoricos": sl.get("sinonimos", {}).get("valores_categoricos", {}),
        "regras_de_negocio": sl.get("regras_de_negocio", []),
    }
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)


def get_metric(metric_name: str | None = None) -> str:
    """
    Retorna a definição oficial de uma métrica (ou de todas, se metric_name
    for omitido ou 'todas') a partir da camada semântica, incluindo fórmula SQL
    e sinônimos.
    """
    sl = _load_semantic_layer()
    metricas = sl.get("metricas", {})
    sinonimos_metricas = sl.get("sinonimos", {}).get("metricas", {})

    if not metric_name or metric_name.strip().lower() in {"todas", "all", "*"}:
        payload = {
            "metricas": metricas,
            "sinonimos_metricas": sinonimos_metricas,
        }
        return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)

    termo = metric_name.strip().lower()
    encontradas = {}

    for nome, definicao in metricas.items():
        lista_sinonimos = [s.lower() for s in sinonimos_metricas.get(nome, [])]
        if (
            termo == nome.lower()
            or termo in nome.lower()
            or any(termo in s or s in termo for s in lista_sinonimos)
        ):
            encontradas[nome] = {
                **definicao,
                "sinonimos": sinonimos_metricas.get(nome, []),
            }

    if not encontradas:
        disponiveis = ", ".join(metricas.keys())
        return (
            f"Métrica '{metric_name}' não encontrada. "
            f"Métricas oficiais disponíveis: {disponiveis}."
        )

    return yaml.safe_dump(
        {"metricas_encontradas": encontradas}, allow_unicode=True, sort_keys=False
    )


def validate_sql(sql: str) -> dict[str, Any]:
    """
    Valida uma consulta SQL antes da execução:
    - Apenas uma instrução SELECT (somente leitura)
    - Bloqueia INSERT, UPDATE, DELETE, DROP, ALTER, CREATE
    - Verifica tabelas permitidas e regras críticas de negócio/JOIN
    - Aplica LIMIT padrão (100) se ausente
    - Executa EXPLAIN em modo read_only no DuckDB para validar sintaxe e colunas
    """
    cleaned = _clean_sql(sql)
    if not cleaned:
        return {"valido": False, "erro": "A consulta SQL está vazia."}

    # Remove ponto e vírgula final opcional antes do parse
    cleaned_no_semi = cleaned.rstrip(";").strip()

    try:
        statements = sqlglot.parse(cleaned_no_semi, read="duckdb")
    except Exception as e:
        return {"valido": False, "erro": f"Erro de sintaxe SQL: {e}"}

    statements = [st for st in statements if st is not None]
    if len(statements) != 1:
        return {
            "valido": False,
            "erro": "Apenas uma única instrução SELECT por vez é permitida.",
        }

    parsed = statements[0]

    # 1. Bloqueio de comandos de escrita/alteração na árvore sintática
    if not isinstance(parsed, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return {
            "valido": False,
            "erro": "Operação bloqueada: apenas consultas SELECT (somente leitura) são permitidas.",
        }

    for node in parsed.walk():
        if isinstance(node, FORBIDDEN_AST_NODES):
            return {
                "valido": False,
                "erro": "Operação bloqueada: comandos de modificação ou DDL não são permitidos.",
            }

    # 2. Verificação de tabelas permitidas (ignorando aliases de CTEs)
    cte_names = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)}
    referenced_tables = {
        tbl.name.lower()
        for tbl in parsed.find_all(exp.Table)
        if tbl.name and tbl.name.lower() not in cte_names
    }

    if not referenced_tables:
        return {
            "valido": False,
            "erro": "A consulta deve referenciar ao menos uma tabela da base.",
        }

    invalid_tables = referenced_tables - ALLOWED_TABLES
    if invalid_tables:
        return {
            "valido": False,
            "erro": (
                f"Tabela(s) desconhecida(s): {', '.join(sorted(invalid_tables))}. "
                f"Tabelas válidas: {', '.join(sorted(ALLOWED_TABLES))}."
            ),
        }

    # 3. Validação de regras de negócio críticas (JOINs e faturamento)
    for join in parsed.find_all(exp.Join):
        on_clause = join.args.get("on")
        if on_clause:
            join_cols = {col.name.lower() for col in on_clause.find_all(exp.Column)}
            if "product_id" in join_cols:
                return {
                    "valido": False,
                    "erro": (
                        "JOIN incorreto: use product_sk (e nunca product_id) "
                        "para relacionar fato_vendas com dim_produtos."
                    ),
                }
            if "postal_code" in join_cols:
                return {
                    "valido": False,
                    "erro": (
                        "JOIN incorreto: use location_sk (e nunca postal_code) "
                        "para relacionar fato_vendas com dim_localizacao."
                    ),
                }

    for mul in parsed.find_all(exp.Mul):
        mul_cols = {col.name.lower() for col in mul.find_all(exp.Column)}
        if {"sales", "quantity"}.issubset(mul_cols):
            return {
                "valido": False,
                "erro": (
                    "Regra de negócio violada: a coluna sales já representa o valor total "
                    "da linha. Nunca calcule sales * quantity."
                ),
            }

    # 4. Aplica LIMIT padrão caso a consulta externa não defina um LIMIT
    sql_final = cleaned_no_semi
    limit_aplicado = False
    if parsed.args.get("limit") is None:
        sql_final = f"{cleaned_no_semi}\nLIMIT {DEFAULT_LIMIT}"
        limit_aplicado = True

    # 5. Validação real de plano/colunas no DuckDB em modo somente leitura
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        con.execute(f"EXPLAIN {sql_final}")
    except Exception as e:
        return {"valido": False, "erro": f"Erro de validação no DuckDB: {e}"}
    finally:
        con.close()

    return {
        "valido": True,
        "sql_validado": sql_final,
        "limit_padrao_aplicado": limit_aplicado,
    }


def run_sql(sql: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """
    Executa uma consulta SQL validada no DuckDB em modo somente leitura,
    com limite padrão de linhas e timeout de execução.
    """
    validacao = validate_sql(sql)
    if not validacao["valido"]:
        return {"ok": False, "erro": validacao["erro"], "sql_executado": sql}

    sql_final = validacao["sql_validado"]
    con = duckdb.connect(str(DB_PATH), read_only=True)
    timed_out = False

    def _abort_query() -> None:
        nonlocal timed_out
        timed_out = True
        con.interrupt()

    timer = threading.Timer(timeout_seconds, _abort_query)
    timer.start()
    try:
        cursor = con.execute(sql_final)
        colunas = [desc[0] for desc in cursor.description]
        rows_raw = cursor.fetchall()
        linhas = [
            {col: _serialize_value(val) for col, val in zip(colunas, row)}
            for row in rows_raw
        ]
        return {
            "ok": True,
            "sql_executado": sql_final,
            "colunas": colunas,
            "linhas": linhas,
            "total_linhas": len(linhas),
        }
    except Exception as e:
        if timed_out:
            return {
                "ok": False,
                "erro": f"Consulta cancelada por exceder o timeout de {timeout_seconds} segundos.",
                "sql_executado": sql_final,
            }
        return {"ok": False, "erro": f"Erro ao executar SQL: {e}", "sql_executado": sql_final}
    finally:
        timer.cancel()
        con.close()