"""
Analista Virtual — Superstore BI (interface Streamlit).

Rodar a partir da raiz do projeto:
    streamlit run src/app.py

Este arquivo cuida apenas da apresentação. O fluxo dos agentes continua em
src/orchestrator.py e src/agents.py, sem alterações.
"""

import csv
import logging
import re
import sys
import time
import uuid
from datetime import date, datetime
from numbers import Number
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import yaml

# Garante que os imports a partir da raiz do projeto funcionem ao rodar `streamlit run src/app.py`
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.orchestrator import answer_question  # noqa: E402

try:
    from src.agents import FALLBACK_MODEL, PRIMARY_MODEL  # noqa: E402
except Exception:  # pragma: no cover - apenas para exibição na aba "Como funciona"
    PRIMARY_MODEL, FALLBACK_MODEL = "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

CACHE_TTL_SEGUNDOS = 24 * 60 * 60   # Respostas bem-sucedidas ficam em cache por 24h
COOLDOWN_SEGUNDOS = 5               # Intervalo mínimo entre chamadas reais à API
LIMITE_LINHAS = 100                 # Mesmo LIMIT imposto pelo validador de SQL
MAX_BARRAS_GRAFICO = 30             # Acima disso o gráfico de barras fica ilegível
TOTAL_REGISTROS = "9.993"           # Linhas da fato_vendas (não consta na camada semântica)

CAMINHO_SEMANTICA = BASE_DIR / "src" / "semantic_layer.yaml"
CAMINHO_AVALIACAO = BASE_DIR / "eval" / "resultados.csv"
CAMINHO_FEEDBACK = BASE_DIR / "eval" / "feedback_usuarios.csv"

# Paleta (validada para daltonismo): azul para valores positivos/séries, vermelho para negativos
COR_SERIE = "#2a78d6"
COR_NEGATIVO = "#e34948"
CORES_NIVEL = {  # ordem fixa das cores categóricas
    "Fácil": "#2a78d6",
    "Médio": "#eb6834",
    "Difícil": "#1baf7a",
    "Armadilha": "#eda100",
}
PALETA_CATEGORICA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ROTULOS_NIVEL = {"facil": "Fácil", "medio": "Médio", "dificil": "Difícil", "armadilha": "Armadilha"}

PERGUNTAS_SUGERIDAS = [
    ("Faturamento e lucro em 2017", "Qual foi o faturamento total e o lucro total da operação em 2017?"),
    ("Faturamento por ano", "Qual foi o faturamento total de cada ano entre 2014 e 2017?"),
    ("Top 5 estados com prejuízo em Móveis", "Quais são os cinco estados com maior prejuízo acumulado na categoria Móveis?"),
    ("Margem por faixa de desconto", "Como a margem de lucro varia entre vendas sem desconto, com desconto moderado (até 20%) e com desconto alto (acima de 20%)?"),
    ("Teste fora de escopo (estoque)", "Qual é o estoque atual de cadeiras no depósito?"),
    ("Teste ambíguo (pior região)", "Qual foi a pior região da empresa em 2017?"),
]

LABELS_CLASSIFICACAO = {
    "dados": "✅ Consulta de dados",
    "fora_de_escopo": "🚫 Fora de escopo (recusa segura)",
    "ambigua": "❓ Pergunta ambígua (pedido de esclarecimento)",
    "falha_sql": "⚠️ Consulta não concluída",
    "erro": "⚠️ Erro na execução",
}

# Palavras nos nomes das colunas usadas para formatar os cards de KPI
PALAVRAS_PERCENTUAL = ("pct", "percent", "margem", "desconto", "participacao", "crescimento", "variacao", "taxa")
PALAVRAS_MONETARIAS = ("faturamento", "lucro", "sales", "profit", "receita", "ticket", "tiquete", "valor", "venda", "prejuizo")
PALAVRAS_DIAS = ("prazo", "dias", "days")
PADRAO_TEMPO = re.compile(
    r"(^|_)(ano|year|mes|month|trimestre|quarter|data|date|periodo|semana|week|dia|day)(_|$)"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("analista_virtual")

st.set_page_config(
    page_title="Analista Virtual — Superstore BI",
    page_icon="📊",
    layout="wide",
)


# =============================================================================
# CAMADA SEMÂNTICA (para exibir o escopo da base sem duplicar informação)
# =============================================================================

@st.cache_data(show_spinner=False)
def carregar_semantica(_mtime: float) -> dict[str, Any]:
    """Lê o YAML da camada semântica. O argumento invalida o cache quando o arquivo muda."""
    try:
        with open(CAMINHO_SEMANTICA, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        logger.exception("Não foi possível ler a camada semântica em %s", CAMINHO_SEMANTICA)
        return {}


def _mtime(caminho: Path) -> float:
    return caminho.stat().st_mtime if caminho.exists() else 0.0


SEMANTICA = carregar_semantica(_mtime(CAMINHO_SEMANTICA))


# =============================================================================
# EXECUÇÃO DOS AGENTES: CACHE, COOLDOWN E ERROS AMIGÁVEIS
# =============================================================================

class _ResultadoNaoCacheavel(Exception):
    """Sinaliza um resultado sem sucesso: exceções não são guardadas pelo st.cache_data."""

    def __init__(self, resultado: dict[str, Any]):
        super().__init__("Resultado sem sucesso não é armazenado em cache.")
        self.resultado = resultado


# Marca, a cada execução do script, se a função cacheada rodou de fato (cache miss)
_CONTROLE_CACHE = {"executou": False}


@st.cache_data(ttl=CACHE_TTL_SEGUNDOS, max_entries=200, show_spinner=False)
def _consultar_agentes(chave_cache: str, _pergunta: str) -> dict[str, Any]:
    """
    Chama o fluxo dos 3 agentes. A chave é a pergunta normalizada; o parâmetro com
    underscore fica fora do hash, então o orquestrador recebe o texto original.
    """
    _CONTROLE_CACHE["executou"] = True
    resultado = answer_question(_pergunta)
    if not resultado.get("sucesso"):
        raise _ResultadoNaoCacheavel(resultado)
    return resultado


def normalizar_pergunta(pergunta: str) -> str:
    return " ".join(pergunta.split()).lower()


def mensagem_erro_amigavel(exc: Exception) -> str:
    texto = str(exc)
    codigo = getattr(exc, "code", None)

    if isinstance(exc, ValueError) and "GEMINI_API_KEY" in texto:
        return "A chave da API do Gemini não está configurada. Verifique a variável GEMINI_API_KEY no arquivo `.env`."
    if codigo == 429 or re.search(r"\b429\b|RESOURCE_EXHAUSTED", texto):
        return (
            "A cota de uso da API do Gemini foi atingida no momento. "
            "Aguarde alguns instantes e tente novamente."
        )
    if codigo in (500, 503, 504) or re.search(r"\b(500|503|504)\b|UNAVAILABLE", texto):
        return (
            "O serviço de IA está sobrecarregado ou indisponível no momento. "
            "Tente novamente em alguns segundos."
        )
    return "Não foi possível concluir a análise devido a um erro técnico. Os detalhes foram registrados no log."


def executar_pergunta(pergunta: str) -> tuple[dict[str, Any], bool]:
    """Retorna (resultado, veio_do_cache)."""
    _CONTROLE_CACHE["executou"] = False
    try:
        resultado = _consultar_agentes(normalizar_pergunta(pergunta), pergunta)
    except _ResultadoNaoCacheavel as nao_cacheavel:
        resultado = nao_cacheavel.resultado
    except Exception as exc:
        logger.exception("Falha em answer_question para a pergunta: %s", pergunta)
        resultado = {
            "sucesso": False,
            "classificacao": "erro",
            "pergunta": pergunta,
            "resposta": mensagem_erro_amigavel(exc),
        }
    return resultado, not _CONTROLE_CACHE["executou"]


def segundos_restantes_cooldown() -> float:
    ultima = st.session_state.get("ultima_chamada_api", 0.0)
    return max(0.0, COOLDOWN_SEGUNDOS - (time.time() - ultima))


# =============================================================================
# PREPARAÇÃO DOS DADOS RETORNADOS
# =============================================================================

def preparar_dataframe(dados: list[dict[str, Any]]) -> pd.DataFrame | None:
    """Converte as linhas do DuckDB em DataFrame com tipos numéricos e de data corretos."""
    if not dados:
        return None
    df = pd.DataFrame(dados)
    for coluna in df.columns:
        if df[coluna].dtype != object:
            continue
        valores = df[coluna].dropna()
        if valores.empty:
            continue
        if all(isinstance(v, Number) and not isinstance(v, bool) for v in valores):
            df[coluna] = pd.to_numeric(df[coluna])  # ex.: Decimal vindo do DuckDB
        elif all(isinstance(v, (date, datetime)) for v in valores):
            df[coluna] = pd.to_datetime(df[coluna])
    return df


def colunas_numericas(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]


def eh_coluna_tempo(df: pd.DataFrame, coluna: str) -> bool:
    return bool(PADRAO_TEMPO.search(coluna.lower())) or pd.api.types.is_datetime64_any_dtype(df[coluna])


def rotulo_coluna(coluna: str) -> str:
    return coluna.replace("_", " ").strip().capitalize()


def formatar_numero_br(valor: float, casas: int = 2) -> str:
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def formatar_kpi(coluna: str, valor: Any) -> str:
    if valor is None or pd.isna(valor):
        return "—"
    nome = coluna.lower()
    valor = float(valor)
    if any(p in nome for p in PALAVRAS_PERCENTUAL):
        return f"{formatar_numero_br(valor)}%"
    if any(p in nome for p in PALAVRAS_MONETARIAS):
        return f"US$ {formatar_numero_br(valor)}"
    if any(p in nome for p in PALAVRAS_DIAS):
        return f"{formatar_numero_br(valor)} dias"
    casas = 0 if valor.is_integer() else 2
    return formatar_numero_br(valor, casas)


# =============================================================================
# VISUALIZAÇÕES AUTOMÁTICAS (KPI, linha e barras)
# =============================================================================

def escolher_metrica(metricas: list[str], msg_id: str) -> str:
    if len(metricas) == 1:
        return metricas[0]
    return st.selectbox(
        "Métrica exibida no gráfico",
        metricas,
        format_func=rotulo_coluna,
        key=f"metrica_{msg_id}",
    )


def render_kpis(df: pd.DataFrame) -> None:
    colunas = st.columns(len(df.columns))
    for col_ui, coluna in zip(colunas, df.columns):
        col_ui.metric(rotulo_coluna(coluna), formatar_kpi(coluna, df.iloc[0][coluna]), border=True)


def render_grafico_linha(df: pd.DataFrame, eixo_tempo: str, dimensao: str | None, metrica: str) -> None:
    df_plot = df.sort_values(eixo_tempo)
    tipo_x = "T" if pd.api.types.is_datetime64_any_dtype(df_plot[eixo_tempo]) else "O"
    tooltip = [alt.Tooltip(f"{eixo_tempo}:{tipo_x}", title=rotulo_coluna(eixo_tempo))]
    if dimensao:
        tooltip.append(alt.Tooltip(f"{dimensao}:N", title=rotulo_coluna(dimensao)))
    tooltip.append(alt.Tooltip(f"{metrica}:Q", title=rotulo_coluna(metrica), format=",.2f"))

    cor = (
        alt.Color(f"{dimensao}:N", title=rotulo_coluna(dimensao), scale=alt.Scale(range=PALETA_CATEGORICA))
        if dimensao
        else alt.value(COR_SERIE)
    )
    grafico = (
        alt.Chart(df_plot)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True))
        .encode(
            x=alt.X(f"{eixo_tempo}:{tipo_x}", title=rotulo_coluna(eixo_tempo), axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{metrica}:Q", title=rotulo_coluna(metrica), scale=alt.Scale(zero=False)),
            color=cor,
            tooltip=tooltip,
        )
        .properties(height=320)
    )
    st.altair_chart(grafico, width="stretch")


def render_grafico_barras(df: pd.DataFrame, dimensao: str, metrica: str) -> None:
    df_plot = df.head(MAX_BARRAS_GRAFICO)  # mantém a ordenação definida pelo SQL
    if len(df) > MAX_BARRAS_GRAFICO:
        st.caption(f"Gráfico exibindo as primeiras {MAX_BARRAS_GRAFICO} de {len(df)} linhas (tabela completa nos detalhes).")

    grafico = (
        alt.Chart(df_plot)
        .mark_bar(cornerRadiusEnd=4, height={"band": 0.7})
        .encode(
            y=alt.Y(f"{dimensao}:N", sort=None, title=rotulo_coluna(dimensao)),
            x=alt.X(f"{metrica}:Q", title=rotulo_coluna(metrica)),
            color=alt.condition(
                alt.datum[metrica] < 0, alt.value(COR_NEGATIVO), alt.value(COR_SERIE)
            ),
            tooltip=[
                alt.Tooltip(f"{dimensao}:N", title=rotulo_coluna(dimensao)),
                alt.Tooltip(f"{metrica}:Q", title=rotulo_coluna(metrica), format=",.2f"),
            ],
        )
        .properties(height=max(120, 32 * len(df_plot) + 40))
    )
    st.altair_chart(grafico, width="stretch")


def render_visualizacao(df: pd.DataFrame, msg_id: str) -> None:
    """Escolhe automaticamente KPI, gráfico de linha ou de barras conforme o formato do resultado."""
    numericas = colunas_numericas(df)
    if not numericas:
        return

    # 1 linha, todas as colunas numéricas: cards de KPI
    if len(df) == 1:
        if len(numericas) == len(df.columns) <= 4:
            render_kpis(df)
        return

    primeira = df.columns[0]

    # Série temporal: 1ª coluna de tempo + até 1 dimensão categórica (vira uma linha por categoria)
    if eh_coluna_tempo(df, primeira):
        metricas = [c for c in numericas if c != primeira]
        dimensoes = [c for c in df.columns if c != primeira and c not in numericas]
        poucas_categorias = not dimensoes or df[dimensoes[0]].nunique() <= len(PALETA_CATEGORICA)
        if metricas and len(dimensoes) <= 1 and poucas_categorias:
            metrica = escolher_metrica(metricas, msg_id)
            render_grafico_linha(df, primeira, dimensoes[0] if dimensoes else None, metrica)
        return

    # Categoria + valor: gráfico de barras horizontais
    dimensoes = [c for c in df.columns if c not in numericas]
    if len(dimensoes) == 1:
        metrica = escolher_metrica(numericas, msg_id)
        render_grafico_barras(df, dimensoes[0], metrica)


# =============================================================================
# FEEDBACK 👍/👎
# =============================================================================

def registrar_feedback(msg_id: str) -> None:
    valor = st.session_state.get(f"feedback_{msg_id}")
    if valor is None:
        return
    msg = next((m for m in st.session_state["messages"] if m.get("id") == msg_id), None)
    if msg is None:
        return

    res = msg["result"]
    linha = {
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        "pergunta": res.get("pergunta", ""),
        "classificacao": status_resposta(res),
        "feedback": "positivo" if valor == 1 else "negativo",
        "sql_executado": res.get("sql_executado") or "",
        "modelos_usados": ", ".join(res.get("modelos_usados") or []),
        "tokens_total": res.get("tokens_total", ""),
        "tempo_segundos": res.get("tempo_segundos", ""),
        "do_cache": msg.get("do_cache", False),
    }
    try:
        CAMINHO_FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
        arquivo_novo = not CAMINHO_FEEDBACK.exists()
        # BOM apenas na criação, para o Excel reconhecer os acentos
        with open(CAMINHO_FEEDBACK, "a", newline="", encoding="utf-8-sig" if arquivo_novo else "utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(linha.keys()))
            if arquivo_novo:
                writer.writeheader()
            writer.writerow(linha)
        st.toast("Feedback registrado. Obrigado!", icon="📝")
    except Exception:
        logger.exception("Não foi possível gravar o feedback em %s", CAMINHO_FEEDBACK)
        st.toast("Não foi possível salvar o feedback.", icon="⚠️")


# =============================================================================
# RENDERIZAÇÃO DAS MENSAGENS
# =============================================================================

def status_resposta(res: dict[str, Any]) -> str:
    classificacao = res.get("classificacao", "dados")
    if classificacao == "erro":
        return "erro"
    if not res.get("sucesso", True):
        return "falha_sql"
    return classificacao


def render_rastro_tecnico(res: dict[str, Any]) -> None:
    etapas = " → ".join(res.get("etapas_executadas") or []) or "-"
    ferramentas = ", ".join(res.get("ferramentas_chamadas") or []) or "Nenhuma"
    partes = [f"**Etapas:** `{etapas}`", f"**Ferramentas:** `{ferramentas}`"]
    if res.get("sql_valido_primeira_tentativa") is not None:
        partes.append(f"**SQL válido na 1ª tentativa:** `{res['sql_valido_primeira_tentativa']}`")
    st.markdown(" | ".join(partes))


def render_assistant_message(msg: dict[str, Any]) -> None:
    """Exibe a resposta do agente com transparência de SQL, dados, gráficos e métricas."""
    res = msg["result"]
    df: pd.DataFrame | None = msg.get("df")
    msg_id = msg["id"]
    status = status_resposta(res)

    # Resposta principal
    if status == "erro":
        st.error(res.get("resposta", "Ocorreu um erro ao processar a pergunta."))
    elif status == "falha_sql":
        st.warning(
            f"Não foi possível gerar uma consulta SQL válida após {res.get('tentativas_sql', 0)} tentativa(s). "
            "Tente reformular a pergunta. O erro técnico está nos detalhes abaixo."
        )
    else:
        st.markdown(res.get("resposta", ""))

    # Visualização e aviso de truncamento (visíveis, fora do expander)
    if status == "dados" and df is not None:
        if len(df) >= LIMITE_LINHAS:
            st.warning(
                f"O resultado atingiu o limite de {LIMITE_LINHAS} linhas: a tabela, o gráfico e o CSV "
                "podem estar incompletos. Refine a pergunta (ex.: filtre por ano, região ou categoria)."
            )
        render_visualizacao(df, msg_id)

    # Métricas da execução (também exibidas em falhas de SQL)
    if "tempo_segundos" in res:
        label_cls = LABELS_CLASSIFICACAO.get(status, status)
        modelos = ", ".join(res.get("modelos_usados") or []) or "-"
        origem = " · ⚡ Resposta do cache (sem custo de API)" if msg.get("do_cache") else ""
        st.caption(
            f"{label_cls} · Modelo: `{modelos}` · Tempo: `{res.get('tempo_segundos', 0.0):.2f}s` · "
            f"Tokens: `{res.get('tokens_total', 0):,}` · Correções SQL: `{res.get('tentativas_correcao', 0)}`{origem}"
        )

    # Detalhes técnicos
    if res.get("sql_executado"):
        with st.expander("Ver consulta SQL, tabela de resultados e detalhes técnicos", expanded=False):
            st.markdown("**Consulta SQL executada (somente leitura no DuckDB):**")
            st.code(res["sql_executado"], language="sql")

            if status == "falha_sql" and res.get("erro"):
                st.markdown("**Último erro registrado:**")
                st.code(str(res["erro"]), language="text")
            elif df is not None:
                st.markdown(f"**Dados retornados ({len(df)} linha(s)):**")
                st.dataframe(df, hide_index=True)
                st.download_button(
                    label="Baixar tabela em CSV",
                    data=df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"resultado_consulta_{msg_id[:8]}.csv",
                    mime="text/csv",
                    key=f"download_csv_{msg_id}",
                    on_click="ignore",
                )
            elif status == "dados":
                st.info("A consulta foi executada sem erros, mas não retornou registros.")

            render_rastro_tecnico(res)
    elif status in ("fora_de_escopo", "ambigua"):
        with st.expander("Ver detalhes da decisão do Orquestrador", expanded=False):
            st.markdown(f"**Motivo:** {res.get('motivo') or 'Não informado.'}")
            render_rastro_tecnico(res)

    # Avaliação humana (não exibida para erros técnicos)
    if status != "erro":
        st.feedback("thumbs", key=f"feedback_{msg_id}", on_change=registrar_feedback, args=(msg_id,))


# =============================================================================
# PAINEL DE MÉTRICAS DA SESSÃO (barra lateral)
# =============================================================================

def render_painel_sessao() -> None:
    respostas = [m for m in st.session_state["messages"] if m["role"] == "assistant"]
    st.subheader("Métricas da sessão")
    if not respostas:
        st.caption("As métricas aparecem após a primeira pergunta.")
        return

    via_api = [m for m in respostas if not m.get("do_cache") and "tempo_segundos" in m["result"]]
    do_cache = sum(1 for m in respostas if m.get("do_cache"))
    tokens = sum(int(m["result"].get("tokens_total", 0) or 0) for m in via_api)
    tempos = [float(m["result"]["tempo_segundos"]) for m in via_api]
    avaliaveis = [
        m["result"]["sql_valido_primeira_tentativa"]
        for m in respostas
        if m["result"].get("sql_valido_primeira_tentativa") is not None
    ]

    c1, c2 = st.columns(2)
    c1.metric("Perguntas", len(respostas))
    c2.metric("Do cache", do_cache)
    c1.metric("Tokens (API)", f"{tokens:,}".replace(",", "."))
    c2.metric("Tempo médio", f"{sum(tempos) / len(tempos):.1f}s" if tempos else "—")
    st.metric(
        "SQL válido na 1ª tentativa",
        f"{100 * sum(avaliaveis) / len(avaliaveis):.0f}%" if avaliaveis else "—",
        help="Considera apenas as perguntas que geraram SQL.",
    )

    historico = pd.DataFrame(
        [
            {
                "pergunta_n": f"#{i}",
                "pergunta": m["result"].get("pergunta", "")[:80],
                "tokens": int(m["result"].get("tokens_total", 0) or 0),
                "tempo_s": float(m["result"].get("tempo_segundos", 0.0) or 0.0),
            }
            for i, m in enumerate(via_api, start=1)
        ]
    )
    if not historico.empty:
        grafico = (
            alt.Chart(historico)
            .mark_bar(cornerRadiusEnd=4, color=COR_SERIE)
            .encode(
                x=alt.X("pergunta_n:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("tokens:Q", title="Tokens"),
                tooltip=[
                    alt.Tooltip("pergunta:N", title="Pergunta"),
                    alt.Tooltip("tokens:Q", title="Tokens", format=","),
                    alt.Tooltip("tempo_s:Q", title="Tempo (s)", format=".2f"),
                ],
            )
            .properties(height=160, title="Tokens por pergunta (chamadas à API)")
        )
        st.altair_chart(grafico, width="stretch")


# =============================================================================
# ABA "COMO FUNCIONA"
# =============================================================================

DIAGRAMA_FLUXO = """
digraph G {
  rankdir=TB; bgcolor="transparent"; nodesep=0.5; ranksep=0.45; newrank=true;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13,
        color="#52514e", fillcolor="#f0efec", fontcolor="#0b0b0b", margin="0.2,0.1"];
  edge [color="#8a8984", fontname="Helvetica", fontsize=11, fontcolor="#6f6e69"];

  pergunta [label="Pergunta do usuário", shape=oval, fillcolor="#cde2fb"];
  orq [label="Agente 1 · Orquestrador\\n(temperatura 0)", fillcolor="#9ec5f4"];
  semantica [label="Camada semântica\\nsemantic_layer.yaml", shape=cylinder];
  desvio [label="Recusa ou pedido de\\nesclarecimento (sem SQL)", fillcolor="#fde2d6"];
  sql [label="Agente 2 · SQL\\n(temperatura 0)", fillcolor="#9ec5f4"];
  valida [label="tool_validate_sql\\nSELECT · LIMIT · regras · EXPLAIN"];
  executa [label="tool_run_sql\\nsomente leitura"];
  duckdb [label="DuckDB\\nesquema estrela", shape=cylinder];
  falha [label="Falha após\\n3 tentativas", fillcolor="#fde2d6"];
  analista [label="Agente 3 · Analista\\n(temperatura 0.2)", fillcolor="#9ec5f4"];
  resposta [label="Resposta executiva\\n+ SQL, tabela e gráfico", shape=oval, fillcolor="#cde2fb"];

  pergunta -> orq;
  orq -> semantica [label=" get_schema\\n get_metric", dir=both];
  orq -> desvio [label=" fora_de_escopo\\n ambigua"];
  orq -> sql [label=" dados: plano\\n + contexto"];
  sql -> valida;
  valida -> executa [label=" válido"];
  valida -> sql [label=" inválido", style=dashed, constraint=false];
  executa -> duckdb [dir=both];
  executa -> sql [label=" erro", style=dashed, constraint=false];
  sql -> falha [style=dashed, constraint=false];
  executa -> analista [label=" colunas + linhas"];
  analista -> resposta;

  {rank=same; orq; semantica;}
  {rank=same; sql; desvio;}
  {rank=same; executa; duckdb;}
  {rank=same; valida; falha;}
}
"""


def render_como_funciona() -> None:
    st.subheader("Arquitetura multiagente")
    st.markdown(
        "Cada pergunta passa por até **três agentes especializados**, todos baseados no Gemini e "
        "orquestrados por `answer_question` em `src/orchestrator.py`. A interface apenas exibe o resultado."
    )

    _, col_diag, _ = st.columns([1, 3, 1])
    with col_diag:
        st.graphviz_chart(DIAGRAMA_FLUXO, width="stretch")

    col_orq, col_sql, col_analista = st.columns(3, gap="large")
    with col_orq:
        st.markdown(
            """
**1. Orquestrador**
- Chama `tool_get_schema` e `tool_get_metric` na camada semântica antes de decidir.
- Classifica a pergunta em `dados`, `fora_de_escopo` ou `ambigua`.
- Para `dados`, entrega ao Agente SQL um **plano de consulta** (tabelas, JOINs, métricas e traduções).
            """
        )
    with col_sql:
        st.markdown(
            """
**2. Agente SQL**
- Escreve uma única consulta `SELECT`/`WITH` seguindo as regras da camada semântica.
- Fluxo obrigatório: `tool_validate_sql` → `tool_run_sql`.
- Se a validação ou a execução falhar, corrige o SQL sozinho (até **3 tentativas**).
            """
        )
    with col_analista:
        st.markdown(
            """
**3. Analista**
- Transforma o resultado em resposta executiva: valores em `US$`, percentuais com `%`.
- Não inventa causas fora dos dados e sugere **1 próxima pergunta**.
            """
        )

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🔒 Travas de segurança**")
        st.markdown(
            "- Somente `SELECT` em modo leitura\n"
            f"- `LIMIT {LIMITE_LINHAS}` padrão e tempo limite de 10s\n"
            "- JOINs apenas por chaves substitutas\n"
            "- Sem nomes de clientes na base"
        )
    with c2:
        st.markdown("**🤖 Modelos e resiliência**")
        st.markdown(
            f"- Principal: `{PRIMARY_MODEL}`\n"
            f"- Reserva: `{FALLBACK_MODEL}` (cota diária esgotada)\n"
            "- Retentativa em 429/500/503/504 (2x, espera de 4s e 8s)\n"
            "- Temperatura 0 no Orquestrador e no SQL"
        )
    with c3:
        st.markdown("**📡 Custo por pergunta (aprox.)**")
        st.markdown(
            "- Pergunta de dados: ~5 chamadas à API (2 orquestrador, 2 SQL, 1 analista)\n"
            "- Recusa ou esclarecimento: ~2 chamadas\n"
            "- Cada correção de SQL adiciona chamadas\n"
            "- Respostas em cache não chamam a API"
        )

    metricas = (SEMANTICA.get("metricas") or {}) if SEMANTICA else {}
    if metricas:
        with st.expander(f"Métricas oficiais da camada semântica ({len(metricas)})"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Métrica": nome, "Descrição": cfg.get("descricao", ""), "SQL": cfg.get("sql", ""), "Unidade": cfg.get("unidade", "")}
                        for nome, cfg in metricas.items()
                    ]
                ),
                hide_index=True,
            )


# =============================================================================
# ABA "AVALIAÇÃO"
# =============================================================================

@st.cache_data(show_spinner=False)
def carregar_avaliacao(_mtime: float) -> pd.DataFrame | None:
    if not CAMINHO_AVALIACAO.exists():
        return None
    df = pd.read_csv(CAMINHO_AVALIACAO, encoding="utf-8")
    df["acertou"] = df["acertou"].astype(str).str.strip().str.lower().eq("true")
    sql_1a = df["sql_valido_1a_tentativa"].astype(str).str.strip().str.lower()
    df["sql_valido_1a_tentativa"] = sql_1a.map({"true": True, "false": False})
    df["nivel_rotulo"] = df["nivel"].map(ROTULOS_NIVEL).fillna(df["nivel"])
    return df


def render_avaliacao() -> None:
    st.subheader("Avaliação automática do agente")
    df = carregar_avaliacao(_mtime(CAMINHO_AVALIACAO))
    if df is None or df.empty:
        st.info(f"Arquivo de avaliação não encontrado em `{CAMINHO_AVALIACAO.relative_to(BASE_DIR)}`.")
        return

    st.markdown(
        f"Resultado do conjunto de **{len(df)} perguntas** de teste (`eval/resultados.csv`), "
        "comparando o SQL e os dados do agente com uma resposta de referência."
    )

    com_sql = df["sql_valido_1a_tentativa"].dropna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acurácia geral", f"{100 * df['acertou'].mean():.0f}%", help=f"{int(df['acertou'].sum())} de {len(df)} perguntas", border=True)
    c2.metric(
        "SQL válido na 1ª tentativa",
        f"{100 * com_sql.astype(bool).mean():.0f}%" if not com_sql.empty else "—",
        help="Considera apenas as perguntas que geraram SQL.",
        border=True,
    )
    c3.metric("Tempo médio", f"{df['tempo_segundos'].mean():.1f}s", border=True)
    c4.metric("Tokens médios", f"{df['tokens_total'].mean():,.0f}".replace(",", "."), border=True)

    ordem_niveis = [n for n in CORES_NIVEL if n in set(df["nivel_rotulo"])]
    escala_nivel = alt.Scale(domain=ordem_niveis, range=[CORES_NIVEL[n] for n in ordem_niveis])

    # Acurácia por nível
    por_nivel = (
        df.groupby("nivel_rotulo", sort=False)
        .agg(total=("id", "count"), acertos=("acertou", "sum"))
        .reset_index()
    )
    por_nivel["acuracia"] = 100 * por_nivel["acertos"] / por_nivel["total"]
    por_nivel["rotulo"] = por_nivel.apply(lambda r: f"{int(r['acertos'])}/{int(r['total'])}", axis=1)

    base = alt.Chart(por_nivel).encode(
        x=alt.X("nivel_rotulo:N", sort=ordem_niveis, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("acuracia:Q", title="Acurácia (%)", scale=alt.Scale(domain=[0, 110])),
    )
    grafico_acuracia = (
        base.mark_bar(cornerRadiusEnd=4, width={"band": 0.6})
        .encode(
            color=alt.Color("nivel_rotulo:N", scale=escala_nivel, legend=None),
            tooltip=[
                alt.Tooltip("nivel_rotulo:N", title="Nível"),
                alt.Tooltip("rotulo:N", title="Acertos"),
                alt.Tooltip("acuracia:Q", title="Acurácia (%)", format=".0f"),
            ],
        )
        + base.mark_text(dy=-8).encode(text="rotulo:N")
    ).properties(height=260, title="Acurácia por nível de dificuldade")

    def grafico_por_pergunta(campo: str, titulo: str, formato: str) -> alt.Chart:
        return (
            alt.Chart(df)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("id:O", title="Pergunta (id)", axis=alt.Axis(labelAngle=0)),
                y=alt.Y(f"{campo}:Q", title=titulo),
                color=alt.Color("nivel_rotulo:N", scale=escala_nivel, title="Nível"),
                tooltip=[
                    alt.Tooltip("id:O", title="Id"),
                    alt.Tooltip("pergunta:N", title="Pergunta"),
                    alt.Tooltip("nivel_rotulo:N", title="Nível"),
                    alt.Tooltip(f"{campo}:Q", title=titulo, format=formato),
                ],
            )
            .properties(height=260, title=f"{titulo} por pergunta")
        )

    g1, g2 = st.columns([1, 2])
    with g1:
        st.altair_chart(grafico_acuracia, width="stretch")
    with g2:
        aba_tokens, aba_tempo = st.tabs(["Tokens", "Tempo"])
        with aba_tokens:
            st.altair_chart(grafico_por_pergunta("tokens_total", "Tokens", ","), width="stretch")
        with aba_tempo:
            st.altair_chart(grafico_por_pergunta("tempo_segundos", "Tempo (s)", ".2f"), width="stretch")

    st.markdown("**Resultados por pergunta**")
    tabela = df[["id", "nivel_rotulo", "pergunta", "acertou", "classificacao", "tempo_segundos", "tokens_total"]].copy()
    tabela["acertou"] = tabela["acertou"].map({True: "✅", False: "❌"})
    st.dataframe(
        tabela,
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("Id", width="small"),
            "nivel_rotulo": "Nível",
            "pergunta": st.column_config.TextColumn("Pergunta", width="large"),
            "acertou": st.column_config.TextColumn("Acertou", width="small"),
            "classificacao": "Classificação",
            "tempo_segundos": st.column_config.NumberColumn("Tempo (s)", format="%.2f"),
            "tokens_total": st.column_config.NumberColumn("Tokens", format="%d"),
        },
    )

    st.markdown("**Inspecionar uma pergunta**")
    opcoes = df["id"].tolist()
    escolhido = st.selectbox(
        "Pergunta",
        opcoes,
        format_func=lambda i: f"#{i} · {df.loc[df['id'] == i, 'pergunta'].iloc[0]}",
        label_visibility="collapsed",
        key="avaliacao_pergunta",
    )
    linha = df.loc[df["id"] == escolhido].iloc[0]
    st.caption(
        f"Nível: {linha['nivel_rotulo']} · Classificação: {linha['classificacao']} · "
        f"Avaliação: {'✅' if linha['acertou'] else '❌'} {linha['motivo_avaliacao']}"
    )
    col_ref, col_agente = st.columns(2)
    with col_ref:
        st.markdown("**Referência**")
        st.code(str(linha["sql_referencia"]) if pd.notna(linha["sql_referencia"]) else "N/A", language="sql")
        st.markdown("Resposta esperada:")
        st.code(str(linha["resposta_esperada"]), language="json")
    with col_agente:
        st.markdown("**Agente**")
        st.code(str(linha["sql_gerado"]) if pd.notna(linha["sql_gerado"]) else "Sem SQL (desvio do Orquestrador)", language="sql")
        st.markdown("Dados obtidos:")
        st.code(str(linha["dados_obtidos"]) if pd.notna(linha["dados_obtidos"]) else "[]", language="json")
    st.markdown("**Resposta do agente:**")
    st.markdown(str(linha["resposta_agente"]))


# =============================================================================
# ESTADO DA SESSÃO
# =============================================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# =============================================================================
# BARRA LATERAL
# =============================================================================

contexto = SEMANTICA.get("contexto_geral", {}) if SEMANTICA else {}
tabelas = list((SEMANTICA.get("tabelas") or {}).keys()) if SEMANTICA else []

with st.sidebar:
    st.title("📊 Analista Virtual")
    st.markdown(
        "Sistema multiagente de BI em **Python + DuckDB + Gemini** sobre a base "
        "**Sample Superstore (EUA)**."
    )

    st.divider()
    st.subheader("Escopo da base")
    periodo = contexto.get("periodo_disponivel", "2014 a 2017")
    tabelas_md = ", ".join(f"`{t}`" for t in tabelas) or "`fato_vendas`, `dim_clientes`, `dim_produtos`, `dim_localizacao`"
    st.markdown(
        f"- **Período:** {periodo} ({TOTAL_REGISTROS} registros)\n"
        "- **Moeda:** Dólar americano (`US$`)\n"
        f"- **Modelo:** Esquema estrela ({tabelas_md})\n"
        f"- **Travas:** Somente `SELECT`, `LIMIT {LIMITE_LINHAS}`, tempo limite de 10s e sem nomes de clientes"
    )

    st.divider()
    st.subheader("Perguntas de exemplo")
    st.caption("Clique em um atalho ou digite sua própria pergunta no campo do chat.")
    for idx_btn, (rotulo, texto_pergunta) in enumerate(PERGUNTAS_SUGERIDAS):
        if st.button(rotulo, key=f"sugestao_{idx_btn}", width="stretch"):
            st.session_state["pending_question"] = texto_pergunta

    st.divider()
    painel_sessao = st.container()  # preenchido no fim do script, já com a última resposta

    st.divider()
    if st.button("Limpar conversa", width="stretch"):
        st.session_state["messages"] = []
        st.rerun()


# =============================================================================
# ÁREA PRINCIPAL
# =============================================================================

st.header("Do BI ao Analista Virtual")
st.markdown(
    "Faça perguntas de negócio em português sobre vendas, lucro, descontos, prazos de envio, "
    "regiões, segmentos e categorias. Cada consulta é validada antes de rodar no DuckDB."
)

aba_chat, aba_como_funciona, aba_avaliacao = st.tabs(["💬 Chat", "🧭 Como funciona", "📈 Avaliação"])

with aba_chat:
    st.info(
        "Cada pergunta é analisada de forma **independente**, sem memória das anteriores. "
        "Para continuar uma análise, repita o contexto: *\"faturamento do Oeste em 2016\"* "
        "em vez de *\"e em 2016?\"*.",
        icon="ℹ️",
    )

    area_conversa = st.container()
    entrada_usuario = st.chat_input(
        "Pergunta completa sobre a Superstore (2014–2017). Ex.: Qual foi o lucro por região em 2016?"
    )
    pergunta_atalho = st.session_state.pop("pending_question", None)
    pergunta_atual = (entrada_usuario or pergunta_atalho or "").strip()

    with area_conversa:
        # Histórico anterior
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    render_assistant_message(msg)

        if pergunta_atual:
            restante = segundos_restantes_cooldown()
            if restante > 0:
                st.warning(
                    f"Aguarde {restante:.0f}s antes da próxima pergunta (proteção da cota da API). "
                    f"Depois, envie novamente: *{pergunta_atual}*",
                    icon="⏳",
                )
            else:
                st.session_state["messages"].append(
                    {"id": uuid.uuid4().hex, "role": "user", "content": pergunta_atual}
                )
                with st.chat_message("user"):
                    st.markdown(pergunta_atual)

                with st.chat_message("assistant"):
                    with st.spinner("Consultando camada semântica, validando SQL e analisando os dados..."):
                        resultado, veio_do_cache = executar_pergunta(pergunta_atual)
                    if not veio_do_cache:
                        st.session_state["ultima_chamada_api"] = time.time()

                    msg_assistente = {
                        "id": uuid.uuid4().hex,
                        "role": "assistant",
                        "result": resultado,
                        "df": preparar_dataframe(resultado.get("dados") or []),
                        "do_cache": veio_do_cache,
                    }
                    st.session_state["messages"].append(msg_assistente)
                    render_assistant_message(msg_assistente)

with aba_como_funciona:
    render_como_funciona()

with aba_avaliacao:
    render_avaliacao()

# Painel da barra lateral renderizado por último para já incluir a resposta recém-gerada
with painel_sessao:
    render_painel_sessao()