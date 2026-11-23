from pathlib import Path
import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DB_PATH = PROCESSED_DIR / "superstore.duckdb"


def find_raw_csv() -> Path:
    """Localiza o arquivo CSV da Superstore dentro de data/raw/."""
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Nenhum arquivo CSV encontrado em {RAW_DIR}. "
            "Coloque o arquivo da Superstore em data/raw/ antes de rodar este script."
        )
    return csv_files[0]


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Carrega o CSV bruto, remove dados pessoais e aplica limpezas estruturais."""
    df = pd.read_csv(csv_path, encoding="latin1")
    initial_rows = len(df)

    # Remove coluna de nome de cliente para não manter dados pessoais
    if "Customer Name" in df.columns:
        df = df.drop(columns=["Customer Name"])

    # Remove espaços extras nas colunas de texto
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    # Remove duplicatas exatas desconsiderando o identificador sequencial Row ID
    cols_sem_row_id = [col for col in df.columns if col != "Row ID"]
    dup_count = int(df.duplicated(subset=cols_sem_row_id).sum())
    df = df.drop_duplicates(subset=cols_sem_row_id, keep="first").copy()

    # Converte datas do padrão americano (MM/DD/AAAA) para tipo date
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y").dt.date
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y").dt.date

    # Padroniza Postal Code como texto de 5 dígitos preservando zeros à esquerda
    df["Postal Code"] = (
        pd.to_numeric(df["Postal Code"], errors="raise")
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )

    # Padroniza nomes das colunas para snake_case
    df = df.rename(
        columns={
            "Row ID": "row_id",
            "Order ID": "order_id",
            "Order Date": "order_date",
            "Ship Date": "ship_date",
            "Ship Mode": "ship_mode",
            "Customer ID": "customer_id",
            "Segment": "segment",
            "Country": "country",
            "City": "city",
            "State": "state",
            "Postal Code": "postal_code",
            "Region": "region",
            "Product ID": "product_id",
            "Category": "category",
            "Sub-Category": "sub_category",
            "Product Name": "product_name",
            "Sales": "sales",
            "Quantity": "quantity",
            "Discount": "discount",
            "Profit": "profit",
        }
    )

    print(
        f"CSV carregado: {csv_path.name} | Linhas iniciais: {initial_rows} | "
        f"Duplicatas removidas: {dup_count} | Linhas finais: {len(df)}"
    )
    return df


def build_star_schema(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separa a base limpa em 1 tabela fato e 3 dimensões com chaves substitutas."""
    # Dimensão Clientes (chave natural limpa)
    dim_clientes = (
        df[["customer_id", "segment"]]
        .drop_duplicates()
        .sort_values("customer_id")
        .reset_index(drop=True)
    )

    # Dimensão Produtos (chave substituta para resolver Product IDs com múltiplos nomes)
    prod_cols = ["product_id", "category", "sub_category", "product_name"]
    dim_produtos = (
        df[prod_cols]
        .drop_duplicates()
        .sort_values(prod_cols)
        .reset_index(drop=True)
    )
    dim_produtos.insert(0, "product_sk", range(1, len(dim_produtos) + 1))

    # Dimensão Localização (chave substituta para resolver Postal Code com múltiplas cidades)
    loc_cols = ["postal_code", "city", "state", "region", "country"]
    dim_localizacao = (
        df[loc_cols]
        .drop_duplicates()
        .sort_values(loc_cols)
        .reset_index(drop=True)
    )
    dim_localizacao.insert(0, "location_sk", range(1, len(dim_localizacao) + 1))

    # Tabela Fato com associação das chaves substitutas (validando cardinalidade N:1)
    fato_vendas = df.merge(dim_produtos, on=prod_cols, how="left", validate="many_to_one")
    fato_vendas = fato_vendas.merge(
        dim_localizacao, on=loc_cols, how="left", validate="many_to_one"
    )

    fato_cols = [
        "row_id",
        "order_id",
        "order_date",
        "ship_date",
        "ship_mode",
        "customer_id",
        "product_sk",
        "location_sk",
        "sales",
        "quantity",
        "discount",
        "profit",
    ]
    fato_vendas = fato_vendas[fato_cols].sort_values("row_id").reset_index(drop=True)

    return fato_vendas, dim_clientes, dim_produtos, dim_localizacao


def write_to_duckdb(
    fato_vendas: pd.DataFrame,
    dim_clientes: pd.DataFrame,
    dim_produtos: pd.DataFrame,
    dim_localizacao: pd.DataFrame,
) -> duckdb.DuckDBPyConnection:
    """Cria o banco DuckDB do zero com esquema explícito e carrega as 4 tabelas."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))

    con.execute(
        """
        CREATE TABLE dim_clientes (
            customer_id VARCHAR PRIMARY KEY,
            segment VARCHAR NOT NULL
        );

        CREATE TABLE dim_produtos (
            product_sk INTEGER PRIMARY KEY,
            product_id VARCHAR NOT NULL,
            category VARCHAR NOT NULL,
            sub_category VARCHAR NOT NULL,
            product_name VARCHAR NOT NULL
        );

        CREATE TABLE dim_localizacao (
            location_sk INTEGER PRIMARY KEY,
            postal_code VARCHAR NOT NULL,
            city VARCHAR NOT NULL,
            state VARCHAR NOT NULL,
            region VARCHAR NOT NULL,
            country VARCHAR NOT NULL
        );

        CREATE TABLE fato_vendas (
            row_id INTEGER PRIMARY KEY,
            order_id VARCHAR NOT NULL,
            order_date DATE NOT NULL,
            ship_date DATE NOT NULL,
            ship_mode VARCHAR NOT NULL,
            customer_id VARCHAR NOT NULL REFERENCES dim_clientes(customer_id),
            product_sk INTEGER NOT NULL REFERENCES dim_produtos(product_sk),
            location_sk INTEGER NOT NULL REFERENCES dim_localizacao(location_sk),
            sales DOUBLE NOT NULL,
            quantity INTEGER NOT NULL,
            discount DOUBLE NOT NULL,
            profit DOUBLE NOT NULL
        );
        """
    )

    con.register("df_dim_clientes", dim_clientes)
    con.register("df_dim_produtos", dim_produtos)
    con.register("df_dim_localizacao", dim_localizacao)
    con.register("df_fato_vendas", fato_vendas)

    con.execute("INSERT INTO dim_clientes SELECT * FROM df_dim_clientes")
    con.execute("INSERT INTO dim_produtos SELECT * FROM df_dim_produtos")
    con.execute("INSERT INTO dim_localizacao SELECT * FROM df_dim_localizacao")
    con.execute("INSERT INTO fato_vendas SELECT * FROM df_fato_vendas")

    return con


def run_quality_tests(con: duckdb.DuckDBPyConnection, df_clean: pd.DataFrame) -> None:
    """Executa testes de volume, nulos, duplicatas, chaves órfãs, datas e reconciliação."""
    print("\n--- Executando testes de qualidade ---")
    falhas = []

    # 1. Volume mínimo de linhas na tabela fato
    total_fato = con.execute("SELECT COUNT(*) FROM fato_vendas").fetchone()[0]
    if total_fato >= 5000 and total_fato == len(df_clean):
        print(f"[OK] Volume da fato_vendas: {total_fato} linhas (>= 5.000)")
    else:
        falhas.append(f"Volume inválido na fato_vendas: {total_fato}")

    # 2. Verificação de nulos em todas as tabelas e colunas
    tables = ["fato_vendas", "dim_clientes", "dim_produtos", "dim_localizacao"]
    null_issues = []
    for table in tables:
        cols = [row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()]
        for col in cols:
            n_nulls = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL"
            ).fetchone()[0]
            if n_nulls > 0:
                null_issues.append(f"{table}.{col}={n_nulls}")
    if not null_issues:
        print("[OK] Valores nulos: 0 encontrados em todas as 4 tabelas")
    else:
        falhas.append(f"Valores nulos encontrados: {', '.join(null_issues)}")

    # 3. Duplicatas nas chaves primárias
    pk_checks = [
        ("fato_vendas", "row_id"),
        ("dim_clientes", "customer_id"),
        ("dim_produtos", "product_sk"),
        ("dim_localizacao", "location_sk"),
    ]
    pk_issues = []
    for table, pk in pk_checks:
        dups = con.execute(
            f"SELECT COUNT(*) - COUNT(DISTINCT {pk}) FROM {table}"
        ).fetchone()[0]
        if dups > 0:
            pk_issues.append(f"{table}.{pk}={dups}")
    if not pk_issues:
        print("[OK] Chaves primárias: 0 duplicatas nas 4 tabelas")
    else:
        falhas.append(f"Duplicatas de PK: {', '.join(pk_issues)}")

    # 4. Chaves órfãs (integridade referencial entre fato e dimensões)
    orphan_queries = {
        "customer_id": """
            SELECT COUNT(*) FROM fato_vendas f
            LEFT JOIN dim_clientes c ON f.customer_id = c.customer_id
            WHERE c.customer_id IS NULL
        """,
        "product_sk": """
            SELECT COUNT(*) FROM fato_vendas f
            LEFT JOIN dim_produtos p ON f.product_sk = p.product_sk
            WHERE p.product_sk IS NULL
        """,
        "location_sk": """
            SELECT COUNT(*) FROM fato_vendas f
            LEFT JOIN dim_localizacao l ON f.location_sk = l.location_sk
            WHERE l.location_sk IS NULL
        """,
    }
    orphan_issues = []
    for fk, query in orphan_queries.items():
        orphans = con.execute(query).fetchone()[0]
        if orphans > 0:
            orphan_issues.append(f"{fk}={orphans}")
    if not orphan_issues:
        print("[OK] Chaves órfãs: 0 registros sem correspondência nas dimensões")
    else:
        falhas.append(f"Chaves órfãs encontradas: {', '.join(orphan_issues)}")

    # 5. Consistência temporal (envio nunca anterior ao pedido)
    bad_dates = con.execute(
        "SELECT COUNT(*) FROM fato_vendas WHERE ship_date < order_date"
    ).fetchone()[0]
    if bad_dates == 0:
        print("[OK] Consistência temporal: 0 pedidos com ship_date < order_date")
    else:
        falhas.append(f"Datas inválidas (ship_date < order_date): {bad_dates}")

    # 6. Reconciliação financeira após JOIN completo das 4 tabelas
    joined_count, joined_sales, joined_profit = con.execute(
        """
        SELECT COUNT(*), SUM(f.sales), SUM(f.profit)
        FROM fato_vendas f
        INNER JOIN dim_clientes c ON f.customer_id = c.customer_id
        INNER JOIN dim_produtos p ON f.product_sk = p.product_sk
        INNER JOIN dim_localizacao l ON f.location_sk = l.location_sk
        """
    ).fetchone()

    expected_sales = float(df_clean["sales"].sum())
    expected_profit = float(df_clean["profit"].sum())
    sales_diff = abs(joined_sales - expected_sales)
    profit_diff = abs(joined_profit - expected_profit)

    if joined_count == len(df_clean) and sales_diff < 1e-4 and profit_diff < 1e-4:
        print(
            f"[OK] Reconciliação após JOIN completo: {joined_count} linhas | "
            f"Sales = US$ {joined_sales:,.2f} | Profit = US$ {joined_profit:,.2f}"
        )
    else:
        falhas.append(
            f"Falha na reconciliação: linhas={joined_count}/{len(df_clean)}, "
            f"diff_sales={sales_diff:.4f}, diff_profit={profit_diff:.4f}"
        )

    # Resumo de contagens por tabela
    print("\n--- Resumo das tabelas carregadas ---")
    for table in tables:
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  - {table}: {n:,} linhas")

    if falhas:
        raise RuntimeError("Testes de qualidade falharam:\n- " + "\n- ".join(falhas))

    print(f"\nCarga concluída com sucesso em: {DB_PATH}")


def main() -> None:
    csv_path = find_raw_csv()
    df_clean = load_and_clean(csv_path)
    fato, dim_cli, dim_prod, dim_loc = build_star_schema(df_clean)
    con = write_to_duckdb(fato, dim_cli, dim_prod, dim_loc)
    try:
        run_quality_tests(con, df_clean)
    finally:
        con.close()


if __name__ == "__main__":
    main()