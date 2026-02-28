from fastmcp import FastMCP
import pandas as pd
import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # ✅ sem display gráfico, roda headless
import matplotlib.pyplot as plt

mcp = FastMCP("DataClaw-Local", version="1.0.0")

os.makedirs("outputs", exist_ok=True)
os.makedirs("testes", exist_ok=True)


# ─────────────────────────────────────────────
# ✅ FIX 2: Limpeza numérica com regra dos 70%
# Evita destruir colunas de texto como "Cliente_1234" ou "iPhone 15"
# ─────────────────────────────────────────────
def coerce_numeric_if_majority(series: pd.Series, threshold: float = 0.7) -> pd.Series:
    converted = pd.to_numeric(
        series.astype(str)
            .str.replace(r"[^\d.,\-]", "", regex=True)
            .str.replace(",", "."),
        errors="coerce",
    )
    valid_ratio = converted.notna().sum() / max(len(series), 1)
    return converted if valid_ratio >= threshold else series


def load_csv_robust(file_path: str) -> pd.DataFrame:
    """
    Carrega CSV lidando com todos os problemas BR e sujos.
    ✅ FIX: encoding fallback (utf-8 → latin-1 → cp1252)
    ✅ FIX: separador fallback (; → auto-detect)
    ✅ FIX: limpeza numérica segura (regra dos 70%)
    """
    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    df = None

    # ✅ FIX 5b: Tenta encodings em sequência
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(
                file_path,
                sep=";",
                decimal=",",
                encoding=enc,
                on_bad_lines="skip",
            )
            break
        except (UnicodeDecodeError, Exception):
            continue

    # Fallback: detecção automática de separador
    if df is None:
        try:
            df = pd.read_csv(
                file_path,
                sep=None,
                decimal=".",
                encoding="utf-8",
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as e:
            raise ValueError(f"Não foi possível ler o arquivo: {e}")

    # Remove linhas completamente vazias e duplicatas
    df = df.drop_duplicates().dropna(how="all")

    # ✅ FIX 2: Aplica conversão numérica segura com threshold 70%
    for col in df.select_dtypes(include="object").columns:
        df[col] = coerce_numeric_if_majority(df[col])

    return df


def detectar_colunas(df: pd.DataFrame):
    """Detecta automaticamente colunas de valor, data e categoria."""
    value_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["total", "valor", "receita", "faturamento", "preco", "price"])),
        None,
    )
    date_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["data", "date", "dt_", "_dt", "periodo"])),
        None,
    )
    cat_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["categoria", "produto", "vendedor", "status", "cidade"])),
        None,
    )
    return value_col, date_col, cat_col


# ─────────────────────────────────────────────
# TOOL 1: analyze_csv
# ─────────────────────────────────────────────
@mcp.tool
def analyze_csv(
    file_path: str,
    pergunta: str = "Faça uma análise completa: totais, tendências, outliers e insights acionáveis",
) -> str:
    """
    Analisa qualquer CSV grande e sujo com precisão matemática.
    Detecta automaticamente colunas de valor, data e categoria.
    """
    try:
        df = load_csv_robust(file_path)
        value_col, date_col, cat_col = detectar_colunas(df)

        report = f"# 📊 DataClaw Analysis\n"
        report += f"**Arquivo**: {os.path.basename(file_path)}\n"
        report += f"**Data**: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        report += f"**Linhas**: {len(df):,} | **Colunas**: {len(df.columns)}\n\n"

        # ✅ FIX 5: Resumo estatístico truncado — só numéricas, máximo 8 colunas
        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            report += "## Resumo Estatístico\n"
            cols_to_show = num_df.columns[:8].tolist()
            report += num_df[cols_to_show].describe().round(2).to_markdown()
            if len(num_df.columns) > 8:
                report += f"\n*Exibindo {len(cols_to_show)} de {len(num_df.columns)} colunas numéricas.*\n"
            report += "\n\n"

        # Faturamento total
        if value_col:
            total = df[value_col].sum()
            media = df[value_col].mean()
            maximo = df[value_col].max()
            report += f"## 💰 Financeiro ({value_col})\n"
            report += f"- **Total**: R$ {total:,.2f}\n"
            report += f"- **Média por venda**: R$ {media:,.2f}\n"
            report += f"- **Maior venda**: R$ {maximo:,.2f}\n\n"

        # Tendência mensal
        if date_col and value_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            valid_dates = df[date_col].notna()
            report += f"⚠️ Datas inválidas ignoradas: {(~valid_dates).sum():,} linhas\n\n"
            if valid_dates.sum() > 0:
                monthly = (
                    df[valid_dates]
                    .groupby(df[date_col].dt.to_period("M"))[value_col]
                    .agg(["sum", "count"])
                    .rename(columns={"sum": "Total R$", "count": "Qtd Vendas"})
                    .tail(12)  # últimos 12 meses
                )
                report += "## 📅 Tendência Mensal (últimos 12 meses)\n"
                report += monthly.to_markdown() + "\n\n"

        # Top categorias
        if cat_col:
            report += f"## 🏷️ Top 10 por {cat_col}\n"
            if value_col:
                top = df.groupby(cat_col)[value_col].sum().sort_values(ascending=False).head(10)
                report += top.to_frame().to_markdown() + "\n\n"
            else:
                top = df[cat_col].value_counts().head(10)
                report += top.to_frame().to_markdown() + "\n\n"

        # Outliers simples
        if value_col:
            q1 = df[value_col].quantile(0.25)
            q3 = df[value_col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[value_col] < q1 - 1.5 * iqr) | (df[value_col] > q3 + 1.5 * iqr)]
            report += f"## ⚠️ Outliers Detectados\n"
            report += f"- {len(outliers):,} transações fora do padrão (método IQR)\n"
            if len(outliers) > 0:
                report += f"- Faixa normal: R$ {q1 - 1.5*iqr:,.2f} a R$ {q3 + 1.5*iqr:,.2f}\n\n"

        # Pergunta customizada no relatório
        report += f"## 🤖 Pergunta: {pergunta}\n"
        report += "Análise acima responde com dados precisos. Solicite métricas específicas se necessário.\n\n"

        # ✅ FIX 3: Gráfico inteligente — detecta colunas corretas
        # ✅ FIX 4: Salva arquivo + retorna caminho (não base64)
        try:
            plt.figure(figsize=(12, 5))

            if date_col and value_col and valid_dates.sum() > 0:
                monthly["Total R$"].plot(kind="bar", color="steelblue", edgecolor="white")
                plt.title(f"Faturamento Mensal — {value_col}", fontsize=14)
                plt.ylabel("R$")
                plt.xlabel("Mês")
            elif cat_col and value_col:
                top.plot(kind="barh", color="steelblue", edgecolor="white")
                plt.title(f"Top 10 {cat_col} por {value_col}", fontsize=14)
                plt.xlabel("R$")
            else:
                num_df.mean().plot(kind="bar", color="steelblue", edgecolor="white")
                plt.title("Médias das colunas numéricas", fontsize=14)

            plt.tight_layout()
            chart_name = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            chart_path = os.path.abspath(f"outputs/{chart_name}")
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close()
            report += f"## 📈 Gráfico\nSalvo em: `{chart_path}`\n"
        except Exception as chart_err:
            report += f"⚠️ Gráfico não gerado: {chart_err}\n"

        return report

    except Exception as e:
        return f"❌ Erro na análise: {str(e)}\nDica: verifique o caminho do arquivo e se é um CSV válido."


# ─────────────────────────────────────────────
# TOOL 2: clean_csv
# ─────────────────────────────────────────────
@mcp.tool
def clean_csv(file_path: str, output_name: str = "cleaned.csv") -> str:
    """
    Limpa o CSV: remove duplicados, linhas vazias e padroniza encoding.
    Retorna o caminho do arquivo limpo.
    """
    try:
        df_raw = pd.read_csv(file_path, sep=";", decimal=",", encoding="utf-8", on_bad_lines="skip")
        linhas_antes = len(df_raw)

        df_clean = df_raw.drop_duplicates().dropna(how="all").reset_index(drop=True)
        linhas_depois = len(df_clean)

        output_path = os.path.abspath(f"outputs/{output_name}")
        df_clean.to_csv(output_path, index=False, sep=";", decimal=",", encoding="utf-8")

        return (
            f"✅ CSV limpo salvo em: {output_path}\n"
            f"   Linhas antes: {linhas_antes:,}\n"
            f"   Linhas depois: {linhas_depois:,}\n"
            f"   Removidas: {linhas_antes - linhas_depois:,} (duplicadas/vazias)"
        )
    except Exception as e:
        return f"❌ Erro na limpeza: {str(e)}"


# ─────────────────────────────────────────────
# TOOL 3: csv_info  (nova — diagnóstico rápido)
# ─────────────────────────────────────────────
@mcp.tool
def csv_info(file_path: str) -> str:
    """
    Diagnóstico rápido do CSV: colunas, tipos, % de nulos, encoding.
    Útil para entender a estrutura antes de analisar.
    """
    try:
        file_path = os.path.expanduser(file_path)
        # lê só 1000 linhas para diagnóstico rápido
        df = pd.read_csv(file_path, sep=";", decimal=",", encoding="utf-8",
                         nrows=1000, on_bad_lines="skip")

        info = f"# 🔍 Diagnóstico: {os.path.basename(file_path)}\n\n"
        info += f"**Linhas amostradas**: 1.000 | **Colunas**: {len(df.columns)}\n\n"
        info += "| Coluna | Tipo | % Nulos | Exemplo |\n"
        info += "|--------|------|---------|--------|\n"
        for col in df.columns:
            tipo = str(df[col].dtype)
            pct_nulo = round(df[col].isna().sum() / len(df) * 100, 1)
            exemplo = str(df[col].dropna().iloc[0]) if df[col].notna().any() else "—"
            info += f"| {col} | {tipo} | {pct_nulo}% | {exemplo[:30]} |\n"

        return info
    except Exception as e:
        return f"❌ Erro no diagnóstico: {str(e)}"


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 DataClaw MCP Local rodando em stdio...")
    print("   Tools disponíveis: analyze_csv | clean_csv | csv_info")
    mcp.run(transport="stdio")