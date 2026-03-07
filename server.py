"""
DataClaw MCP v3.0 — JSON Architecture
======================================
PRINCÍPIO FUNDAMENTAL:
  - Pandas calcula TUDO
  - Server retorna JSON com números prontos e labels inequívocos
  - LLM só formata e apresenta — NUNCA recalcula

Por que isso resolve o problema de 20k linhas:
  - O CSV nunca entra no contexto do LLM
  - Apenas o JSON final (< 2KB) vai para o LLM
  - Impossível reinterpretação: labels são autoexplicativos
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

from fastmcp import FastMCP
import pandas as pd
import numpy as np

mcp = FastMCP("DataClaw", version="3.0.0")

_CACHE: dict[str, tuple] = {}
_CACHE_LIMIT = 3

BASE_DIR   = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS INTERNOS — pandas puro, zero LLM
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:8]


def _detect_format(file_path: str) -> tuple:
    """Detecta separador, decimal e encoding lendo apenas 4KB."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(file_path, "r", encoding=enc, errors="strict") as f:
                sample = f.read(4096)
            counts = {s: sample.count(s) for s in [";", ",", "\t"]}
            sep     = max(counts, key=counts.get)
            decimal = "," if sep == ";" else "."
            return sep, decimal, enc
        except Exception:
            continue
    return ",", ".", "utf-8"


_PTBR = {
    "zero":0,"um":1,"uma":1,"dois":2,"duas":2,"tres":3,"três":3,
    "quatro":4,"cinco":5,"seis":6,"sete":7,"oito":8,"nove":9,
    "dez":10,"onze":11,"doze":12,"treze":13,"catorze":14,
    "quatorze":14,"quinze":15,"dezesseis":16,"dezessete":17,
    "dezoito":18,"dezenove":19,"vinte":20
}

def _coerce_numeric(series: pd.Series, threshold: float = 0.75) -> pd.Series:
    """Converte object→número com segurança. Resolve PT-BR ('dez'→10)."""
    mapped = series.astype(str).str.strip().str.lower().map(
        lambda x: str(_PTBR[x]) if x in _PTBR else x
    )
    cleaned = (
        mapped
        .str.replace(r"[^\d.,\-]", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    converted = pd.to_numeric(cleaned, errors="coerce")
    ratio = converted.notna().sum() / max(len(series.dropna()), 1)
    return converted if ratio >= threshold else series


def _normalize_text(series: pd.Series) -> pd.Series:
    """Title case + strip em colunas categóricas."""
    if series.dtype != object:
        return series
    return series.astype(str).str.strip().str.title()


def _load(file_path: str) -> tuple:
    """
    Retorna (df_raw, df_clean).
    df_raw:  original — para métricas brutas
    df_clean: limpo e normalizado — para todas as análises
    Usa cache para evitar releitura.
    """
    key = _cache_key(file_path)
    if key in _CACHE:
        return _CACHE[key]

    sep, decimal, enc = _detect_format(file_path)

    with open(file_path, "r", encoding=enc, errors="replace") as f:
        total_lines = sum(1 for _ in f) - 1

    kw = dict(sep=sep, decimal=decimal, encoding=enc,
              on_bad_lines="skip", low_memory=False)

    if total_lines <= 50_000:
        df_raw = pd.read_csv(file_path, **kw)
    else:
        chunks = [c for c in pd.read_csv(file_path, chunksize=5000, **kw)]
        df_raw = pd.concat(chunks, ignore_index=True)

    df_clean = df_raw.drop_duplicates().dropna(how="all").reset_index(drop=True)

    # Converte numéricos (resolve "dez" → 10)
    for col in df_clean.select_dtypes(include=["object","string"]).columns:
        df_clean[col] = _coerce_numeric(df_clean[col])

    # Normaliza categóricas (resolve "são paulo" → "São Paulo")
    for col in df_clean.select_dtypes(include=["object","string"]).columns:
        if df_clean[col].nunique() < 200:
            df_clean[col] = _normalize_text(df_clean[col])

    if len(_CACHE) >= _CACHE_LIMIT:
        del _CACHE[next(iter(_CACHE))]
    _CACHE[key] = (df_raw, df_clean)
    return df_raw, df_clean


def _find_col(df: pd.DataFrame, keywords: list[str],
              dtype_filter: str = None) -> str | None:
    """Encontra coluna por keywords no nome."""
    num_cols = set(df.select_dtypes(include="number").columns)
    for kw in keywords:
        for col in df.columns:
            if kw in col.lower():
                if dtype_filter == "numeric" and col not in num_cols:
                    continue
                return col
    return None


def _safe(val):
    """Converte tipos numpy para Python nativo (para json.dumps)."""
    if isinstance(val, (np.integer,)):  return int(val)
    if isinstance(val, (np.floating,)): return round(float(val), 2)
    if isinstance(val, (np.bool_,)):    return bool(val)
    if pd.isna(val):                    return None
    return val


def _to_json(data: dict) -> str:
    """Serializa dict para JSON com indentação. Limite de 3500 chars."""
    raw = json.dumps(data, ensure_ascii=False, indent=2, default=_safe)
    if len(raw) > 3500:
        # Trunca listas longas antes de serializar novamente
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 8:
                data[k] = v[:8]
                data[f"{k}_truncated"] = True
        raw = json.dumps(data, ensure_ascii=False, indent=2, default=_safe)
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — csv_info
# Diagnóstico de estrutura. Não calcula métricas de negócio.
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool
def csv_info(file_path: str) -> str:
    """
    Diagnóstico da estrutura do CSV: colunas, tipos, nulos e tamanho.
    Retorna JSON. Chame ANTES de analyze_csv.

    Args:
        file_path: Caminho absoluto do arquivo CSV.
    """
    try:
        file_path = os.path.expanduser(file_path)
        if not os.path.exists(file_path):
            return _to_json({"error": f"Arquivo não encontrado: {file_path}"})

        size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
        sep, decimal, enc = _detect_format(file_path)

        df = pd.read_csv(file_path, sep=sep, decimal=decimal,
                         encoding=enc, nrows=2000, on_bad_lines="skip")

        with open(file_path, "r", encoding=enc, errors="replace") as f:
            total_lines = sum(1 for _ in f) - 1

        columns = []
        for col in df.columns:
            columns.append({
                "name":         col,
                "type":         str(df[col].dtype),
                "null_pct":     round(df[col].isna().sum() / len(df) * 100, 1),
                "unique_count": int(df[col].nunique()),
                "sample_value": str(df[col].dropna().iloc[0])[:40] if df[col].notna().any() else None
            })

        return _to_json({
            "file_name":        Path(file_path).name,
            "total_rows":       total_lines,
            "total_columns":    len(df.columns),
            "file_size_mb":     size_mb,
            "separator":        sep,
            "decimal":          decimal,
            "encoding":         enc,
            "columns":          columns
        })

    except Exception as e:
        return _to_json({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — analyze_csv
# Análise completa. Pandas calcula TUDO. LLM recebe JSON com números prontos.
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool
def analyze_csv(
    file_path: str,
    focus: str = "full"
) -> str:
    """
    Análise completa de CSV grande (10k–500k linhas).
    Pandas processa tudo localmente. LLM recebe apenas JSON com números calculados.
    Nunca envia dados brutos para o contexto.

    Args:
        file_path: Caminho absoluto do arquivo CSV.
        focus: "full" | "financial" | "quality" | "trends" | "ranking"
    """
    try:
        df_raw, df = _load(file_path)

        value_col = _find_col(df, ["total_venda","total","receita","faturamento",
                                   "revenue","amount","valor","price"], "numeric")
        date_col  = _find_col(df, ["data","date","dt_","periodo","timestamp","created"])
        cat_col   = _find_col(df, ["produto","product","vendedor","seller",
                                   "categoria","category","cidade","city"])
        status_col = _find_col(df, ["status","situacao","estado_venda"])

        dups   = len(df_raw) - len(df)
        result = {
            "file_name":  Path(file_path).name,
            "analyzed_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
        }

        # ── BLOCO: qualidade ─────────────────────────────────────────────────
        null_cells  = int(df.isna().sum().sum())
        total_cells = df.shape[0] * df.shape[1]

        result["data_quality"] = {
            "rows_in_file":               len(df_raw),
            "rows_after_dedup":           len(df),
            "duplicate_rows_removed":     dups,
            "null_cells":                 null_cells,
            "total_cells":                total_cells,
            "null_pct":                   round(null_cells / total_cells * 100, 2),
        }

        # ── BLOCO: financeiro ────────────────────────────────────────────────
        if value_col and focus in ["full", "financial"]:
            t_bruto = float(df_raw[value_col].sum())
            t_limpo = float(df[value_col].sum())
            media   = float(df[value_col].mean())
            maximo  = float(df[value_col].max())
            minimo  = float(df[value_col].min())
            nulos   = int(df[value_col].isna().sum())

            result["financial"] = {
                "value_column":                      value_col,
                "total_WITH_duplicates":             round(t_bruto, 2),
                "total_WITHOUT_duplicates":          round(t_limpo, 2),
                "difference_caused_by_duplicates":   round(t_bruto - t_limpo, 2),
                "average_per_row":                   round(media, 2),
                "maximum_value":                     round(maximo, 2),
                "minimum_value":                     round(minimo, 2),
                "null_values_ignored":               nulos,
                "CORRECT_VALUE_TO_USE":              round(t_limpo, 2),
                "WARNING": "Always use total_WITHOUT_duplicates for reporting"
            }

        # ── BLOCO: datas e tendência ─────────────────────────────────────────
        if date_col and focus in ["full", "trends"]:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
            valid    = df[date_col].notna()
            inv_n    = int((~valid).sum())
            inv_pct  = round(inv_n / len(df) * 100, 2)

            result["dates"] = {
                "date_column":          date_col,
                "invalid_dates_count":  inv_n,
                "invalid_dates_pct":    inv_pct,
                "valid_dates_count":    int(valid.sum()),
                "WARNING": f"{inv_n} rows ({inv_pct}%) excluded from trend analysis"
            }

            if valid.sum() > 0 and value_col:
                monthly = (
                    df[valid]
                    .groupby(df.loc[valid, date_col].dt.to_period("M"))[value_col]
                    .agg(total="sum", count="count")
                    .round(2)
                    .tail(12)
                )
                trend = []
                for period, row in monthly.iterrows():
                    trend.append({
                        "month":              str(period),
                        "total_revenue":      round(float(row["total"]), 2),
                        "transaction_count":  int(row["count"])
                    })
                result["monthly_trend"] = trend

        # ── BLOCO: rankings ──────────────────────────────────────────────────
        if cat_col and focus in ["full", "ranking"]:
            if value_col:
                top = (df.groupby(cat_col)[value_col]
                         .sum()
                         .sort_values(ascending=False)
                         .head(8))
                ranking = [
                    {"rank": i+1, "name": str(k), "total_revenue": round(float(v), 2)}
                    for i, (k, v) in enumerate(top.items())
                ]
            else:
                top = df[cat_col].value_counts().head(8)
                ranking = [
                    {"rank": i+1, "name": str(k), "count": int(v)}
                    for i, (k, v) in enumerate(top.items())
                ]
            result[f"ranking_by_{cat_col}"] = ranking

        # ── BLOCO: outliers (IQR + Z-score duplo) ───────────────────────────
        if value_col and focus in ["full", "financial"]:
            col_data = df[value_col].dropna()

            # IQR
            q1, q3 = col_data.quantile([0.25, 0.75])
            iqr    = q3 - q1
            lo_iqr = float(q1 - 1.5 * iqr)
            hi_iqr = float(q3 + 1.5 * iqr)
            out_iqr = df[(df[value_col] < lo_iqr) | (df[value_col] > hi_iqr)]

            # Z-score (detecta outliers extremos que IQR pode perder)
            z_scores = (col_data - col_data.mean()) / col_data.std()
            out_z    = df.loc[col_data.index[z_scores.abs() > 3]]

            # Top 5 outliers mais extremos para inspeção
            top_outliers = (
                df.nlargest(5, value_col)[[value_col] + ([cat_col] if cat_col else [])]
                .reset_index(drop=True)
            )
            extreme = [
                {
                    "rank": i+1,
                    "value": round(float(row[value_col]), 2),
                    "category": str(row[cat_col]) if cat_col and cat_col in row else None
                }
                for i, row in top_outliers.iterrows()
            ]

            result["outliers"] = {
                "method_iqr": {
                    "normal_range_min": round(lo_iqr, 2),
                    "normal_range_max": round(hi_iqr, 2),
                    "outliers_count":   len(out_iqr),
                    "outliers_pct":     round(len(out_iqr) / len(df) * 100, 2)
                },
                "method_zscore": {
                    "threshold":       "z > 3",
                    "outliers_count":  len(out_z),
                    "outliers_pct":    round(len(out_z) / len(df) * 100, 2)
                },
                "top_5_extreme_values": extreme
            }

        # ── BLOCO: anomalias por categoria (Carlos suspeito, etc) ────────────
        if status_col and cat_col and focus in ["full", "ranking"]:
            cancel_vals = ["cancelada", "cancelado", "cancelled", "canceled"]
            df["_is_cancel"] = df[status_col].astype(str).str.lower().str.strip().isin(cancel_vals)

            cancel_by_cat = df.groupby(cat_col)["_is_cancel"].agg(
                total="count", cancellations="sum"
            )
            cancel_by_cat["cancel_rate_pct"] = (
                cancel_by_cat["cancellations"] / cancel_by_cat["total"] * 100
            ).round(2)

            mean_rate = float(cancel_by_cat["cancel_rate_pct"].mean())
            std_rate  = float(cancel_by_cat["cancel_rate_pct"].std())
            threshold = mean_rate + 1.5 * std_rate

            anomalies = []
            for name, row in cancel_by_cat.iterrows():
                rate = float(row["cancel_rate_pct"])
                anomalies.append({
                    "name":                str(name),
                    "total_sales":         int(row["total"]),
                    "cancellations":       int(row["cancellations"]),
                    "cancel_rate_pct":     rate,
                    "is_anomaly":          rate > threshold,
                    "deviation_from_mean": round(rate - mean_rate, 2)
                })

            anomalies.sort(key=lambda x: x["cancel_rate_pct"], reverse=True)
            df.drop(columns=["_is_cancel"], inplace=True)

            result[f"cancellation_analysis_by_{cat_col}"] = {
                "team_average_cancel_rate_pct": round(mean_rate, 2),
                "anomaly_threshold_pct":        round(threshold, 2),
                "details":                      anomalies
            }

        # ── BLOCO: qualidade de colunas específicas ──────────────────────────
        nota_col = _find_col(df, ["nota","rating","score","avaliacao"])
        if nota_col and focus in ["full", "quality"]:
            invalid_high = int((df[nota_col] > 5.0).sum())
            invalid_low  = int((df[nota_col] < 1.0).sum())
            result["rating_quality"] = {
                "column":              nota_col,
                "expected_range":      "1.0 to 5.0",
                "values_above_5":      invalid_high,
                "values_below_1":      invalid_low,
                "total_invalid":       invalid_high + invalid_low,
                "is_clean":            (invalid_high + invalid_low) == 0
            }

        # ── BLOCO: sazonalidade por categoria ────────────────────────────────
        if date_col and value_col and cat_col and focus in ["full", "trends"]:
            if df[date_col].notna().sum() > 0:
                valid_df = df[df[date_col].notna()].copy()

                # Coeficiente de variação mensal por categoria
                seasonal = []
                for name, grp in valid_df.groupby(cat_col):
                    monthly_rev = (
                        grp.groupby(grp[date_col].dt.month)[value_col].sum()
                    )
                    if len(monthly_rev) < 3:
                        continue
                    cv   = float(monthly_rev.std() / monthly_rev.mean() * 100)
                    peak = int(monthly_rev.idxmax())
                    months = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
                              7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
                    seasonal.append({
                        "category":                str(name),
                        "seasonality_cv_pct":      round(cv, 1),
                        "peak_month":              months.get(peak, str(peak)),
                        "peak_month_revenue":      round(float(monthly_rev.max()), 2),
                        "average_monthly_revenue": round(float(monthly_rev.mean()), 2),
                        "peak_vs_average_ratio":   round(float(monthly_rev.max() / monthly_rev.mean()), 2),
                        "has_strong_seasonality":  cv > 20
                    })

                seasonal.sort(key=lambda x: x["seasonality_cv_pct"], reverse=True)
                result["seasonality_by_category"] = seasonal

        return _to_json(result)

    except Exception as e:
        return _to_json({"error": str(e), "tip": "Run csv_info() first to validate the file."})


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — query_csv
# Filtro + agrupamento ad-hoc. Retorna JSON.
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool
def query_csv(
    file_path: str,
    group_by: str = "",
    filter_col: str = "",
    filter_val: str = "",
    metric: str = "sum",
    top_n: int = 10,
) -> str:
    """
    Consulta ad-hoc: agrupa e filtra sem enviar dados ao LLM.
    Retorna JSON com resultados calculados pelo pandas.

    Args:
        file_path:  Caminho absoluto do arquivo CSV.
        group_by:   Coluna para agrupar. Ex: "Vendedor", "Produto"
        filter_col: Coluna para filtrar. Ex: "Status"
        filter_val: Valor do filtro (case-insensitive). Ex: "Cancelada"
        metric:     "sum" | "count" | "mean" | "min" | "max"
        top_n:      Máximo de resultados (máx 15)
    """
    try:
        _, df = _load(file_path)
        top_n = min(top_n, 15)

        # Filtro
        if filter_col and filter_val:
            if filter_col not in df.columns:
                return _to_json({
                    "error": f"Column '{filter_col}' not found",
                    "available_columns": list(df.columns)
                })
            mask = df[filter_col].astype(str).str.strip().str.lower() == filter_val.strip().lower()
            df   = df[mask]
            if df.empty:
                return _to_json({
                    "result": "no_rows_found",
                    "filter": f"{filter_col} = '{filter_val}'"
                })

        # Agrupamento
        rows_total = len(df)
        if group_by:
            if group_by not in df.columns:
                return _to_json({
                    "error": f"Column '{group_by}' not found",
                    "available_columns": list(df.columns)
                })
            num_cols = df.select_dtypes(include="number").columns.tolist()
            agg_func = metric if metric in ["sum","mean","count","min","max"] else "sum"

            if not num_cols:
                counts = df[group_by].value_counts().head(top_n)
                rows   = [{"rank": i+1, "name": str(k), "count": int(v)}
                          for i, (k, v) in enumerate(counts.items())]
            else:
                grouped = (
                    df.groupby(group_by)[num_cols]
                    .agg(agg_func)
                    .round(2)
                    .sort_values(num_cols[0], ascending=False)
                    .head(top_n)
                    .reset_index()
                )
                rows = []
                for i, row in grouped.iterrows():
                    entry = {"rank": i+1, "name": str(row[group_by])}
                    for col in num_cols:
                        entry[col] = _safe(row[col])
                    rows.append(entry)
        else:
            rows = df.head(top_n).to_dict(orient="records")

        return _to_json({
            "file":          Path(file_path).name,
            "filter":        f"{filter_col}='{filter_val}'" if filter_col else None,
            "group_by":      group_by or None,
            "metric":        metric,
            "rows_matched":  rows_total,
            "results":       rows
        })

    except Exception as e:
        return _to_json({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — clean_csv
# Limpa e salva. Retorna JSON com estatísticas. NUNCA retorna dados brutos.
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool
def clean_csv(file_path: str, output_name: str = "") -> str:
    """
    Limpa o CSV: remove duplicatas, linhas vazias, normaliza texto categórico.
    Salva em outputs/ e retorna JSON com estatísticas. Nunca retorna dados brutos.

    Args:
        file_path:   Caminho absoluto do arquivo CSV original.
        output_name: Nome do arquivo de saída. Ex: vendas_limpo.csv
    """
    try:
        file_path = os.path.expanduser(file_path)
        sep, decimal, enc = _detect_format(file_path)

        df_raw = pd.read_csv(file_path, sep=sep, decimal=decimal,
                             encoding=enc, on_bad_lines="skip", low_memory=False)
        n_raw  = len(df_raw)

        df_clean = df_raw.drop_duplicates().dropna(how="all").reset_index(drop=True)

        # Normaliza categóricas
        for col in df_clean.select_dtypes(include=["object","string"]).columns:
            if df_clean[col].nunique() < 200:
                df_clean[col] = _normalize_text(df_clean[col])

        n_clean = len(df_clean)

        if not output_name:
            output_name = f"{Path(file_path).stem}_clean.csv"

        out_path = OUTPUT_DIR / output_name
        df_clean.to_csv(out_path, index=False, sep=sep, decimal=decimal, encoding="utf-8")

        # Impacto financeiro das linhas removidas
        value_col = _find_col(df_raw, ["total_venda","total","receita","revenue",
                                        "amount","valor"], "numeric")
        financial_impact = None
        if value_col:
            t_raw   = float(df_raw[value_col].sum())
            t_clean = float(df_clean[value_col].sum())
            financial_impact = {
                "total_before": round(t_raw, 2),
                "total_after":  round(t_clean, 2),
                "difference":   round(t_raw - t_clean, 2)
            }

        return _to_json({
            "status":                "success",
            "output_file":           str(out_path.absolute()),
            "rows_before":           n_raw,
            "rows_after":            n_clean,
            "rows_removed":          n_raw - n_clean,
            "text_normalized":       True,
            "file_size_kb":          round(out_path.stat().st_size / 1024, 1),
            "financial_impact":      financial_impact,
            "next_step":             f"analyze_csv('{out_path.absolute()}')"
        })

    except Exception as e:
        return _to_json({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.stderr.write("DataClaw MCP v3.0 — JSON Architecture\n")
    sys.stderr.write(f"  BASE_DIR:   {BASE_DIR}\n")
    sys.stderr.write(f"  OUTPUT_DIR: {OUTPUT_DIR}\n")
    sys.stderr.flush()
    mcp.run(transport="stdio")