import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ✅ FIX 1: Cria a pasta antes de tentar salvar
os.makedirs("testes", exist_ok=True)

np.random.seed(42)
random.seed(42)

n = 20000

data = {
    "ID_Venda": range(1, n + 1),
    "Data": [
        (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))).strftime(
            random.choice(["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"])
        )
        for _ in range(n)
    ],
    "Cliente": [f"Cliente_{random.randint(1000, 9999)}" for _ in range(n)],
    "Produto": [
        random.choice(["Notebook Dell", "iPhone 15", "Monitor LG", "Teclado Mecânico", "Mouse Logitech"])
        for _ in range(n)
    ],
    "Categoria": [random.choice(["Eletrônicos", "Acessórios", "Periféricos"]) for _ in range(n)],
    "Valor_Unitario": [round(random.uniform(89.9, 8999.9), 2) for _ in range(n)],
    "Quantidade": [random.randint(1, 15) for _ in range(n)],
    "Total_Venda": [round(random.uniform(100, 15000), 2) for _ in range(n)],
    "Desconto_Pct": [round(random.uniform(0, 30), 1) for _ in range(n)],
    "Status": [random.choice(["Concluída", "Pendente", "Cancelada", ""]) for _ in range(n)],
    "Cidade": [
        random.choice(["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba", "Porto Alegre"])
        for _ in range(n)
    ],
    "Vendedor": [random.choice(["João", "Maria", "Pedro", "Ana", "Carlos"]) for _ in range(n)],
}

df = pd.DataFrame(data)

# Problemas reais que agentes falham:
df.loc[::7, "Data"] = "inválida"           # datas ruins
df.loc[::13, "Valor_Unitario"] = np.nan    # NaN espalhados
df = pd.concat([df, df.head(500)], ignore_index=True)  # 500 duplicados
df.loc[::19, "Quantidade"] = "dez"         # texto onde deveria ser número
df.loc[500:550, "Cidade"] = "são paulo"    # case/acentos inconsistentes
df.loc[::31, "Status"] = ""               # status vazio

df.to_csv(
    "testes/vendas_sujos_20k.csv",
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8",
)

print("✅ CSV DIABÓLICO gerado: testes/vendas_sujos_20k.csv")
print(f"   Total de linhas: {len(df):,}")
print("   Problemas incluídos: datas mistas, NaN, duplicados, texto em número, acentos, vazios")