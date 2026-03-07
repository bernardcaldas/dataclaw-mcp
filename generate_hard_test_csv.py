"""
generate_v3_test.py — CSV de Teste para DataClaw MCP v3.0
Desafios projetados especificamente para validar a arquitetura JSON.
Roda: python generate_v3_test.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, os, json

os.makedirs("testes", exist_ok=True)
np.random.seed(99)
random.seed(99)

n = 20000
produtos   = ["Notebook Dell", "iPhone 15", "Monitor LG", "Teclado Mecânico", "Mouse Logitech"]
cidades    = ["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba", "Porto Alegre"]
vendedores = ["João", "Maria", "Pedro", "Ana", "Carlos"]
categorias = ["Eletrônicos", "Acessórios", "Periféricos"]
regioes    = ["Sudeste", "Sul", "Nordeste", "Norte", "Centro-Oeste"]
canais     = ["Online", "Loja Física", "Revendedor", "Televendas"]
status_ok  = ["Concluída", "Pendente", "Cancelada"]

# ── BASE LIMPA ────────────────────────────────────────────────────────────────
produto_l  = [random.choice(produtos)   for _ in range(n)]
cidade_l   = [random.choice(cidades)    for _ in range(n)]
vendedor_l = [random.choice(vendedores) for _ in range(n)]
cat_l      = [random.choice(categorias) for _ in range(n)]
status_l   = [random.choice(status_ok)  for _ in range(n)]
qtd_l      = [random.randint(1, 15)     for _ in range(n)]
val_l      = [round(random.uniform(89.9, 8999.9), 2) for _ in range(n)]
desc_l     = [round(random.uniform(0, 30), 1) for _ in range(n)]
total_l    = [round(val_l[i] * qtd_l[i] * (1 - desc_l[i]/100), 2) for i in range(n)]

df = pd.DataFrame({
    "ID_Venda":       range(1, n+1),
    "Data":           [(datetime(2024,1,1)+timedelta(days=random.randint(0,364)))
                       .strftime(random.choice(["%d/%m/%Y","%Y-%m-%d","%d-%m-%Y"]))
                       for _ in range(n)],
    "Cliente":        [f"CLI-{random.randint(10000,99999)}" for _ in range(n)],
    "Produto":        produto_l,
    "Categoria":      cat_l,
    "Regiao":         [random.choice(regioes)  for _ in range(n)],
    "Canal_Venda":    [random.choice(canais)   for _ in range(n)],
    "Vendedor":       vendedor_l,
    "Cidade":         cidade_l,
    "Valor_Unitario": val_l,
    "Quantidade":     [str(q) for q in qtd_l],
    "Desconto_Pct":   desc_l,
    "Total_Venda":    total_l,
    "Status":         status_l,
    "Nota_Cliente":   [round(random.uniform(1.0,5.0),1) for _ in range(n)],
    "Prazo_Entrega":  [random.randint(1,30) for _ in range(n)],
    "Margem_Pct":     [round(random.uniform(5.0,45.0),1) for _ in range(n)],
})

# ─────────────────────────────────────────────────────────────────────────────
# DESAFIO 1 — DUPLICATAS PARCIAIS (mesmos dados, ID diferente)
# Mais difícil que duplicatas exatas — testa se o MCP detecta
# ─────────────────────────────────────────────────────────────────────────────
dup_parcial = df.iloc[200:350].copy()
dup_parcial["ID_Venda"] = range(n+1, n+1+len(dup_parcial))  # ID diferente, resto igual
df = pd.concat([df, dup_parcial], ignore_index=True)
N_DUP_PARCIAL = len(dup_parcial)

# ─────────────────────────────────────────────────────────────────────────────
# DESAFIO 2 — DATAS EM 4 FORMATOS + TIMESTAMPS
# ─────────────────────────────────────────────────────────────────────────────
df.loc[::9,  "Data"] = "inválida"
df.loc[::41, "Data"] = "2024/06/15 14:30:00"         # timestamp ISO
df.loc[::53, "Data"] = "15-Jun-2024"                  # formato inglês

# ─────────────────────────────────────────────────────────────────────────────
# DESAFIO 3 — TEXTO NUMÉRICO MISTO PT-BR + EN
# ─────────────────────────────────────────────────────────────────────────────
df.loc[::19, "Quantidade"] = "dez"
df.loc[::29, "Quantidade"] = "five"       # inglês — NÃO deve converter
df.loc[::37, "Quantidade"] = "12.0"       # float como string

# ─────────────────────────────────────────────────────────────────────────────
# DESAFIO 4 — NORMALIZAÇÃO DE TEXTO (5 variantes de cidade)
# ─────────────────────────────────────────────────────────────────────────────
df.loc[500:520,  "Cidade"] = "são paulo"
df.loc[1000:1015,"Cidade"] = "SAO PAULO"
df.loc[2000:2008,"Cidade"] = "Sao Paulo"
df.loc[3000:3005,"Cidade"] = "S. Paulo"        # abreviação — NÃO normalizar
df.loc[4000:4003,"Cidade"] = "SP"              # sigla — NÃO normalizar

# ─────────────────────────────────────────────────────────────────────────────
# DESAFIO 5 — OUTLIERS EM 3 COLUNAS DIFERENTES
# ─────────────────────────────────────────────────────────────────────────────
# Outliers em Total_Venda
for i in [100,500,1000,3000,7000,12000,15000,18000,19000]:
    if i < len(df): df.loc[i,"Total_Venda"] = round(random.uniform(120000,250000),2)
for i in [200,600,1500,5000,9000]:
    if i < len(df): df.loc[i,"Total_Venda"] = round(random.uniform(0.01,0.99),2)

# Outliers em Prazo_Entrega (entregas impossíveis)
for i in [300,800,2000,4000]: 
    if i < len(df): df.loc[i,"Prazo_Entrega"] = random.randint(180,365)

# Outliers em Margem_Pct (margens impossíveis)
for i in [400,900,2500]: 
    if i < len(df): df.loc[i,"Margem_Pct"] = round(random.uniform(95.0,150.0),1)

N_OUTLIERS_VENDA   = 9 + 5
N_OUTLIERS_PRAZO   = 4
N_OUTLIERS_MARGEM  = 3

# ─────────────────────────────────────────────────────────────────────────────
# DESAFIO 6 — SAZONALIDADE DUPLA
# iPhone 15: pico em novembro (Black Friday)
# Monitor LG: pico em janeiro e julho (início semestre)
# ─────────────────────────────────────────────────────────────────────────────
df["_dt"] = pd.to_datetime(df["Data"], errors="coerce", format="mixed", dayfirst=True)

mask_iphone_nov = (df["_dt"].dt.month == 11) & (df["Produto"] == "iPhone 15")
mask_monitor_jan = (df["_dt"].dt.month == 1)  & (df["Produto"] == "Monitor LG")
mask_monitor_jul = (df["_dt"].dt.month == 7)  & (df["Produto"] == "Monitor LG")

df.loc[mask_iphone_nov,  "Total_Venda"] = (df.loc[mask_iphone_nov,  "Total_Venda"] * 3.1).round(2)
df.loc[mask_monitor_jan, "Total_Venda"] = (df.loc[mask_monitor_jan, "Total_Venda"] * 2.4).round(2)
df.loc[mask_monitor_jul, "Total_Venda"] = (df.loc[mask_monitor_jul, "Total_Venda"] * 2.1).round(2)
df.drop(columns=["_dt"], inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# DESAFIO 7 — DOIS VENDEDORES SUSPEITOS (padrões diferentes)
# Carlos: taxa de cancelamento alta
# Maria: ticket médio artificialmente alto (fraude de valor?)
# ─────────────────────────────────────────────────────────────────────────────
idx_carlos = df[df["Vendedor"] == "Carlos"].index[:350]
df.loc[idx_carlos, "Status"] = "Cancelada"

idx_maria = df[df["Vendedor"] == "Maria"].index[:200]
df.loc[idx_maria, "Total_Venda"] = (df.loc[idx_maria, "Total_Venda"] * 4.5).round(2)

# ─────────────────────────────────────────────────────────────────────────────
# DESAFIO 8 — NOTAS IMPOSSÍVEIS E PRAZO ZERO
# ─────────────────────────────────────────────────────────────────────────────
df.loc[350:360,  "Nota_Cliente"] = 9.5
df.loc[700:705,  "Nota_Cliente"] = -2.0
df.loc[1200:1202,"Nota_Cliente"] = 0.0
df.loc[2000:2003,"Prazo_Entrega"] = 0     # prazo zero = impossível

# ─────────────────────────────────────────────────────────────────────────────
# DESAFIO 9 — CANAL ONLINE DOMINANTE EM VALOR, NÃO EM VOLUME
# Testa se o MCP consegue distinguir volume vs valor por canal
# ─────────────────────────────────────────────────────────────────────────────
idx_online = df[df["Canal_Venda"] == "Online"].index[:300]
df.loc[idx_online, "Total_Venda"] = (df.loc[idx_online, "Total_Venda"] * 3.8).round(2)

# ─────────────────────────────────────────────────────────────────────────────
# DESAFIO 10 — NaN ESTRATÉGICOS EM COLUNAS CRÍTICAS
# ─────────────────────────────────────────────────────────────────────────────
df.loc[::13, "Valor_Unitario"] = np.nan
df.loc[::41, "Margem_Pct"]    = np.nan
df.loc[::53, "Status"]        = ""
df.loc[::61, "Nota_Cliente"]  = np.nan

# ─────────────────────────────────────────────────────────────────────────────
# SALVA
# ─────────────────────────────────────────────────────────────────────────────
df.to_csv("testes/vendas_v3_desafio.csv", index=False,
          sep=";", decimal=",", encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# GABARITO OFICIAL — calculado aqui, não pelo MCP
# ─────────────────────────────────────────────────────────────────────────────
df_c = df.drop_duplicates(
    subset=[c for c in df.columns if c != "ID_Venda"]
).dropna(how="all").reset_index(drop=True)

t_bruto = df["Total_Venda"].sum()
t_limpo = df_c["Total_Venda"].sum()
dups    = len(df) - len(df_c)

df_c["_dt"] = pd.to_datetime(df_c["Data"], errors="coerce", format="mixed", dayfirst=True)
inv_datas   = df_c["_dt"].isna().sum()

top5_prod = df_c.groupby("Produto")["Total_Venda"].sum().sort_values(ascending=False).head(5)
top5_vend = df_c.groupby("Vendedor")["Total_Venda"].sum().sort_values(ascending=False).head(5)
top_canal_vol = df_c["Canal_Venda"].value_counts().head(4)
top_canal_val = df_c.groupby("Canal_Venda")["Total_Venda"].sum().sort_values(ascending=False).head(4)

q1,q3 = df_c["Total_Venda"].quantile([0.25,0.75])
iqr   = q3-q1
lo,hi = q1-1.5*iqr, q3+1.5*iqr
out_iqr = df_c[(df_c["Total_Venda"]<lo)|(df_c["Total_Venda"]>hi)]
z   = (df_c["Total_Venda"]-df_c["Total_Venda"].mean())/df_c["Total_Venda"].std()
out_z = df_c[z.abs()>3]

carlos_c = (df_c[df_c["Vendedor"]=="Carlos"]["Status"]=="Cancelada").sum()
outros_c = df_c[df_c["Vendedor"]!="Carlos"].groupby("Vendedor").apply(
    lambda x: (x["Status"]=="Cancelada").sum()).mean()

maria_ticket  = df_c[df_c["Vendedor"]=="Maria"]["Total_Venda"].mean()
outros_ticket = df_c[df_c["Vendedor"]!="Maria"].groupby("Vendedor")["Total_Venda"].mean().mean()

inv_notas = df_c[(df_c["Nota_Cliente"]>5.0)|(df_c["Nota_Cliente"]<1.0)].dropna(subset=["Nota_Cliente"])
inv_prazo  = df_c[df_c["Prazo_Entrega"]==0]

sp_var = df_c["Cidade"].value_counts()
sp_var = sp_var[[i for i in sp_var.index
                 if any(x in str(i).lower() for x in ["paulo","sp","s."])]]

# sazonalidade
months_map = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
              7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
saz = {}
for prod, grp in df_c.groupby("Produto"):
    m = grp.groupby(grp["_dt"].dt.month)["Total_Venda"].sum()
    if len(m)<3: continue
    saz[prod] = {
        "peak_month": months_map.get(int(m.idxmax()),"?"),
        "peak_value": round(m.max(),2),
        "avg_value":  round(m.mean(),2),
        "ratio":      round(m.max()/m.mean(),2),
    }

gabarito = {
    "arquivo":          "testes/vendas_v3_desafio.csv",
    "total_linhas_bruto": len(df),
    "total_linhas_limpo": len(df_c),
    "duplicatas_removidas": dups,
    "faturamento_bruto":  round(t_bruto,2),
    "faturamento_limpo":  round(t_limpo,2),
    "diferenca_duplicatas": round(t_bruto-t_limpo,2),
    "datas_invalidas_count": int(inv_datas),
    "datas_invalidas_pct":   round(inv_datas/len(df_c)*100,2),
    "top5_produtos": {k: round(v,2) for k,v in top5_prod.items()},
    "top5_vendedores": {k: round(v,2) for k,v in top5_vend.items()},
    "canal_maior_volume":    str(top_canal_vol.index[0]),
    "canal_maior_faturamento": str(top_canal_val.index[0]),
    "outliers_iqr_count":   len(out_iqr),
    "outliers_iqr_pct":     round(len(out_iqr)/len(df_c)*100,2),
    "outliers_zscore_count":len(out_z),
    "iqr_range_min": round(lo,2),
    "iqr_range_max": round(hi,2),
    "carlos_cancelamentos":  int(carlos_c),
    "outros_cancelamentos_media": round(outros_c,1),
    "carlos_ratio":          round(carlos_c/max(outros_c,1),2),
    "maria_ticket_medio":    round(maria_ticket,2),
    "outros_ticket_medio":   round(outros_ticket,2),
    "maria_ratio_ticket":    round(maria_ticket/max(outros_ticket,1),2),
    "notas_invalidas_count": len(inv_notas),
    "prazo_zero_count":      len(inv_prazo),
    "variantes_sao_paulo":   {str(k):int(v) for k,v in sp_var.items()},
    "sazonalidade": saz,
}

with open("testes/gabarito_v3.json","w",encoding="utf-8") as f:
    json.dump(gabarito, f, ensure_ascii=False, indent=2)

# ── PRINT ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("📋  GABARITO OFICIAL — vendas_v3_desafio.csv")
print("="*65)

print(f"\n📁  ARQUIVO")
print(f"    Linhas brutas:    {gabarito['total_linhas_bruto']:>8,}")
print(f"    Linhas limpas:    {gabarito['total_linhas_limpo']:>8,}  (rem. {dups} dups parciais)")
print(f"    Colunas:          {len(df.columns)}")

print(f"\n💰  FATURAMENTO")
print(f"    Bruto:   R$ {gabarito['faturamento_bruto']:>14,.2f}")
print(f"    Limpo:   R$ {gabarito['faturamento_limpo']:>14,.2f}")
print(f"    Dif.:    R$ {gabarito['diferenca_duplicatas']:>14,.2f}")

print(f"\n📅  DATAS")
print(f"    Inválidas: {gabarito['datas_invalidas_count']:,} ({gabarito['datas_invalidas_pct']}%)")

print(f"\n🏆  TOP 5 PRODUTOS")
for i,(p,v) in enumerate(gabarito["top5_produtos"].items(),1):
    flag = " ← sazonalidade" if saz.get(p,{}).get("ratio",0)>1.8 else ""
    print(f"    {i}. {p:<22} R$ {v:>14,.2f}{flag}")

print(f"\n👤  TOP 5 VENDEDORES")
for i,(v,val) in enumerate(gabarito["top5_vendedores"].items(),1):
    flags = []
    if v=="Carlos": flags.append(f"cancelamentos: {carlos_c} ({round(carlos_c/max(outros_c,1),1)}x média)")
    if v=="Maria":  flags.append(f"ticket {round(maria_ticket/max(outros_ticket,1),1)}x acima da média ⚠️")
    flag = f"  ← {' | '.join(flags)}" if flags else ""
    print(f"    {i}. {v:<10} R$ {val:>14,.2f}{flag}")

print(f"\n⚠️   OUTLIERS")
print(f"    IQR:     {gabarito['outliers_iqr_count']:,} outliers ({gabarito['outliers_iqr_pct']}%)")
print(f"    Z-score: {gabarito['outliers_zscore_count']:,} outliers (z>3)")
print(f"    Range normal: R${lo:,.2f} → R${hi:,.2f}")

print(f"\n📊  CANAL (volume vs valor)")
print(f"    Maior volume:      {gabarito['canal_maior_volume']}")
print(f"    Maior faturamento: {gabarito['canal_maior_faturamento']}")
for c in top_canal_vol.index:
    vol = int(top_canal_vol[c])
    val = round(float(top_canal_val.get(c,0)),2)
    print(f"    {c:<15} {vol:>5} vendas  R$ {val:>14,.2f}")

print(f"\n🌆  VARIANTES 'SÃO PAULO'")
for k,v in gabarito["variantes_sao_paulo"].items():
    print(f"    '{k}': {v}")

print(f"\n⭐  QUALIDADE")
print(f"    Notas inválidas:  {gabarito['notas_invalidas_count']:,}")
print(f"    Prazo zero:       {gabarito['prazo_zero_count']:,}")

print(f"\n📈  SAZONALIDADE")
for prod, s in saz.items():
    strong = " 🔴 FORTE" if s["ratio"]>1.8 else ""
    print(f"    {prod:<22} pico={s['peak_month']}  ratio={s['ratio']}x{strong}")

print(f"\n{'='*65}")
print(f"✅  Arquivo: testes/vendas_v3_desafio.csv  ({len(df):,} linhas | {len(df.columns)} colunas)")
print(f"✅  Gabarito: testes/gabarito_v3.json")
print(f"{'='*65}\n")