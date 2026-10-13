"""
Etapa 2: exploração das reviews da Olist.

Perguntas:
  - Quantas avaliações existem e quantas têm comentário em texto?
  - Como as notas se distribuem?
  - Qual a taxa de comentários por nota? (quem escreve mais: o satisfeito ou o insatisfeito?)
  - Quão longos são os comentários?

Uso (da raiz do projeto, com o CSV em dados/):
    python src/etapa2_explorar_reviews.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "dados", "olist_order_reviews_dataset.csv")
DOCS = os.path.join(RAIZ, "docs")
os.makedirs(DOCS, exist_ok=True)

df = pd.read_csv(CSV)
df["tem_comentario"] = df["review_comment_message"].notna()

print(f"Avaliações: {len(df):,}")
print(f"Pedidos distintos: {df['order_id'].nunique():,}")
print(f"Com comentário em texto: {df['tem_comentario'].sum():,} ({df['tem_comentario'].mean():.1%})")

# Notas
notas = df["review_score"].value_counts().sort_index()
print("\nDistribuição das notas:")
for nota, n in notas.items():
    print(f"  {nota}: {n:>6,} ({n / len(df):.1%})")

# Comentários por nota
por_nota = df.groupby("review_score")["tem_comentario"].agg(["sum", "mean"])
print("\nComentários por nota:")
for nota, linha in por_nota.iterrows():
    print(f"  nota {nota}: {int(linha['sum']):>6,} comentários ({linha['mean']:.1%} das avaliações)")

# Tamanho dos comentários
com = df[df["tem_comentario"]].copy()
com["n_palavras"] = com["review_comment_message"].str.split().str.len()
print("\nTamanho dos comentários (palavras):")
print(com["n_palavras"].describe().round(1).to_string())

# Alvo proposto: 1-2 negativa, 4-5 positiva, 3 descartada
alvo = com[com["review_score"] != 3].copy()
alvo["negativa"] = (alvo["review_score"] <= 2).astype(int)
print(f"\nComentários com nota diferente de 3: {len(alvo):,}")
print(f"  Negativas: {alvo['negativa'].sum():,} ({alvo['negativa'].mean():.1%})")
print(f"  Positivas: {(1 - alvo['negativa']).sum():,.0f} ({1 - alvo['negativa'].mean():.1%})")
print(f"  Comentários descartados (nota 3): {(com['review_score'] == 3).sum():,}")

# Gráfico
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(notas.index, notas.values, color="#4C72B0")
ax[0].set_title("Avaliações por nota")
ax[0].set_xlabel("Nota")
ax[0].set_ylabel("Quantidade")

ax[1].bar(por_nota.index, por_nota["mean"] * 100, color="#DD8452")
ax[1].set_title("% das avaliações com comentário, por nota")
ax[1].set_xlabel("Nota")
ax[1].set_ylabel("%")
plt.tight_layout()
plt.savefig(os.path.join(DOCS, "distribuicao_notas.png"), dpi=130)
print("\nGráfico salvo em docs/distribuicao_notas.png")
