"""
Etapa 3: limpeza do texto e preparação da base do modelo.

O que faz:
  1. Mantém só avaliações com comentário (descarta nota 3).
  2. Cria o alvo: negativa = nota 1-2 (1), positiva = nota 4-5 (0).
  3. Limpa o texto: minúsculas, remove espaços extras, quebras de linha e comentários vazios.
  4. Remove comentários duplicados do mesmo pedido.
  5. Lista as palavras mais associadas a cada classe (razão de chances simples).
  6. Salva dados/reviews_limpas.csv (não versionado).

Uso (da raiz do projeto, com o CSV em dados/):
    python src/etapa3_limpar_texto.py
"""

import os
import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "dados", "olist_order_reviews_dataset.csv")
SAIDA = os.path.join(RAIZ, "dados", "reviews_limpas.csv")

df = pd.read_csv(CSV)
print(f"Avaliações: {len(df):,}")

# 1. Só com comentário e sem nota 3
df = df[df["review_comment_message"].notna() & (df["review_score"] != 3)].copy()
print(f"Com comentário e nota diferente de 3: {len(df):,}")


# 2. Limpeza do texto
def limpar(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"https?://\S+", " ", texto)        # links
    texto = re.sub(r"[\r\n\t]+", " ", texto)           # quebras de linha
    texto = re.sub(r"\s+", " ", texto).strip()         # espaços repetidos
    return texto


df["texto"] = df["review_comment_message"].map(limpar)

# Comentários vazios depois da limpeza
vazios = (df["texto"].str.len() == 0).sum()
df = df[df["texto"].str.len() > 0]
print(f"Comentários vazios removidos: {vazios:,}")

# 3. Duplicados: mesmo texto no mesmo pedido
antes = len(df)
df = df.drop_duplicates(subset=["order_id", "texto"])
print(f"Duplicados removidos (mesmo pedido e texto): {antes - len(df):,}")

# 4. Alvo
df["negativa"] = (df["review_score"] <= 2).astype(int)
print(f"\nBase final: {len(df):,} comentários")
print(f"  Negativas: {df['negativa'].sum():,} ({df['negativa'].mean():.1%})")
print(f"  Positivas: {(1 - df['negativa']).sum():,} ({1 - df['negativa'].mean():.1%})")

# 5. Palavras que mais separam as classes
vec = CountVectorizer(min_df=30, binary=True)  # palavra aparece em no mínimo 30 comentários
X = vec.fit_transform(df["texto"])
palavras = vec.get_feature_names_out()
neg = X[df["negativa"].values == 1].sum(axis=0).A1
pos = X[df["negativa"].values == 0].sum(axis=0).A1
n_neg, n_pos = df["negativa"].sum(), (1 - df["negativa"]).sum()

tabela = pd.DataFrame({"palavra": palavras, "em_negativas": neg, "em_positivas": pos})
tabela["pct_neg"] = tabela["em_negativas"] / n_neg
tabela["pct_pos"] = tabela["em_positivas"] / n_pos
tabela["razao"] = (tabela["pct_neg"] + 1e-4) / (tabela["pct_pos"] + 1e-4)

print("\nPalavras mais associadas a comentários NEGATIVOS (aparecem x vezes mais que nos positivos):")
for _, r in tabela.sort_values("razao", ascending=False).head(15).iterrows():
    print(f"  {r['palavra']:<14} {r['razao']:>6.1f}x   ({int(r['em_negativas'])} neg / {int(r['em_positivas'])} pos)")

print("\nPalavras mais associadas a comentários POSITIVOS:")
for _, r in tabela.sort_values("razao").head(15).iterrows():
    print(f"  {r['palavra']:<14} {1 / r['razao']:>6.1f}x   ({int(r['em_positivas'])} pos / {int(r['em_negativas'])} neg)")

# 6. Salvar
df[["order_id", "review_score", "texto", "negativa"]].to_csv(SAIDA, index=False)
print(f"\nBase limpa salva em dados/reviews_limpas.csv")
