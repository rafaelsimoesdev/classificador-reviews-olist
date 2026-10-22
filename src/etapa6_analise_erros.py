"""
Etapa 6: análise de erros da regressão logística (TF-IDF, acentos normalizados).

Treina o melhor modelo da etapa 5 e mostra onde ele erra no conjunto de teste:
  - falsos negativos: comentário negativo que o modelo chamou de positivo
  - falsos positivos: comentário positivo que o modelo chamou de negativo
Também mede o erro por tamanho do comentário e por nota, e salva os erros em CSV.

Uso (da raiz do projeto, depois de rodar a etapa 3):
    python src/etapa6_analise_erros.py
"""

import os
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(RAIZ, "dados", "reviews_limpas.csv")
DOCS = os.path.join(RAIZ, "docs")
os.makedirs(DOCS, exist_ok=True)

df = pd.read_csv(DADOS, keep_default_na=False)
df["texto_original"] = df["texto"]


def sem_acento(t):
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


df["texto"] = df["texto"].map(sem_acento)

idx_tr, idx_te = train_test_split(
    df.index, test_size=0.20, stratify=df["negativa"], random_state=SEED
)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
X_tr = vec.fit_transform(df.loc[idx_tr, "texto"])
X_te = vec.transform(df.loc[idx_te, "texto"])
modelo = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_tr, df.loc[idx_tr, "negativa"])

teste = df.loc[idx_te].copy()
teste["prob_neg"] = modelo.predict_proba(X_te)[:, 1]
teste["pred"] = (teste["prob_neg"] >= 0.5).astype(int)
teste["erro"] = teste["pred"] != teste["negativa"]
teste["n_palavras"] = teste["texto"].str.split().str.len()

fn = teste[(teste["negativa"] == 1) & (teste["pred"] == 0)]  # perdeu a negativa
fp = teste[(teste["negativa"] == 0) & (teste["pred"] == 1)]  # alarme falso
print(f"Teste: {len(teste):,} | erros: {teste['erro'].sum():,} ({teste['erro'].mean():.1%})")
print(f"  Falsos negativos (negativa classificada como positiva): {len(fn):,}")
print(f"  Falsos positivos (positiva classificada como negativa): {len(fp):,}")

print("\nTaxa de erro por nota original:")
for nota, g in teste.groupby("review_score"):
    print(f"  nota {nota}: {g['erro'].mean():>6.1%} de erro ({len(g):,} comentários)")

faixas = pd.cut(teste["n_palavras"], [0, 3, 8, 15, 100], labels=["1-3", "4-8", "9-15", "16+"])
print("\nTaxa de erro por tamanho do comentário (palavras):")
for faixa, g in teste.groupby(faixas, observed=True):
    print(f"  {faixa:>5}: {g['erro'].mean():>6.1%} de erro ({len(g):,} comentários)")


def mostrar(titulo, base, n=12, ordem=False):
    print(f"\n{titulo}")
    for _, r in base.sort_values("prob_neg", ascending=ordem).head(n).iterrows():
        print(f"  [nota {r['review_score']} | p(neg)={r['prob_neg']:.2f}] {r['texto_original'][:140]}")


# Erros "mais confiantes" do modelo são os mais instrutivos
mostrar("Falsos negativos mais confiantes (negativos que o modelo acha muito positivos):", fn, ordem=True)
mostrar("Falsos positivos mais confiantes (positivos que o modelo acha muito negativos):", fp, ordem=False)

# Hipóteses: mistura de elogio e reclamação, ou menção ao produto/entrega
def contem(base, palavras):
    return base["texto"].str.contains("|".join(palavras)).mean()

misto = ["mas ", "porem", "entretanto", "so que", "apesar"]
print(f"\nComentários com 'mas', 'porém' etc.: {contem(teste, misto):.1%} do teste, "
      f"{contem(fn, misto):.1%} dos falsos negativos, {contem(fp, misto):.1%} dos falsos positivos")

nao_recebi = ["nao recebi", "nao chegou", "ainda nao", "aguardando"]
print(f"Comentários sobre entrega não concluída: {contem(teste, nao_recebi):.1%} do teste, "
      f"{contem(fn, nao_recebi):.1%} dos falsos negativos, {contem(fp, nao_recebi):.1%} dos falsos positivos")

# Limiar: muitos negativos perdidos ficam com p(neg) entre 0,2 e 0,5.
# Quanto um limiar mais baixo recupera, e a que custo?
print("\nLimiar de decisão (classe negativa):")
print(f"  {'limiar':>6}{'precisão':>10}{'recall':>8}{'alertas':>9}")
for lim in (0.5, 0.4, 0.3, 0.2):
    pred = teste["prob_neg"] >= lim
    vp = (pred & (teste["negativa"] == 1)).sum()
    print(f"  {lim:>6.1f}{vp / pred.sum():>10.1%}{vp / teste['negativa'].sum():>8.1%}{pred.mean():>9.1%}")

erros = teste[teste["erro"]][["order_id", "review_score", "prob_neg", "texto_original"]]
erros.sort_values("prob_neg").to_csv(os.path.join(DOCS, "erros_modelo.csv"), index=False)
print(f"\n{len(erros):,} erros salvos em docs/erros_modelo.csv")
