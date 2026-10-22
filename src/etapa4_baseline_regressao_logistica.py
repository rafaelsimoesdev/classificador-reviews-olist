"""
Etapa 4: baseline e regressão logística com TF-IDF.

Compara, no mesmo conjunto de teste (20% dos comentários, separados antes de tudo):
  1. Baseline "sempre positiva" (palpite sem informação)
  2. TF-IDF + regressão logística, texto como está
  3. TF-IDF + regressão logística, com acentos normalizados (péssima = pessima)

Métricas: ROC-AUC e PR-AUC (não dependem de limiar), além de precisão e recall
da classe negativa no limiar padrão de 0,5. A classe negativa é a de interesse (29%).

Uso (da raiz do projeto, depois de rodar a etapa 3):
    python src/etapa4_baseline_regressao_logistica.py
"""

import json
import os
import unicodedata

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

SEED = 42
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(RAIZ, "dados", "reviews_limpas.csv")
DOCS = os.path.join(RAIZ, "docs")
os.makedirs(DOCS, exist_ok=True)

# keep_default_na=False: um comentário que é literalmente "n/a" não pode virar valor ausente
df = pd.read_csv(DADOS, keep_default_na=False)


def sem_acento(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
    )


df["texto_sem_acento"] = df["texto"].map(sem_acento)

# Divisão estratificada, feita uma vez e usada em todos os modelos
idx_tr, idx_te = train_test_split(
    df.index, test_size=0.20, stratify=df["negativa"], random_state=SEED
)
y_tr, y_te = df.loc[idx_tr, "negativa"], df.loc[idx_te, "negativa"]
print(f"Treino: {len(idx_tr):,} | Teste: {len(idx_te):,} | negativas no teste: {y_te.mean():.1%}")

resultados = {}


def registrar(nome, prob, pred):
    resultados[nome] = {
        "roc_auc": round(float(roc_auc_score(y_te, prob)), 4),
        "pr_auc": round(float(average_precision_score(y_te, prob)), 4),
        "precisao_neg": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "recall_neg": round(float(recall_score(y_te, pred, zero_division=0)), 4),
        "acuracia": round(float((pred == y_te).mean()), 4),
    }


# 1. Baseline: "sempre positiva"
dummy = DummyClassifier(strategy="prior").fit(df.loc[idx_tr, ["texto"]], y_tr)
prob = dummy.predict_proba(df.loc[idx_te, ["texto"]])[:, 1]
registrar("baseline_sempre_positiva", prob, (prob >= 0.5).astype(int))


# 2 e 3. TF-IDF + regressão logística
def treinar(coluna, nome):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X_tr = vec.fit_transform(df.loc[idx_tr, coluna])  # vocabulário só do treino
    X_te = vec.transform(df.loc[idx_te, coluna])
    modelo = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    modelo.fit(X_tr, y_tr)
    prob = modelo.predict_proba(X_te)[:, 1]
    registrar(nome, prob, (prob >= 0.5).astype(int))
    return vec, modelo


treinar("texto", "tfidf_logistica_com_acento")
vec, modelo = treinar("texto_sem_acento", "tfidf_logistica_sem_acento")

# Resultados
print(f"\n{'Modelo':<30}{'ROC-AUC':>9}{'PR-AUC':>8}{'Prec.neg':>10}{'Rec.neg':>9}{'Acur.':>8}")
for nome, r in resultados.items():
    print(f"{nome:<30}{r['roc_auc']:>9.3f}{r['pr_auc']:>8.3f}"
          f"{r['precisao_neg']:>10.1%}{r['recall_neg']:>9.1%}{r['acuracia']:>8.1%}")

# Coeficientes do melhor modelo: o que a regressão logística "aprendeu"
palavras = vec.get_feature_names_out()
coef = pd.Series(modelo.coef_[0], index=palavras).sort_values()
print("\nTermos que mais empurram para POSITIVA (coeficiente negativo):")
for p, c in coef.head(12).items():
    print(f"  {p:<22}{c:>7.2f}")
print("\nTermos que mais empurram para NEGATIVA (coeficiente positivo):")
for p, c in coef.tail(12).iloc[::-1].items():
    print(f"  {p:<22}{c:>7.2f}")

with open(os.path.join(DOCS, "metricas_baseline.json"), "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
print("\nResultados salvos em docs/metricas_baseline.json")
