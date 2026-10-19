"""
Etapa 7: rede neural simples (MLP) contra a regressão logística.

Mesma divisão (80/20 estratificada, seed 42) e mesmo TF-IDF das etapas 4 e 5.
A rede tem uma camada escondida e para de treinar sozinha quando a validação
interna (10% do treino) deixa de melhorar (early stopping).

Como uma rede neural muda a cada inicialização, o treino é repetido com 5
sementes diferentes e o resultado é mostrado como média e variação.

Uso (da raiz do projeto, depois de rodar a etapa 3):
    python src/etapa7_rede_neural.py
"""

import json
import os
import time
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

SEED = 42
SEMENTES_REDE = [1, 2, 3, 4, 5]
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(RAIZ, "dados", "reviews_limpas.csv")
DOCS = os.path.join(RAIZ, "docs")
os.makedirs(DOCS, exist_ok=True)

df = pd.read_csv(DADOS, keep_default_na=False)
df["texto"] = df["texto"].map(
    lambda t: "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
)

idx_tr, idx_te = train_test_split(
    df.index, test_size=0.20, stratify=df["negativa"], random_state=SEED
)
y_tr, y_te = df.loc[idx_tr, "negativa"], df.loc[idx_te, "negativa"]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
X_tr = vec.fit_transform(df.loc[idx_tr, "texto"])
X_te = vec.transform(df.loc[idx_te, "texto"])
print(f"Treino: {X_tr.shape[0]:,} | Teste: {X_te.shape[0]:,} | entradas: {X_tr.shape[1]:,}")


def medir(prob):
    return roc_auc_score(y_te, prob), average_precision_score(y_te, prob)


resultados = {}

# Regressão logística (referência)
t0 = time.time()
lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_tr, y_tr)
roc, pr = medir(lr.predict_proba(X_te)[:, 1])
resultados["regressao_logistica"] = {"roc_auc": round(roc, 4), "pr_auc": round(pr, 4),
                                     "segundos_treino": round(time.time() - t0, 1)}

# Rede neural: uma camada escondida, 5 sementes
arquiteturas = {"rede_1_camada_64": (64,), "rede_2_camadas_128_32": (128, 32)}
for nome, camadas in arquiteturas.items():
    rocs, prs, segs, iters = [], [], [], []
    for s in SEMENTES_REDE:
        t0 = time.time()
        rede = MLPClassifier(
            hidden_layer_sizes=camadas, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=3, max_iter=50, random_state=s,
        ).fit(X_tr, y_tr)
        segs.append(time.time() - t0)
        iters.append(rede.n_iter_)
        r, p = medir(rede.predict_proba(X_te)[:, 1])
        rocs.append(r)
        prs.append(p)
    resultados[nome] = {
        "roc_auc_media": round(float(np.mean(rocs)), 4), "roc_auc_dp": round(float(np.std(rocs)), 4),
        "pr_auc_media": round(float(np.mean(prs)), 4), "pr_auc_dp": round(float(np.std(prs)), 4),
        "epocas_media": round(float(np.mean(iters)), 1),
        "segundos_treino_media": round(float(np.mean(segs)), 1),
    }

print(f"\n{'Modelo':<26}{'ROC-AUC':>14}{'PR-AUC':>14}{'Treino':>10}")
r = resultados["regressao_logistica"]
print(f"{'regressao_logistica':<26}{r['roc_auc']:>14.3f}{r['pr_auc']:>14.3f}{r['segundos_treino']:>9.1f}s")
for nome in arquiteturas:
    r = resultados[nome]
    print(f"{nome:<26}{r['roc_auc_media']:>8.3f}±{r['roc_auc_dp']:.3f}{r['pr_auc_media']:>8.3f}±{r['pr_auc_dp']:.3f}"
          f"{r['segundos_treino_media']:>9.1f}s  ({r['epocas_media']:.0f} épocas)")

with open(os.path.join(DOCS, "metricas_rede_neural.json"), "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
print("\nResultados salvos em docs/metricas_rede_neural.json")
