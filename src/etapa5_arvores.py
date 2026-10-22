"""
Etapa 5: árvore de decisão e Random Forest contra a regressão logística.

Mesma divisão da etapa 4 (80/20 estratificada, seed 42) e mesmo TF-IDF, para a
comparação ser justa. Texto com acentos normalizados.

Uso (da raiz do projeto, depois de rodar a etapa 3):
    python src/etapa5_arvores.py
"""

import json
import os
import time
import unicodedata

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

SEED = 42
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(RAIZ, "dados", "reviews_limpas.csv")
DOCS = os.path.join(RAIZ, "docs")
os.makedirs(DOCS, exist_ok=True)

df = pd.read_csv(DADOS, keep_default_na=False)


def sem_acento(t):
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


df["texto"] = df["texto"].map(sem_acento)

idx_tr, idx_te = train_test_split(
    df.index, test_size=0.20, stratify=df["negativa"], random_state=SEED
)
y_tr, y_te = df.loc[idx_tr, "negativa"], df.loc[idx_te, "negativa"]

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
X_tr = vec.fit_transform(df.loc[idx_tr, "texto"])
X_te = vec.transform(df.loc[idx_te, "texto"])
print(f"Treino: {X_tr.shape[0]:,} | Teste: {X_te.shape[0]:,} | atributos (palavras e pares): {X_tr.shape[1]:,}")

modelos = {
    "regressao_logistica": LogisticRegression(max_iter=1000, random_state=SEED),
    "arvore_profundidade_5": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED),
    "arvore_profundidade_20": DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=SEED),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=SEED,
    ),
}

resultados = {}
treinados = {}
for nome, modelo in modelos.items():
    t0 = time.time()
    modelo.fit(X_tr, y_tr)
    seg = time.time() - t0
    prob = modelo.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    treinados[nome] = modelo
    resultados[nome] = {
        "roc_auc": round(float(roc_auc_score(y_te, prob)), 4),
        "pr_auc": round(float(average_precision_score(y_te, prob)), 4),
        "precisao_neg": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "recall_neg": round(float(recall_score(y_te, pred, zero_division=0)), 4),
        "segundos_treino": round(seg, 1),
    }

print(f"\n{'Modelo':<26}{'ROC-AUC':>9}{'PR-AUC':>8}{'Prec.neg':>10}{'Rec.neg':>9}{'Treino':>9}")
for nome, r in resultados.items():
    print(f"{nome:<26}{r['roc_auc']:>9.3f}{r['pr_auc']:>8.3f}"
          f"{r['precisao_neg']:>10.1%}{r['recall_neg']:>9.1%}{r['segundos_treino']:>8.1f}s")

# A árvore rasa em texto: dá para ler como o modelo decide
print("\nÁrvore de profundidade 5 (primeiros níveis, 1 = negativa):")
regras = export_text(treinados["arvore_profundidade_5"], feature_names=list(vec.get_feature_names_out()),
                     max_depth=3)
print(regras)

# Palavras mais importantes no Random Forest
imp = pd.Series(treinados["random_forest"].feature_importances_, index=vec.get_feature_names_out())
print("Termos mais importantes no Random Forest:")
for p, v in imp.sort_values(ascending=False).head(12).items():
    print(f"  {p:<22}{v:>8.4f}")

with open(os.path.join(DOCS, "metricas_modelos.json"), "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
print("\nResultados salvos em docs/metricas_modelos.json")
