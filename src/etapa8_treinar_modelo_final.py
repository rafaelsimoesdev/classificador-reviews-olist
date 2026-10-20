"""
Etapa 8: treina e salva o modelo que vai para a API.

Modelo: TF-IDF (palavras e pares) + regressão logística, texto sem acentos.

O limiar de alerta é escolhido só com dados de treino (predições out-of-fold),
buscando recall de pelo menos 95% na classe negativa, como sugeriu a análise
de erros (etapa 6). Depois o modelo é avaliado uma única vez no teste e, por
fim, retreinado com todos os comentários para ir para produção.

Uso (da raiz do projeto, depois de rodar a etapa 3; use as versões do requirements.txt):
    python src/etapa8_treinar_modelo_final.py
"""

import json
import os
import unicodedata

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

SEED = 42
RECALL_ALVO = 0.95
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(RAIZ, "dados", "reviews_limpas.csv")
MODELO = os.path.join(RAIZ, "model", "model.pkl")
DOCS = os.path.join(RAIZ, "docs")


def normalizar(texto: str) -> str:
    """Mesma normalização usada na API (app/main.py)."""
    texto = " ".join(texto.lower().split())
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def novo_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)),
        ("logistica", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])


def enxugar(pipe):
    # stop_words_ guarda todos os termos descartados e só serve para inspeção.
    # Removê-lo deixa o arquivo do modelo bem menor e não muda as previsões.
    if hasattr(pipe.named_steps["tfidf"], "stop_words_"):
        del pipe.named_steps["tfidf"].stop_words_
    return pipe


df = pd.read_csv(DADOS, keep_default_na=False)
df["texto"] = df["texto"].map(normalizar)
X, y = df["texto"], df["negativa"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

# 1. Limiar escolhido só com o treino
oof = cross_val_predict(
    novo_pipeline(), X_tr, y_tr,
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba",
)[:, 1]
_, rec, lim = precision_recall_curve(y_tr, oof)
limiar = float(lim[np.where(rec[:-1] >= RECALL_ALVO)[0][-1]])
print(f"Limiar escolhido no treino (recall >= {RECALL_ALVO:.0%}): {limiar:.3f}")

# 2. Avaliação única no teste
pipe = novo_pipeline().fit(X_tr, y_tr)
prob = pipe.predict_proba(X_te)[:, 1]
metricas = {"roc_auc": round(float(roc_auc_score(y_te, prob)), 4),
            "pr_auc": round(float(average_precision_score(y_te, prob)), 4)}
for nome, l in (("limiar_0_5", 0.5), ("limiar_alerta", limiar)):
    pred = (prob >= l).astype(int)
    metricas[nome] = {"limiar": round(l, 4),
                      "precisao": round(float(precision_score(y_te, pred)), 4),
                      "recall": round(float(recall_score(y_te, pred)), 4),
                      "sinalizados": round(float(pred.mean()), 4)}
print(json.dumps(metricas, indent=2, ensure_ascii=False))

# 3. Modelo final com todos os comentários
final = enxugar(novo_pipeline().fit(X, y))
coef = pd.Series(final.named_steps["logistica"].coef_[0],
                 index=final.named_steps["tfidf"].get_feature_names_out())
os.makedirs(os.path.dirname(MODELO), exist_ok=True)
joblib.dump({
    "pipeline": final,
    "limiar_alerta": round(limiar, 4),
    "metricas_teste": metricas,
    "n_treino": int(len(df)),
    "taxa_negativas": round(float(y.mean()), 4),
    "termos_negativos": coef.sort_values(ascending=False).head(10).index.tolist(),
    "termos_positivos": coef.sort_values().head(10).index.tolist(),
    "sklearn_version": sklearn.__version__,
}, MODELO, compress=3)
with open(os.path.join(DOCS, "metricas_modelo_final.json"), "w", encoding="utf-8") as f:
    json.dump({"limiar_alerta": round(limiar, 4), **metricas}, f, ensure_ascii=False, indent=2)
print(f"\nModelo salvo em model/model.pkl ({os.path.getsize(MODELO) / 1e6:.1f} MB), "
      f"treinado com {len(df):,} comentários (scikit-learn {sklearn.__version__})")
