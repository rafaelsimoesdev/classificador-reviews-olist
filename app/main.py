"""
API do classificador de reviews da Olist.

Recebe o texto de um comentário e devolve se a avaliação é negativa, com a
probabilidade e os termos que mais pesaram na decisão.
"""

import os
import unicodedata

import joblib
import numpy as np
from fastapi import FastAPI

from app.schema import Classificacao, Comentario

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
payload = joblib.load(MODEL_PATH)
PIPELINE = payload["pipeline"]
LIMIAR = payload["limiar_alerta"]
TFIDF = PIPELINE.named_steps["tfidf"]
COEF = PIPELINE.named_steps["logistica"].coef_[0]
TERMOS = TFIDF.get_feature_names_out()

app = FastAPI(
    title="Classificador de Reviews Olist",
    description=(
        "Classifica comentários de clientes de e-commerce como negativos ou positivos.\n\n"
        "Modelo: TF-IDF + regressão logística, treinado com 37 mil avaliações reais da Olist. "
        f"Um comentário é marcado como negativo quando a probabilidade passa de {LIMIAR:.0%}, "
        "limiar escolhido para encontrar cerca de 95% das avaliações negativas."
    ),
    version="1.0.0",
)


def normalizar(texto: str) -> str:
    """Mesma normalização usada no treino (src/etapa8_treinar_modelo_final.py)."""
    texto = " ".join(texto.lower().split())
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def termos_relevantes(vetor, negativa: bool, n: int = 5) -> list[str]:
    # Contribuição de cada termo = peso TF-IDF no comentário x coeficiente do modelo
    idx = vetor.nonzero()[1]
    if len(idx) == 0:
        return []
    contrib = vetor.toarray()[0, idx] * COEF[idx]
    ordem = np.argsort(-contrib) if negativa else np.argsort(contrib)
    sinal = 1 if negativa else -1
    return [TERMOS[idx[i]] for i in ordem[:n] if sinal * contrib[i] > 0]


def confianca(p: float) -> str:
    # Distância até o limiar, relativa ao espaço disponível daquele lado
    # (o limiar não fica em 0,5, então os dois lados têm tamanhos diferentes)
    espaco = (1 - LIMIAR) if p >= LIMIAR else LIMIAR
    relativa = abs(p - LIMIAR) / espaco
    if relativa >= 0.6:
        return "ALTA"
    if relativa >= 0.3:
        return "MÉDIA"
    return "BAIXA"


@app.get("/")
def raiz():
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/classificar", response_model=Classificacao)
def classificar(comentario: Comentario):
    texto = normalizar(comentario.texto)
    vetor = TFIDF.transform([texto])
    p = float(PIPELINE.named_steps["logistica"].predict_proba(vetor)[0, 1])
    negativa = p >= LIMIAR
    return Classificacao(
        negativa=negativa,
        probabilidade_negativa=round(p, 4),
        confianca=confianca(p),
        termos_relevantes=termos_relevantes(vetor, negativa),
    )


@app.get("/model/info")
def model_info():
    return {
        "modelo": "TF-IDF (palavras e pares) + regressão logística",
        "limiar_alerta": LIMIAR,
        "comentarios_treino": payload["n_treino"],
        "taxa_negativas_base": payload["taxa_negativas"],
        "metricas_teste": payload["metricas_teste"],
        "termos_mais_negativos": payload["termos_negativos"],
        "termos_mais_positivos": payload["termos_positivos"],
        "scikit_learn": payload["sklearn_version"],
    }
