"""
Testes da API. Rodam com o modelo salvo em model/model.pkl.

Uso (da raiz do projeto):
    pip install -r requirements.txt -r requirements-dev.txt
    pytest
"""

from fastapi.testclient import TestClient

from app.main import app

cliente = TestClient(app)


def classificar(texto):
    resposta = cliente.post("/classificar", json={"texto": texto})
    assert resposta.status_code == 200
    return resposta.json()


def test_health():
    assert cliente.get("/health").json() == {"status": "healthy"}


def test_comentario_claramente_negativo():
    r = classificar("Péssimo, até agora não recebi meu pedido")
    assert r["negativa"] is True
    assert r["probabilidade_negativa"] > 0.9
    assert "nao recebi" in r["termos_relevantes"]


def test_comentario_claramente_positivo():
    r = classificar("Chegou antes do prazo, amei!")
    assert r["negativa"] is False
    assert r["probabilidade_negativa"] < 0.1


def test_acentos_e_maiusculas_nao_mudam_o_resultado():
    a = classificar("PÉSSIMO ATENDIMENTO")["probabilidade_negativa"]
    b = classificar("pessimo atendimento")["probabilidade_negativa"]
    assert a == b


def test_texto_vazio_e_rejeitado():
    assert cliente.post("/classificar", json={"texto": ""}).status_code == 422


def test_model_info():
    info = cliente.get("/model/info").json()
    assert 0 < info["limiar_alerta"] < 1
    assert info["comentarios_treino"] > 30000
