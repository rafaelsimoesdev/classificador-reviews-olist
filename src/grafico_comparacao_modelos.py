"""
Gráfico de resumo: PR-AUC de cada modelo no mesmo conjunto de teste.

Lê as métricas salvas pelas etapas 4, 5 e 7 (pasta docs/) e gera
docs/comparacao_modelos.png, usado no topo do README.

Uso (da raiz do projeto):
    python src/grafico_comparacao_modelos.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(RAIZ, "docs")


def ler(nome):
    with open(os.path.join(DOCS, nome), encoding="utf-8") as f:
        return json.load(f)


base = ler("metricas_baseline.json")
arv = ler("metricas_modelos.json")
rede = ler("metricas_rede_neural.json")

modelos = [  # (rótulo, PR-AUC, é o modelo escolhido?)
    ("Árvore de decisão (prof. 5)", arv["arvore_profundidade_5"]["pr_auc"], False),
    ("Árvore de decisão (prof. 20)", arv["arvore_profundidade_20"]["pr_auc"], False),
    ("Random Forest", arv["random_forest"]["pr_auc"], False),
    ("Rede neural (1 camada)", rede["rede_1_camada_64"]["pr_auc_media"], False),
    ("Regressão logística", arv["regressao_logistica"]["pr_auc"], True),
]
baseline = base["baseline_sempre_positiva"]["pr_auc"]

AZUL, CINZA, TEXTO, TEXTO2 = "#2a78d6", "#c9c7c0", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

y = range(len(modelos))
ax.barh(y, [m[1] for m in modelos], height=0.55,
        color=[AZUL if m[2] else CINZA for m in modelos], edgecolor="#fcfcfb", linewidth=2)
for i, (rotulo, valor, escolhido) in enumerate(modelos):
    ax.text(valor + 0.01, i, f"{valor:.2f}".replace(".", ","), va="center", fontsize=9,
            color=TEXTO, fontweight="bold" if escolhido else "normal")

ax.axvline(baseline, color=TEXTO2, linestyle=(0, (3, 3)), linewidth=1)
ax.text(baseline + 0.01, -0.55, "palpite sem informação (0,29)",
        fontsize=8, color=TEXTO2, va="bottom")

ax.set_yticks(list(y), [m[0] for m in modelos], fontsize=9, color=TEXTO)
ax.set_xlim(0, 1.05)
ax.set_ylim(-0.75, len(modelos) - 0.5)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "0,2", "0,4", "0,6", "0,8", "1"])
ax.set_xlabel("PR-AUC (quanto maior, melhor)", fontsize=9, color=TEXTO2)
ax.tick_params(axis="x", colors=TEXTO2, labelsize=8)
ax.tick_params(axis="y", length=0)
for lado in ("top", "right", "left"):
    ax.spines[lado].set_visible(False)
ax.spines["bottom"].set_color("#d6d4cd")
ax.set_title("Encontrar avaliações negativas: o modelo mais simples empata com a rede neural",
             fontsize=10, color=TEXTO, loc="left", pad=10)
plt.tight_layout()
plt.savefig(os.path.join(DOCS, "comparacao_modelos.png"), facecolor=fig.get_facecolor())
print("Gráfico salvo em docs/comparacao_modelos.png")
