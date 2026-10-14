# Classificador de Reviews Olist

**Autor:** Rafael Simões  
**Dataset:** [Brazilian E-Commerce Public Dataset by Olist, Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
**Pergunta:** Dado o comentário de um cliente, a avaliação é positiva ou negativa?

> **Status:** em desenvolvimento. Este README é atualizado a cada etapa.

---

## Contexto

Depois de prever atrasos de entrega ([Projeto 2](https://github.com/rafaelsimoesdev/previsao-atraso-entregas) e [Projeto 3](https://github.com/rafaelsimoesdev/deliveryrisk-api)), este projeto olha para o outro lado da operação: o que o cliente escreve depois de receber o pedido. Ler milhares de comentários manualmente não escala. Um classificador de texto permite priorizar as avaliações negativas e descobrir o que as motiva.

## Definição do alvo

| Nota | Classe |
|---|---|
| 1 e 2 | Negativa |
| 4 e 5 | Positiva |
| 3 | Descartada (sentimento ambíguo) |

Só entram na análise as avaliações que têm comentário em texto.

## Plano

- [x] Exploração dos dados (notas, comentários, tamanho do texto)
- [x] Definição do alvo e análise do desbalanceamento
- [ ] Limpeza do texto
- [ ] Baseline e TF-IDF + regressão logística (ROC-AUC e PR-AUC)
- [ ] Comparação com árvore de decisão e Random Forest
- [ ] Análise de erros: o que o modelo erra e por quê
- [ ] API com FastAPI, Docker e deploy no Google Cloud Run
- [ ] Comparação com um LLM e limitações

## Etapa 2: exploração das reviews

Script: [`src/etapa2_explorar_reviews.py`](src/etapa2_explorar_reviews.py)

![Distribuição das notas](docs/distribuicao_notas.png)

| | Valor |
|---|---|
| Avaliações | 99.224 |
| Com comentário em texto | 40.977 (41%) |
| Notas 1-2 com comentário | 10.890 |
| Notas 4-5 com comentário | 26.530 |
| Nota 3 com comentário (descartada) | 3.557 |
| **Base final para o modelo** | **37.420 comentários** |

- **Desbalanceamento:** 29% negativas e 71% positivas. Um modelo que responde "sempre positiva" acerta 71% sem entender nada, então a acurácia sozinha engana. A avaliação usará precisão, recall, ROC-AUC e PR-AUC.
- **Quem escreve mais:** 77% das avaliações com nota 1 têm comentário, contra 36% das de nota 5. Os comentários negativos estão bem representados.
- **Textos curtos:** a mediana é de 9 palavras (máximo de 45). Alguns comentários têm só espaços e serão removidos na limpeza.

---

## Estrutura

```
classificador-reviews-olist/
├── dados/        # CSVs do Kaggle (não versionados, crie a pasta localmente)
├── src/          # scripts de cada etapa
├── docs/         # gráficos e métricas
└── README.md
```

## Como obter os dados

Baixe o dataset no [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) e coloque `olist_order_reviews_dataset.csv` na pasta `dados/`.
