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

- [ ] Exploração dos dados (notas, comentários, tamanho do texto)
- [ ] Definição do alvo e análise do desbalanceamento
- [ ] Limpeza do texto
- [ ] Baseline e TF-IDF + regressão logística (ROC-AUC e PR-AUC)
- [ ] Comparação com árvore de decisão e Random Forest
- [ ] Análise de erros: o que o modelo erra e por quê
- [ ] API com FastAPI, Docker e deploy no Google Cloud Run
- [ ] Comparação com um LLM e limitações

## Estrutura

```
classificador-reviews-olist/
├── dados/        # CSVs do Kaggle (não versionados)
├── src/          # scripts de cada etapa
├── docs/         # gráficos e métricas
└── README.md
```

## Como obter os dados

Baixe o dataset no [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) e coloque `olist_order_reviews_dataset.csv` na pasta `dados/`.
