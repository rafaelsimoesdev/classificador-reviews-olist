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
- [x] Limpeza do texto
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

## Etapa 3: limpeza do texto

Script: [`src/etapa3_limpar_texto.py`](src/etapa3_limpar_texto.py)

O que o script faz: mantém só avaliações com comentário (sem a nota 3), cria o alvo (1 = negativa, 0 = positiva), converte para minúsculas, remove links, quebras de linha e espaços repetidos, e descarta comentários vazios e duplicados do mesmo pedido.

| | Valor |
|---|---|
| Antes da limpeza | 37.420 comentários |
| Vazios removidos | 26 |
| Duplicados removidos (mesmo pedido e texto) | 14 |
| **Base final** | **37.380 comentários** |
| Negativas / positivas | 10.889 (29,1%) / 26.491 (70,9%) |

**Palavras mais associadas a cada classe** (quantas vezes aparecem mais em uma classe que na outra, com no mínimo 30 ocorrências):

| Negativas | Positivas |
|---|---|
| péssima (52x), dinheiro (48x), insatisfeita (45x), reembolso (43x), descaso (41x), providências (39x), estorno (38x) | amei (66x), ótimo (54x), excelente (51x), adorei (44x), parabéns (37x), perfeito (27x) |

- **Queixas de atendimento, não só de produto:** "reembolso", "estorno", "providências" e "descaso" mostram que muitos comentários negativos falam da resposta da loja ou da devolução do dinheiro, além do item entregue.
- **Acentos inconsistentes:** "péssima" e "pessima", "ótima" e "otima" aparecem como palavras diferentes. Isso divide o sinal entre duas grafias, então a etapa seguinte compara o modelo com e sem normalização de acentos.

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
