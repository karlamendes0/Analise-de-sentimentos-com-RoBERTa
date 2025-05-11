# Analise de sentimentos de tweet - Modelo RoBERTA
Neste caderno, desenvolveremos um modelo de Análise de Sentimento para categorizar um tweet como Positivo ou Negativo.

# Dataset
O dataset escolhido foi "Sentiment140 dataset with 1.6 million tweets". Ele contém 1.600.000 tweets extraídos usando a API do Twitter. Os tweets foram anotados (0 = negativo, 2 = neutro, 4 = positivo) e podem ser usados ​​para detectar sentimentos. Foi criado por alunos de Stanford, e essa rotulação possibilita a detecção automatizada de sentimentos nas mensagens, tornando o conjunto de dados extremamente útil para pesquisas no campo da análise de sentimentos em redes sociais. Pode ser consultado nesse link: https://www.kaggle.com/datasets/kazanova/sentiment140

# Modelo Utilizado 
RoBERTA - 89,84% de acurácia

#Métricas finais de avaliação:
=============================
eval_accuracy: 0.8984
eval_f1: 0.8984
eval_precision: 0.8985
eval_recall: 0.8984
eval_pr_auc: 0.9627


# Fluxo do Trabalho 📓

- Realizar o Pré-processamento dos dados 🔎
- Tokenização: Os textos dos tweets são convertidos em tokens que o modelo pode processar. 🗳️ 
- Divisão dos Dados: Os dados são divididos em dois conjuntos: treinamento (75%) e teste (25%). 🗳️
- Treinamento do Modelo RoBERTa: O modelo RoBERTa é treinado com os dados de treinamento. 💪
- Evolução dos Parâmetros: O modelo ajusta seus parâmetros durante o treinamento, otimizando suas previsões. 🧠
- Teste do Modelo RoBERTa: O modelo é avaliado com o conjunto de dados de teste para medir sua capacidade de generalização. 🏆
- Matriz de Confusão: A matriz de confusão é gerada para avaliar o desempenho do modelo. 🏆
