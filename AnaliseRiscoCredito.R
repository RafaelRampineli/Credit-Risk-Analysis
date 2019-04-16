# Mini Projeto 2 - Analise de Risco de Crédito

# Objetivo: Avaliar o risco de concessão de crédito a clientes de instituições financeiras.

# Será necessário identificar as variáveis mais relevantes para a construção do modelo, bem como analisar o dataset
# realizar conversões e normalizações das variáveis e por mim, construir o melhor modelo preditivo possível.

# Carregando o datset
dataset_credit <- read.csv("credit_dataset.csv", header = TRUE, sep = ",")

# Coluna credit.rating é nossa coluna target
head(dataset_credit)
str(dataset_credit)

# Verificar se existe dados missing no dataset.
any(is.na(dataset_credit))

################################################################################################################################################
                        ################## ETAPA 1: ANALISAR O DATASET PARA COMPRENDER AS VARIÁVEIS ##################
################################################################################################################################################

# Identificando variáveis categóricas que não estão como factor e devemos realizar a transformação.
Variaveis_Categoricas <- c('credit.rating', 'account.balance', 'previous.credit.payment.status',
                           'credit.purpose', 'savings', 'employment.duration', 'installment.rate',
                           'marital.status', 'guarantor', 'residence.duration', 'current.assets',
                           'other.credits', 'apartment.type', 'bank.credits', 'occupation', 
                           'dependents', 'telephone', 'foreign.worker')

# As variáveis que não forem Categóricas, ou seja, variáveis numéricas que realmente representam valores e não categorias
# deveremos realizar a normalização dos dados.
# Os dados estão em escalas diferentes e para um melhor resultado do modelo, devemos normalizar todos na mesma escala.
Variaveis_Numericas <- c("credit.duration.months", "age", "credit.amount")

################################################################################################################################################
    ################## ETAPA 2: CRIANDO FUNÇÕES PARA CONVERTER VARIÁVEIS CATEGÓRIAS E NORMALIZAÇÃO DE VARIÁVEIS NUMÉRICAS ################## 
################################################################################################################################################

# Função irá transformar cada uma das "colunas" informadas em factor e salvar novamente no dataset
func_to.factors <- function(dataset, var_categoricas){
  for (n in var_categoricas){
    dataset[[n]] <- as.factor(dataset[[n]])
  }
  return(dataset)
}

# Função irá realizar a normalização de cada uma das "colunas" informadas e salvar novamente no dataset
func_scale.features <- function(dataset, vars_to_scale){
  for (n in vars_to_scale){
    dataset[[n]] <- scale(dataset[[n]], center=T, scale=T) # Função scale está sendo utilizada para realizar a normalização dos dados
  }
  return(dataset)
}


# Realizando a transformação dos dados em variáveis categóricas
dataset_credit_transformed <- func_to.factors(dataset = dataset_credit, var_categoricas = Variaveis_Categoricas)

# Realizando a padronização/normalização dos dados numéricos
# Obs. O dataset a ser passado é o "dataset_credit_transformed" que já sofreu a transformação das variáveis categóricas
dataset_credit_transformed <- func_scale.features(dataset = dataset_credit_transformed, vars_to_scale = Variaveis_Numericas)

str(dataset_credit_transformed)

# Verificar se o número de colunas do novo dataset é igual ao anterior, garantindo assim que todas variáveis foram selecionadas.
if ( ncol(dataset_credit_transformed) != ncol(dataset_credit) ){
  "VERIFICAR A ETAPA DA SELEÇÃO DAS VARIÁVEIS PARA TRANSFORMAÇÃO E NORMALIZAÇÃO.
  EXISTEM VARIÁVEIS QUE FICARAM DE FORA."
}

################################################################################################################################################
                            ################## ETAPA 3: DIVIDINDO OS DADOS EM TREINO E TESTE ################## 
################################################################################################################################################

# Existem várias maneiras de realizar o split dos dados em treino e teste. Abaixo será utilizando o pacote caTools para o split.

# Dividindo os dados utilizando o pacote CaTools.
install.packages("caTools")
library("caTools")

# 70% dos dados serão classificados como TRUE e 30% como FALSE na variavel amostra_Dados,
# Representando assim 70% -> Treino e 30% Teste.
amostra_Dados <- sample.split(dataset_credit_transformed$credit.rating, SplitRatio = 0.70)
dados_treino <- subset(dataset_credit_transformed, amostra_Dados == TRUE)
dados_teste <- subset(dataset_credit_transformed, amostra_Dados == FALSE)

################################################################################################################################################
                      ################## ETAPA 4: IDENTIFICANDO VARIÁVEIS MAIS RELEVANTES DO DATASET ##################
################################################################################################################################################

# Importando o pacote caret para ser possível utilizar a função: varImp
library("caret")

# Existem várias formas de realizar o levantamento das variáveis mais relevantes do dataset.
# Abaixo irei utilizar o algoritmo RandomForest para obter essa informação.
# Obs. Importante: Utilizar os dados de Treino
library(randomForest)

variaveis_relevantes_rf <- randomForest(credit.rating ~ . ,
                                     data = dados_treino,
                                     ntree = 100, nodesize = 10, importance = T)


# Visualizando o resultado das variáveis mais relevantes do dataset
variaveis_relevantes_rf
varImp(variaveis_relevantes_rf)
varImpPlot(variaveis_relevantes_rf)


# Realizando O mesmo levantamento de variáveis mais relevantes do dataset, porém utilizando funções fornecidas pelo pacote Caret.
# Obs: Esse algoritmo requer bastante processamento de CPU e pode levar + tempo para o resultado.
# library("caret")

# Criando uma função que será responsável por fazer a análise de variávéis relevantes.

# Metodo de Cross-Validation divide o conjunto de dados em subconjuntos e cada subconjunto é mantido até 
# o fim do treinamento dos outros subconjuntos. Esse processo ocorre até que cada subconjunto tenha um resultado
# e uma estimativa geral de precisão seja fornecida no final.
func_feature_selection <- function (n_interacoes, var_preditoras, var_target){
  ctrl_rfe <- rfeControl(functions = rfFuncs, method = "cv", # Cross-Validation Method
                         verbose = FALSE, returnResamp = "all",
                         number = n_interacoes) # Realiza a divisão em 10-fold-cross validation
  
  result_rfe <- rfe(x = var_preditoras, y = var_target,
                    sizes = 1:10,
                    rfeControl = ctrl_rfe)
}

variaveis_relevantes_caret <- func_feature_selection(n_interacoes = 20, 
                                                     var_preditoras = dados_treino[,-1], #Ignorando a 1º Coluna
                                                     var_target = dados_treino[,1] #Somente a 1º Coluna
                                                     )


# Visualizando o resultado das variváveis mais relevantes do dataset utilizando pacote caret
variaveis_relevantes_caret
varImp(variaveis_relevantes_caret)

# Realizando a análise de variáveis mais importantes com uma segunda opção de algoritmo utilizando o pacote caret:

# method = repeatedcv : Realiza a cross validation N vezes e o resultado final é a média do número de repetições.
# number = Número de divisões a serem realizadas
# repeats = Número de vezes que será executado

?trainControl
?train

fun_train_control <- function(metodo_trainControl, number_kfold, n_repeats, formula, dataset, method_train){
  train_control <- trainControl(method = metodo_trainControl, 
                                number = number_kfold, 
                                repeats = n_repeats)
  
  train_result <- train(formula, 
                        data = dataset, 
                        method = method_train, 
                        trControl = train_control)
  
  return(train_result)
}

variaveis_relevantes_trainControl <- fun_train_control(metodo_trainControl = "repeatedcv",
                                                       number_kfold = 10,
                                                       n_repeats = 3,
                                                       formula = credit.rating ~ .,
                                                       dataset = dados_treino,
                                                       method_train = "glm")

varImp(variaveis_relevantes_trainControl)
plot(varImp(variaveis_relevantes_trainControl))

# RESULTADO: Análisando os 3 algoritmos utilizandos para identificar variáveis mais relevantes, é possível notar que ambos os 
# algoritmos apresentara quase as mesmas variáveis como sendo "relevantes". 
# A divergência entre um modelo e outro é ACEITÁVEL e sempre vai ocorrer. Cabe ao analista escolher quais são as variáveis com
# maior relevancia e utiliza-las. Os algoritmos apenas auxiliam para essa tomada de decisão.

################################################################################################################################################
                    ################## ETAPA 5: CONSTRUINDO UM MODELO UTILIZANDO OS DADOS DE TREINO ################## 
################################################################################################################################################

# Durante essa etapa iremos realizar a construção de um modelo de classificação utilizando o algoritmo de regressão Logística,
# que é utilizado e recomendado para fazer classificações de 2 possíveis OUTPUTs onde os dados "independentes" possuem
# relações entre sí para realizar o output classificatório.
# No caso da classificação que queremos realizar é se o cliente irá receber crédito/empréstimo da agência ou não.

modelo_logistico <- glm("credit.rating ~ .", data = dados_treino, family = binomial(link = "logit"))
summary(modelo_logistico)

previsao_modelo  <- predict(modelo_logistico, dados_teste, type = "response")

# O resultado do modelo de Regressão Logística consiste em "probabilidade" de um evento acontecer devido ao "response".
# O resultado varia entre 0-1 e iremos realizar um arredondamento nos valores para os valores serem 0 ou 1,
# para posteriormente ser possível realizar a comparação do resultado através do método confusionMatrix.
previsao_modelo <- round(previsao_modelo)
previsao_modelo

# Avaliando o resultado previsto pelo modelo criado com todas as variáveis
# A avaliação é feita utilizando como referencia, ou seja, valores corretos, os dados de teste

# Modelo criado com dados treino
# Previsão feita com dados teste
# E Avaliação feita com dados teste
confusionMatrix(table(data = previsao_modelo, reference = dados_teste[,1]), positive = '1')

# Resultado: Obtivemos 78% de acertividade dos dados utilizando todas as variáveis em um modelo de regressão Logistica

################################################################################################################################################
              ################## ETAPA 6: OTIMIZANDO O MODELO UTILIZANDO SOMENTE AS VARIÁVEIS MAIS RELEVANTES ##################
################################################################################################################################################

# De acordo com o resultado da etapa 4, as variáveis mais relevantes são:
  # account.balance | credit.duration.months | previous.credit.payment.status | credit.purpose | credit.amount | savings

modelo_logistico_v2 <- glm("credit.rating ~ account.balance + credit.duration.months + previous.credit.payment.status + credit.purpose + credit.amount + savings",
                          data = dados_treino, family = binomial(link = "logit"))

summary(modelo_logistico_v2)

previsao_modelo_v2 <- predict(modelo_logistico_v2, dados_teste, type = "response")
previsao_modelo_v2 <- round(previsao_modelo_v2)
previsao_modelo_v2

confusionMatrix(table(data = previsao_modelo_v2, reference = dados_teste[,1]), positive = '1')

# Resultado: Obtivemos 76% de acertividade dos dados utilizando somente variáveis consideradas "mais importantes".


### Realizando um teste de otimização utilizando o algoritmo RandomForest
library("rpart")

modelo_randomForest <- rpart("credit.rating ~ account.balance + credit.duration.months + previous.credit.payment.status + credit.purpose + credit.amount + savings", data = dados_treino, method = "class")
summary(modelo_randomForest)

previsao_randomForest <- predict(modelo_randomForest, dados_teste, type = "class")

confusionMatrix(table(data = previsao_randomForest, reference = dados_teste[,1]), positive = '1')

# Resultado : Obtivemos 77% de acertividade dos dados utilizando o modelo RandomForest


# Resultado Final: De todos os modelos utilizados, O modelo com todas as variáveis de Regressão Logistica obteve um resultado melhor (78%) em comparação ao resultado quando selecionamos somente variáveis consideradas
# "importantes" durante a análise.

################################################################################################################################################
          ################## ETAPA 7: APRESENTANDO O RESULTADO DO MODELO FINAL ATRAVÉS DE GRÁFICOS DE CURVAS ROC ################## 
################################################################################################################################################

# Atribuindo o melhor modelo que obtivemos em uma nova variável.
best_fit_model <- modelo_logistico
best_fit_prediction <- predict(best_fit_model, dados_teste[,-1], type = "response") # Não passando a coluna que queremos prever durante a previsão.


# Carregando Biblioteca de utilitários para construção de gráficos
source("plot_utils.R")

# Pacote útil para gerar curvas ROC
library("ROCR")

previsoes <- prediction(best_fit_prediction, dados_teste[,1])

# Acima da linha vermelha representa "boa Previsão". 
# Quanto mais alinhado ao canto superior esquerdo, melhor é o algoritmo e melhor foi a previsão.
plot.roc.curve(previsoes, title.text = "Curva ROC")
plot.pr.curve(previsoes, title.text = "Curva Precision/Recall")


## OBSERVAÇÕES FINAIS E IMPORTANTES: 
# O MODELO E O RESULTADO AQUI APRESENTADO, PODEM SOFRER ANÁLISES E AJUSTES EM BUSCA DE MELHORAR A ASSERTIVIDADE DO MODELO E AUMENTAR A TAXA DE PERCENTUAL.
# O FIM DA ANÁLISE E ACEITAÇÃO DO RESULTADO, VARIA DE ACORDO COM O QUE ESTÁ SE PREVENDO.
# NO CASO DE CONCEDER CREDITO OU NÃO AO CLIENTE, ESTOU CONSIDERANDO QUE 78% DE ASSERTIVIDADE DO MEU MODELO PODE SER CONSIDERADO UMA TAXA BOA PARA
# PREVISÕES E CONSIDERAÇÕES EM CONCEDER OU NÃO O EMPRÉSTIMO.
# LEMBRANDO TAMBÉM QUE A DECISÃO FINAL DE CONCEDER OU NÃO, CABE A PESSOA FÍSICA FINAL, QUE IRÁ UTILIZAR A PREVISÃO DO ALGORITMO PARA LHE AUXILIAR
# NA TOMADA DE DECISÃO.

