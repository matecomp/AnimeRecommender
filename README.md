# Anime Recommender

###### Este projeto é um sistema de recomendação utilizando a base de dados do MyAnimeList disponível no link:
###### https://www.kaggle.com/CooperUnion/anime-recommendations-database

# Getting Started
##### No local desejado siga as instruções abaixo:
###### 1 passo: Clone o projeto e entre na pasta
#
```sh
$ git clone https://github.com/matecomp/AnimeRecommender.git
$ cd AnimeRecommender
```
###### 2 passo: Execute o arquivo main
#
```sh
$ python3 main.py
```
###### 3 passo: Treine o modelo
#
```sh
Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
Type 'copyright', 'credits' or 'license' for more information
IPython 6.0.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: model.fit()
```
###### 4 passo: Avalie o modelo
#
```sh
Epoch 47 : loss 0.1432888213532736
Epoch 48 : loss 0.13194171226320126
Epoch 49 : loss 0.1217403373945261

In [2]: model.evaluate(log=True)
```
###### 5 passo: Compare a saída do programa com a avaliação do usuário
#
```sh
User(216) and Item(Warau Kangofu The Animation):
Rate = 6.0, Predict = 5.999009590805006
Diff = 0.0009904091949941574

User(160) and Item(CLAMP in Wonderland):
Rate = 9.0, Predict = 9.004352154911363
Diff = 0.00435215491136276

User(135) and Item(Nodame Cantabile):
Rate = 8.0, Predict = 8.009179723361832
Diff = 0.00917972336183226

User(7) and Item(Date A Live: Date to Date):
Rate = 7.0, Predict = 6.99793894123092
Diff = 0.0020610587690796933

Out[3]: 0.28477531486751045

In [4]: "O método evaluate retorna o erro médio"

```
###### 6 passo: salve o modelo
#
```sh
In [4]: model.save('RecommenderSystem/Weights')
```
###### Para recarregar o modelo salvo:
#
```sh
In [5]: model.load('RecommenderSystem/Weights')
```
