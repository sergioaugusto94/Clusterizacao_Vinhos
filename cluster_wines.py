import pandas as pd
from minisom import MiniSom

base = pd.read_csv('wines.csv')
x = base.iloc[:, 1:14].values
y = base.iloc[:,0].values

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range = (0,1))
x = normalizador.fit_transform(x)

#Self Organizing Map (SOM)
# x e y é o tamanho dos neurônios da camada de saída
# no caso temos 64 neurônios (8x8).
# O número total de neurônios é dado pela fórmula:
# 5*raiz(N), N --> numero de registros
#input_len --> numero de atributos de entrada
#sigma -> raio em volta do BMU (best matching unit)
#random_seed -> define uma inicialização dos pesos
som = MiniSom(x = 8, y = 8, input_len = 13, 
              sigma = 1.0, learning_rate = 0.5, 
              random_seed = 2)

#fazendo a inicialização dos pesos
som.random_weights_init(x)

#fazendo o treinamento
som.train_random(data = x, num_iteration = 200)

#obtendo os pesos
som._weights

# o numero de vezes que cada neurõnio da camada de saída
# foi selecionado como BMU
cont_activation = som.activation_response(x)

from pylab import pcolor, colorbar, plot
pcolor(som.distance_map().T)
#MID - Mean Inter Neuron Distance
colorbar()
#quanto mais proximo de um for o neuronio, significa
# que ele é bem diferente dos seus visinhos

# winner retorna a posição do neurônio BMU para um determinado
#registro. No caso abaixo, para o registro 0
w = som.winner(x[0])
markers = ['o', 's', 'D']
color = ['r', 'g', 'b']
y[y == 1] = 0
y[y == 2] = 1
y[y == 3] = 2

for i, j in enumerate(x):
    # print(i) #printa o numero da linha
    # print(j) #printa os registros de cada linha
    w = som.winner(j)
    # print(w)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10, 
         markeredgecolor = color[y[i]], markeredgewidth = 2)





