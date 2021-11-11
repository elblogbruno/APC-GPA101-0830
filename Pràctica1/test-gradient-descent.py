import numpy as np
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import scipy.stats

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset



def descensGradient(x, y, theta, alpha = 0.0001, n_iteracions=50000):
    # alpha = ratio d'aprenentatge
    # theta = vector amb els valors W0, W1... etc. (Wj) que ens indicaran la posició de la recta. Inicialitzats aleatoriament. 
    # x = matriu amb els valors de les variables dependents.
    # y = vector amb els valors de la variable objectiu.
    
    m = len(y)
    costList = np.zeros(n_iteracions) #Vector per poder fer un seguiment del cost que va tenint la recta quan es va executant l'algorisme.

    thetaList = np.zeros((n_iteracions, theta.shape[0])) #Vector per poder fer un seguiment dels valors de Wj quan es va executant l'algorisme.
    prediccio = np.dot(x, theta) #Prediccio que fa la recta amb els Wj inicials.

    for it in range(n_iteracions):
        
        theta = theta - (alpha*(x.T.dot(prediccio-y))/m) #actualitzem els valors de Wj.
        prediccio = np.dot(x, theta)#Prediccio que fa la recta amb els Wj en el moment d'execució.
        
        thetaList[it, :] = theta.T #Ens guardem el theta actual.
        costList[it] = np.sum(np.square(prediccio-y))/(2*m) #tenint la predicció dels punts que faria la recta i els punts real podem calcular el cost
        
    return theta, costList, thetaList

# Carreguem dataset d'exemple
dataset = load_dataset('data/Admission_Predict_Ver1.1.csv')
dataset = dataset.drop(['Serial No.'], axis=1) # no agafem el primer atribut serial number, ja que és un index i no te res de relació amb els altres atributs.
data = dataset.values

x = data[:, :7] # no agafem el primer atribut serial number, ja que és un index i no te res de relació amb els altres atributs.
y = data[:, 7]
index = dataset.columns[:8]

#X2 = np.ones(shape=(x.shape[0], 2)) #Creem X2, una matriu plena de 1 que tingui 2 columnes i tantes fileres com té X1

X2 = np.c_[np.ones((len(x), 1)), x]
print(X2)
theta = np.random.randn(2,1) #Generem una Wj aleatoria per a començar l'algorisme

iteracions=1000
ratio_aprenentatge=0.001

# print(X2, X2.shape)
print(theta, theta.shape)
print(X2.shape)
theta_final, cost_history, theta_history = descensGradient(X2, y, theta, ratio_aprenentatge, iteracions) #Cridem a la funció per calcular el gradient del descens


print(theta_final, cost_history[-5:])


plt.plot(range(iteracions), cost_history)
plt.show()


plt.plot(X2[:,1], Y1, "x")
plt.plot(X2[:,1], X2 @ theta_final, "b")
plt.show()