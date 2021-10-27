#!/usr/bin/env python
# coding: utf-8

# # Practica 1 APC: Graduate Admission 2
# ## Predicting admission from important parameters
# 
# [URL Database Kaggle](https://www.kaggle.com/mohansacharya/graduate-admissions)
# 
# - #### Bruno Moya Ruiz - 1568176
# - #### Marc Garrofé Urrutia - 1565644
# - #### Martí Simon Rojas - 1568180

# # Apartat (C): Analitzant Dades
# En aquest apartat analitzarem la nostra base de dades. Volem primer separar el input (X) del output (Y). En el nostre cas tenim 9 variables, de les quals la ultima es el nostre output, es a dir el resultat que volem obtenir del nostre model.


from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import pyplot as plt
import scipy.stats

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Carreguem dataset d'exemple
dataset = load_dataset('data/Admission_Predict_Ver1.1.csv')
dataset = dataset.drop(['Serial No.'], axis=1) # no agafem el primer atribut serial number, ja que és un index i no te res de relació amb els altres atributs.
data = dataset.values

x = data[:, :7] # no agafem el primer atribut serial number, ja que és un index i no te res de relació amb els altres atributs.
y = data[:, 7]
index = dataset.columns[:8]
print("Dimensionalitat de la BBDD:", dataset.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)


# D'aqui veiem que la nostra base de dades presenta 500 elements. Ara mirarem si presenta valors no existents, es a dir que el seu valor sigui NaN. Aquests valors poden fer variar el resultat del nostre model i és important tenir-los en compte.

print("Per comptar el nombre de valors no existents:")
print(dataset.isnull().sum())


# Podem observar com el nostre dataset no presenta cap valor NaN.

print("Per visualitzar les primeres 5 mostres de la BBDD:")
dataset.head() 


# Podem observar estadístiques de la nostra base de dades, com la desviació standard d'aquesta (std) , la mitjana de cada variable (mean) entre altres. Aixó ens serveix per descobrir quina distribució segueix cada variable i el seu tipus.

dataset.describe()


# Podem mostrar algun atribut del nostre dataset en format de grafica. En aquest cas mostrarem la atribut 0 (GRE Score) i tambe els altres. Aixo ens permetra veure si per exemple segueixen una distribució gaussiana, aixó és important i ja veurem que una variable amb distribució gaussiana pot ser més bona que una altre.

# mostrem atribut 0 fins a 8

for i in range(len(index)):
    if i==len(index)-1:
        break
    plt.figure()
    plt.scatter(x[:, i], y)
    plt.title(index[i])
    plt.ylabel('Chance of Admit')
    plt.xlabel(index[i])
    plt.show()


for i in range(len(index)):
    if i==len(index)-1:
        break
    plt.figure()
    plt.scatter(x[:, i], y)
    plt.title("Histograma de " + index[i])
    plt.ylabel('Count')
    plt.xlabel(index[i])
    hist = plt.hist(x[:,i], bins=11, range=[np.min(x[:,i]), np.max(x[:,i])], histtype="bar", rwidth=0.8)
#     plt.show()


# Aquest histograma ens permet observar si presenta una distribució Gaussiana. Per exemple, el nostre atribut 0 presenta una distribució Gaussiana. També podem estudiar la correlació entre els diferents atributs per tal de saber si estan correlacionats entre ells. Aixo ens sera util per observar quins atributs tenen més pes.

import seaborn as sns

# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
correlacio = dataset.corr()

plt.figure()

ax = sns.heatmap(correlacio, annot=True, linewidths=.5)


# Podem observar pel heatmap, que gariebe tots els atributs estan relacionats amb un grau per sobre del 65% amb el nostre output menys el atribut research i serial number (que és un index) que son els menys relacionats, i per tant potser un dels possibles a descartar en un futur. També podem utilitzar la funció pairplot per tal de veure els atributs que estan relacionats entre si. Pero la relació que ens interessa observar es la dels atributs amb l'ultim atribut ja que es la sortida, podent així observar quin atribut esta més relacionat amb la sortida.

# Mirem la relació entre atributs utilitzant la funció pairplot
relacio = sns.pairplot(dataset)

# # Apartat (B): Primeres regressions


def mse(v1, v2):
    return ((v1 - v2)**2).mean()


# Per a la regressió podeu utilitzar la llibreria sklearn:


from sklearn.linear_model import LinearRegression

def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()

    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)

    # Retornem el model entrenat
    return regr


# Primer estandaritzarem les nostres dades del dataset. Hem decidit mostrar un histograma per a cada atribut del nostre dataset amb les dades normalitzades i sense normalitzar, per entendre com afecta la normalització o estandaritzacio.

def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t


# In[15]:


for i in range(x.shape[1]):
    plt.figure()
    plt.title("Histograma de l'atribut {} amb normalització".format(str(index[i])))
    plt.xlabel("Attribute Value")
    plt.ylabel("Count")
    x_t = standarize(x)
    hist = plt.hist(x_t[:,i], bins=11, range=[np.min(x_t[:,i]), np.max(x_t[:,i])], histtype="bar", rwidth=0.8)

    plt.figure()
    plt.title("Histograma de l'atribut {} sense normalitzar".format(str(index[i])))
    plt.xlabel("Attribute Value")
    plt.ylabel("Count")
    x_t_1 = x
    hist1 = plt.hist(x_t_1[:,i], bins=11, range=[np.min(x_t_1[:,i]), np.max(x_t_1[:,i])], histtype="bar", rwidth=0.8)


# Hem fet el mateix que abans per a la regressió, primer l'entrenem amb dades sense normalitzar per a cada atribut i després  amb dades normalitzades.

from sklearn.metrics import r2_score

for atribut_index in range(x.shape[1]):
    atribut1 = x[:,atribut_index].reshape(x.shape[0], 1) 
    
    regr = regression(atribut1, y) 
    predicted = regr.predict(atribut1)

    # Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
    plt.figure()
    plt.title("Regressió del atribut {} sense normalitzar".format(index[atribut_index]))
    ax = plt.scatter(x[:,atribut_index], y)
    plt.plot(atribut1[:,0], predicted, 'r')

    # Mostrem l'error (MSE i R2)
    MSE = mse(y, predicted)
    r2 = r2_score(y, predicted)

    print("Mean squeared error: ", MSE)
    print("R2 score: ", r2)


from sklearn.metrics import r2_score

for i in range(x.shape[1]):
    atribut_index = i 
    # Extraiem el primer atribut de x i canviem la mida a #exemples, #dimensions de l'atribut.
    # En el vostre cas, haureu de triar un atribut com a y, i utilitzar la resta com a x.
    atribut1 = x[:,atribut_index].reshape(x.shape[0], 1) 
    
    atribut1 = standarize(atribut1)
    
    regr = regression(atribut1, y) 
    predicted = regr.predict(atribut1)

    # Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
    plt.figure()
    plt.title("Regressió del atribut {} normalitzat".format(index[atribut_index]))
    ax = plt.scatter(x[:,atribut_index], predicted)
    plt.plot(atribut1[:, 0], predicted, 'r')

    # Mostrem l'error (MSE i R2)
    MSE = mse(y, predicted)
    r2 = r2_score(y, predicted)

    print("Mean squeared error: ", MSE)
    print("R2 score: ", r2)


# Un cop mostrats de manera adient, (en forma de taula, i/o de gràfics si la dimensionalitat ho permet) els resultats aconseguits amb la regressió, avaluarem de manera independent la idonietat de cadascun dels atributs.

""" Per a assegurar-nos que el model s'ajusta be a dades noves, no vistes, 
cal evaluar-lo en un conjunt de validacio (i un altre de test en situacions reals).
Com que en aquest cas no en tenim, el generarem separant les dades en 
un 80% d'entrenament i un 20% de validació.
"""
def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:] 
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val

# Dividim dades d'entrenament
x_train, y_train, x_val, y_val = split_data(x, y)

for i in range(x_train.shape[1]):
    x_t = x_train[:,i] # seleccionem atribut i en conjunt de train
    x_v = x_val[:,i] # seleccionem atribut i en conjunt de val.
    x_t = np.reshape(x_t,(x_t.shape[0],1))
    x_v = np.reshape(x_v,(x_v.shape[0],1))
    
    regr = regression(x_t, y_train)    
    error = mse(y_val, regr.predict(x_v)) # calculem error
    r2 = r2_score(y_val, regr.predict(x_v))

    
    print("Error en atribut %s (%d): %f" %(str(index[i]), i, error))
    print("R2 score en atribut %s (%d): %f" %(str(index[i]), i, r2))
    print("---")


# Ara observarem com canvia un cop hem estandaritzat les dades. Per estandartizar, cridem a la funció anterior standarize, i li pasem cada atribut del dataset, es a dir les dades de train i les de validació d'aquell atribut.

for i in range(x_train.shape[1]):
    x_t = x_train[:,i] # seleccionem atribut i en conjunt de train
    x_v = x_val[:,i] # seleccionem atribut i en conjunt de val.
    x_t = np.reshape(x_t,(x_t.shape[0],1))
    x_v = np.reshape(x_v,(x_v.shape[0],1))
    
    x_t = standarize(x_t) #estandaratizem les dades de train del atribut i 
    x_v = standarize(x_v) #estandaratizem les dades de validacio del atribut i 
    
    
    regr = regression(x_t, y_train)    
    error = mse(y_val, regr.predict(x_v)) # calculem error
    r2 = r2_score(y_val, regr.predict(x_v))

    
    print("Error en atribut %s (%d): %f" %(str(index[i]), i, error))
    print("R2 score en atribut %s (%d): %f" %(str(index[i]), i, r2))
    print("---")


# Ara comprovarem com canvia si escollim de primera ma les variables que em trobat que presenten més correlació aixi com una distribució Gaussiana:

dataset_1 = load_dataset('data/Admission_Predict_Ver1.1.csv')
dataset_1 = dataset_1.drop(['Serial No.', 'University Rating' ,'SOP', 'LOR ', 'Chance of Admit '], axis=1) # eliminem els atributs que no volem, deixant els que hem decidit son mes importants.

y = dataset['Chance of Admit '] 
x_train, y_train, x_test, y_test = split_data(x, y)

regr = regression(x_train, y_train)  #creem el objecte regresor
 
y_pred = regr.predict(x_test) # fem la predicció

error = mse(y_test, y_pred) # calculem error
r2 = r2_score(y_test, y_pred) # calculem el r2 score

plt.figure(figsize= (8,8))
plt.scatter(y_test, y_pred, c='crimson')
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.show()


print("Error total: %f" %(error))
print("R2 score total: %f" %(r2))
print("---")


# Si estandaritzem, el resultat no estara influenciat pel rang de les dades. Pero tot i aixi, veiem que estandaritzan obtenim un error mes alt, i un r2 score més baix.

# selected_variables_index = [0,1,2]

dataset_1 = load_dataset('data/Admission_Predict_Ver1.1.csv')
dataset_1 = dataset_1.drop(['Serial No.', 'University Rating' ,'SOP', 'LOR ', 'Chance of Admit '], axis=1) # eliminem els atributs que no volem, deixant els que hem decidit son mes importants.

y_1 = dataset['Chance of Admit '] 
x_train, y_train, x_test, y_test = split_data(x, y_1)

x_train = standarize(x_train) # estandaratizem les dades de train del atribut i 
x_test = standarize(x_test) # estandaratizem les dades de validacio del atribut i 
    
regr = regression(x_train, y_train)  #creem el objecte regresor
 
y_pred = regr.predict(x_test) # fem la predicció

error = mse(y_test, y_pred) # calculem error
r2 = r2_score(y_test, y_pred) # calculem el r2 score

plt.figure(figsize= (8,8))
plt.scatter(y_test, y_pred, c='crimson')

plt.xlabel('True values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.show()


print("Error total: %f" %(error))
print("R2 score total: %f" %(r2))
print("---")


# Si en canvi introduim tots els atributs veiem un resultat completament diferent:

# selected_variables_index = [0,1,2]
x_train, y_train, x_test, y_test = split_data(x, y)

regr = regression(x_train, y_train)  #creem el objecte regresor

y_pred = regr.predict(x_test) # fem la predicció

error = mse(y_test, y_pred) # calculem error
r2 = r2_score(y_test, y_pred) # calculem el r2 score

plt.figure(figsize= (8,8))
plt.scatter(y_test, y_pred, c='crimson')

plt.xlabel('True values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.show()


print("Error total: %f" %(error))
print("R2 score total: %f" %(r2))
print("---")


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 
from sklearn.metrics import r2_score

# x_train, y_train, x_test, y_test = split_data(x, y)

errors = []
r2Scores = []
for i in range(1,6):
    pca = PCA(n_components=i)
    data = pca.fit_transform(scale(x))
    
    lineal = LinearRegression()
    lineal.fit(data, y)
    y_pred = lineal.predict(data)
    
    error = mse(y, y_pred) # calculem error
    r2 = r2_score(y, y_pred) # calculem el r2 score
    
    print("Error total (%d): %f" %(i, error))
    print("R2 score total (%d): %f" %(i, r2))
    
    errors.append(error)
    r2Scores.append(r2)

plt.figure(figsize= (8,8))
    
# plt.scatter(i, y_pred, c='crimson')
plt.plot([1,2,3,4,5], errors, '-v')
    
plt.xlabel('Nombre de components en el PCA')
plt.ylabel('MSE')
plt.xticks([1,2,3,4,5])
plt.show()    

plt.figure(figsize= (8,8))
    
plt.scatter(x, y, c='crimson')
plt.plot([1,2,3,4,5], r2Scores, '-v')
    
plt.xlabel('Nombre de components en el PCA')
plt.ylabel('R2 Score')
plt.xticks([1,2,3,4,5])
plt.show()    


# # Apartat (A): El descens del gradient  


class Regressor(object):
    def __init__(self,wj, alpha, x, y):
        # Inicialitzem w0 i w1 (per ser ampliat amb altres w's)
        print(" WJ Inicial : " + str(wj) + " alpha: " + str(alpha))
        self.wj = wj 
        self.alpha = alpha
        self.x = x
        self.y = y
        self.m = len(y)

    def predict(self, x):
        Y_pred = np.array([])
        b = self.wj
        for x in X:
            Y_pred = np.append(Y_pred, b[0] + (b[1] * x))

        return Y_pred

    def __update(self, hy, y):
        # actualitzar aqui els pesos donada la prediccio (hy) i la y real.
        self.wj = self.wj - (self.alpha * (self.x.T.dot(hy-y))/self.m) #actualitzem els valors de Wj.

    def train(self, max_iter, epsilon):
        # Entrenar durant max_iter iteracions o fins que la millora sigui inferior a epsilon
        cost_list = np.zeros(max_iter) #Vector per poder fer un seguiment del cost que va tenint la recta quan es va executant l'algorisme.

        wj_list = np.zeros((max_iter, self.wj.shape[0])) #Vector per poder fer un seguiment dels valors de Wj quan es va executant l'algorisme.
        prediccio =  np.dot(self.x, self.wj) #Prediccio que fa la recta amb els Wj inicials.

        for i in range(max_iter):
            prediccio =  np.dot(self.x, self.wj) # Prediccio que fa la recta amb els Wj en el moment d'execució.
            
            self.__update(prediccio, self.y)

            wj_list[i, :] = self.wj.T #Ens guardem el theta actual.
            
            cost = np.sum( np.square(prediccio- self.y) ) / (2*self.m)
            
            cost_list[i] = cost
            
#             print("Iteracio (%d): %f" %(i, cost))

            if cost < epsilon:
                print("Cost mes petit que epsilon")
                break
            
            
        return self.wj, cost_list, wj_list

    
    
  


# Fem funcionar el algoritme de gradient descent, i mostrem en grafiques.


import sys

x_train, y_train, x_test, y_test = split_data(x, y)

y_modified = y_train.reshape((y_train.shape[0], 1))

X2 = np.ones(shape=(x_train.shape[0], 2)) #Creem X2, una matriu plena de 1 que tingui 2 columnes i tantes fileres com té X1

for i in range(x_train.shape[0]):
    X2[i][1] = x_train[i][5] #assignem els valors de la columna 8 de X1 (millor valor per fer regressió) a la columna 1 de X2
    
ratio_aprenentatge = 0.0001 # tambe se li diu alpha 

wj = np.random.randn(2,1) 

regr = Regressor(wj, ratio_aprenentatge, X2, y_modified)

max_iteracions = 100000

theta_final, cost_history, theta_history = regr.train(max_iteracions, sys.float_info.epsilon) #Cridem a la funció per calcular el gradient del descens

plt.figure(figsize= (8,8))
plt.plot(range(max_iteracions), cost_history)
plt.xlabel('Nombre de iteracions')
plt.ylabel('Cost')
plt.show()

plt.figure(figsize= (8,8))
plt.plot(X2[:,1], y_train, "x")
plt.plot(X2[:,1], X2 @ theta_final, "b")

plt.xlabel('Dades ')
plt.ylabel('Valor de theta')
plt.show()


# Podem optimitzar el regresor i despres predir utilitzant el regresor i veure com millora.

import sys

x_train, y_train, x_test, y_test = split_data(x, y)

print(x_train.shape, y_train.shape, y_test.shape, y_train.shape)

y_modified = y_train.reshape((y_train.shape[0], 1))

X2 = np.ones(shape=(x_train.shape[0], 2)) #Creem X2, una matriu plena de 1 que tingui 2 columnes i tantes fileres com té X1

for i in range(x_train.shape[0]):
    X2[i][1] = x_train[i][5] #assignem els valors de la columna 8 de X1 (millor valor per fer regressió) a la columna 1 de X2
    
ratio_aprenentatge = 0.0001 # tambe se li diu alpha 

wj = np.random.randn(2,1) 

regr = Regressor(wj, ratio_aprenentatge, X2, y_modified)

max_iteracions = 100000

theta_final, cost_history, theta_history = regr.train(max_iteracions, sys.float_info.epsilon) #Cridem a la funció per calcular el gradient del descens

y_pred = regr.predict(x_train)[:100]

error = mse(y_test, y_pred) # calculem error
r2 = r2_score(y_test, y_pred) # calculem el r2 score

print("Error total: %f" %(error))
print("R2 score total: %f" %(r2))

plt.figure(figsize= (8,8))
plt.scatter(y_test, y_pred, c='crimson')

plt.xlabel('True values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.show()


import sys

y = y.reshape((y.shape[0], 1))

X2 = np.ones(shape=(x.shape[0], 2)) #Creem X2, una matriu plena de 1 que tingui 2 columnes i tantes fileres com té X1

for i in range(x.shape[0]):
    X2[i][1] = x[i][5] #assignem els valors de la columna 8 de X1 (millor valor per fer regressió) a la columna 1 de X2
    
ratio_aprenentatge = 0.005 # tambe se li diu alpha 

wj = np.random.randn(2,1) 

epochs = []
sum_cost = []

for i in range(1, 10):
    
    max_iteracions = 10000*i
    
    epochs.append(max_iteracions)
    
    print("Comprovant max_iter = %d alpha = %d" %(max_iteracions, ratio_aprenentatge))
    
    regr = Regressor(wj, ratio_aprenentatge, X2, y)

    theta_final, cost_history, theta_history = regr.train(max_iteracions, sys.float_info.epsilon) #Cridem a la funció per calcular el gradient del descens
    
    sum_cost.append(cost_history)
    
plt.figure(figsize= (8,8))

for index, i in enumerate(sum_cost):
    plt.plot(range(epochs[index]), i)
    
plt.xlabel('Nombre de iteracions (Max_iter = %d)' %(max_iteracions))
plt.ylabel('Cost')
plt.show()

plt.figure(figsize= (8,8))
plt.plot(X2[:,1], y, "x")
plt.plot(X2[:,1], X2 @ theta_final, "b")

plt.xlabel('Dades ')
plt.ylabel('Valor de theta')
plt.show()    


import sys

y = y.reshape((y.shape[0], 1))

X2 = np.ones(shape=(x.shape[0], 2)) #Creem X2, una matriu plena de 1 que tingui 2 columnes i tantes fileres com té X1

for i in range(x.shape[0]):
    X2[i][1] = x[i][5] #assignem els valors de la columna 8 de X1 (millor valor per fer regressió) a la columna 1 de X2
    
ratio_aprenentatge = 0.001 # tambe se li diu alpha 

wj = np.random.randn(2,1) 

epochs = []
sum_cost = []

for i in range(1, 5):
    
    max_iteracions = 100000
    
    epochs.append(max_iteracions)
    
    print("Comprovant max_iter = %d alpha = %d" %(max_iteracions, ratio_aprenentatge))
    
    regr = Regressor(wj, ratio_aprenentatge, X2, y)

    theta_final, cost_history, theta_history = regr.train(max_iteracions, sys.float_info.epsilon) #Cridem a la funció per calcular el gradient del descens
    
    sum_cost.append(cost_history)
    
    plt.figure(figsize= (8,8))

    plt.plot(range(max_iteracions), cost_history)

    plt.xlabel('Nombre de iteracions (Max_iter = %d)' %(max_iteracions))
    plt.ylabel('Cost')
    plt.show()


import sys

y = y.reshape((y.shape[0], 1))

X2 = np.ones(shape=(x.shape[0], 2)) #Creem X2, una matriu plena de 1 que tingui 2 columnes i tantes fileres com té X1

for i in range(x.shape[0]):
    X2[i][1] = x[i][5] #assignem els valors de la columna 8 de X1 (millor valor per fer regressió) a la columna 1 de X2
    
ratio_aprenentatge = 0.0001 # tambe se li diu alpha 

wj = np.random.randn(2,1) 

epochs = []
sum_cost = []

for i in range(1, 5):
    
    max_iteracions = 100000
    
    epochs.append(max_iteracions)
    
    print("Comprovant max_iter = %d alpha = %d" %(max_iteracions, ratio_aprenentatge))
    
    regr = Regressor(wj, ratio_aprenentatge, X2, y)

    theta_final, cost_history, theta_history = regr.train(max_iteracions, sys.float_info.epsilon) #Cridem a la funció per calcular el gradient del descens
    
    sum_cost.append(cost_history)
    
    plt.figure(figsize= (8,8))

    plt.plot(range(max_iteracions), cost_history)

    plt.xlabel('Nombre de iteracions (Max_iter = %d)' %(max_iteracions))
    plt.ylabel('Cost')
    plt.show()


# L'últim pas serà validar el regressor trobat pel descent del gradient desenvolupat en aquest apartat visualment, aplicat a un model de recta i un model de pla. Per a això, caldrà considerar el millor atribut identificat en el primer punt de l'anterior entrega per visualitzar la línia regressora en 2D (podeu mostrar dades 2d amb la funció scatter). Després, dos dels atributs identificats a l'últim punt del primer exercici per visualitzar el pla regressor en 3D (En el cas 3D l’scatter s’ha de fer sobre una figura amb projecció 3D).


get_ipython().run_line_magic('matplotlib', 'notebook')

# Creem figura 3d
from mpl_toolkits.mplot3d import axes3d, Axes3D
# generem dades 3D d'exemple
x_val = np.random.random((100, 2))
y_val = np.random.random((100, 1))
regr = regression(x_val, y_val)
predX3D = regr.predict(x_val)

# Afegim els 1's
A = np.hstack((x_val,np.ones([x_val.shape[0],1])))
w = np.linalg.lstsq(A,predX3D)[0]

#Dibuixem
#1r creem una malla acoplada a la zona de punts per tal de representar el pla
malla = (range(20) + 0 * np.ones(20)) / 10 
malla_x1 =  malla * (max(x_val[:,0]) - min(x_val[:,0]))/2 + min(x_val[:,0])
malla_x2 =  malla * (max(x_val[:,1]) - min(x_val[:,1]))/2 + min(x_val[:,1])

#la funcio meshgrid ens aparella un de malla_x1 amb un de malla_x2, per atot
#element de mallax_1 i per a tot element de malla_x2.
xplot, yplot = np.meshgrid(malla_x1 ,malla_x2)

# Cal desnormalitzar les dades
def desnormalitzar(x, mean, std):
    return x * std + mean

#ara creem la superficies que es un pla
zplot = w[0] * xplot + w[1] * yplot + w[2]

#Dibuixem punts i superficie
plt3d = plt.figure('Coeficiente prismatico -- Relacio longitud desplacament 3D', dpi=100.0).gca(projection='3d')
plt3d.plot_surface(xplot,yplot,zplot, color='red')
plt3d.scatter(x_val[:,0],x_val[:,1],y_val)
