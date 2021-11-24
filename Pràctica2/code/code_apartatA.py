%matplotlib notebook
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
from ipywidgets import interact

#remove warnings
import warnings
warnings.filterwarnings('ignore')
# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Carreguem dataset d'exemple
dataset = load_dataset('pokemon.csv')
# dataset = dataset.drop(['japanese_name', 'name', 'type1', 'type2','abilities'], axis=1) # no agafem el primer atribut serial number, ja que és un index i no te res de relació amb els altres atributs.
dataset = dataset.drop(['japanese_name', 'name', 'type1', 'type2'], axis=1) # no agafem el primer atribut serial number, ja que és un index i no te res de relació amb els altres atributs.
data = dataset.values

x = data[:, 0:-2] # no agafem el primer atribut serial number, ja que és un index i no te res de relació amb els altres atributs.
y = data[:, -1]

index = dataset.columns[:40]

print("Dimensionalitat de la BBDD:", dataset.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)

print("Per comptar el nombre de valors no existents:")
print(dataset.isnull().sum())

# CORRELACIO

import seaborn as sns

# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
correlacio = dataset.corr()

correlacio.is_legendary.to_frame().plot.bar()

values = correlacio.is_legendary.values
values = values[values > 0.3]

print(values)

# PREPROCESSING

def clean_dataset(dataset: pd.core.frame.DataFrame, type = 'default') -> pd.core.frame.DataFrame:
    """
    Funció que processa els valors "NULLS" d'un dataset aplicant l'estratègia o tractament especificat
    :param dataset: DataFrame amb la informació que es vol filtrar
    :param type: Indica el tipus de tractament dels  "NaN"
    :return: DataFrame aplicant el mètode especificat a totes les línies amb valors "NaN"
    """
    if type == 'default':
        # Deletes all rows with missing values
        return dataset.dropna()
    elif type == 'backfill':
        # Applies pandas method of backfilling
        return dataset.fillna(method='backfill', axis=1)
    elif type == 'mean':
        # Replaces missing values with the mean of the column
        return dataset.fillna(dataset.mean())
    else:
        print("::-> ERROR : clean_dataset - " + str(type) + " is not a valid option...")

from sklearn.preprocessing import OneHotEncoder

def preprocessing_model(light = False):
    # PREPROCESSING

    dataset = load_dataset('pokemon.csv')
    
    # modify row 773 column capture_rate
    dataset.loc[773, 'capture_rate'] = 30
    
    # Feature Selection : Elimianció atributs que no aporten informació
    dataset = dataset.drop(['japanese_name', 'name', 'pokedex_number'], axis=1)

    # Eliminació de classification, doncs és un atribut que expressa si és legendary o no i aquest és el nostre objectiu a determinar
    dataset = dataset.drop(['classfication'], axis=1)
    
    if not light:
        # Categoritzar Type1
        # Definim encoder
        encoder = OneHotEncoder(sparse=False)
        # Creem variable auxiliar per guardar la informació del atribut type1
        new_data_1 = pd.DataFrame (encoder.fit_transform(dataset[['type1']]))
        # Assignem els noms a les columnes
        new_data_1.columns = encoder.get_feature_names(['type1'])
        # Eliminem la columna del dataset
        dataset.drop(['type1'] ,axis=1, inplace=True)

        # Categoritzar Type2
        # Modifiquem nom de la columna type 2 a type 1 per tal de poder fer un merge posteriorment
        # Utilitzarem el mateix procés que amb la variable type1
        dataset = dataset.rename(columns = {'type2' : 'type1'})
        new_data_2 = pd.DataFrame (encoder.fit_transform(dataset[['type1']]))
        new_data_2.columns = encoder.get_feature_names(['type1'])
        dataset.drop(['type1'] ,axis=1, inplace=True)

        # Sumem les dues taules type
        new_data_type = new_data_1 + new_data_2
        # Afegim al final del dataset les taules type1 i type2 codificades amb OneHotEncoder
        dataset = pd.concat([dataset, new_data_type], axis=1)

        # Eliminem la columna type amb valors NULLS:
        dataset = dataset.drop(['type1_nan'], axis=1)

        #Categoritzar Abilities
        
        import ast
        ab = dataset['abilities'].to_numpy()
        a = []
        for index, row in enumerate(ab):
            if str(row) != 'nan':
                row = ast.literal_eval(row)
                a.append(row)
            else:
                print("Error", index)

        a = np.array(a).flatten()

        flat_list = [item for sublist in a for item in sublist]

        abilitats_uniques = np.unique(flat_list).reshape(-1,1)

        # Declarem encoder tipus OneHotEncoder
        enc = OneHotEncoder(handle_unknown='ignore')
        # Declarem Dataframe auxiliar amb les noves columnes
        enc_df = pd.DataFrame(enc.fit_transform(abilitats_uniques).toarray(), columns=['abilities_' + str(i[0]) for i in abilitats_uniques])

        dataset = dataset.join(enc_df)
        
        dataset = dataset.drop(['abilities'], axis=1)
    else:
        dataset = dataset.drop(['abilities'], axis=1)
    # Tractament valors NULLS -> Aplicarem mètode de la mitjana
    dataset = clean_dataset(dataset, type='mean')

    return dataset

# BALANCEJAR

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from collections import Counter

light = False
dataset = preprocessing_model(light)
dataset['is_legendary'].value_counts()

y = dataset.values[:,34].astype(int)
x = dataset.drop(['is_legendary'] ,axis=1, inplace=True)
x = dataset.values[:,:]

# Random over sampler crea dades 'falses' a partir de les dades 'reals'
# Esta el SMOTE que utilitza KNN.
over_sampler = RandomOverSampler(random_state=0)
over_sampler = SMOTE()
X_res, y_res = over_sampler.fit_resample(x, y)

print(f"Training target statistics: {Counter(y_res)}")
print(f"Testing target statistics: {Counter(y)}")


