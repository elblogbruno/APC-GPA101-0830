from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

def printModelScore(model, X_train, y_train, X_test, y_test):
    print("\n")
    print(type(model).__name__)
    print ('Training Score:', model.score(X_train, y_train) )
    print ('Testing Score:', model.score(X_test, y_test) )
    print ('Training MSE: ', np.mean((model.predict(X_train) - y_train)**2))
    print ('Testing MSE: ', np.mean((model.predict(X_test) - y_test)**2))
    print ('')
    report = classification_report(y_test, model.predict(X_test))
    print(report)
    print("\n")

x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, train_size=0.5, test_size=0.5, random_state=0)

param_grid = {"n_neighbors": [5, 10, 20, 50], "weights":['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

log_reg = KNeighborsClassifier()
# ValueError: Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
grid_search = GridSearchCV(log_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error', verbose=1)
 
grid_search.fit(x_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
print("Dades sense balancejar ni standaritzar")
log_reg = KNeighborsClassifier(algorithm='auto', n_neighbors=5, weights='distance')
log_reg.fit(x, y)
# y_pred = log_reg.predict(x)

printModelScore(log_reg, x, y, x, y)



print("Dades sense balancejar standaritzar")
x_scaled = StandardScaler().fit_transform(x)
log_reg = KNeighborsClassifier(algorithm='auto', n_neighbors=5, weights='distance')
log_reg.fit(x_scaled, y)
y_pred = log_reg.predict(x)

printModelScore(log_reg, x_scaled, y, x_scaled, y)


print("Dades  balancejar sense standaritzar")
log_reg = KNeighborsClassifier(algorithm='auto', n_neighbors=5, weights='distance')
log_reg.fit(X_res, y_res)
y_pred = log_reg.predict(X_res)

printModelScore(log_reg, X_res, y_res, X_res, y_res)


print("Dades  balancejar standaritzar")
x_scaled = StandardScaler().fit_transform(X_res)
log_reg = KNeighborsClassifier(algorithm='auto', n_neighbors=5, weights='distance')
log_reg.fit(x_scaled, y_res)
y_pred = log_reg.predict(x_scaled)

printModelScore(log_reg, x_scaled, y_res, x_scaled, y_res)


x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, train_size=0.8, test_size=0.2, random_state=0)

log_reg = KNeighborsClassifier(algorithm='auto', n_neighbors=5, weights='distance')
log_reg.fit(x_train, y_train)

probs = log_reg.predict_proba(x_test)
roc_and_pr(y_test, probs)

printModelScore(log_reg, x_train, y_train, x_test, y_test)

# cross-validation

scores = cross_val_score(log_reg, x_train, y_train, cv=6, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Mean:", scores.mean())



