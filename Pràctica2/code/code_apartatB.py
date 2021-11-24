%matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# import some data to play with
iris = datasets.load_iris()

# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target
print(y)

n_classes = 3
    
fig, sub = plt.subplots(1, 2, figsize=(16,6))
plt.ylabel(['Iris-Setosa (0)', 'Iris-Versicolour (1)', 'Iris-Virginica (2)'])
sub[0].scatter(X[:,0], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
sub[1].scatter(X[:,1], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')


particions = [0.5, 0.7, 0.8]

def printModelScore(model, X_train, y_train, X_test, y_test):
    print("\n")
    print(type(model).__name__)
    print ('Training Score:', model.score(X_train, y_train) )
    print ('Testing Score:', model.score(X_test, y_test) )
    print ('Training MSE: ', np.mean((model.predict(X_train) - y_train)**2))
    print ('Testing MSE: ', np.mean((model.predict(X_test) - y_test)**2))

for part in particions:
    x_t, x_v, y_t, y_v = train_test_split(X, y, train_size=part)
    
    #Creem el regresor logístic
    logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
    # l'entrenem
    logireg.fit(x_t, y_t)
    # probs = logireg.predict_proba(x_v)

    # print ("Correct classification Logistic ", part*100, "% of the data: ", logireg.score(x_v, y_v))
    printModelScore(logireg, x_t, y_t, x_v, y_v)
    #Creem el regresor logístic
    svc = svm.SVC(C=10.0, kernel='rbf', gamma=0.9, probability=True)

    # l'entrenem 
    svc.fit(x_t, y_t)
    # probs = svc.predict_proba(x_v)
    printModelScore(svc, x_t, y_t, x_v, y_v)
    # print ("Correct classification SVM      ", part*100, "% of the data: ", svc.score(x_v, y_v))


    # RANDOM FOREST
    rf = RandomForestClassifier()
    rf.fit(x_t, y_t)
    # probs = rf.predict_proba(x_v)
    printModelScore(rf, x_t, y_t, x_v, y_v)
    # print ("Correct classification Random Forest ", part*100, "% of the data: ", rf.score(x_v, y_v))
    
    # KNN
    knn = KNeighborsClassifier()
    knn.fit(x_t, y_t)
    probs = rf.predict_proba(x_v)
    printModelScore(knn, x_t, y_t, x_v, y_v)
    # print ("Correct classification KNN ", part*100, "% of the data: ", knn.score(x_v, y_v))

def printModelScore(model, X_train, y_train, X_test, y_test):
    print("\n")
    print(type(model).__name__)
    print ('Training Score:', model.score(X_train, y_train) )
    print ('Testing Score:', model.score(X_test, y_test) )
    print ('Training MSE: ', np.mean((model.predict(X_train) - y_train)**2))
    print ('Testing MSE: ', np.mean((model.predict(X_test) - y_test)**2))

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0) # dades partides en train i test sense standaritzar

scaler_training = preprocessing.StandardScaler().fit(x_train)
X_scaled_train = scaler_training.transform(x_train)

scaler_test = preprocessing.StandardScaler().fit(x_test)
X_scaled_test = scaler_test.transform(x_test)


regr = RandomForestRegressor(n_estimators=150, criterion='absolute_error', bootstrap=True, random_state=0, warm_start=True)     # Definim model
regr.fit(X_scaled_train, y_train)

printModelScore(regr, X_scaled_train, y_train, X_scaled_test, y_test)

# COMPARANT MODELS

from sklearn.model_selection import GridSearchCV

# param_grid = {'n_estimators': [125, 150, 175, 200], 'criterion': ['squared_error', 'absolute_error', 'poisson'], 'bootstrap': [False, True], 'warm_start' : [True, False]  }
param_grid = {'n_estimators': [125, 150, 175, 200], 'criterion': ['squared_error', 'absolute_error', 'poisson'] }
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error', verbose=2)
 
grid_search.fit(X_scaled_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

param_grid = {'kernel' : ['linear', 'sigmoid', 'poly', 'rbf'], 'gamma' : ['scale', 'auto'], 'class_weight' : ['balanced', 'None'] }
# svc = make_pipeline(StandardScaler(), SVC())
svc = SVC()

grid_search = GridSearchCV(svc, param_grid, cv=5,
                          scoring='neg_mean_squared_error', verbose=2)
 
grid_search.fit(X_scaled_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)


svc = SVC(class_weight='balanced', gamma='scale', kernel='rbf')
svc.fit(X_scaled_train, y_train)
printModelScore(svc, X_scaled_train, y_train, X_scaled_test, y_test)

from sklearn.model_selection import GridSearchCV

param_grid = {"C": np.logspace(-3,3,7), "penalty":["l1","l2"], 'solver': ['newton-cg', 'saga' ,'sag','lbfgs'], 'multi_class': ['auto', 'ovr', 'multinomial']}
log_reg = LogisticRegression()

grid_search = GridSearchCV(log_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error', verbose=2)
 
grid_search.fit(X_scaled_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)

log_reg = LogisticRegression(C=0.1, penalty='l1', solver='saga', multi_class='auto')
log_reg.fit(x_t, y_t)
printModelScore(log_reg, x_t, y_t, x_v, y_v)

from sklearn.model_selection import GridSearchCV

param_grid = {"n_neighbors": [5, 10, 20, 50], "weights":['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=5,
                          scoring='neg_mean_squared_error', verbose=2)
 
grid_search.fit(X_scaled_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)

knn = KNeighborsClassifier(n_neighbors=20, algorithm='auto', weights='uniform')
knn.fit(x_t, y_t)
printModelScore(knn, x_t, y_t, x_v, y_v)

def roc_and_pr(y_v, probs):
    n_classes = 2
    from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc

    # Compute Precision-Recall and plot curve
    precision = {}
    recall = {}
    average_precision = {}
    plt.figure()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_v == i, probs[:, i])
        average_precision[i] = average_precision_score(y_v == i, probs[:, i])

        plt.plot(recall[i], precision[i],
        label='Precision-recall curve of class {0} (area = {1:0.2f})'
                            ''.format(i, average_precision[i]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="upper right")

        
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_v == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # Plot ROC curve
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
    plt.legend()