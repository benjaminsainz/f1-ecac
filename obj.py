"""
@authors: Benjamin Mario Sainz-Tinajero, Andres Eduardo Gutierrez-Rodriguez,
Héctor Gibrán Ceballos-Cancino, Francisco Javier Cantu-Ortiz
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


def fitness_value(X, ind):
    X_train, X_test, y_train, y_test = train_test_split(X, ind, test_size=0.75, random_state=0)
    svc_model = SVC(kernel='linear', gamma=0.1, max_iter=-1, C=1).fit(X_train, y_train)
    knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2, metric='minkowski').fit(X_train, y_train)
    dtc_model = DecisionTreeClassifier(criterion='entropy', max_depth=None).fit(X_train, y_train)
    y_pred_svc = svc_model.predict(X_test)
    y_pred_knn = knn_model.predict(X_test)
    y_pred_dtc = dtc_model.predict(X_test)
    svm_f1 = metrics.f1_score(y_test, y_pred_svc, average='macro')
    knn_f1 = metrics.f1_score(y_test, y_pred_knn, average='macro')
    dtc_f1 = metrics.f1_score(y_test, y_pred_dtc, average='macro')
    return (svm_f1 + knn_f1 + dtc_f1) / 3
