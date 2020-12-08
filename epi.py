import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from helpers import plot_results, show_cm
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

def epi():
    # load dataset
    df = pd.read_csv("./datasets/epi/data.csv")
    
    # get features and target values
    X = df.drop(columns=['y', 'Unnamed: 0'])
    # y = df['y'] # multiclass
    y = df['y'].replace({2:0, 3:0, 4:0, 5:0}) # binary class

    # calculate the most correlated features with the output
    number_of_features = 10
    threshold = sorted(X.corrwith(y),reverse=True)[number_of_features]
    print(f'Correlation threshold: {threshold}')
    most_corr_cols = X.columns[X.corrwith(y) > threshold]
    print('Number of Corr Columns: ' + str(len(most_corr_cols)))
    if(len(most_corr_cols) ==0):
        return
    
    X = X[most_corr_cols].values
    y = y.values

    # preprocess
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.4, random_state=0)

    parameters = [
        {'kernel':['rbf'], 'C':[ 0.01, 0.1, 1, 10], 'gamma': [0.01, 1, 10, 100]},
        {'kernel': ['linear'], 'C':[0.01, 0.1, 1, 10]}
    ]
    clf = GridSearchCV(SVC(), parameters, scoring='f1_macro', return_train_score=True)

    # clf = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3,5,7,9,11]}, return_train_score=True)

    # clf = GridSearchCV(NearestCentroid(), {}, return_train_score=True)

    clf.fit(X_train, y_train)

    print("BEST PARAMS COMBINATION: ", clf.best_params_)
    y_pred = clf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'F1 Score: {f1_score(y_test, y_pred,average="macro")}')
    # confusion matrix
    show_cm(y_test, y_pred)
    
    # plot times and score charts
    plot_results(clf.cv_results_)


if __name__ == "__main__":
    epi()