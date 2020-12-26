import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from helpers import title, plot_gallery, plot_results, plot_pca_n
from train import train_svm, train_knn_nc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def main():
    df = pd.read_csv("./datasets/mnist/train.csv")
    IMG_WIDTH = 28
    IMG_HEIGHT = 28

    y = df['label']
    X = df.drop('label', axis=1)
   
    # Plot labels distribution
    # df['label'].value_counts().plot.bar()
    # plt.show()
    # return

    X_train, X_test, y_train, y_test =  train_test_split(X, y,test_size=0.4,random_state=0)
   
    # preprocess
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # plot_pca_n(X_train) # plot the chart of n component compared to Cumulative explained variance
    PCA_COMPONENTS = 300
    svm_params = [        
      {"kernel": "poly", "C":1, "degree":2},
      {"kernel": "poly", "C":1, "degree":3},
      {"kernel": "linear", "C":1},
      {"kernel": "linear", "C":10},
      {"kernel": "rbf", "C":0.001, "gamma": 1000},
      {"kernel": "rbf", "C":1, "gamma": 0.01},
      {"kernel": "rbf", "C":100, "gamma": 0.01},
      {"kernel": "sigmoid", "C":1, "gamma":1}
    ]
    
    svm_models, svm_data = train_svm(X_train, X_test, y_train, y_test, svm_params, PCA_COMPONENTS)
    plot_results(svm_data, ["mean_fit_time", "mean_score_time", "f1_test_score","f1_train_score","acc_test_score","acc_train_score"])

    knn_nc,_models, knn_nc_data = train_knn_nc(X_train, X_test, y_train, y_test, PCA_COMPONENTS)
    plot_results(knn_nc_data, ["mean_fit_time", "mean_score_time", "f1_test_score","f1_train_score","acc_test_score","acc_train_score"])

    # plot correct and false prediction examples
    y_preds = svm_data[0]['y_preds']
    y_true = y_test.values
   
    correct = [i for i in range(len(y_true)) if y_preds[i] == y_true[i]][:12]
    wrong = [i for i in range(len(y_true)) if y_preds[i] != y_true[i]][:12]
    
    correct_images = [X_test[i] for i in correct]
    wrong_images = [X_test[i] for i in wrong]

    prediction_titles = ['predicted: %s\ntrue: %s' % (y_preds[i], y_true[i]) for i in correct]
    plot_gallery(correct_images, prediction_titles, IMG_WIDTH, IMG_HEIGHT, 3, 4)

    prediction_titles = ['predicted: %s\ntrue: %s' % (y_preds[i], y_true[i]) for i in wrong]
    plot_gallery(wrong_images, prediction_titles, IMG_WIDTH, IMG_HEIGHT, 3, 4)

if __name__ == "__main__":
    main()
     