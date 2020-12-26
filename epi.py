import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from helpers import plot_results, plot_pca_n
from train import train_svm, train_knn_nc
import matplotlib.pyplot as plt

def main():
    # load dataset
    df = pd.read_csv("./datasets/epi/data.csv")
    
    # get features and target values
    X = df.drop(columns=['y', 'Unnamed: 0'])
    y = df['y'] # multiclass

    # Plot labels distribution
    # df['y'].value_counts().plot.bar()
    # plt.show()
    # return

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.4, random_state=0)

    # preprocess
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # plot_pca_n(X_train)
    # return
    PCA_COMPONENTS=50
    svm_params = [        
      {"kernel": "poly", "C":1, "gamma": 1, "degree":2},
      {"kernel": "linear", "C":1},
      {"kernel": "linear", "C":10},
      {"kernel": "linear", "C":0.1},
      {"kernel": "rbf", "C":1, "gamma": 0.01},
      {"kernel": "rbf", "C":100, "gamma": 0.01},
      {"kernel": "sigmoid", "C":1, "gamma":1},
    ]
    # svm_models, svm_data = train_svm(X_train, X_test, y_train, y_test, svm_params, PCA_COMPONENTS)
    # plot_results(svm_data, ["mean_fit_time", "mean_score_time", "f1_test_score","f1_train_score","acc_test_score","acc_train_score"])

    knn_nc_models, knn_nc_data = train_knn_nc(X_train, X_test, y_train, y_test, PCA_COMPONENTS)
    plot_results(knn_nc_data, ["mean_fit_time", "mean_score_time", "f1_test_score","f1_train_score","acc_test_score","acc_train_score"])


if __name__ == "__main__":
    main()