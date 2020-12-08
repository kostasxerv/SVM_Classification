import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from helpers import show_cm, title, plot_gallery, plot_results, get_grey_img, plot_pca_n
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

# subset of https://www.kaggle.com/omkargurav/face-mask-dataset
# choose images resolution of 225x225
IMG_WIDTH = 225
IMG_HEIGHT = 225

def load_dataset():
  base_path = 'datasets/masks/'

  names = ['without_mask', 'with_mask']

  X = []
  y = []
  # load images into array
  for cl in names: 
      for f in os.listdir(base_path + cl):
          img_path = base_path + cl + '/' + f
          y.append(names.index(cl))
          X.append(get_grey_img(img_path))
  X = np.array(X)
  y = np.array(y)

  # reshape to be compatible with sklearn
  nsamples, nx, ny = X.shape
  X = X.reshape((nsamples,nx*ny))

  return X, y, names


def olivetti():
    X, y, target_names = load_dataset()

    X_train, X_test, y_train, y_test =  train_test_split(X, y,test_size=0.4,random_state=0)

    pca = PCA(n_components=0.85)
    pca.fit(X)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # plot_pca_n(pca, X) # plot the chart of n component compared to Cumulative explained variance


    parameters = [
        {'kernel':['rbf'], 'C':[0.1, 1, 10, 100], 'gamma': [0.00001, 0.0001, 0.001, 0.005, 0.01, 1]},
        {'kernel': ['linear'], 'C':[0.005, 0.01, 0.1, 1, 10, 100]}
    ]
    clf = GridSearchCV(SVC(), parameters, scoring='f1_macro', return_train_score=True)

    # clf = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3,5,7,9,11]}, return_train_score=True)

    # clf = GridSearchCV(NearestCentroid(), {}, return_train_score=True)


    clf.fit(X_train_pca, y_train)

    print("BEST PARAMS COMBINATION: ", clf.best_params_)
    y_pred = clf.predict(X_test_pca)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'F1 score: {f1_score(y_test, y_pred, average="macro")}')
    # confusion matrix
    show_cm(y_test, y_pred)

    # plot correct and false prediction examples
    prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
    plot_gallery(X_test, prediction_titles, IMG_WIDTH, IMG_HEIGHT, 5, 5)

    # plot times and score charts
    plot_results(clf.cv_results_)


if __name__ == "__main__":
    olivetti()
     