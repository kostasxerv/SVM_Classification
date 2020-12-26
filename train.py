from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datetime import datetime
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def train_svm(X_train, X_test, y_train, y_test, svm_params, n_components, with_pca=True):
  models=[]
  # WITH PCA
  if with_pca:
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

  chart_data = []
  for par in svm_params:
      model = SVC(**par)
      model, data = train({"name": "SVM", **par}, model, X_train, X_test, y_train, y_test)
      chart_data.append(data)
      models.append({
        "params": par,
        "model": model
      })
  
  return models, chart_data


def train_knn_nc(X_train, X_test, y_train, y_test, n_components, with_kpca=True, with_lda=True):
  models = []
  # Apply KPCA
  if with_kpca:
    kpca = KernelPCA(n_components=n_components, random_state=0)
    kpca.fit(X_train)
    X_train = kpca.transform(X_train)
    X_test = kpca.transform(X_test)

  # Apply LDA
  if with_lda:
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    X_train = lda.transform(X_train)
    X_test = lda.transform(X_test)

  params = [
      {"n_neighbors": 3},
      {"n_neighbors": 5},
      {"n_neighbors": 7},
      {"n_neighbors": 9},
  ]
  chart_data = []
  for par in params:
    model = KNeighborsClassifier(**par)
    model, data = train({"name": "KNN", **par}, model, X_train, X_test, y_train, y_test)
    chart_data.append(data)
    models.append({
      "params": {"name": "KNN", **par},
      "model": model
    })

# NearestCentroid
  model = NearestCentroid()
  model, data = train({"name": "NeCentroid"}, model, X_train, X_test, y_train, y_test)
  chart_data.append(data)
  models.append({
    "params": {"name": "NeCentroid"},
    "model": model
  })

  return models, chart_data

def train(par, model,  X_train, X_test, y_train, y_test):
  print(par)
  t1 = datetime.now()
  model.fit(X_train, y_train)
  train_time = datetime.now() - t1

  y_preds_train = model.predict(X_train)
  acc_train = accuracy_score(y_train, y_preds_train)
  f1_train = f1_score(y_train, y_preds_train, average="macro")

  t1 = datetime.now()
  y_preds = model.predict(X_test)
  pred_time = datetime.now() - t1

  acc_test = accuracy_score(y_test, y_preds)
  f1_test = f1_score(y_test, y_preds, average="macro")

  data = {
      "params": par,
      "mean_fit_time": train_time.total_seconds(),
      "mean_score_time": pred_time.total_seconds(),
      "f1_test_score": f1_test,
      "f1_train_score": f1_train,
      "acc_test_score": acc_test,
      "acc_train_score": acc_train,
      "y_preds": y_preds
  }

  return model, data