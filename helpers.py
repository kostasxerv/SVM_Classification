import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from skimage.color import rgb2gray

# show time and scores charts
def plot_results(results):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    df = pd.DataFrame(results)
    df['name'] = df['params'].apply(lambda x: '-'.join([str(x[k]) for k in x]))

    df.sort_values(by='mean_fit_time')\
        .plot.barh(x='name', y=["mean_fit_time",],  figsize=(10,10), title='Mean fit time', legend=False, ax=axes[0,0])

    df.sort_values(by='mean_score_time')\
        .plot.barh(x='name', y=["mean_score_time"], figsize=(10,10), title='Mean score time', legend=False, ax=axes[0,1])

    df.sort_values(by='mean_test_score')\
        .plot.barh(x='name', y=["mean_test_score"], figsize=(10,10), title='Mean test score', legend=False, ax=axes[1,0])

    df.sort_values(by='mean_train_score')\
        .plot.barh(x='name', y=["mean_train_score"], figsize=(10,10), title='Mean train score', legend=False, ax=axes[1,1])
    
    plt.show()

# show confusion matrix results
def show_cm(y_test, y_pred):
    mat = confusion_matrix(y_test, y_pred)
    # get TP, FP, FN, TN from confusion matrix
    TN = mat[0][0]
    FN = mat[1][0]
    TP = mat[1][1]
    FP = mat[0][1]
    print(f"True Positive:{TP}, True Negative:{TN}, False Positive:{FP}, False Negative:{FN}")
    return TP, FP, TN, FN

# plot images as grid
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

    plt.show()

# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

# get black and white image
def get_grey_img(img_path):
    return rgb2gray(cv2.imread(img_path))

# plot pca components and explained variance
def plot_pca_n(pca, data):
  perc_var_expl = pca.explained_variance_ / np.sum(pca.explained_variance_)
  
  cum_var_expl = np.cumsum(perc_var_expl)

  plt.clf()
  plt.plot(cum_var_expl, linewidth=2)
  plt.axis('tight')
  plt.grid()
  plt.xlabel('n_componets')
  plt.ylabel('Cumulative explained variance')
  plt.show()