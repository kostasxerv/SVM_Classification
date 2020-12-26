import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA

# show time and scores charts
DEFAULT_COLS = ["mean_fit_time", "mean_score_time", "mean_train_score", "mean_test_score"]
def plot_results(results, columns=DEFAULT_COLS, ncols=2, limit=10):
    fig, axes = plt.subplots(nrows=math.ceil(len(columns) / ncols), ncols=ncols)
    df = pd.DataFrame(results)
    df['name'] = df['params'].apply(lambda x: '-'.join([str(x[k]) for k in x]))

    for col in columns:
        i = columns.index(col)
        ax1, ax0 = (i % ncols), int(i/ncols)
        df.sort_values(by=col)[:limit].plot.barh(x='name', y=[col],  figsize=(10,10), title=col, legend=False, ax=axes[ax0,ax1])

    plt.tight_layout()
    plt.show()


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
    plt.tight_layout()
    plt.show()

# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    print(target_names, y_pred, y_test, i)
    pred_name = target_names[y_pred[i]]
    true_name = target_names[y_test[i]]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


# plot pca components and explained variance
def plot_pca_n(X):
    pca = PCA()
    pca.fit(X)
    perc_var_expl = pca.explained_variance_ / np.sum(pca.explained_variance_)
    cum_var_expl = np.cumsum(perc_var_expl)

    plt.clf()
    plt.plot(cum_var_expl, linewidth=2)
    plt.axis('tight')
    plt.grid()
    plt.xlabel('n_componets')
    plt.ylabel('Cumulative explained variance')
    plt.show()