import numpy as np
import scipy.spatial.distance as sp_dist

from scratch.abstract import AbstractModel


class KMeans(AbstractModel):
    def __init__(self, n_clusts):
        self.n_clusts = n_clusts

    def fit(self, X):
        n = X.shape[0]

        # initialize model convergence lists
        # these lists capture the labels and centers for each iteration of the model training
        trajectory_labels = []
        trajectory_centers = []

        it = 0
        while True:
            # if iteration number is zero, then randomly select labels
            # otherwise pull previous label assignments
            if it == 0:
                plabels = np.random.choice(range(self.n_clusts), n)
                trajectory_labels.append(plabels)
            else:
                trajectory_labels.append(labels)
                plabels = trajectory_labels[it]
            
            # calculate mean values within column and grouped by labels
            centers = np.array([X[plabels == i].mean(axis=0) for i in range(self.n_clusts)])
            trajectory_centers.append(centers)

            # calculate difference between data points and centroids
            dists = sp_dist.cdist(X, centers)

            # calculate minimum distance within row
            labels = np.argmin(dists, axis=1)

            # if labels remain the same, then calculate centers and break
            # otherwise save labels and centers and increase iteration number
            if (plabels == labels).all():
                break
            else:
                it += 1

        self.centers = centers
        self.labels = labels

        self.trajectory_centers = trajectory_centers
        self.trajectory_labels = trajectory_labels

    def predict(self, X):
        # calculate difference between new dataset and centroids
        dists = sp_dist.cdist(X, self.centers)

        # calculate labels as centroids with minimum distance
        labels = np.argmin(dists, axis=1)

        # output label assignments
        return labels
