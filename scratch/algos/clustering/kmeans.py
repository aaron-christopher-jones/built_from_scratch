import numpy as np
import scipy.spatial.distance as sp_dist


class KMeans:
    """
    
    """

    def __init__(self, num_clusts):
        """
        """
        self.num_clusts = num_clusts

    def fit(self, Xtrain):
        """
        """
        list_labels = []
        list_centers = []

        it = 0
        while True:
            # if iteration number is zero, then random select labels
            # otherwise pull previous label assignments
            if it == 0:
                pre_labels = np.random.choice(range(self.num_clusts), Xtrain.shape[0])
                list_labels.append(pre_labels)
            else:
                pre_labels = list_labels[it]

            # append updated centroids to centers
            # calculate mean values within column and grouped by labels
            centers = np.array([
                Xtrain[pre_labels == i].mean(axis=0) 
                for i in range(self.num_clusts)
            ])
            list_centers.append(centers)

            # calculate difference between data points and centroids
            dists = sp_dist.cdist(Xtrain, centers)

            # append updated cluster assignments to labels
            # calculate minimum distance within row
            labels = np.argmin(dists, axis=1)
            list_labels.append(labels)

            # if labels remain the same, then calculate centers and break
            # otherwise increase iteration number
            if (pre_labels == labels).all():
                centers = np.array([
                    Xtrain[labels == i].mean(axis=0) 
                    for i in range(self.num_clusts)
                ])
                list_centers.append(centers)
                break
            else:
                it += 1

        self.centers = centers
        self.labels = labels
        self.list_centers = list_centers
        self.list_labels = list_labels

    def predict(self, Xvalid):
        """
        """
        # calculate difference between data points and centroids
        dists = sp_dist.cdist(Xvalid, self.centers)

        # append updated cluster assignments to labels
        # calculate minimum distance within row
        labels = np.argmin(dists, axis=1)

        return labels
