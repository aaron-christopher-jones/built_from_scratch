import numpy as np
import scipy.spatial.distance as sp_dist

from scratch.abstract import AbstractModel


class KNearestNeighbors(AbstractModel):
    """
    Despite virtually all statistical / machine learning algorithms being 
    available as easily callable functions from open source libraries, I 
    believe a strong working knowledge of the algorithms is still imperative. 
    It has been a hobby of mine to code these algorithms from scratch to 
    confirm and expand my working knowledge. I am going to start sharing some 
    of this work in a series of blog posts. This first post will be on k-nearest 
    neighbors.

    K-nearest neighbors is an algorithm that can be used for both classification 
    and regression, although it is probably most commonly used for classification. 
    The flow of the algorithm is nearly identical for continuous and categorical 
    dependent variables. The only difference is how the nearest neighbors are 
    combined to produce the final prediction.

    The process of k-nearest neighbors is:

    1. Compute the distance between every observation in the training dataset and 
    every observation in the prediction dataset. The distance metric used in my 
    version below is the Minkowski distance of order p. Minkowski distance of order 
    p is $D(X, Y) = \sqrt(\sum_{i=1}^{n}(x_{i} - y_{i})^{2})$ where each *i* is a 
    different feature of the observation vector.
    2. Select k observations from the training dataset for each observation in the 
    prediction dataset. The k hyperparameter is either set in advance by the user or 
    optimized through some search procedure. In this case, the k value is set to 5. 
    The k observations are those with the minimum distances.
    3. Aggregate the k observations to produce the final predicted value for the 
    observation vector in the prediction dataset. In the case of this classification 
    example, make the prediction the category value that occurs most often in the k 
    observations of the training dataset. For a regression problem, the output value 
    from the k observations might instead be the mean average.

    Below is the coded from scratch version of the k-nearest neighbors algorithm and 
    the results of running the algorithm on the breast cancer dataset.
    """
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.Xtrain = X
        self.ytrain = y
        
    def predict(self, X):
        n = X.shape[0]
        dists = sp_dist.cdist(X, self.Xtrain, "minkowski", p=2)
        min_idx = [dists[i, :].argsort()[:self.n_neighbors] for i in range(n)]
        yhat = np.array([np.argmax(np.bincount(self.ytrain[i])) for i in min_idx])
        return yhat
