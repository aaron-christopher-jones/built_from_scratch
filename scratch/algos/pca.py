import numpy as np


class PCA:
    def __init__(self):
        pass

    @staticmethod
    def normalizing(v):
        return (v - np.mean(v)) / np.std(v)

    def fit(self, X):
        # step 1: normalizing
        Xarray = X.to_numpy()
        self.Xscale = np.apply_along_axis(self.normalizing, 0, Xarray)

        # step 2: compute covariances
        Xcov = np.cov(self.Xscale.T)

        # step 3: compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(Xcov)
        eigenvectors = eigenvectors.T

        # step 4: construct feature vector
        idx = np.flip(np.argsort(eigenvalues))
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[idx]

    def transform(self):
        return np.dot(self.Xscale, self.eigenvectors.T)
