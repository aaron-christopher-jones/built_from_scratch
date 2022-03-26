from abc import ABC, abstractmethod

class AbstractModel(ABC):
 
    def fit(self):
        pass

    def predict(self):
        pass
