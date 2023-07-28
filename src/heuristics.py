import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin

class TopKSelector():
    """
    Generates prediction sets by including the top-K classes.

    Args:
        * estimator (sklearn.Classifier): an sklearn classifier
        * k (int): the number of classes to return
    """
    def __init__(self,
                 estimator: ClassifierMixin,
                 k: int):
        self.estimator = estimator
        self.k = k
    
    def fit(self):
        # does nothing
        return self
    
    def predict(self, X: ArrayLike):
        """
        Returns prediction sets by including the top-K classes.

        Args:
            X (pandas.DataFrame, numpy.array): the input to create prediction sets for
        """
        probas = self.estimator.predict_proba(X)
        # get indices of top-k scores
        top_k = np.argpartition(probas, -self.k, axis=1)[:, -self.k:]
        # allocate multi-hot index
        multihot = np.zeros(probas.shape)
        # create index to set values in multi-hot encoding
        I = np.arange(top_k.shape[0])[:, np.newaxis]
        # set value of top-k to 1
        multihot[I, top_k] = 1
        # cast to boolean for compatibility with MAPIE
        return multihot.astype(bool)

class HeuristicCutoffSelector():
    """
    Generates prediction sets by including all classes with a score above (exclusive) a user-defined ``cutoff``.
    
    Args:
        * estimator(sklearn.Classifier): an sklearn classifier
        * cutoff (float): all classes with a prediction above this value are included in the prediction set
        * always_predict (bool, optional): whether to always include the top-1 class, even if the predicted value
            for its class does not meet the cutoff. Defaults to True.
    """
    def __init__(self,
                 estimator: ClassifierMixin,
                 cutoff: float,
                 always_predict: bool = True):
        self.estimator = estimator
        self.cutoff = cutoff
        self.always_predict = always_predict
    
    def fit(self, *args, **kwargs):
        # do nothing
        return self
    
    def predict(self, X: ArrayLike):
        """
        Returns prediction sets by including all classes with a prediction > cutoff.

        Args:
            X (pandas.DataFrame, numpy.array): the input to create prediction sets for
        """
        probas = self.estimator.predict_proba(X)
        cutoff_selection = probas > self.cutoff
        if self.always_predict:
            # get index rgument of highest class
            top1 = probas.argmax(axis=1)[...,None]
            # allocate one-hot index
            onehot = np.zeros(probas.shape)
            # create index to set values in one-hot encoding
            I = np.arange(top1.shape[0])[:,np.newaxis]
            # set value of top-1 to 1
            onehot[I, top1] = 1
            # cast to boolean
            return cutoff_selection | onehot.astype(bool)
        else:
            return cutoff_selection

class TopKCutoffSelector():
    """
    Generates prediction sets by including all classes with a score above (exclusive) a user-defined ``cutoff`` or includes
    the top-K if no predictions meet this requirement.
    
    Args:
        * estimator(sklearn.Classifier): an sklearn classifier
        * cutoff (float): all classes with a prediction above this value are included in the prediction set
        * k (int): the top-K classes to include in the prediction set when no classes meet the cutoff.
    """
    def __init__(self,
                 estimator: ClassifierMixin,
                 cutoff: float,
                 k: int):
        self.estimator = estimator
        self.cutoff = cutoff
        self.k = k
    
    def fit(self, *args, **kwargs):
        # do nothing
        return self
    
    def predict(self, X):
        probas = self.estimator.predict_proba(X)
        cutoff_selection = probas > self.cutoff
        for i in range(X.shape[0]):
            if cutoff_selection[i].any():
                pass
            else:
                # get index of k highest class
                top_k = np.argpartition(probas[i,:], -self.k)[-self.k:]
                # cast to boolean for compatibility with MAPIE
                cutoff_selection[i, top_k] = True
        return cutoff_selection
