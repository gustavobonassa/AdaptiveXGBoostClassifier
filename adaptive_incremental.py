import numpy as np

import xgboost as xgb

from skmultiflow.core.base import BaseSKMObject, ClassifierMixin
from skmultiflow.drift_detection import ADWIN
from skmultiflow.utils import get_dimensions
import threading
import concurrent.futures
import multiprocessing
from pathos.multiprocessing import ProcessPool as Pool
import time

def _thread(x, y):
    # print(args["ensemble"])
    print(args)
    # margins = args["ensemble"].predict(args["data"], output_margin=True)
    # return margins
    return x+y

def _teste(ensemble, data):
    pool = Pool(8)
    results = pool.map(pow, [1,2,3,4], [5,6,7,8])
    print(results)

class Adaptive2(BaseSKMObject, ClassifierMixin):

    def __init__(self,
                 learning_rate=0.3,
                 max_depth=6,
                 max_window_size=1000,
                 min_window_size=None):
        """
        Adaptive XGBoost classifier.

        Parameters
        ----------

        learning_rate:
            Learning rate, a.k.a eta.

        max_depth: int (default = 6)
            Max tree depth.

        max_window_size: int (default=1000)
            Max window size.

        min_window_size: int (default=None)
            Min window size. If this parameters is not set, then a fixed size
            window of size ``max_window_size`` will be used.

        update_strategy: str (default='replace')
            | The update strategy to use:
            | 'push' - the ensemble resembles a queue
            | 'replace' - oldest ensemble members are replaced by newer ones

        Notes
        -----
        The Adaptive XGBoost [1]_ (AXGB) classifier is an adaptation of the
        XGBoost algorithm for evolving data streams. AXGB creates new members
        of the ensemble from mini-batches of data as new data becomes
        available.  The maximum ensemble  size is fixed, but learning does not
        stop once this size is reached, the ensemble is updated on new data to
        ensure consistency with the current data distribution.

        References
        ----------
        .. [1] Montiel, Jacob, Mitchell, Rory, Frank, Eibe, Pfahringer,
           Bernhard, Abdessalem, Talel, and Bifet, Albert. “AdaptiveXGBoost for
           Evolving Data Streams”. In:IJCNN’20. International Joint Conference
           on Neural Networks. 2020. Forthcoming.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self._first_run = True
        self._ensemble = None
        self._drift_detector = None
        self._X_buffer = np.array([])
        self._y_buffer = np.array([])
        self._samples_seen = 0
        self._model_idx = 0

        self._configure()

    def _configure(self):
        self._reset_window_size()
        self._init_margin = 0.0
        self._boosting_params = {"silent": True,
                                 "objective": "binary:logistic",
                                 "eta": self.learning_rate,
                                 "max_depth": self.max_depth}

    def reset(self):
        """
        Reset the estimator.
        """
        self._first_run = True
        self._configure()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Partially (incrementally) fit the model.

        Parameters
        ----------
        X: numpy.ndarray
            An array of shape (n_samples, n_features) with the data upon which
            the algorithm will create its model.

        y: Array-like
            An array of shape (, n_samples) containing the classification
            targets for all samples in X. Only binary data is supported.

        classes: Not used.

        sample_weight: Not used.

        Returns
        -------
        AdaptiveXGBoostClassifier
            self
        """
        for i in range(X.shape[0]):
            self._partial_fit(np.array([X[i, :]]), np.array([y[i]]))
        return self

    def _partial_fit(self, X, y):
        if self._first_run:
            self._X_buffer = np.array([]).reshape(0, get_dimensions(X)[1])
            self._y_buffer = np.array([])
            self._first_run = False
        self._X_buffer = np.concatenate((self._X_buffer, X))
        self._y_buffer = np.concatenate((self._y_buffer, y))
        # print("---- DEBBUGER ----")
        # print("--- _X_buffer: %s ---" % len(self._X_buffer))
        # print("--- _y_buffer: %s ---" % len(self._y_buffer))
        # print("--- window size: %s ---" % self.window_size)
        while self._X_buffer.shape[0] >= self.window_size:
            self._train_on_mini_batch(X=self._X_buffer[0:self.window_size, :],
                                      y=self._y_buffer[0:self.window_size])
            delete_idx = [i for i in range(self.window_size)]
            self._X_buffer = np.delete(self._X_buffer, delete_idx, axis=0)
            self._y_buffer = np.delete(self._y_buffer, delete_idx, axis=0)

            # Check window size and adjust it if necessary
            self._adjust_window_size()

    def _adjust_window_size(self):
        if self._dynamic_window_size < self.max_window_size:
            self._dynamic_window_size *= 2
            if self._dynamic_window_size > self.max_window_size:
                self.window_size = self.max_window_size
            else:
                self.window_size = self._dynamic_window_size

    def _reset_window_size(self):
        if self.min_window_size:
            self._dynamic_window_size = self.min_window_size
        else:
            self._dynamic_window_size = self.max_window_size
        self.window_size = self._dynamic_window_size

    def _train_on_mini_batch(self, X, y):
        booster = self._train_booster(X, y, self._model_idx)
        # Update ensemble
        self._ensemble = booster


    def _train_booster(self, X: np.ndarray, y: np.ndarray, last_model_idx: int):
        d_mini_batch_train = xgb.DMatrix(X, y.astype(int))
        # Get margins from trees in the ensemble
        margins = np.asarray([self._init_margin] * d_mini_batch_train.num_row())
        
        if self._ensemble:
            margins = self._ensemble.predict(d_mini_batch_train, output_margin=True)

        d_mini_batch_train.set_base_margin(margin=margins)
        booster = xgb.train(params=self._boosting_params,
                            dtrain=d_mini_batch_train,
                            num_boost_round=1,
                            verbose_eval=False)
        return booster

    def predict(self, X):
        """
        Predict the class label for sample X

        Parameters
        ----------
        X: numpy.ndarray
            An array of shape (n_samples, n_features) with the samples to
            predict the class label for.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (, n_samples), containing the
            predicted class labels for all instances in X.

        """
        # start_time = time.time()
        if self._ensemble:
            d_test = xgb.DMatrix(X)
            predicted = self._ensemble.predict(d_test)
            # print("--- %s seconds ---" % (time.time() - start_time))
            # print("--- %s tamanho ---" % len(X))
            return np.array(predicted > 0.5).astype(int)
        # Ensemble is empty, return default values (0)
        return np.zeros(get_dimensions(X)[0])

    def predict_proba(self, X):
        """
        Not implemented for this method.
        """
        raise NotImplementedError("predict_proba is not implemented for this method.")
