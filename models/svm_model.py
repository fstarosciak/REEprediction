import numpy as np
from sklearn.svm import SVR


class SVMModel:

    def __init__(self, kernel="rbf", C=1.0, epsilon=0.1):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon

        self._model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
        )

    def train(self, X_train, y_train):
        y_flat = y_train.ravel()
        self._model.fit(X_train, y_flat)

    def predict(self, X_test):
        return self._model.predict(X_test).reshape(-1, 1)

    def forward_propagation(self, X):
        return self.predict(X)

    def get_params(self):
        return {
            "model":   "SVM",
            "kernel":  self.kernel,
            "C":       self.C,
            "epsilon": self.epsilon,
        }
