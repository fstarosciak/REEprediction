import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestModel:

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
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
            "model":        "RandomForest",
            "n_estimators": self.n_estimators,
            "max_depth":    self.max_depth,
            "random_state": self.random_state,
        }
