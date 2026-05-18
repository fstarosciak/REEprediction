from sklearn.ensemble import HistGradientBoostingRegressor


class HistGradientBoostingModel:

    def __init__(
            self,
            learning_rate=0.05,
            max_depth=4,
            max_iter=300,
            random_state=42,
    ):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.random_state = random_state

        self._model = HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_iter=self.max_iter,
            random_state=self.random_state,
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
            "model": "HistGradientBoosting",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }