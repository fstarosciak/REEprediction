import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def direction_accuracy(y_true, y_pred):
    direction_true = np.sign(y_true.flatten())
    direction_pred = np.sign(y_pred.flatten())

    mask = direction_true != 0
    if mask.sum() == 0:
        return 0.0

    hits = (direction_true[mask] == direction_pred[mask]).sum()
    return float(hits / mask.sum() * 100.0)


def evaluate_model(model, X_test, y_test):
    y_pred = model.forward_propagation(X_test)

    results = {
        "mae":                mae(y_test, y_pred),
        "mse":                mse(y_test, y_pred),
        "rmse":               rmse(y_test, y_pred),
        "direction_accuracy": direction_accuracy(y_test, y_pred),
    }
    return results


if __name__ == "__main__":
    np.random.seed(0)
    y_true = np.array([1.0, -0.5, 0.8, -1.2, 0.3])
    y_pred = np.array([0.9, -0.4, 0.6, -1.0, -0.1])

    print("Test metrics.py")
    print(f"  MAE:  {mae(y_true, y_pred):.4f}")
    print(f"  MSE:  {mse(y_true, y_pred):.4f}")
    print(f"  RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"  Dir.Acc: {direction_accuracy(y_true, y_pred):.1f}%")
    print("  OK!")
