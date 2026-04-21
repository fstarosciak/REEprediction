def train_model(model, X, y, epochs=100, lr=0.01, verbose=False):
    loss_history = []

    for epoch in range(epochs):
        y_pred = model.forward_propagation(X)

        loss = ((y_pred - y) ** 2).mean()
        loss_history.append(float(loss))

        grads_W, grads_b = model.backward_propagation(y)
        model.update(grads_W, grads_b, lr)

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:>4d}/{epochs}  Loss: {loss:.6f}")

    return loss_history
