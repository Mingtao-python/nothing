import numpy as np

def run_gradient_demo():
    X = np.linspace(0, 10, 50)
    Y = 2*X + 1

    w = 0
    b = 0
    lr = 0.01

    losses = []

    for epoch in range(200):
        pred = w*X + b
        loss = np.mean((pred - Y)**2)
        losses.append(float(loss))

        dw = np.mean(2*(pred - Y)*X)
        db = np.mean(2*(pred - Y))

        w -= lr * dw
        b -= lr * db

    return losses
