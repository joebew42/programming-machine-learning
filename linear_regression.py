import numpy as np


def predict(X, w):
    return X * w


def loss(X, Y, w):
    return np.average((predict(X, w) - Y)**2)


def train(X, Y, iterations, learning_rate):
    w = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + learning_rate) < current_loss:
            w += learning_rate
        elif loss(X, Y, w - learning_rate) < current_loss:
            w -= learning_rate
        else:
            return w

    raise Exception("Couldn't converge within %d iterations" % iterations)


if __name__ == "__main__":
    X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

    w = train(X, Y, 1000, 0.01)
    print("w=%.3f" % w)

    for x, y in zip(X, Y):
        print("Reservations: %4d, Pizzas: %4d, Predicted Pizzas: %4d" %
              (x, y, predict(x, w)))
