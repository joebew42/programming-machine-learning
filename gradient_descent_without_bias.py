import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import seaborn as sns

matplotlib.use("Qt5Agg")


def predict(X, w, b):
    return X * w + b


def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y)**2)


def gradient(X, Y, w):
    return 2 * np.average(X * (predict(X, w, 0) - Y))


def train(X, Y, iterations, learning_rate):
    w = 0
    for i in range(iterations):
        print("Iteration %4d => Loss: %.6f" % (i, loss(X, Y, w, 0)))
        w -= gradient(X, Y, w) * learning_rate
    return w


if __name__ == "__main__":
    X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

    w = train(X, Y, 100, 0.001)
    print("w=%.3f" % w)

    YPREDICT = []
    for x, y in zip(X, Y):
        prediction = predict(x, w, 0)
        YPREDICT.append(prediction)
        print("Reservations: %4d, Pizzas: %4d, Predicted Pizzas: %4d" %
              (x, y, prediction))

    sns.set()
    plot.axis([0, 50, 0, 50])
    plot.xticks(fontsize=15)
    plot.yticks(fontsize=15)
    plot.xlabel("Reservations", fontsize=30)
    plot.ylabel("Pizzas", fontsize=30)
    plot.plot(X, Y, "bo")
    plot.plot(X, YPREDICT, "ro", linestyle='-')
    plot.show()
