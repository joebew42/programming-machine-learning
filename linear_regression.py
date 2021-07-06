import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import seaborn as sns

matplotlib.use("Qt5Agg")


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

    YPREDICT = []
    for x, y in zip(X, Y):
        YPREDICT.append(predict(x, w))
        print("Reservations: %4d, Pizzas: %4d, Predicted Pizzas: %4d" %
              (x, y, predict(x, w)))

    sns.set()
    plot.axis([0, 50, 0, 50])
    plot.xticks(fontsize=15)
    plot.yticks(fontsize=15)
    plot.xlabel("Reservations", fontsize=30)
    plot.ylabel("Pizzas", fontsize=30)
    plot.plot(X, Y, "bo")
    plot.plot(X, YPREDICT, "ro", linestyle='-')
    plot.show()
