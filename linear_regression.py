import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import seaborn as sns

matplotlib.use("Qt5Agg")


def predict(X, w, b):
    return X * w + b


def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y)**2)


def train(X, Y, iterations, learning_rate):
    w = b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + learning_rate, b) < current_loss:
            w += learning_rate
        elif loss(X, Y, w - learning_rate, b) < current_loss:
            w -= learning_rate
        elif loss(X, Y, w, b + learning_rate) < current_loss:
            b += learning_rate
        elif loss(X, Y, w, b - learning_rate) < current_loss:
            b -= learning_rate
        else:
            return w, b

    raise Exception("Couldn't converge within %d iterations" % iterations)


if __name__ == "__main__":
    X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

    w, b = train(X, Y, 10000, 0.01)
    print("w=%.3f, b=%.3f" % (w, b))

    YPREDICT = []
    for x, y in zip(X, Y):
        prediction = predict(x, w, b)
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
