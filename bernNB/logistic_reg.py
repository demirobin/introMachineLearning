import numpy as np
import math
from numpy import linalg as LA


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


class LogisticRegression:
    def __init__(self, training):
        self.dataset = training
        self.X = training[:, :-1]
        self.y = training[:, -1]
        self.feature = {}  # w coefficients

    def gradient_ascent(self, w):
        gradient = np.zeros(len(self.X[0]))
        for i in range(len(self.y)):
            xi = np.array(self.X[i])
            wx = np.dot(w, self.X[i])
            sig = sigmoid(wx)
            scalar = self.y[i] - sig
            gradient = np.add(gradient, xi * scalar)
        return gradient

    # We suggest that you implement a fit(X, y, max_iters=5000, lr=0.01, tol=0.0005) function
    # for your logistic regression class, which takes the training data as input
    # and sets the model parameter attributes as a side effect.
    # You should use gradient descent to learn the parameters
    # and we suggest using the default hyperparameters noted above
    # for the maximum number of gradient descent iterations (max_iters),
    # the learning rate (lr), and the tolerance epsilon for stopping (tol).
    def fit(self, max_iter, lr, tol):
        w = np.zeros(len(self.X[0]))  # w vector
        k = 0
        miss = 1
        while k < max_iter and miss == 1:
            # print(k)
            miss = 0
            grad = np.array(self.gradient_ascent(w))
            w = np.add(w, grad * lr)
            epsilon = np.array(grad * lr)
            if LA.norm(epsilon) < tol:
                break
            k += 1
            miss = 1
        for i in range(0, len(self.X[0])):
            self.feature[i + 1] = w[i]
        self.feature[0] = 1  # bias term
        self.feature["w"] = w
        print("fit done")

    # We suggest you implement a predict(X) function,
    # which takes an matrix containing n training examples and outputs n predictions for the estimated ˆy values,
    # using the currently stored model parameters.
    # You should logistic function with a threshold of 0.5 to implement this function.
    def predict(self, matrix):
        w = self.feature["w"]
        bias = self.feature[0]
        f_w = np.zeros(len(matrix))
        for i in range(0, len(matrix)):
            x = np.array(matrix[i])
            if np.dot(w, x) + bias > 0.5:
                f_w[i] = 1
            else:
                f_w[i] = 0
        return f_w

    def save_parameters(self):
        weights = np.append(self.feature[0], self.feature["w"])
        np.savetxt("weights.tsv", weights, delimiter="\\t")
        print("weights saved")


if __name__ == '__main__':
    data = np.loadtxt(".//train_dataset.tsv", delimiter="\t")
    model = LogisticRegression(data)
    model.fit(5000, 0.01, 0.0005)
    model.save_parameters()
