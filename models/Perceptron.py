import numpy as np
import copy


class Perceptron:
    def __init__(self, class_weight={-1: 1, 1: 1}):
        self.class_weight = copy.deepcopy(class_weight)
        self.w = None
        self.b = None

    def partial_fit(self, X, y):
        if self.w is None:
            self.w = np.zeros(X.shape[1])
            self.b = 0
        hat_p = np.dot(self.w, X.T) + self.b
        for i in range(len(y)):
            if y[i] * hat_p[i] <= 0:
                self.w += self.class_weight[y[i]] * y[i] * X[i]
                self.b += self.class_weight[y[i]] * y[i]
        return self

    def predict(self, X):
        y = np.sign(np.dot(self.w, X.T) + self.b).astype(int)
        y[y == 0] = 1
        return y

    def __deepcopy__(self, memo={}):
        replica = Perceptron(class_weight=self.class_weight)
        replica.w = copy.deepcopy(self.w)
        replica.b = self.b
        return replica
