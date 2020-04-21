import numpy as np
import copy


class CSOGDClassifier:
    def __init__(self, eta, class_weight, loss_func):
        self.eta = eta
        self.class_weight = copy.deepcopy(class_weight)
        self.rho = class_weight[1] / class_weight[-1]
        self.loss_func = loss_func
        self.w = None

    def partial_fit(self, X, y):
        # X = np.hstack((X, np.ones(shape=(len(X), 1))))
        if self.w is None:
            self.w = np.zeros(X.shape[1])
        hat_y = np.dot(self.w, X.T)
        for i in range(len(y)):
            if self.loss_func == 'I':
                loss = max(0, (self.rho * (y[i] == 1) + (y[i] == -1)) - y[i] * hat_y[i])
                if loss > 0:
                    self.w += self.eta * y[i] * X[i]
            elif self.loss_func == 'II':
                loss = (self.rho * (y[i] == 1) + (y[i] == -1)) * max(0, 1 - y[i] * hat_y[i])
                if loss > 0:
                    self.w += self.eta * (self.rho * (y[i] == 1) + (y[i] == -1)) * y[i] * X[i]
        return self

    def predict(self, X):
        # X = np.hstack((X, np.ones(shape=(len(X), 1))))
        y = np.sign(np.dot(self.w, X.T)).astype(int)
        y[y == 0] = 1
        return y

    def __deepcopy__(self, memo={}):
        replica = CSOGDClassifier(eta=self.eta, class_weight=self.class_weight, loss_func=self.loss_func)
        replica.w = copy.deepcopy(self.w)
        return replica
