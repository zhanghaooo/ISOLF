import numpy as np
import copy


class OA3Classifier:
    def __init__(self, eta, class_weight, query_weight, gamma):
        self.eta = eta
        self.class_weight = copy.deepcopy(class_weight)
        self.query_weight = copy.deepcopy(query_weight)
        self.gamma = gamma
        self.rho = class_weight[1] / class_weight[-1]
        self.coef_ = None
        self.cov = None

    def query(self, X):
        X = np.hstack((X, np.ones(shape=(len(X), 1))))
        p = np.dot(self.coef_, X.T)
        hat_y = np.sign(p)
        hat_y[hat_y == 0] = 1
        label_requests = []
        for i in range(len(X)):
            v = X[i].dot(self.cov).dot(X[i].T)
            q = np.abs(p[i]) - (self.eta * v * self.gamma * max(self.rho, 1)) / (2 * v + 2 * self.gamma)
            if q < 0:
                label_requests.append(1)
            else:
                pr = self.query_weight[hat_y[i]] / (self.query_weight[hat_y[i]] + q)
                label_requests.append(np.random.binomial(1, pr))
        return np.array(label_requests).astype(bool)

    def partial_fit(self, X, y):
        X = np.hstack((X, np.ones(shape=(len(X), 1))))
        if self.coef_ is None:
            self.coef_ = np.zeros(X.shape[1])
            self.cov = np.eye(X.shape[1])
        hat_y = np.dot(self.coef_, X.T)
        for i in range(len(y)):
            if hat_y[i] * y[i] < 1:
                self.cov -= (self.cov.dot(X[i].reshape(-1, 1)).dot(X[i].reshape(1, -1)).dot(self.cov)) / \
                                   (self.gamma + X[i].reshape(1, -1).dot(self.cov).dot(X[i].reshape(-1, 1)))
                if y[i] == 1:
                    self.coef_ += self.rho * self.eta * y[i] * X[i].dot(self.cov)
                else:
                    self.coef_ += self.eta * y[i] * X[i].dot(self.cov)
        return self

    def predict(self, X):
        X = np.hstack((X, np.ones(shape=(len(X), 1))))
        if self.coef_ is None:
            self.coef_ = np.zeros(X.shape[1])
            self.cov = np.eye(X.shape[1])
        y = np.sign(np.dot(self.coef_, X.T)).astype(int)
        y[y == 0] = 1
        return y

    def __deepcopy__(self, memo={}):
        replica = OA3Classifier(eta=self.eta, class_weight=self.class_weight, query_weight=self.query_weight,
                                gamma=self.gamma)
        replica.coef_ = copy.deepcopy(self.coef_)
        replica.cov = copy.deepcopy(self.cov)
        return replica


