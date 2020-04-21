import numpy as np
import copy


class SGDClassifier:
    def __init__(self,
                 learning_rate='optimal',
                 eta0=0.1,
                 alpha=0.0001,
                 class_weight={-1: 1, 1: 1},
                 budget=1.0,
                 query_strategy='FCFQ',
                 query_ratio=1.0,  # for RQ
                 query_threshold=1.0,  # for DPDQ
                 query_beta=1,  # for RPDQ
                 query_weight={-1: 1, 1: 10},  # for AQ
                 ):

        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.alpha = alpha
        self.class_weight = copy.deepcopy(class_weight)
        self.budget = budget
        self.query_strategy = query_strategy
        self.query_ratio = query_ratio
        self.query_threshold = query_threshold
        self.query_weight = copy.deepcopy(query_weight)
        self.query_beta = query_beta
        self.t_ = 0
        self.coef_ = None
        self.intercept_ = None
        self.trained_number = 0
        self.query_number = 0
        if query_strategy == 'AQ':
            self.K = 0
            self.max_X = 0
        elif query_strategy == 'BAAQ':
            self.K = {-1: 0, 1: 0}
            self.max_X = {-1: 0, 1: 0}
        else:
            self.K = None
            self.max_X = None

    def partial_fit(self, X, y):
        if self.coef_ is None:
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 0
        for i in range(len(X)):
            hat_p = np.sum(self.coef_ * X[i]) + self.intercept_
            hat_y = np.sign(hat_p)
            if hat_y == 0:
                hat_y = 1
            if self.learning_rate == 'optimal':
                eta = 1 / (self.alpha * (self.t_ + 1000))
            elif self.learning_rate == 'constant':
                eta = self.eta0
            else:
                raise ValueError("invaluable learning rate")
            if y[i] * hat_p <= 1:
                self.coef_ += self.class_weight[y[i]] * eta * y[i] * X[i] - eta * self.alpha * self.coef_
                self.intercept_ += eta * y[i]
                if self.query_strategy == 'AQ':
                    self.K += 1
                    self.max_X = max(self.max_X, sum(X[i] ** 2))
                elif self.query_strategy == 'BAAQ':
                    self.K[hat_y] += 1
                    self.max_X[hat_y] = max(self.max_X[hat_y], sum(X[i] ** 2))
            else:
                self.coef_ -= eta * self.alpha * self.coef_
            self.t_ += 1
        return self

    def predict(self, X):
        y = np.sign(np.dot(self.coef_, X.T) + self.intercept_).astype(int)
        y[y == 0] = 1
        self.trained_number += len(X)
        return y

    def query(self, X):
        if self.query_strategy == 'FCFQ':
            label_requests = np.ones(len(X)).astype(bool)
            self.query_number += len(X)
        elif self.query_strategy == 'RQ':
            label_requests = np.random.binomial(1, self.budget, len(X)).astype(bool)
            self.query_number += label_requests.sum()
        elif self.query_strategy == 'DPDQ':
            hat_p = np.dot(self.coef_, X.T) + self.intercept_
            label_requests = np.abs(hat_p) < self.query_threshold
            self.query_number += label_requests.sum()
        elif self.query_strategy == 'RPDQ':
            hat_p = np.dot(self.coef_, X.T) + self.intercept_
            label_requests = np.random.binomial(1, self.query_weight / (self.query_weight + np.abs(hat_p))).astype(bool)
            self.query_number += label_requests.sum()
        elif self.query_strategy == 'AQ':
            hat_p = np.dot(self.coef_, X.T) + self.intercept_
            hat_y = np.sign(hat_p)
            hat_y[hat_y == 0] = 1
            label_requests = []
            for i in range(len(X)):
                pr = self.query_weight[hat_y[i]] / (self.query_weight[hat_y[i]] + np.abs(hat_p[i]))
                label_requests.append(np.random.binomial(1, pr))
                if label_requests[-1] == 1:
                    self.query_number += 1
            label_requests = np.array(label_requests).astype(bool)
        elif self.query_strategy == 'BAAQ':
            hat_p = np.dot(self.coef_, X.T) + self.intercept_
            hat_y = np.sign(hat_p)
            hat_y[hat_y == 0] = 1
            label_requests = []
            for i in range(len(X)):
                self.query_beta = self.budget * 2 / (1 + np.exp(self.query_number / self.trained_number - self.budget))
                query_weight = self.query_beta * max(self.max_X[hat_y[i]], sum(X[i] ** 2)) * \
                               np.sqrt(1 + self.K[hat_y[i]]) / (np.sqrt(1 + self.K[-1]) + np.sqrt(1 + self.K[1]))
                pr = query_weight / (query_weight + np.abs(hat_p[i]))
                label_requests.append(np.random.binomial(1, pr))
                if label_requests[-1] == 1:
                    self.query_number += 1
            label_requests = np.array(label_requests).astype(bool)
        else:
            raise ValueError("Invalid active learning mode!")
        return label_requests

    def __deepcopy__(self, memo={}):
        replica = SGDClassifier(learning_rate=self.learning_rate,
                                eta0=self.eta0,
                                alpha=self.alpha,
                                class_weight=self.class_weight,
                                query_strategy=self.query_strategy,
                                query_ratio=self.query_ratio,
                                query_threshold=self.query_threshold,
                                query_weight=self.query_weight,
                                query_beta=self.query_beta)
        replica.t_ = self.t_
        replica.coef_ = copy.deepcopy(self.coef_)
        replica.intercept_ = self.intercept_
        replica.K = copy.deepcopy(self.K)
        replica.max_X = copy.deepcopy(self.max_X)
        return replica
