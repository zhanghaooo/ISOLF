from evaluator.confusion_matrix import ConfusionMatrix
import numpy as np
import copy


class OFOClassifier:
    def __init__(self, alpha=0.0001):
        self.alpha = alpha
        self.t_ = 0
        self.coef_ = None
        self.intercept_ = None
        self.confusion_matrix = ConfusionMatrix()
        self.tau = 0.5
        self.class_weight = {-1: self.tau, 1: 1 - self.tau}

    def partial_fit(self, X, y):
        if self.coef_ is None:
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 0

        for i in range(len(y)):
            hat_p = np.sum(self.coef_ * X[i]) + self.intercept_
            eta = 1 / (self.alpha * (self.t_ + 1000))
            self.t_ += 1
            if y[i] * hat_p < 1:
                self.coef_ += eta * y[i] * X[i] - eta * self.alpha * self.coef_
                self.intercept_ += eta * y[i]
            else:
                self.coef_ -= eta * self.alpha * self.coef_
            hat_y = self.predict(X[i])
            self.confusion_matrix.add_a_result(y[i], hat_y)
        return self

    def predict(self, X):
        hat_p = self.predict_proba(X)
        FP = self.confusion_matrix.n_false_positive()
        FN = self.confusion_matrix.n_false_negative()
        TP = self.confusion_matrix.n_true_positive()
        if 2 * TP + FN + FP > 0:
            self.tau = TP / (2 * TP + FN + FP)
            self.class_weight = {-1: self.tau, 1: 1 - self.tau}
        hat_y = np.sign(hat_p - self.tau).astype(int)
        if hat_y == 0:
            hat_y = 1
        return hat_y

    def predict_proba(self, X):
        return self._sigmoid(np.dot(self.coef_, X.T) + self.intercept_)

    def _sigmoid(self, p):
        return 1.0 / (np.exp(-p) + 1.0)

    def __deepcopy__(self, memo={}):
        replica = OFOClassifier()
        replica.t_ = self.t_
        replica.coef_ = copy.deepcopy(self.coef_)
        replica.intercept_ = self.intercept_
        replica.confusion_matrix = copy.deepcopy(self.confusion_matrix)
        return replica
