import numpy as np
from evaluator.classification_measurements import ClassificationMeasurements
import copy


class OMCSL:
    def __init__(self, base_model, n_models=4, gamma=1, measurement='F1_score'):
        self.base_model = base_model
        self.n_models = n_models
        self.gamma = gamma
        self.measurement = measurement
        self.models = []
        cost_n = np.linspace(1, n_models, n_models) / (n_models + 1) / 2
        cost_p = 1 - cost_n
        for i in range(n_models):
            self.models.append(copy.deepcopy(base_model))
            self.models[i].class_weight = {-1: cost_n[i], 1: cost_p[i]}
        self.performances = np.zeros(n_models)
        self.mean_eval_measurements = [ClassificationMeasurements() for _ in range(n_models)]
        self.representative = 0

    def query(self, X):
        return self.models[self.representative].query(X)

    def partial_fit(self, X, y):
        prediction = [[] for _ in range(self.n_models)]
        for i in range(self.n_models):
            # partial_fit
            self.models[i].partial_fit(X, y)
            # record performances
            prediction[i].extend(self.models[i].predict(X))
            self.mean_eval_measurements[i].add_result(y, np.array(prediction[i]))
            if self.measurement == 'accuracy':
                self.performances[i] = self.mean_eval_measurements[i].get_accuracy()
            elif self.measurement == 'precision':
                self.performances[i] = self.mean_eval_measurements[i].get_precision()
            elif self.measurement == 'recall':
                self.performances[i] = self.mean_eval_measurements[i].get_recall()
            elif self.measurement == 'specificity':
                self.performances[i] = self.mean_eval_measurements[i].get_specificity()
            elif self.measurement == 'F1_score':
                self.performances[i] = self.mean_eval_measurements[i].get_F_score(beta=1)
            elif self.measurement == 'G_mean':
                self.performances[i] = self.mean_eval_measurements[i].get_G_mean()
            elif self.measurement == 'MCC':
                self.performances[i] = self.mean_eval_measurements[i].get_MCC()
            elif self.measurement == 'OP':
                self.performances[i] = self.mean_eval_measurements[i].get_OP()
            else:
                raise ValueError("invalid metric type!")
        select_proba = self.__soft_max(self.gamma * self.performances)
        self.representative = np.random.choice(a=self.n_models, p=select_proba)
        return self

    @staticmethod
    def __soft_max(x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def predict(self, X):
        return self.models[self.representative].predict(X)

    def __deepcopy__(self, memo={}):
        replica = OMCSL(base_model=self.base_model,
                        n_models=self.n_models,
                        gamma=self.gamma,
                        measurement=self.measurement)
        replica.models = copy.deepcopy(self.models)
        replica.performances = copy.deepcopy(self.performances)
        replica.mean_eval_measurements = copy.deepcopy(self.mean_eval_measurements)
        replica.representative = self.representative
        return replica

    def __getattr__(self, item):
        return getattr(self.models[self.representative], item)
