from evaluator.window_classfication_measurement import WindowClassificationMeasurements
import numpy as np
import copy


class OACSL:
    def __init__(self,
                 base_model,
                 window_size=100,
                 rho=0,
                 delta_rho=1,
                 eta_rho=1,
                 measurement='F1_score'):
        self.base_model = base_model
        self.window_size = window_size
        self.rho = rho
        self.delta_rho = delta_rho
        self.eta_rho = eta_rho
        self.measurement = measurement
        self.models = [copy.deepcopy(base_model), copy.deepcopy(base_model), copy.deepcopy(base_model)]
        self.models[0].class_weight = {-1: 1.0, 1: np.log2(1 + np.exp2(rho))}
        self.models[1].class_weight = {-1: 1.0, 1: np.log2(1 + np.exp2(rho - delta_rho / 2))}
        self.models[2].class_weight = {-1: 1.0, 1: np.log2(1 + np.exp2(rho + delta_rho / 2))}
        self.n_models = 3
        self.performances = [0, 0, 0]
        self.current_eval_measurements = []
        for i in range(self.n_models):
            self.current_eval_measurements.append(WindowClassificationMeasurements(window_size=self.window_size))
        self.trained_number = 0
        self.diff_predictions = 0

    def partial_fit(self, X, y):
        # partial_fit
        for i in range(self.n_models):
            self.models[i].partial_fit(X, y)
        self.trained_number += len(X)
        # record performances
        prediction = [[] for _ in range(self.n_models)]
        for i in range(self.n_models):
            prediction[i].extend(self.models[i].predict(X))
            self.current_eval_measurements[i].add_results(y, np.array(prediction[i]))
            if self.measurement == 'accuracy':
                self.performances[i] = self.current_eval_measurements[i].get_accuracy()
            elif self.measurement == 'precision':
                self.performances[i] = self.current_eval_measurements[i].get_precision()
            elif self.measurement == 'recall':
                self.performances[i] = self.current_eval_measurements[i].get_recall()
            elif self.measurement == 'specificity':
                self.performances[i] = self.current_eval_measurements[i].get_specificity()
            elif self.measurement == 'F1_score':
                self.performances[i] = self.current_eval_measurements[i].get_F_score(beta=1)
            elif self.measurement == 'G_mean':
                self.performances[i] = self.current_eval_measurements[i].get_G_mean()
            elif self.measurement == 'MCC':
                self.performances[i] = self.current_eval_measurements[i].get_MCC()
            elif self.measurement == 'OP':
                self.performances[i] = self.current_eval_measurements[i].get_OP()
        # cost-sensitive detect
        for i in range(len(X)):
            if prediction[1][i] != prediction[2][i]:
                self.diff_predictions += 1
        if self.trained_number % self.window_size == 0 and self.diff_predictions == self.window_size:
            self.diff_predictions = 0
            cp = self.models[0].class_weight[1]
            cp_l = self.models[1].class_weight[1]
            cp_r = self.models[2].class_weight[1]
            rho = np.log2(np.exp2(cp) - 1)
            rho += self.eta_rho * (self.performances[1] - self.performances[2]) / (cp_l - cp_r) / (1 + np.exp2(-rho))
            self.models[0].class_weight[1] = np.log2(1 + np.exp2(rho))
            self.models[1].class_weight[1] = np.log2(1 + np.exp2(rho - self.delta_rho / 2))
            self.models[2].class_weight[1] = np.log2(1 + np.exp2(rho + self.delta_rho / 2))
        return self

    def query(self, X):
        label_requests = self.models[0].query(X)
        return label_requests

    def predict(self, X):
        return self.models[0].predict(X)

    def __deepcopy__(self, memo={}):
        replica = OACSL(base_model=self.base_model,
                        window_size=self.window_size,
                        rho=self.rho,
                        delta_rho=self.delta_rho,
                        eta_rho=self.eta_rho,
                        measurement=self.measurement)
        replica.models = copy.deepcopy(self.models)
        replica.performances = copy.deepcopy(self.performances)
        replica.current_eval_measurements = copy.deepcopy(self.current_eval_measurements)
        replica.trained_number = self.trained_number
        return replica

    def __getattr__(self, item):
        return getattr(self.models[0], item)
