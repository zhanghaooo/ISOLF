from evaluator.classification_measurements import ClassificationMeasurements
import numpy as np


class Evaluator:
    def __init__(self,
                 measurement,
                 pretrain_size=100,
                 batch_size=1,
                 budget=1.0):
        self.measurement = measurement
        self.pretrain_size = pretrain_size
        self.batch_size = batch_size
        self.budget = budget
        self.mean_eval_measurement = ClassificationMeasurements()

    @staticmethod
    def __hing_loss(z):
        return max(0, 1 - z)

    def evaluate(self, X, y, model):
        trained_number = 0
        budgets = self.budget * len(X)
        if self.pretrain_size > len(X):
            raise ValueError('pretrain size > number of total sample!')
        if self.pretrain_size > 0:
            model.partial_fit(X[0:self.pretrain_size], y[0:self.pretrain_size])
            trained_number += self.pretrain_size
            budgets -= self.pretrain_size
        mean_scores = []
        class_weight_records = []
        while trained_number + self.batch_size <= len(X):
            # get data
            batch_X = X[trained_number:trained_number + self.batch_size]
            batch_y = y[trained_number:trained_number + self.batch_size]
            # records cost weight
            try:
                rho = model.class_weight[1] / model.class_weight[-1]
            except ZeroDivisionError:
                rho = 0.0
            class_weight_records.append(rho)
            # predict and record result
            hat_y = model.predict(batch_X)
            self.mean_eval_measurement.add_result(batch_y, hat_y)
            if self.measurement == 'accuracy':
                mean_scores.append(self.mean_eval_measurement.get_accuracy())
            elif self.measurement == 'precision':
                mean_scores.append(self.mean_eval_measurement.get_precision())
            elif self.measurement == 'recall':
                mean_scores.append(self.mean_eval_measurement.get_recall())
            elif self.measurement == 'specificity':
                mean_scores.append(self.mean_eval_measurement.get_specificity())
            elif self.measurement == 'F1_score':
                mean_scores.append(self.mean_eval_measurement.get_F_score(beta=1))
            elif self.measurement == 'G_mean':
                mean_scores.append(self.mean_eval_measurement.get_G_mean())
            elif self.measurement == 'MCC':
                mean_scores.append(self.mean_eval_measurement.get_MCC())
            elif self.measurement == 'OP':
                mean_scores.append(self.mean_eval_measurement.get_OP())
            # active learning
            try:
                label_requests = model.query(batch_X)
            except AttributeError:
                label_requests = np.ones(len(batch_X)).astype(bool)
            for i in range(len(label_requests)):
                if budgets == 0:
                    label_requests[i:] = False
                    break
                if label_requests[i]:
                    budgets -= 1
            # train models
            if np.any(label_requests):
                model.partial_fit(batch_X[label_requests], batch_y[label_requests])
            trained_number += self.batch_size
        return np.array(mean_scores)
