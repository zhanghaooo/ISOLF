from evaluator.confusion_matrix import ConfusionMatrix
import numpy as np
import copy


class ClassificationMeasurements:
    def __init__(self):
        self.confusion_matrix = ConfusionMatrix()

    def add_result(self, y, hat_y):
        y_ = np.array(y)
        hat_y_ = np.array(hat_y)
        for i in range(len(y_)):
            self.confusion_matrix.add_a_result(y_[i], hat_y_[i])

    def get_accuracy(self, cost={-1: 1, 1: 1}):
        tp = self.confusion_matrix.n_true_positive()
        tn = self.confusion_matrix.n_true_negative()
        fp = self.confusion_matrix.n_false_positive()
        fn = self.confusion_matrix.n_false_negative()
        try:
            accuracy = (cost[1] * tp + cost[-1] * tn) / (tp + tn + fp + fn)
        except ZeroDivisionError:
            accuracy = 0.0
        return accuracy

    def get_loss(self, cost={-1: 1, 1: 1}):
        tp = self.confusion_matrix.n_true_positive()
        tn = self.confusion_matrix.n_true_negative()
        fp = self.confusion_matrix.n_false_positive()
        fn = self.confusion_matrix.n_false_negative()
        try:
            loss = (cost[-1] * fp + cost[1] * fn) / (tp + tn + fp + fn)
        except ZeroDivisionError:
            loss = 0.0
        return loss

    def get_precision(self):
        tp = self.confusion_matrix.n_true_positive()
        fp = self.confusion_matrix.n_false_positive()
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0.0
        return precision

    def get_recall(self):
        tp = self.confusion_matrix.n_true_positive()
        fn = self.confusion_matrix.n_false_negative()
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0.0
        return recall

    def get_specificity(self):
        tn = self.confusion_matrix.n_true_negative()
        fp = self.confusion_matrix.n_false_positive()
        try:
            specificity = tn / (fp + tn)
        except ZeroDivisionError:
            specificity = 0.0
        return specificity

    def get_F_score(self, beta=1):
        if beta < 0:
            ValueError('beta < 0 for F_measure!')
        precision = self.get_precision()
        recall = self.get_recall()
        try:
            F_score = (1 + beta) * (precision * recall) / float(beta * precision + recall)
        except ZeroDivisionError:
            F_score = 0.0
        return F_score

    def get_G_mean(self):
        return np.sqrt(self.get_recall() * self.get_specificity())

    def get_MCC(self):
        tp = self.confusion_matrix.n_true_positive()
        tn = self.confusion_matrix.n_true_negative()
        fp = self.confusion_matrix.n_false_positive()
        fn = self.confusion_matrix.n_false_negative()
        p = tp + fn
        n = tn + fp
        hat_p = tp + fp
        hat_n = tn + fn
        try:
            MCC = (tp * tn - fp * fn) / float(np.sqrt(p * n * hat_p * hat_n))
        except ZeroDivisionError:
            MCC = 0.0
        return MCC

    def get_OP(self):
        accuracy = self.get_accuracy()
        specificity = self.get_specificity()
        recall = self.get_recall()
        if abs(specificity + recall) < 1e-6:
            OP = accuracy
        else:
            OP = accuracy - np.abs(specificity - recall) / float(specificity + recall)
        return OP

    def __deepcopy__(self, memo={}):
        replica = ClassificationMeasurements()
        replica.confusion_matrix = copy.deepcopy(self.confusion_matrix)
        return replica
