import numpy as np
import copy


class ConfusionMatrix:
    def __init__(self):
        self.confusion_matrix = np.zeros((2, 2)).astype(int)  # [[TN, FP], [FN, TP]]

    def add_a_result(self, y, hat_y):
        if y == -1:
            y = 0
        if hat_y == -1:
            hat_y = 0
        self.confusion_matrix[y, hat_y] += 1

    def del_a_result(self, y, hat_y):
        if y == -1:
            y = 0
        if hat_y == -1:
            hat_y = 0
        self.confusion_matrix[y, hat_y] -= 1

    def get_value(self, y, hat_y):
        return self.confusion_matrix[y, hat_y]

    def n_true_negative(self):
        return int(self.confusion_matrix[0, 0])

    def n_false_negative(self):
        return int(self.confusion_matrix[1, 0])

    def n_false_positive(self):
        return int(self.confusion_matrix[0, 1])

    def n_true_positive(self):
        return int(self.confusion_matrix[1, 1])

    def clear(self):
        self.confusion_matrix = np.zeros((2, 2))

    def __deepcopy__(self, memo={}):
        replica = ConfusionMatrix()
        replica.confusion_matrix = copy.deepcopy(self.confusion_matrix)
        return replica
