import numpy as np


class HyperplaneStreamGenerator:
    def __init__(self,
                 n_features=2,
                 n_drift_features=0,
                 concept_drift_proba=0.0,
                 concept_drift_speed=0.0,
                 imbalance_rate=1,
                 imbalance_rate_min=1,
                 imbalance_rate_max=1,
                 imbalance_drift_proba=0.0,
                 imbalance_drift_speed=0.0,
                 noise_percentage=0.0):
        self.n_features = n_features
        self.n_drift_features = n_drift_features
        self.concept_drift_proba = concept_drift_proba
        self.concept_drift_speed = concept_drift_speed
        self.imbalance_rate = imbalance_rate
        self.imbalance_rate_min = imbalance_rate_min
        self.imbalance_rate_max = imbalance_rate_max
        self.imbalance_drift_proba = imbalance_drift_proba
        self.imbalance_drift_speed = imbalance_drift_speed
        self.noise_percentage = noise_percentage

        self.coef_ = np.random.random(self.n_features) * 2 - 1
        self.coef_ = self.coef_ / np.sqrt(np.sum(self.coef_ ** 2))
        self.intercept_ = -np.sum(self.coef_) * 0.5  # Hyperplane passing (0.5, 0.5, ... , 0.5)

    def get_samples(self, n_samples=1):
        X = np.zeros((n_samples, self.n_features))
        y = np.zeros(n_samples)
        i = 0
        while i < n_samples:
            xt = np.random.random(self.n_features)  # all dimension belongs to [0.0, 1.0)
            while np.abs(xt.dot(self.coef_) + self.intercept_) <= 1e-6:
                xt = np.random.random(self.n_features)
            yt = np.sign(xt.dot(self.coef_) + self.intercept_)
            sample_proba = {-1: 1, 1: 1 / self.imbalance_rate}
            if np.random.random() <= sample_proba[yt]:
                X[i] = xt
                y[i] = yt
                i += 1
                self.noise_percentage *= min(sample_proba.values())
                if np.random.random() <= self.noise_percentage:
                    y[i] = -y[i]
                if np.random.random() <= self.concept_drift_proba:
                    self.coef_ = self.__concept_drift(self.coef_)
                    self.intercept_ = -np.sum(self.coef_) * 0.5
                if np.random.random() <= self.imbalance_drift_proba:
                    new_imbalance_rate = [max(self.imbalance_rate - self.imbalance_drift_speed, self.imbalance_rate_min),
                                          min(self.imbalance_rate + self.imbalance_drift_speed, self.imbalance_rate_max)]
                    self.imbalance_rate = new_imbalance_rate[np.random.randint(2)]
        return X, y.astype(int)

    def __concept_drift(self, v):
        ov = np.ones(self.n_drift_features)
        i = 0
        while i < self.n_drift_features:
            if np.abs(v[i]) > 1e-6:
                break
            i += 1
        if i < self.n_drift_features:
            ov[i] = - (v.sum() - v[i]) / v[i]
        ov = ov / np.sqrt(np.sum(ov ** 2))
        v = v / np.sqrt(np.sum(v[0:self.n_drift_features] ** 2))
        v[0:self.n_drift_features] = v[0:self.n_drift_features] + ov * self.concept_drift_speed
        v = v / np.sqrt(np.sum(v ** 2))
        return v


