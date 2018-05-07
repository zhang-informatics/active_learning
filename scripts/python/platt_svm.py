import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier


class PlattScaledSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, **svm_kwargs):
        self.svm_kwargs = svm_kwargs
        self.svm  = SGDClassifier(loss="hinge", **self.svm_kwargs)
        self.lr = LogisticRegression()

    def fit(self, X, y):
        self.svm.fit(X, y)
        dists = self.svm.decision_function(X)
        self.lr.fit(dists.reshape(-1, 1), y)
        return self

    def predict(self, X, y=None):
        dists = self.svm.decision_function(X)
        preds = self.lr.predict(dists.reshape(-1, 1))

    def predict_proba(self, X, y=None):
        dists = self.svm.decision_function(X)
        probs = self.lr.predict_proba(dists.reshape(-1, 1))
        return probs

    def get_params(self, deep=True):
        return self.svm_kwargs

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

def reliability_curve(y_true, y_score, bins=10, normalize=False):
    if normalize:
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    bin_width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2

    y_score_bin_mean = np.empty(bins)
    empirical_prob_pos = np.empty(bins)
    for i, threshold in enumerate(bin_centers):
        # Find all samples that fit into the i-th bin.
        bin_idx = np.logical_and((threshold - bin_width / 2) < y_score,
                                 y_score <= (threshold + bin_width / 2))
        y_score_bin_mean[i] = y_score[bin_idx].mean()
        empirical_prob_pos[i] = y_true[bin_idx].mean()
    return y_score_bin_mean, empirical_prob_pos

def test(X, y):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    sgd_params = {"penalty": "l2", "tol": 1e-3,
                  "alpha": 1e-4, "random_state": 8}
    sgd = SGDClassifier(loss="hinge", **sgd_params)
    platt_sgd = PlattScaledSVM(**sgd_params)

    sgd.fit(X_train, y_train)
    platt_sgd.fit(X_train, y_train)

    sgd_scores = sgd.decision_function(X_test)
    platt_sgd_scores = platt_sgd.predict_proba(X_test)[:, 1]
    print(sgd_scores.shape)
    print(platt_sgd_scores.shape)

    reliability_scores = {}
    reliability_scores["sgd"] = reliability_curve(y_test, sgd_scores, normalize=True)
    reliability_scores["platt_sgd"] = reliability_curve(y_test, platt_sgd_scores)

    plt.figure(0, figsize=(8, 8))
    plt.plot([0.0, 1.0], [0.0, 1.0], 'k', label="Perfect")
    for method, (y_score_bin_mean, empirical_prob_pos) in reliability_scores.items():
        plt.plot(y_score_bin_mean, empirical_prob_pos, label=method)
    plt.ylabel("Empirical Probability")
    plt.xlabel("Predicted Score")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str)
    args = parser.parse_args()
    data = pd.read_csv(args.infile)
    y = data["T/F"].astype(int)
    X = data.drop("T/F", axis=1)
    test(X, y)
