import argparse
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets

from misc import load_dataset, convert_su_data_sklearn_compatible


class SU_Base(BaseEstimator, ClassifierMixin):

    def __init__(self, prior=.7, lam=1):
        self.prior = prior
        self.lam = lam

    def fit(self, x, y):
        pass

    def predict(self, x):
        check_is_fitted(self, 'coef_')
        x = check_array(x)
        return np.sign(.1 + np.sign(self._basis(x).dot(self.coef_)))

    def score(self, x, y):
        x_s, x_u = x[y == 1, :], x[y == 0, :]
        f = self.predict
        p_p = self.prior
        p_n = 1 - self.prior
        p_s = p_p ** 2 + p_n ** 2

        # SU risk estimator with zero-one loss
        r_s = (np.sign(-f(x_s)) - np.sign(f(x_s))) * p_s / (p_p - p_n)
        r_u = (-p_n * (1 - np.sign(f(x_u))) + p_p * (1 - np.sign(-f(x_u)))) / (p_p - p_n)
        return r_s.mean() + r_u.mean()

    def _basis(self, x):
        # linear basis
        return np.hstack((x, np.ones((len(x), 1))))


class SU_SL(SU_Base):

    def fit(self, x, y):
        check_classification_targets(y)
        x, y = check_X_y(x, y)
        x_s, x_u = x[y == +1, :], x[y == 0, :]
        n_s, n_u = len(x_s), len(x_u)

        p_p = self.prior
        p_n = 1 - self.prior
        p_s = p_p ** 2 + p_n ** 2
        k_s = self._basis(x_s)
        k_u = self._basis(x_u)
        d = k_u.shape[1]

        A = (p_p - p_n) / n_u * (k_u.T.dot(k_u) + 2 * self.lam * n_u * np.eye(d))
        b = 2 * p_s * k_s.T.mean(axis=1) - k_u.T.mean(axis=1)
        self.coef_ = np.linalg.solve(A, b)

        return self


class SU_DH(SU_Base):

    def fit(self, x, y):
        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False

        check_classification_targets(y)
        x, y = check_X_y(x, y)
        x_s, x_u = x[y == +1, :], x[y == 0, :]
        n_s, n_u = len(x_s), len(x_u)

        p_p = self.prior
        p_n = 1 - self.prior
        p_s = p_p ** 2 + p_n ** 2
        k_s = self._basis(x_s)
        k_u = self._basis(x_u)
        d = k_u.shape[1]

        P = np.zeros((d + 2 * n_u, d + 2 * n_u))
        P[:d, :d] = self.lam * np.eye(d)
        q = np.vstack((
            -p_s / (n_s * (p_p - p_n)) * k_s.T.dot(np.ones((n_s, 1))),
            -p_n / (n_u * (p_p - p_n)) * np.ones((n_u, 1)),
            -p_p / (n_u * (p_p - p_n)) * np.ones((n_u, 1))
        ))
        G = np.vstack((
            np.hstack((np.zeros((n_u, d)), -np.eye(n_u), np.zeros((n_u, n_u)))),
            np.hstack((0.5 * k_u, -np.eye(n_u), np.zeros((n_u, n_u)))),
            np.hstack((k_u, -np.eye(n_u), np.zeros((n_u, n_u)))),
            np.hstack((np.zeros((n_u, d)), np.zeros((n_u, n_u)), -np.eye(n_u))),
            np.hstack((-0.5 * k_u, np.zeros((n_u, n_u)), -np.eye(n_u))),
            np.hstack((-k_u, np.zeros((n_u, n_u)), -np.eye(n_u)))
        ))
        h = np.vstack((
            np.zeros((n_u, 1)),
            -0.5 * np.ones((n_u, 1)),
            np.zeros((n_u, 1)),
            np.zeros((n_u, 1)),
            -0.5 * np.ones((n_u, 1)),
            np.zeros((n_u, 1))
        ))
        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        self.coef_ = np.array(sol['x'])[:d]


def class_prior_estimation(DS, DU):
    # class-prior estimation using MPE method in Ramaswamy et al. (2016)
    from mpe import wrapper
    km1, km2 = wrapper(DU, DS.reshape(-1, DS.shape[1]//2))
    prior_p = km2
    return 0.5 * (np.sqrt(2 * prior_p - 1) + 1)


def main(loss_name, prior=0.7, n_s=500, n_u=500, end_to_end=False):
    if loss_name == 'squared':
        SU = SU_SL
    elif loss_name == 'double-hinge':
        SU = SU_DH

    # load dataset
    n_test = 100
    x_s, x_u, x_test, y_test = load_dataset(n_s, n_u, n_test, prior)
    x_train, y_train = convert_su_data_sklearn_compatible(x_s, x_u)

    if end_to_end:
        # use KM2 (Ramaswamy et al., 2016)
        est_prior = class_prior_estimation(x_s, x_u)
    else:
        # use the pre-fixed class-prior
        est_prior = prior

    # cross-validation
    lam_list = [1e-01, 1e-04, 1e-07]
    score_cv_list = []
    for lam in lam_list:
        clf = SU(prior=est_prior, lam=lam)
        score_cv = cross_val_score(clf, x_train, y_train, cv=5).mean()
        score_cv_list.append(score_cv)

    # training with the best hyperparameter
    lam_best = lam_list[np.argmax(score_cv_list)]
    clf = SU(prior=est_prior, lam=lam_best)
    clf.fit(x_train, y_train)

    # test prediction
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--loss',
        action   = 'store',
        required = True,
        type     = str,
        choices  = ['squared', 'double-hinge'],
        help     = 'loss function')

    parser.add_argument('--ns',
        action   = 'store',
        required = False,
        type     = int,
        default  = 500,
        help     = 'number of similar data pairs')

    parser.add_argument('--nu',
        action   = 'store',
        required = False,
        type     = int,
        default  = 500,
        help     = 'number of unlabeled data points')

    parser.add_argument('--prior',
        action   = 'store',
        required = False,
        type     = float,
        default  = 0.7,
        help     = 'true class-prior (ratio of positive data)')

    parser.add_argument('--full',
        action   = 'store_true',
        default  = False,
        help     = 'do end-to-end experiment including class-prior estimation (default: false)')

    args = parser.parse_args()
    main(args.loss, args.prior, args.ns, args.nu, args.full)
