import argparse
import numpy as np
from misc import load_dataset, accuracy


def SUQD(DS, DU, prior, reg):
    dim = DU.shape[1]
    pp = prior
    pn = 1 - prior
    DS = DS.reshape(-1, dim)
    NS = len(DS)
    NU = len(DU)
    PS = np.hstack((DS, np.ones((NS, 1))))
    PU = np.hstack((DU, np.ones((NU, 1))))
    w = np.linalg.solve(
        PU.T.dot(PU) + 2*reg*NU*np.eye(dim + 1),
        NU/(pp-pn)*(2*(pp**2+pn**2)/NS*PS.T.dot(np.ones((NS,1))) - 1/NU*PU.T.dot(np.ones((NU,1)))))
    return lambda x: x.dot(w[:-1]) + w[-1]


def SUDH(DS, DU, prior, reg):
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False

    dim = DU.shape[1]
    pp = prior
    pn = 1 - prior
    DS = DS.reshape(-1, dim)
    NS = len(DS)
    NU = len(DU)
    PS = np.hstack((DS, np.ones((NS, 1))))
    PU = np.hstack((DU, np.ones((NU, 1))))
    dim += 1

    P = np.zeros((dim + 2 * NU, dim + 2 * NU))
    P[:dim, :dim] = reg * np.eye(dim)
    q = np.vstack((
        -(pp**2+pn**2)/(NS*(2*pp-1)) * PS.T.dot(np.ones((NS, 1))),
        -pn/(NU*(2*pp-1)) * np.ones((NU, 1)),
        -pp/(NU*(2*pp-1)) * np.ones((NU, 1))
    ))
    G = np.vstack((
        np.hstack((np.zeros((NU, dim)), -np.eye(NU), np.zeros((NU, NU)))),
        np.hstack((0.5*PU, -np.eye(NU), np.zeros((NU, NU)))),
        np.hstack((PU, -np.eye(NU), np.zeros((NU, NU)))),
        np.hstack((np.zeros((NU, dim)), np.zeros((NU, NU)), -np.eye(NU))),
        np.hstack((-0.5*PU, np.zeros((NU, NU)), -np.eye(NU))),
        np.hstack((-PU, np.zeros((NU, NU)), -np.eye(NU)))
    ))
    h = np.vstack((
        np.zeros((NU, 1)),
        -0.5*np.ones((NU, 1)),
        np.zeros((NU, 1)),
        np.zeros((NU, 1)),
        -0.5*np.ones((NU, 1)),
        np.zeros((NU, 1))
    ))
    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    w = np.array(sol['x'])[:dim]
    return lambda x: x.dot(w[:-1]) + w[-1]


def concat_su_data(DS, DU):
    D = list(zip(DS, 0*np.ones(len(DS)))) \
        + list(zip(DU, 1*np.ones(len(DU))))
    D = np.array(D)
    return D


def split_su_data(D):
    X, pseudoY = zip(*D)
    X = np.array(X)
    pseudoY = np.array(pseudoY)
    DS = np.stack(X[pseudoY == 0])
    DU = np.stack(X[pseudoY == 1])
    return DS, DU


def class_prior_estimation(DS, DU):
    from mpe import wrapper
    km1, km2 = wrapper(DU, DS.reshape(-1, DS.shape[1]//2))
    prior_p = km2
    return 0.5 * (np.sqrt(2 * prior_p - 1) + 1)


def estimate_risk(xs, xu, f, prior):
    xs = xs.reshape(-1, xs.shape[1] // 2)
    rs = (np.sign(-f(xs)) - np.sign(f(xs))) * prior / (2 * prior - 1)
    ru = (-(1 - prior) * (1 - np.sign(f(xu))) + prior * (1 - np.sign(-f(xu)))) / (2 * prior - 1)
    return rs.mean() + ru.mean()


def main(loss_name, prior=0.7, NS=500, NU=500, end_to_end=False):
    Ntest = 100
    val_ratio = 0.2

    if loss_name == 'squared':
        method = SUQD
    elif loss_name == 'double-hinge':
        method = SUDH

    DS, DU, DtestP, DtestN = load_dataset(NS, NU, Ntest, prior)

    if end_to_end:
        # use KM2 (Ramaswamy et al., 2016)
        est_prior = class_prior_estimation(DS, DU)
    else:
        # use the pre-fixed class-prior
        est_prior = prior

    # split train-val
    D = concat_su_data(DS, DU)
    np.random.shuffle(D)
    val_size = int(len(D) * val_ratio)
    D, valD = D[:len(D)-val_size], D[len(D)-val_size:]
    valDS, valDU = split_su_data(valD)

    N = len(D)
    decay_rates = [1e-01, 1e-04, 1e-07]

    validation_risks = []
    fs = []
    for decay_rate in decay_rates:
        f = method(DS, DU, est_prior, decay_rate)
        val_risk = estimate_risk(valDS, valDU, f, est_prior)
        validation_risks.append(val_risk)
        fs.append(f)
    f = fs[np.argmin(validation_risks)]
    test_accuracy = accuracy(f, DtestP, DtestN)

    print(test_accuracy)


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
        default  = 200,
        help     = 'number of similar data pairs')

    parser.add_argument('--nu',
        action   = 'store',
        required = False,
        type     = int,
        default  = 200,
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
