#!/usr/bin/env python
# coding: utf-8

import numpy as np


def gen1(n, dim, mean=2, var=1):
    return np.random.normal(mean, var, size=(n, dim))


def gen0(n, dim, mean=-2, var=1):
    return np.random.normal(mean, var, size=(n, dim))


def synth_dataset(ns, nu, prior, dim=2):
    nsp = np.random.binomial(ns, prior**2 / (prior**2 + (1-prior)**2))
    nsn = ns - nsp
    DS = np.concatenate((
        np.hstack((gen1(nsp, dim), gen1(nsp, dim))),
        np.hstack((gen0(nsn, dim), gen0(nsn, dim)))))

    nup = np.random.binomial(nu, prior)
    nun = nu - nup
    DU = np.concatenate((gen1(nup, dim), gen0(nun, dim)))

    return DS, DU


def synth_dataset_test(n, prior, dim=2):
    n1 = np.random.binomial(n, prior)
    n0 = n - n1
    return gen1(n1, dim), gen0(n0, dim)


def load_dataset(NS, NU, Ntest, prior):
    dim = 2
    DS, DU = synth_dataset(NS, NU, prior, dim)
    DtestP, DtestN = synth_dataset_test(Ntest, prior, dim)
    return DS, DU, DtestP, DtestN


def accuracy(model, DP, DN):
    YP = np.sign(model(DP))
    YN = np.sign(model(DN))
    return (len(YP[YP == +1]) + len(YN[YN == -1])) / (len(DP) + len(DN))
