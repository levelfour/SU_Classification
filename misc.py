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
    xs = np.concatenate((
        np.hstack((gen1(nsp, dim), gen1(nsp, dim))),
        np.hstack((gen0(nsn, dim), gen0(nsn, dim)))))

    nup = np.random.binomial(nu, prior)
    nun = nu - nup
    xu = np.concatenate((gen1(nup, dim), gen0(nun, dim)))

    return xs, xu


def synth_dataset_test(n, prior, dim=2):
    n1 = np.random.binomial(n, prior)
    n0 = n - n1
    x = np.concatenate((gen1(n1, dim), gen0(n0, dim)))
    y = np.concatenate((np.ones(n1), -np.ones(n0)))
    return x, y


def load_dataset(n_s, n_u, n_test, prior, dim=2):
    x_s, x_u = synth_dataset(n_s, n_u, prior, dim)
    x_test, y_test = synth_dataset_test(n_test, prior, dim)
    return x_s, x_u, x_test, y_test


def convert_su_data_sklearn_compatible(x_s, x_u):
    x = np.concatenate((x_s.reshape(-1, x_s.shape[1] // 2), x_u))
    y = np.concatenate((np.ones(x_s.shape[0] * 2), np.zeros(x_u.shape[0])))
    return x, y
