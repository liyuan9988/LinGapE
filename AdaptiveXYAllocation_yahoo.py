# -*- coding: utf-8 -*-

import numpy as np
from math import pi
import itertools
import click
import multiprocessing as mp
import sys

# Add observation


def add_observe(A, b, x, theta, sigma):
    A += np.expand_dims(x, -1).dot(np.expand_dims(x, axis=0))
    b += observe(x, theta, sigma) * x
    return A, b


def observe(x, theta, sigma):
    return x.dot(theta) + np.random.randn() * sigma


# search for next arm in greedy manner
# returns arm and rho
def search_next_arm(A, X, Y, verbose=False):
    score = []
    for x in X:
        A_tmp = A + np.expand_dims(x, -1).dot(np.expand_dims(x, axis=0))
        A_inv_tmp = np.linalg.inv(A_tmp)
        err = np.max([y.T.dot(A_inv_tmp.dot(y)) for y in Y])
        score.append(err)
    if(verbose):
        print(score)
    score = np.array(score)
    idx = np.random.choice(np.arange(len(X))[score == np.min(score)])
    return idx, np.min(score)

# calculate confidence bound


def confidence_bound(y, n, A_inv, K, sigma, delta):
    tmp = y.T.dot(A_inv.dot(y))
    return 2 * np.sqrt(2) * sigma * np.sqrt(tmp) * (np.sqrt(np.log(6 / pi / pi * n * n * K * K / delta)))


# constrct Y which contations all directions to be distinguished
def const_Y(X, candidate):
    Y = []
    for x0, x1 in itertools.combinations(X[candidate], 2):
        Y.append(x0 - x1)
    return np.array(Y)

# add obervation in one phase


def one_round(X, theta, sigma, Y, thresh, arm_count):
    t = 0
    d = X.shape[1]
    #A = np.zeros((d,d))
    A = np.eye(d) * 0.01
    b = np.zeros(d)
    for arm, x in enumerate(X):
        A, b = add_observe(A, b, X[arm], theta, sigma)
        arm_count[arm] += 1
        t += 1
    rho = 10000.0
    while(rho > thresh):
        t += 1
        arm, rho = search_next_arm(A, X, Y)
        A, b = add_observe(A, b, X[arm], theta, sigma)
        arm_count[arm] += 1
    return A, b, rho, t

# check wether arm i is dominated or not


def test_dominate(i, n, X, A_inv, theta_hat, sigma, delta):
    x = X[i]
    K = len(X)
    for j in range(len(X)):
        if(i == j):
            continue
        est_dif = (X[j] - x).dot(theta_hat)
        conf_bound = confidence_bound(X[j] - x, n, A_inv, K, sigma, delta)
        if(est_dif > conf_bound):
            return True
    return False

# give all arms not dominated


def const_candidate(A, b, X, sigma, delta, n):
    candidate = []
    A_inv = np.linalg.inv(A)
    theta_hat = np.linalg.solve(A, b)
    for i in range(len(X)):
        if(not test_dominate(i, n, X, A_inv, theta_hat, sigma, delta)):
            candidate.append(i)
    return candidate

#


def check_inequality(A, X, rho):
    A_inv = np.linalg.inv(A)
    res = []
    for x in X[1:]:
        for x_dash in X:
            tmp = (x - x_dash).T.dot(A_inv).dot(x - x_dash)
            res.append(tmp)
            if(rho < tmp):
                return x, x_dash
    print("max_norm=%lf" % np.max(res))
    return 0


def check_inequality2(X, theta, sigma, rho, n, delta):
    Delta_min = (X[0] - X[-1]).dot(theta)
    K = len(X)
    a = 2 * np.sqrt(2) * sigma * np.sqrt(rho *
                                         np.log(6 / pi / pi * n * n * K * K / delta))
    print(a)
    return a < Delta_min

# do one simulation


def simulation(X, theta, d, alpha, sigma, delta, verbose=True):
    # initilization
    rho = 1.0
    n = (d * (d + 1) + 1)
    whole_count = 0
    K = len(X)
    arm_count = np.zeros(K, dtype=int)
    candidate = range(K)  # arms not dominated
    Y = const_Y(X, candidate)
    i = 0
    while(len(candidate) > 1):
        A, b, rho, n = one_round(X, theta, sigma, Y, alpha * rho, arm_count)
        whole_count += n
        candidate = const_candidate(A, b, X, sigma, delta, n)
        Y = const_Y(X, candidate)
        i += 1
        if(verbose):
            print("-----phase %d ended------" % i)
            print("n=%d" % n)
            print("rho=%lf" % rho)
            print("rho/n=%lf" % (rho / n))
            print("total arm choosed=%d" % whole_count)
            print("remained arm:")
            print(candidate)
            print("counts for each arm")
            print(arm_count)
            print("inv matrix A")
            print(np.linalg.inv(A))
            tmp = check_inequality2(X, theta, sigma, rho, n, delta)
            if(tmp):
                # print(np.linalg.inv(A))
                # print(tmp[0])
                # print(tmp[1])
                # print(rho/n)
                sys.exit(1)

    return whole_count, arm_count, candidate[0]


def cal_theta(whole_X, dim):
    target = np.load("targets.npy")
    target[target < 0.5] = -1.0
    A = np.eye(dim) * 0.01
    b = np.zeros(dim)
    for i, x in enumerate(whole_X):
        A += np.expand_dims(x, -1).dot(np.expand_dims(x, axis=0))
        b += target[i] * x
    theta = np.linalg.solve(A, b)
    np.save("theta.npy", theta)
    return theta


def one_proc(K, randseed):
    sigma = 2.0
    delta = 0.05
    alpha = 0.1
    # set X and theta
    np.random.seed(randseed)
    whole_X = np.load("features.npy")
    whole_X = whole_X[whole_X[:, 0] > 0]
    dim = whole_X.shape[1]
    #theta = cal_theta(whole_X,dim)
    theta = np.load("theta.npy")
    tmp = whole_X.dot(theta)
    bad_X = whole_X[np.logical_and(tmp < -0.93, tmp > -1.0)]
    good_X = whole_X[np.logical_and(tmp < 1.0, tmp > -0.88)]
    X = bad_X[np.random.choice(len(bad_X), K - 1, replace=False)]
    good_arm = good_X[np.random.choice(len(good_X))]
    X = np.concatenate((np.expand_dims(good_arm, 0), X))
    print(X.dot(theta))
    return simulation(X, theta, dim, alpha, sigma, delta, True)


@click.command()
@click.option('--k', '-k', default=2)
@click.option('--nexperiments', default=10)
@click.option('--nparallel', default=1)
@click.option('--show_armcount', default=False)
def cmd(k, nexperiments, nparallel, show_armcount):
    if(nparallel == 1):
        res_list = [one_proc(k, i) for i in range(nexperiments)]
    else:
        pool = mp.Pool(nparallel)
        res_list = pool.starmap(one_proc, [(k, i)
                                           for i in range(nexperiments)])
    f = open("adaptive_result_K=%d_yahoo.txt" % k, "w")
    for a in res_list:
        f.write(str(a[0]) + "," + str(a[2]) + "\n")
    f.close()
    if(show_armcount):
        a = res_list[0][1]
        f = open("armcount_adaptive.txt", "w")
        for b in a:
            f.write(str(b) + "\n")
        f.close()


def main():
    cmd()

if __name__ == "__main__":
    main()
