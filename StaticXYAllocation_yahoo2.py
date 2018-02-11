# -*- coding: utf-8 -*-

import numpy as np
from math import pi
import itertools
import click
import multiprocessing as mp
from cvxpy import *
import sys

# Add observation

def optimal_ratio(X):
    Y = const_Y(X, range(len(X)))
    Y = np.array([y for y in Y])
    K, d = X.shape
    C = Variable(K, 1)
    A = np.eye(d)*0.01
    for i in range(K):
        A = A + C[i] * np.expand_dims(X[i, :], -
                                      1).dot(np.expand_dims(X[i, :], axis=0))
    tmp = bmat([[matrix_frac(y, A) for y in Y]])
    obj = Minimize(max_entries(tmp))
    constraints = [C >= 0, sum_entries(C) == 1.0e3]
    prob = Problem(obj, constraints)
    prob.solve()
    return np.array(C.value)[:, 0]




def add_observe(A, b, x, theta, sigma):
    A += np.expand_dims(x, -1).dot(np.expand_dims(x, axis=0))
    b += observe(x, theta, sigma) * x
    return A, b


def observe(x, theta, sigma):
    tmp = x.dot(theta)
    if(tmp > 1):
        return 1
    elif(tmp < -1):
        return -1
    else:
        p = (tmp + 1) / 2
        return np.random.choice((1.0, -1.0), p=[p, 1 - p])

# search for next arm in greedy manner
# returns arm id


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
    return idx

# calculate confidence bound

def search_next_arm_from_opt_ratio(arm_count, opt_ratio):
    return np.argmin((arm_count + 1) / opt_ratio)


def confidence_bound(y, n, A_inv, K, sigma, delta):
    tmp = y.T.dot(A_inv.dot(y))
    if(tmp <= 0):
        print(y)
        print(n)
        print(A_inv)
        sys.exit(1)
    return 2 * np.sqrt(2) * sigma * np.sqrt(tmp) * (np.sqrt(np.log(6 / pi / pi * n * n * K * K / delta)))


def check_stop_condition(X, n, A, b, sigma, delta, verbose=False):
    theta_hat = np.linalg.solve(A, b)
    A_inv = np.linalg.inv(A)
    est_reward = X.dot(theta_hat)
    best_arm = np.argmax(est_reward)
    for i in range(len(X)):
        if(i == best_arm):
            continue
        est_dif = est_reward[best_arm] - est_reward[i]
        arm_dif = X[best_arm] - X[i]
        conf_bound = confidence_bound(arm_dif, n, A_inv, len(X), sigma, delta)
        if(est_dif < conf_bound):
            if(verbose):
                print("--------at %d step-------" % n)
                print("arm %d violates stop condition" % i)
                print("best estimated arm: %d" % best_arm)
                print("best estimated reward: %f" % est_reward[best_arm])
                print("estimated rewards difference:%f" % est_dif)
                print("confidence bound:%f" % conf_bound)
            return -1
    return best_arm

# constrct Y which contations all directions to be distinguished


def const_Y(X, candidate):
    Y = []
    for x0, x1 in itertools.combinations(X[candidate], 2):
        Y.append(x0 - x1)
    return np.array(Y)

# do one simulation


def simulation(X, theta, d, sigma, delta, verbose=True):
    # initilization
    A = np.eye(d) * 0.01
    b = np.zeros(d)
    n = 0
    K = len(X)
    arm_count = np.zeros(K, dtype=int)
    Y = const_Y(X, range(K))
    # select arms one times for each
    for i, x in enumerate(X):
        add_observe(A, b, x, theta, sigma)
        n += 1
        arm_count[i] += 1
    # select arm in a greedy manner until stop condition is satisified
    opt_ratio = optimal_ratio(X)
    print(opt_ratio)
    while(check_stop_condition(X, n, A, b, sigma, delta) < 0):
        if(verbose):
            if(n % 1000 == 0):
                check_stop_condition(X, n, A, b, sigma, delta, True)
        #arm = search_next_arm(A, X, Y)
        arm = search_next_arm_from_opt_ratio(arm_count,opt_ratio)
        A, b = add_observe(A, b, X[arm], theta, sigma)
        n += 1
        arm_count[arm] += 1
    best_arm = check_stop_condition(X, n, A, b, sigma, delta)
    return n, arm_count, best_arm


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
    sigma = 1.0
    delta = 0.05
    # set X and theta
    np.random.seed(randseed)
    whole_X = np.load("features.npy")
    whole_X = whole_X[whole_X[:, 0] > 0]
    dim = whole_X.shape[1]
    #theta = cal_theta(whole_X,dim)
    theta = np.load("theta.npy")
    tmp = whole_X.dot(theta)
    bad_X = whole_X[tmp < -0.93]
    good_X = whole_X[tmp > -0.88]
    X = bad_X[np.random.choice(len(bad_X), K - 1, replace=False)]
    good_arm = good_X[np.random.choice(len(good_X))]
    X = np.concatenate((np.expand_dims(good_arm, 0), X))
    return simulation(X, theta, dim, sigma, delta, True)


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
    f = open("static_result_K=%d_yahoo.txt" % k, "w")
    for a in res_list:
        f.write(str(a[0]) + "," + str(a[2]) + "\n")
    f.close()
    if(show_armcount):
        a = res_list[0][1]
        f = open("armcount_static.txt", "w")
        for b in a:
            f.write(str(b) + "\n")
        f.close()


def main():
    cmd()

if __name__ == "__main__":
    main()
