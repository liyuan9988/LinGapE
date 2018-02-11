# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:33:52 2017

@author: User1
"""
import numpy as np
from math import pi
import itertools
import click
import multiprocessing as mp

# Add observation


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
def search_next_arm(A, X, y, sigma, delta, verbose=False):
    score = []
    for i in range(len(X)):
        x = X[i]
        A_tmp = A + np.expand_dims(x, -1).dot(np.expand_dims(x, axis=0))
        A_inv_tmp = np.linalg.inv(A_tmp)
        err = y.T.dot(A_inv_tmp.dot(y))
        score.append(err)
    if(verbose):
        print(score)
    score = np.array(score)
    idx = np.random.choice(np.arange(len(X))[score == np.min(score)])
    return idx

# calculate confidence bound


def cal_err_width(x, A, K, sigma, delta, S, A_inv=None):
    if(A_inv is None):
        A_inv = np.linalg.inv(A)
    tmp = x.T.dot(A_inv.dot(x))
    det = np.linalg.det(A)
    return np.sqrt(tmp) * (sigma * np.sqrt(2 * np.log(K * K * np.sqrt(det) / delta)) + S)


def get_next_arm(A, b, X, epsilon, delta, sigma, S, verbose=False):
    theta_hat = np.linalg.solve(A, b)
    est_rewards = X.dot(theta_hat)
    best_est_arm = X[np.argmax(est_rewards)]
    best_reward = np.max(est_rewards)
    error_bounds = []
    A_inv = np.linalg.inv(A)
    K = len(X)
    for x in X:
        error_bounds.append(cal_err_width(
            x - best_est_arm, A, K, sigma, delta, S, A_inv))
    error_bounds = np.array(error_bounds)
    error_UCB = est_rewards - best_reward + error_bounds
    best_candi_arm = X[np.argmax(error_UCB)]
    best_candi_ucb = error_UCB[np.argmax(error_UCB)]
    if(verbose):
        print("best_est_arm=" + str(best_est_arm))
        print("best_est_reward=" + str(best_reward))
        print("candi_arm=" + str(best_candi_arm))
        print("candi_est_reward=" + str(est_rewards))
        print("candi_arm_ucb=" + str(best_candi_ucb))
        print("error_UCBs=" + str(error_UCB))
        print("Next pulled arm=" + str(search_next_arm(A, X,
                                                       best_est_arm - best_candi_arm, sigma, delta)))
    if(best_candi_ucb < epsilon or np.argmax(error_UCB) == np.argmax(est_rewards)):
        return -1 * (np.argmax(est_rewards) + 1)
    else:
        return search_next_arm(A, X, best_est_arm - best_candi_arm, sigma, delta)
# do one simulation


def simulation(X, theta, d, sigma, delta, epsilon, S, verbose=True):
    # initilization
    A = np.eye(d)
    b = np.zeros(d)
    K = len(X)
    arm_count = np.zeros(K, dtype=int)
    n = 0
    for i, x in enumerate(X):
        A, b = add_observe(A, b, x, theta, sigma)
        arm_count[i] = 1
        n += 1
    next_arm = get_next_arm(A, b, X, epsilon, delta, sigma, S)
    while(next_arm >= 0):
        x = X[next_arm]
        A, b = add_observe(A, b, x, theta, sigma)
        arm_count[next_arm] += 1
        n += 1
        if(n % 10000 == 0 and verbose):
            print("=========%d the iter===========" % n)
            print("arm_count=" + str(arm_count))
            print("epsilon=" + str(epsilon))
        next_arm = get_next_arm(A, b, X, epsilon, delta,
                                sigma,S, (n % 10000 == 0 and verbose))
    best_arm = -(next_arm + 1)
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
    sigma = 2.0
    delta = 0.05
    # set X and theta
    np.random.seed(randseed)
    whole_X = np.load("features.npy")
    whole_X = whole_X[whole_X[:, 0] > 0]
    dim = whole_X.shape[1]
    # theta = cal_theta(whole_X,dim)
    theta = np.load("theta.npy")
    tmp = whole_X.dot(theta)
    bad_X = whole_X[np.logical_and(tmp < -0.93, tmp > -1.0)]
    good_X = whole_X[np.logical_and(tmp < 1.0, tmp > -0.88)]
    X = bad_X[np.random.choice(len(bad_X), K - 1, replace=False)]
    good_arm = good_X[np.random.choice(len(good_X))]
    X = np.concatenate((np.expand_dims(good_arm, 0), X))
    epsilon = 0.0
    print(X.dot(theta))
    S = np.linalg.norm(theta)
    return simulation(X, theta, dim, sigma, delta, epsilon, S, False)


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
    f = open("LinUgapE_result_K=%d_yahoo.txt" % k, "w")
    for a in res_list:
        f.write(str(a[0]) + "," + str(a[2]) + "\n")
    f.close()
    if(show_armcount):
        a = res_list[0][1]
        f = open("armcount_lingapE.txt", "w")
        for b in a:
            f.write(str(b) + "\n")
        f.close()


def main():
    cmd()

if __name__ == "__main__":
    main()
