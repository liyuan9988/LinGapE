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
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

# Add observation


def add_observe(A, b, x, theta, sigma):
    A += np.expand_dims(x, -1).dot(np.expand_dims(x, axis=0))
    b += observe(x, theta, sigma) * x
    return A, b


def observe(x, theta, sigma):
    return x.dot(theta) + np.random.randn() * sigma


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


def get_optimal(X, y):
    K, d = X.shape
    names = [str(i) for i in range(K)]
    prob = LpProblem("The Whiskas Problem", LpMinimize)
    w = LpVariable.dicts("w", names)
    abs_w = LpVariable.dicts("abs_w", names, lowBound=0)
    prob += lpSum([abs_w[i] for i in names])
    for j in range(d):
        prob += (lpSum([X[int(i), j] * w[i] for i in names]) == y[j])
    for i in names:
        prob += (abs_w[i] >= w[i])
        prob += (abs_w[i] >= -w[i])
    prob.solve()
    ratio = np.array([abs_w[i].value() for i in names])
    return ratio


def search_next_arm_optimal_ratio(ratio, arm_count):
    return np.argmin([arm_count[i] / (ratio[i] + 1.0e-10) for i in range(K)])


def cal_err_width(x, A, K, sigma, delta, A_inv=None):
    if(A_inv is None):
        A_inv = np.linalg.inv(A)
    tmp = x.T.dot(A_inv.dot(x))
    det = np.linalg.det(A)
    return np.sqrt(tmp) * (sigma * np.sqrt(2 * np.log(K * K * np.sqrt(det) / delta)) + 2)


def get_next_arm(A, b, X, epsilon, delta, sigma, verbose=False):
    theta_hat = np.linalg.solve(A, b)
    est_rewards = X.dot(theta_hat)
    best_est_arm = X[np.argmax(est_rewards)]
    best_reward = np.max(est_rewards)
    error_bounds = []
    A_inv = np.linalg.inv(A)
    K = len(X)
    for x in X:
        error_bounds.append(cal_err_width(
            x - best_est_arm, A, K, sigma, delta, A_inv))
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
        return (-1 * (np.argmax(est_rewards) + 1), -1)
    else:
        # return search_next_arm(A, X, best_est_arm - best_candi_arm,
        # sigma,delta)
        # return search_next_arm_optimal_ratio(X, y, arm_count)
        return np.argmax(est_rewards), np.argmax(error_UCB)
# do one simulation


def simulation(X, theta, d, sigma, delta, epsilon, verbose=True):
    # initilization
    A = np.eye(d)
    b = np.zeros(d)
    K = len(X)
    arm_count = np.zeros(K, dtype=int)
    ratio = dict()
    for i in range(K):
        for j in range(K):
            ratio[(i, j)] = get_optimal(X, X[i] - X[j])
    n = 0
    for i, x in enumerate(X):
        A, b = add_observe(A, b, x, theta, sigma)
        arm_count[i] = 1
        n += 1
    arm_i, arm_j = get_next_arm(A, b, X, epsilon, delta, sigma)
    while(arm_i >= 0):
        x = X[next_arm]
        A, b = add_observe(A, b, x, theta, sigma)
        arm_count[next_arm] += 1
        n += 1
        if(n % 10000 == 0 and verbose):
            print("=========%d the iter===========" % n)
            print("arm_count=" + str(arm_count))
            print("epsilon=" + str(epsilon))
        arm_i, arm_j = get_next_arm(A, b, X, epsilon, delta,
                                    sigma, (n % 10000 == 0 and verbose))
        if(arm_i >= 0):
            #next_arm = search_next_arm(A, X, X[arm_i]-X[arm_j],sigma,delta)
            next_arm = search_next_arm_optimal_ratio(ratio[(arm_i,arm_j)],arm_count)
    best_arm = -(arm_i + 1)
    return n, arm_count, best_arm


def one_proc(dim, randseed):
    sigma = 1.0
    delta = 0.05
    # set X and theta
    np.random.seed(randseed)
    theta = np.zeros(dim)
    theta[0] = 2.0
    X = np.eye(dim)
    tmp = np.zeros(dim)
    tmp[0] = np.cos(0.01)
    tmp[1] = np.sin(0.01)
    X = np.r_[X, np.expand_dims(tmp, 0)]
    #epsilon = theta.dot(X[0]-X[-1])/2.0
    epsilon = 0.0
    return simulation(X, theta, dim, sigma, delta, epsilon, False)


"""
def one_proc(dim, randseed):
    sigma = 1.0
    delta = 0.05
    org_dim = 5
    # set X and theta
    np.random.seed(randseed)
    theta = np.zeros(org_dim)
    theta[0] = 0.1**(dim - 2)
    X = np.eye(org_dim)
    #epsilon = theta.dot(X[0]-X[-1])/2.0
    epsilon = 0.0
    return simulation(X, theta, org_dim, sigma, delta, epsilon, False)
"""


@click.command()
@click.option('--dim', '-d', default=2)
@click.option('--nexperiments', default=10)
@click.option('--nparallel', default=1)
@click.option('--show_armcount', default=False)
def cmd(dim, nexperiments, nparallel, show_armcount):
    if(nparallel == 1):
        res_list = [one_proc(dim, i) for i in range(nexperiments)]
    else:
        pool = mp.Pool(nparallel)
        res_list = pool.starmap(one_proc, [(dim, i)
                                           for i in range(nexperiments)])
    f = open("LinUgapE_result_d=%d.txt" % dim, "w")
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
