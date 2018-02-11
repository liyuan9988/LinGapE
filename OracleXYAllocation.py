# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:58:16 2017

@author: User1
"""

# -*- coding: utf-8 -*-

import numpy as np
from math import pi
import itertools
import click
import multiprocessing as mp
import sys
from cvxpy import *

#get optimal ratio
def optimal_ratio(X,theta):
    Y = const_Y(X,theta)
    Y = np.array([y/y.dot(theta) for y in Y])
    print(Y)
    K,d = X.shape
    C = Variable(K,1)
    #A = np.eye(d)
    A = np.zeros((d,d))
    for i in range(K):
        A = A + C[i]* np.expand_dims(X[i,:],-1).dot(np.expand_dims(X[i,:],axis=0))
    tmp = bmat([[matrix_frac(y, A) for y in Y]])
    obj = Minimize(max_entries(tmp))
    constraints = [C>=0, sum_entries(C) == 1.0e3]
    prob = Problem(obj, constraints)
    prob.solve()
    return np.array(C.value)[:,0]


#Add observation
def add_observe(A,b,x,theta,sigma):
    A += np.expand_dims(x,-1).dot(np.expand_dims(x,axis=0))
    b += observe(x,theta,sigma)*x
    return A,b

def observe(x,theta,sigma):
    return x.dot(theta) + np.random.randn()*sigma


def next_arm_score(A,x,Y,theta,varbose=False):
    A_tmp = A + np.expand_dims(x,-1).dot(np.expand_dims(x,axis=0))
    A_inv_tmp = np.linalg.inv(A_tmp)
    res = []
    for y in Y:
        tmp1 = y.T.dot(A_inv_tmp.dot(y))
        tmp2 = y.dot(theta)
        res.append(np.sqrt(tmp1)/tmp2)
    if(varbose):
        print(res)
    return np.max(res)

def max_randomly(a):
    a = np.array(a)
    
#search for next arm in greedy manner
#returns arm id
def search_next_arm(A,X,Y,theta,verbose=False):
    score = []
    for x in X:
        score.append(next_arm_score(A,x,Y,theta))
    score= np.array(score)
    idx = np.random.choice(np.arange(len(X))[score <= np.min(score)*(1.0+1.0e-10)])
    if(verbose):
        print("scores=")
        print(score)
        print("candidate=")
        print(np.arange(len(X))[score <= np.min(score)])
    return idx

def search_next_arm_from_opt_ratio(arm_count,opt_ratio):
    return np.argmin((arm_count+1)/opt_ratio)

#calculate confidence bound
def confidence_bound(y,n,A_inv,K,sigma,delta):
    tmp = y.T.dot(A_inv.dot(y))
    return 2*np.sqrt(2)*sigma*np.sqrt(tmp)*(np.sqrt(np.log(6/pi/pi*n*n*K/delta)))

def cal_err_width(x,A,K,sigma,delta,A_inv=None):
    if(A_inv is None):
        A_inv = np.linalg.inv(A)
    tmp = x.T.dot(A_inv.dot(x))
    det = np.linalg.det(A)
    return np.sqrt(tmp)*(sigma*np.sqrt(2*np.log(K*K*np.sqrt(det)/delta) )+ 2)

#check whether the stop condition is satisified
#if stop condition is satisified, returns the best arm
#otherwise returns -1
def check_stop_condition(Y,K,theta,n,A,sigma,delta,verbose = False):
    A_inv = np.linalg.inv(A)
    for y in Y:
        score_dif = y.dot(theta)
        conf_bound = confidence_bound(y,n,A_inv,K,sigma,delta)
        if(score_dif < conf_bound):
            if(verbose):
                print("--------at %d step-------"%n)
                print(y)
                print("estimated rewards difference:%f"%score_dif)
                print("confidence bound:%f"%conf_bound)
                print("lingape confidence bound:%f"%cal_err_width(y,A,K,sigma,delta,A_inv))
            return -1
    return 1

#constrct Y which contations all directions to be distinguished
def const_Y(X,theta):
    rewards = X.dot(theta)
    best_arm_id = np.argmax(rewards)
    Y = []
    for i,x in enumerate(X):
        if(i != best_arm_id):
            Y.append(X[best_arm_id]-x)
    return np.array(Y)

#do one simulation
def simulation(X,theta,d,sigma,delta,verbose=True):
    #initilization
    #A = np.zeros((d,d))
    A = np.eye(d)
    b = np.zeros(d)
    n = 0
    K = len(X)
    arm_count = np.zeros(K,dtype=int)
    Y = const_Y(X,theta)
    #select arms one times for each
    opt_ratio = optimal_ratio(X,theta)
    print(opt_ratio)
    for i,x in enumerate(X):
        add_observe(A,b,x,theta,sigma)
        n+=1
        arm_count[i] += 1
    #select arm in a greedy manner until stop condition is satisified
    while(check_stop_condition(Y,K,theta,n,A,sigma,delta) < 0):
        if(verbose):
            if(n % 10000 == 0):
                check_stop_condition(Y,K,theta,n,A,sigma,delta,True)
                print(arm_count)
                print(next_arm_score(A,np.zeros(d),Y,theta,True))
                #print(np.linalg.inv(A))
        #arm = search_next_arm(A,X,Y,theta)
        arm = search_next_arm_from_opt_ratio(arm_count,opt_ratio)
        A,b = add_observe(A,b,X[arm],theta,sigma)
        n += 1
        arm_count[arm] += 1
    best_arm = check_stop_condition(Y,K,theta,n,A,sigma,delta)
    return n,arm_count,best_arm



def one_proc(dim,randseed):
    sigma = 1.0
    delta = 0.05
    np.random.seed(randseed)
    #set X and theta
    theta = np.zeros(dim); theta[0] = 2.0
    X = np.eye(dim)
    tmp = np.zeros(dim)
    tmp[0] = np.cos(0.01); tmp[1] = np.sin(0.01)
    X = np.r_[X,np.expand_dims(tmp,0)]
    return simulation(X,theta,dim,sigma,delta,False)

@click.command()
@click.option('--dim', '-d', default=5)
@click.option('--nexperiments', default=10)
@click.option('--nparallel', default=1)
@click.option('--show_armcount',default=False)
def cmd(dim,nexperiments,nparallel,show_armcount):
    if(nparallel==1):
        res_list = [one_proc(dim,i) for i in range(nexperiments)]
    else:
        pool = mp.Pool(nparallel)
        res_list = pool.starmap(one_proc, [(dim,i) for i in range(nexperiments)])
    f = open("oracle_result_d=%d.txt"%dim ,"w")
    for a in res_list:
        f.write(str(a[0]) + "," + str(a[2]) + "\n")
    f.close()
    if(show_armcount):
       a = res_list[0][1]
       f = open("armcount_oracle.txt" ,"w")
       for b in a:
           f.write(str(b)+"\n")
       f.close()


def main():
    cmd()
    
if __name__ == "__main__":
    main()
    



