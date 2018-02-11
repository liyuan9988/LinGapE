import numpy as np
import sys

d = 5
X = np.eye(d, dtype=float)
theta = np.zeros(d, dtype=float)
theta[0] = 2.0
tmp = np.zeros(d, dtype=float)
tmp[0] = np.cos(0.01)
tmp[1] = np.sin(0.01)
X = np.r_[X, np.expand_dims(tmp, axis=0)]
sigma = 1.0
K = d + 1
S = np.linalg.norm(theta)
reg = 1
epsilon = 2*(1-np.cos(0.01))
delta = 0.05


def confidence_bound(x, A, t):
    L = 1
    tmp = np.sqrt(x.dot(np.linalg.inv(A)).dot(x))
    return tmp * (sigma * np.sqrt(d * np.log(K * K * (1 + t * L * L) / reg / delta)) + np.sqrt(reg) * 2)


def decide_arm(y, A):
    tmp = [y.dot(np.linalg.inv(A + matrix_dot(x))).dot(y)
           for x in X]
    # print tmp
    return np.argmin(tmp)


def matrix_dot(a):
    return np.expand_dims(a, axis=1).dot(np.expand_dims(a, axis=0))

A = np.eye(d) * reg
b = np.zeros(d)
arm_selections = np.ones(K)
t = K
for i in range(K):
    A += matrix_dot(X[i])
    r = (theta.dot(X[i]) + np.random.randn() * sigma)
    b += X[i] * r

theta_hat = np.linalg.solve(A, b)
est_reward = X.dot(theta_hat)
it = np.argmax(est_reward)
jt = np.argmax(est_reward - est_reward[it] +
               np.array([confidence_bound(x - X[it], A, t) for x in X]))
B = est_reward[jt] - est_reward[it] + confidence_bound(X[it] - X[jt], A, t)
while(B > epsilon):
    a = decide_arm(X[it] - X[jt], A)
    A += matrix_dot(X[a])
    b += X[a] * (theta.dot(X[a]) + np.random.randn() * sigma)
    arm_selections[a] += 1
    t += 1
    theta_hat = np.linalg.solve(A, b)
    est_reward = X.dot(theta_hat)
    if(t%5000==0):
        print arm_selections
        print B
        print (it,jt)
    it = np.argmax(est_reward)
    jt = np.argmax(est_reward - np.max(est_reward) +
                   np.array([confidence_bound(x - X[it], A, t) for x in X]))
    B = est_reward[jt] - est_reward[it] + confidence_bound(X[it] - X[jt], A, t)

print t
print arm_selections
