import gzip
import numpy as np


def encode_line(line):
    user_id = line.index("|user")
    user_raw_feature = line[user_id + 1:user_id + 7]
    user_feature = np.zeros(6)
    for a in user_raw_feature:
        a = a.split(":")
        if(int(a[0]) == 7):
            break
        user_feature[int(a[0]) - 1] = float(a[1])
    article_id = line[1]
    article_feature_id = line.index("|" + article_id)
    article_raw_feature = line[article_feature_id + 1:article_feature_id + 7]
    article_feature = np.zeros(6)
    for a in article_raw_feature:
        a = a.split(":")
        if(int(a[0]) == 7):
            break
        else:
            article_feature[int(a[0]) - 1] = float(a[1])
    feature = np.outer(user_feature, article_feature)
    return np.reshape(feature,36)


if __name__ == '__main__':
    f = gzip.open("ydata-fp-td-clicks-v1_0.20090501.gz")
    nrow = 0
    for i in f:
        nrow += 1
    f.close()
    f = gzip.open("ydata-fp-td-clicks-v1_0.20090501.gz")
    X = np.empty((nrow, 36))
    y = np.empty(nrow)
    for i, line in enumerate(f):
        line = line.decode()
        line = line.rstrip()
        line = line.split(" ")
        X[i] = encode_line(line)
        y[i] = int(line[2])
    f.close()
    np.save("features.npy",X)
    np.save("targets.npy",y)
