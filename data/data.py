import _pickle as cPickle
import numpy as np
import gzip
import os

f = gzip.open('mnist.pkl.gz', 'rb')
f.seek(0)
train_data, valid_data, tests_data = cPickle.load(f, encoding='latin1')
f.close()

train_img = train_data[0]
tests_img = tests_data[0]
valid_img = valid_data[0]

train_res = train_data[1]
tests_res = tests_data[1]
valid_res = valid_data[1]

train_res = np.eye(10)[train_res]
tests_res = np.eye(10)[tests_res]
valid_res = np.eye(10)[valid_res]

np.savetxt("train_img.csv", train_img, delimiter=",")
np.savetxt("tests_img.csv", tests_img, delimiter=",")
np.savetxt("valid_img.csv", valid_img, delimiter=",")
np.savetxt("train_res.csv", train_res, delimiter=",")
np.savetxt("tests_res.csv", tests_res, delimiter=",")
np.savetxt("valid_res.csv", valid_res, delimiter=",")

