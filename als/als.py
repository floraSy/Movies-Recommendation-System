#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This is an example implementation of ALS for learning how to use Spark. Please refer to
pyspark.ml.recommendation.ALS for more conventional use.

This example requires numpy (http://www.numpy.org/)
"""
from __future__ import print_function
from __future__ import division

import sys
import time

import numpy as np
from numpy.random import randn
from numpy import matrix
from pyspark.sql import SparkSession
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAMBDA = 0.01   # regularization
M = 2509
U = 45115
np.random.seed(233)


def rmse(dataset, ms, us):
    e = 0.0
    Rp = ms * us.T
    Rp = Rp.getA()
    for um, rating in dataset.items():
        e += (rating - Rp[um[0]][um[1]]) ** 2
    e = np.sqrt(e / len(dataset))
    return e


def update(i, mat, ratings):
    uu = mat.shape[0]
    ff = mat.shape[1]
#
    XtX = mat.T * mat
    Xty = mat.T * ratings[i, :].T
#
    for j in range(ff):
        XtX[j, j] += LAMBDA * uu
#
    return np.linalg.solve(XtX, Xty)


def main():

    """
    Usage: als [F] [iterations] [partitions]"

    print(""WARN: This is a naive implementation of ALS and is given as an
      example. Please use pyspark.ml.recommendation.ALS for more
      conventional use."", file=sys.stderr)
    """

    F = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    LAMBDA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    partitions = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    spark = SparkSession\
        .builder\
        .appName("PythonALS_%d_%d_%d_%d" % (F, int(1000*LAMBDA), ITERATIONS, partitions))\
        .getOrCreate()

    sc = spark.sparkContext

    print("Running ALS with M=%d, U=%d, F=%d, LAMBDA=%f, iters=%d, partitions=%d" %
          (M, U, F, LAMBDA, ITERATIONS, partitions))

    R = np.zeros((M, U))
    ms = matrix(randn(M, F))
    us = matrix(randn(U, F))

    trainset = sc.textFile("file:///home/ec2-user/train.csv") \
                .map(lambda x:x.split(',')) \
                .map(lambda x:((int(x[0]), int(x[1])), float(x[2]))) \
                .collectAsMap()
    '''
    valset = sc.textFile("file:///home/ec2-user/validation.csv") \
                .map(lambda x:x.split(',')) \
                .map(lambda x:((int(x[0]), int(x[1])), float(x[2]))) \
                .collectAsMap()
    testset = sc.textFile("file:///home/ec2-user/test.csv") \
                .map(lambda x:x.split(',')) \
                .map(lambda x:((int(x[0]), int(x[1])), float(x[2]))) \
                .collectAsMap()
    '''

    for um, rating in trainset.items():
        R[um[0]][um[1]] = rating

    R = matrix(R)
    Rb = sc.broadcast(R)
    msb = sc.broadcast(ms)
    usb = sc.broadcast(us)
    x = [i for i in range(ITERATIONS+1)]
    train_list = [rmse(trainset, ms, us)]
    # val_list = [rmse(valset, ms, us)]
    # test_list = [rmse(testset, ms, us)]

    start = time.time()
    for i in range(ITERATIONS):
        ms = sc.parallelize(range(M), partitions) \
               .map(lambda x: update(x, usb.value, Rb.value)) \
               .collect()
        # collect() returns a list, so array ends up being
        # a 3-d array, we take the first 2 dims for the matrix
        ms = matrix(np.array(ms)[:, :, 0])
        msb = sc.broadcast(ms)

        us = sc.parallelize(range(U), partitions) \
               .map(lambda x: update(x, msb.value, Rb.value.T)) \
               .collect()
        us = matrix(np.array(us)[:, :, 0])
        usb = sc.broadcast(us)

        train_error = rmse(trainset, ms, us)
        # val_error = rmse(valset, ms, us)
        # test_error = rmse(testset, ms, us)
        train_list.append(train_error)
        # val_list.append(val_error)
        # test_list.append(test_error)
        print("\nIteration %d:" % i)
        print("RMSE: {}".format(train_error))
        # print("validation RMSE: {}".format(val_error))
        # print("test RMSE: {}".format(test_error))
    end = time.time()
    print(end - start)

    plt.title("F = {}, LAMBDA = {}".format(F, LAMBDA))
    # plt.ylim(0.3, 0.7)
    # plt.plot(x, test_list, color="green", label="test")
    # plt.plot(x, val_list, color="blue", label="validation")
    plt.plot(x, train_list, color="blue", label="train")
    # plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("RMSE")
    plt.savefig("/home/ec2-user/%d_%d_%d.jpg" % (F, int(1000*LAMBDA), int(100*(end-start))))

    spark.stop()


if __name__ == "__main__":
    main()
