from __future__ import print_function
from __future__ import division

import time
import random


def outputdataset(dataset, filename):
    with open(filename, 'w') as fo:
        for uid, ratings in enumerate(dataset):
            for mid, rating in ratings.items():
                print(','.join([str(uid),str(mid),str(rating)]), file=fo)


def outputid(select_ids, filename):
    with open(filename, 'w') as fo:
        print('\n'.join(map(lambda x:str(x), select_ids)), file=fo)

def main():
    ratingfile = "ratings.csv"
    maxuid = 270896
    threshold = 1000
    users = [dict() for i in range(maxuid)]
    r = set()
    with open(ratingfile, 'r') as fi:
        head = fi.readline()
        line = fi.readline()
        while line:
            data = line.split(',')
            r.add(data[2])
            users[int(data[0])-1][int(data[1])-1] = (float(data[2]) - 2.75) / 2.25
            line = fi.readline()
    print(r)
    usermap = []
    susers = []
    for uid, ratings in enumerate(users):
        if len(ratings) >= threshold:
            usermap.append(uid)
            susers.append(ratings)
    print("number of selected users: " + str(len(susers)))
    outputid(usermap, "users.csv")

    movies = set()
    for ratings in users:
        for mid in ratings.keys():
            movies.add(mid)
    print("number of selected movies: " + str(len(movies)))
    smovies = list(sorted(movies))
    outputid(smovies, "movies.csv")

    moviemap = {}
    for i, mid in enumerate(smovies):
        moviemap[mid] = i

    random.seed(666)
    trainset = [dict() for i in range(len(susers))]
    valset = [dict() for i in range(len(susers))]
    testset = [dict() for i in range(len(susers))]
    for uid, ratings in enumerate(susers):
        for mid, rating in ratings.items():
            trainset[uid][moviemap[mid]] = rating
    '''
    for uid, ratings in enumerate(trainset):
        num = len(ratings) // 10
        for i in range(num):
            d = random.randint(0, len(ratings)-1)
            mid, rating = list(ratings.items())[d]
            valset[uid][mid] = rating
            ratings.pop(mid)
        for i in range(num):
            d = random.randint(0, len(ratings)-1)
            mid, rating = list(ratings.items())[d]
            testset[uid][mid] = rating
            ratings.pop(mid)
    outputdataset(valset, "validation.csv")
    outputdataset(testset, "test.csv")
    '''
    outputdataset(trainset, "train.csv")


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end-start)
