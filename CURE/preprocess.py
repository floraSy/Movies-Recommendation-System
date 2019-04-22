import sys
import random
import pandas as pd
import numpy as np
import time as tm
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
data_file = pd.read_csv("movies_metadata.csv")

metadata = data_file[['budget','genres','popularity','revenue','release_date','runtime','vote_average','vote_count','id']]
metadata.dropna(inplace=True)

metadata['budget'] = metadata['budget'].astype(float)
metadata['popularity'] = metadata['popularity'].astype(float)
#
# new_data=[]
min_tuple = (1874, 12, 9)
timestamp = []
for times in metadata['release_date']:
    time_arr = tuple(map(lambda x:float(x),times.split('-')))
    timestamp.append((time_arr[0]-min_tuple[0])*365+ (time_arr[1] - min_tuple[1])*30 + (time_arr[2] - min_tuple[2]) * 30)
metadata['timestamp'] = timestamp

genres = metadata['genres']
nameL = []
Genres = []
for i in genres:
    newG = []
    row = i.strip('[').strip(']')
    if len(row)!= 0:
        newRow = row.split(',')
        for i in range(len(newRow)):
            if i%2 == 1:
                name = newRow[i].strip('}').split(':')
                nameL.append(eval(name[1]))
                newG.append(eval(name[1]))
    Genres.append(newG)
nameL = list(set(nameL))
nameL.sort()
numberG = np.zeros([len(nameL),len(Genres)])
for i in range(len(Genres)):
    for p in Genres[i]:
        index = nameL.index(p)
        numberG[index][i] = 0.1
for i in range(len(nameL)):
    metadata[nameL[i]] = numberG[i]
metadata.drop(columns=['release_date'],inplace=True)
metadata.drop(columns=['genres'],inplace=True)
metadata['newid'] = metadata['id']
metadata.drop(columns=['id'],inplace=True)
print metadata[metadata.columns[:-1]]

minmax_processed = preprocessing.MinMaxScaler().fit_transform(metadata[metadata.columns[:-21]])
metadata_scaled = pd.DataFrame(minmax_processed,columns=metadata.columns[:-21])
metadata_scaled.dropna(inplace=True)
for i in range(len(nameL)):
    metadata_scaled[nameL[i]] = numberG[i]
print metadata_scaled
newMeta = metadata_scaled.values.tolist()


a = []
while len(a) <500:
    a.append(random.randint(0,45000))
    a = list(set(a))
a.sort()

file = open('sample.dat', 'w')
for i in a:
    file.write("%r,"%(i))
    for p in range(27):
        if p == 26:
            file.write("%r\n"%newMeta[i][p])
        else:
            file.write("%r,"%(newMeta[i][p]))
file.close()

file = open('metadata.txt', 'w')
for i in range(45130):
    for p in range(27):
        if p == 26:
            file.write("%r\n"%(newMeta[i][p]))
        #elif p>6 and p<26:
            #file.write("%r,"%(newMeta[i][p]*0.1))
        else:
            file.write("%r,"%(newMeta[i][p]))
file.close()

K = [2**i for i in range(5)] + [i for i in range(17, 32)] + [2**i for i in range(5, 10)]
ad = []

for k in K:
    print k
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(metadata_scaled)
    '''
    labels = pd.Series(kmeans.labels_)
    d = 0
    for i in range(k):
        xi = metadata_scaled[labels.values == i].values
        di = 0.0
        for a in range(len(xi)-1):
            for b in range(a, len(xi)):
                dist = 0.0
                for ai, bi in zip(xi[a], xi[b]):
                    dist += (ai - bi) ** 2
                dist = np.sqrt(dist)
                di = max(dist, di)
        d += di
    d = d / k
    '''
    ad.append(sum(np.min(cdist(metadata_scaled, kmeans.cluster_centers_, 'euclidean'),axis=1))/metadata_scaled.shape[0])
plt.axvline(20,color = "red")
plt.title("Elbow Method")
plt.plot(K, ad, color="blue")
plt.xlabel("K")
plt.ylabel("Average Radius")
plt.savefig("EM.jpg")
plt.plot()