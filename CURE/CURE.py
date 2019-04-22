import matplotlib.pyplot as plt
k = 20
n = 3
alpha = 0.2
sampleid = []
clusters = []
with open('sampleclu.txt','r') as filereader:
    for i in filereader:
        clusters.append([])
        each = []
        means = []
        p = i.split("]")
        q = p[0].strip('[').split(',')
        z = p[1].strip(',').split('[')
        l = z[1].split(',')
        for j in q:
            each.append(int(j))
        for j in l:
            means.append(float(j))
        sampleid.append([each,means])
print(sampleid)
data = []
with open('metadata.txt','r') as filereader:
    for i in filereader:
        eachdata = []
        p = i.strip('\n').split(',')
        for j in p:
            eachdata.append(float(j))
        data.append(eachdata)
print(data[0])

dataID = []
for i in sampleid:
    detail = []
    for j in i[0]:
        detail.append(data[j])
    dataID.append(detail)
print(dataID[0])

def dist(a,b):
    dist = 0
    for i in range(4):
        dist = dist+ (a[i]-b[i])**2
    return dist

res = []
for i in range(k):
    tem = []
    for j in range(n):
        maxDist = 0
        for p in dataID[i]:
            print(p)
            if j == 0:
                minDist = dist(p,sampleid[i][1])
            else:
                aa = []
                for q in tem:
                    aa.append(dist(p,q))
                minDist = min(aa)
            if minDist>=maxDist:
                maxDist = minDist
                maxPoint = p
        tem.append(maxPoint)
    rep = []
    for p in tem:
        reps = []
        for q in range(27):
            reps.append(p[q]+alpha*(sampleid[i][1][q]-p[q]))
        rep.append(reps)
        rep.sort()
    res.append(rep)
print(res[0])
for i in range(len(data)):
    bb = []
    for j in res:
        aa = []
        for l in j:
            aa.append(dist(data[i],l))
        bb.append(min(aa))
    ind = bb.index(min(bb))
    clusters[ind].append(i)
lenL = []
for i in range(k):
    print("Cluster %r:%r"%(i+1,clusters[i]))
    lenL.append(len(clusters[i]))
print(lenL)
plt.title("Cluster Number")
plt.bar(range(20),lenL)
plt.xlabel("Clusters")
plt.ylabel("Number")
plt.savefig("Cluster Result.jpg")
