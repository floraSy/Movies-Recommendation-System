import sys

def Dis(a,b):
    Distance = 0
    for i in range(27):
        Distance = Distance + (a[i]-b[i])**2
    return Distance

def merge(a,b):
    newData = []
    for i in range(27):
        lenA = len(a[0])
        lenB = len(b[0])
        newData.append((a[1][i]*lenA+b[1][i]*lenB)/(lenA+lenB))
    return [a[0]+b[0],newData]

k = 20
n = 3
alpha = 0.2
sample = []
data = []
with open('sample.dat','r') as filereader:
    for i in filereader:
        info = i.strip('\n').split(',')
        sample.append(info)
        data.append(info)

sampledata = []
for i in sample:
    row = []
    for j in range(len(i)):
        if j == 0:
            id = int(i[j])
        elif j != 0:
            row.append(float(i[j]))
    sampledata.append([[id],row])
print(sampledata[0])
dataSet = []
for i in sampledata:
    p = 0
    row = []
    for j in i[1]:
        if p!=0:
            row.append(j)
        p = p+1
    dataSet.append(row)
print(dataSet)

L = len(sampledata)

while L > k:
    distance = []
    for i in range(L):
        for j in range(i+1,L):
            distance.append([[i,j],Dis(sampledata[i][1],sampledata[j][1])])
    distance.sort(key=lambda x:x[1])
    A = sampledata[distance[0][0][0]]
    B = sampledata[distance[0][0][1]]
    sampledata.remove(A)
    sampledata.remove(B)
    sampledata.append(merge(A,B))
    L = len(sampledata)
    print(L)
print(sampledata)
file = open('sampleclu.txt', 'w')
for i in sampledata:
    file.write("%r\n"%i)  
file.close()
