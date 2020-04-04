# -*- coding: utf-8 -*-

import numpy as np
import sys as sys
import scipy.io.wavfile
from scipy.spatial import distance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import copy

sample=sys.argv[1]
fs,y=scipy.io.wavfile.read(sample)
x=np.array(y.copy())
#chosing the centroids
kmeans = KMeans(n_clusters=16).fit(x)
centroids = kmeans.cluster_centers_

centers_new = np.zeros(centroids.shape)
dst=np.zeros((len(x),len(centroids)))
clusters = np.zeros(len(x))
clusters2 = np.zeros(len(x))

arr=np.zeros((10))
for k in range(10):
   # over all the point and calculate the distance from the centoroids.    
   for index1,i in enumerate(x):
        for index2,r in enumerate(centroids):
            dst[index1,index2]=distance.euclidean(i, r)
   clusters=np.min(dst,axis=1)
   clusters2=np.argmin(dst,axis=1)
   #save the average of the distance in each iteration 
   arr[k]=np.average(clusters)
   for i in range(len(centroids)):
       j=x[np.where(clusters2 == i)]
       if(j== 'NaNs'):
           centers_new[i]=centroids[i]
       else:
           centers_new[i]=np.round(np.mean(j,axis=0))         
   centroids=copy.deepcopy(centers_new)
        
plt.plot(range(1,11), arr)
plt.title('The average loss value as function of the iterations')
plt.xlabel('Iterations')
plt.ylabel('Average loss value')
plt.show()
