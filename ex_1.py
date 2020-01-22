import numpy as np
import sys as sys
import scipy.io.wavfile
from scipy.spatial import distance
import copy


sample,centroids=sys.argv[1],sys.argv[2]
fs,y=scipy.io.wavfile.read(sample)
x=np.array(y.copy())
#centroid initialization
centroids=np.loadtxt(centroids)
if (centroids.size ==2):
    centroids.shape=(1,2)
centers_new = np.zeros(centroids.shape)
dst=np.zeros((len(x),len(centroids)))
clusters = np.zeros(len(x))
#file txt to write the new centroids in.
file_cents = open("output.txt", 'w') 
for k in range(30):
    # over all the point and calculate the distance from the centoroids.    
    for index1,i in enumerate(x):
          for index2,r in enumerate(centroids):
               dst[index1,index2]=distance.euclidean(i, r)
    clusters=np.argmin(dst,axis=1)
    for i in range(len(centroids)):
         j=x[np.where(clusters == i)]
         if(j== 'NaNs'):
              centers_new[i]=centroids[i]
         else:
              centers_new[i]=np.round(np.mean(j,axis=0))
     #print the centers_new
    print(f"[iter {k}]:{','.join([str(i) for i in centers_new])}")
     #output of cents.txt
    file_cents.write(f"[iter {k}]:{','.join([str(i) for i in centers_new])}\n")
     #if it convergence stop the loop
    if (np.linalg.norm(centers_new-centroids)==0):
          break
    centroids=copy.deepcopy(centers_new)   
file_cents.close   
         



