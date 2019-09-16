

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('lungcancer.csv')
X1=dataset.loc[:, ['Agerecodewith1yearolds', 'CStumorsize2004','DerivedAJCCT6thed2004','DerivedAJCCN6thed2004','DerivedAJCCM6thed2004']]
from sklearn.cluster import KMeans 
clusters = 7
kmeans = KMeans(n_clusters = clusters) 
kmeans.fit(X1) 
XY=kmeans.labels_
print(XY)
pca_=dataset
str_labels = list(map(lambda label:'% s' % label, kmeans.labels_)) 
from sklearn.decomposition import PCA 
pca = PCA(2) 
pca.fit(X1) 
pca_data = pd.DataFrame(pca.transform(X1)) 
print(pca_data.head())
X = pca_.iloc[:, [2,3]].values
kmeans = KMeans(n_clusters = 7, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'grey', label = 'Cluster 6')
plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 100, c = 'olive', label = 'Cluster 7')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of cancer patient')
plt.xlabel('pca 1')
plt.ylabel('pca 2')
plt.legend()
plt.show()
a=dataset.DerivedAJCCStageGroup6thed2004.tolist()
zz=[]
j1=0
j2=0
j3=0
j4=0
j5=0
j6=0
j7=0
for i in range(0,752300):
    if(a[i]=="\'IA\'"):
        j1=i
        break;
for i in range(0,752300):
    if(a[i]=="\'IB\'"):
        j2=i
        break;
for i in range(0,752300):
    if(a[i]=="\'IIA\'"):
        j3=i
        break;
for i in range(0,752300):
    if(a[i]=="\'IIB\'"):
        j4=i
        break;
for i in range(0,752300):
    if(a[i]=="\'IIIA\'"):
        j5=i
        break;
for i in range(0,752300):
    if(a[i]=="\'IIIB\'"):
        j6=i
        break;
for i in range(0,752300):
    if(a[i]=="\'IV\'"):
        j7=i
        break;
z1=XY[j1]
z2=XY[j2]
z3=XY[j3]
z4=XY[j4]
z5=XY[j5]
z6=XY[j6]
z7=XY[j7]
for i in range(0,75235):
    if(XY[i]==z1):
        zz.append(a[j1])
    elif(XY[i]==z2):
         zz.append(a[j2])
    elif(XY[i]==z3):
         zz.append(a[j3])
    elif(XY[i]==z4):
         zz.append(a[j4])
    elif(XY[i]==z5):
         zz.append(a[j5])
    elif(XY[i]==z6):
         zz.append(a[j6])
    elif(XY[i]==z7):
         zz.append(a[j7])
F=len(zz)

count=0
for i in range (0,F):
    if(zz[i]==a[i]):
        count=count
    else:
        count=count+1
acc=(count/75236)*100
print("the accuracy of the given dataset of cancer patient is ",acc) 
