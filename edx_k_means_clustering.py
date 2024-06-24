
def warn(*args,**kwargs):
    pass
import warnings
warnings.warn=warn

#import libraries
import random 
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#There are many models for clustering.
#Despite its simplicity, the K-means is vastly used for clustering in many data science applications, 
#it is especially useful if you need to quickly discover insights from unlabeled data

#k-means on a random generated dataset

#Creating our own dataset

#set random seed
np.random.seed(0)

#making random clusters of points by using the make_blobs class
x,y=make_blobs(n_samples=5000,centers=[[4,4],[-2,-1],[2,-3],[1,1]],cluster_std=0.9)

#display scatter plot
plt.scatter(x[:,0],x[:,1],marker=',')
plt.show()


#setting up k means
#used parameters
#init: Initialization method of the centroids.
#n_clusters: The number of clusters to form as well as the number of centroids to generate.
#n_init: Number of time the k-means algorithm will be run with different centroid seeds.
K_means=KMeans(init='k-means++',n_clusters=4,n_init=12)
#fit the model
K_means.fit(x)

#grab the labels
K_means_labels=K_means.labels_
K_means_labels

#creating the visual plot
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()


#cluster into 3 clusters
K_means3=KMeans(init='k-means++',n_clusters=3,n_init=12)
K_means3.fit(x)
fig=plt.figure(figsize=(6,4))
colors=plt.cm.Spectral(np.linspace(0,1,len(set(K_means3.labels_))))
ax=fig.add_subplot(1,1,1)
for k,col in zip(range(len(K_means3.cluster_centers_)),colors):
    my_members=K_means3.labels_==k
    cluster_center=K_means3.cluster_centers_[k]
    ax.plot(x[my_members,0],x[my_members,1],'w',markerfacecolor=col,marker=',')
    ax.plot(cluster_center[0],cluster_center[1],'o',markerfacecolor=col,markeredgecolor='k',markersize=6)
plt.show()    




#COSTUMER SEGMENTATION WITH K-MEANS

#Customer segmentation is the practice of partitioning a customer base into groups of individuals that have similar characteristics. 
#It is a significant strategy as a business can target these specific groups of customers and effectively allocate marketing resources.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")

#Preprocessing

# Drop the 'Address' column(categorical data)
df = cust_df.drop('Address', axis=1)
df.head()

# Normalizing over the std deviation
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

#Modeling
# Perform K-means clustering
clusterNum = 3
K_means = KMeans(init='k-means++', n_clusters=clusterNum, n_init=12)
K_means.fit(x)
labels = K_means.labels_
print(labels)


#Insights
# Assign the label to each row in the dataframe
df['Clus_km'] = labels
df.head(5)

#check the centroid values by averaging the features in each cluster.
df.groupby('Clus_km').mean()

#distribution of customers based on their age and income:
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

# Plot the 3D scatter plot
fig = plt.figure(figsize=(8, 6))
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))

# #k-means will partition your customers into mutually exclusive groups, for example, into 3 clusters. 
# The customers in each cluster are similar to each other demographically.
# Now we can create a profile for each group, considering the common characteristics of each cluster. 
# For example, the 3 clusters can be:

# - AFFLUENT, EDUCATED AND OLD AGED
# - MIDDLE AGED AND MIDDLE INCOME
# - YOUNG AND LOW INCOME