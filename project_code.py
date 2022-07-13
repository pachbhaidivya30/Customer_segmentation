#!/usr/bin/env python
# coding: utf-8

# # CUSTOMER SEGMENTATION MODEL

# ### Importing all relevent Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score
scalar=StandardScaler()
import warnings
warnings.filterwarnings('ignore')


# ### Loading data

# In[2]:


customer=pd.read_csv('Mall_Customers.csv')
customer.size


# In[3]:


customer.head()


# ###  understanding dataset

# In[4]:


customer.info()


# In[5]:


customer.describe()


# ### Plotting density plots

# In[6]:


plt.figure(figsize=(30,45))
for i,col in enumerate(customer.columns):
    if customer[col].dtype != 'object':
        ax = plt.subplot(9, 2, i+1)
        sns.kdeplot(customer[col], ax=ax)
        plt.xlabel(col)
        
plt.show()


# ### Plotting heatmap to understand correlation between attributes

# In[7]:


plt.figure(figsize=(8,6))
sns.heatmap(customer.corr(), annot=True)
plt.show()


# In[8]:


corr_matrix=customer.corr()
corr_matrix['Spending Score (1-100)'].sort_values(ascending=False)


# ### Segmentation using two attributes (Annual Income and Spending Score)

# In[9]:


x=customer.iloc[:,[3,4]].values


# In[10]:


print(x)


# ### WCSS (Within Cluster Sum of Squares)

# #### Finding WCSS values for different number of clusters

# In[11]:


wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=400,random_state=42)
    kmeans.fit(x)
    
    wcss.append(kmeans.inertia_)


# #### Plotting elbow graph

# In[12]:


sns.set()
plt.plot(range(1,11),wcss,linewidth=3,markersize=8,marker='o',color='red')
plt.title("The elbow point graph")
plt.xlabel('number of cluster')
plt.ylabel('wcss')
plt.show()


# In[13]:


# Taking number of cluster as 5 and applying k-mean model


# In[14]:


kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
labels=kmeans.fit_predict(x)


# In[15]:


# counting number of customers in each segment
sns.countplot(labels)


# ### Visualization
# #### plotting all the cluster and centroids

# In[16]:


plt.figure(figsize=(12,10))

plt.scatter(x[labels==0,0],x[labels==0,1],s=80,c='green',label='High Income-Less Spending')
plt.scatter(x[labels==1,0],x[labels==1,1],s=80,c='blue',label='Mid Income-Mid Spending')
plt.scatter(x[labels==2,0],x[labels==2,1],s=80,c='red',label='High Income-High Spending')
plt.scatter(x[labels==3,0],x[labels==3,1],s=80,c='orange',label='Low Income-High Spending')
plt.scatter(x[labels==4,0],x[labels==4,1],s=80,c='purple',label='Low Income-Low Spending')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='black',label='centroid')
plt.title("Customer Group")
plt.xlabel("Annual income(k$)")
plt.ylabel("Spending score")
plt.legend()
plt.show()


# In[17]:


cluster_2d = pd.concat([customer,pd.DataFrame({'Cluster':kmeans.labels_})],axis=1)
cluster_2d


# In[18]:


datamaps_gender={
    "Male":1,
    "Female":0
}
cluster_2d['Gender']=cluster_2d['Gender'].map(datamaps_gender)
cluster_2d


# In[19]:


cluster_2d.to_csv("Clustered_2d_Data.csv")


# In[20]:


x = cluster_2d.drop(['Cluster'],axis=1)
Y= cluster_2d[['Cluster']]
x_train, x_test, Y_train, Y_test =train_test_split(x, Y, test_size=0.3,random_state=42)


# In[21]:


ax = DecisionTreeClassifier()

# Train Decision Tree Classifer
ax = ax.fit(x_train,Y_train)

#Predict the response for test dataset
Y_pred = ax.predict(x_test)


# In[22]:


print(metrics.confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


# In[23]:


accuracy_score(Y_test, Y_pred)


# ## 3D PLOT

# ### Segmentation using 3 attributes (Age, Income,Spending score)

# In[24]:


X=customer.iloc[:,[2,3,4]].values


# In[25]:


wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)
    
sns.set()
plt.plot(range(1,11),wcss,linewidth=3,markersize=10,marker='o',color='green')
plt.title("The elbow point graph")
plt.xlabel('number of cluster')
plt.ylabel('wcss')
plt.show()


# In[26]:



model = KMeans(n_clusters = 5, init = "k-means++", max_iter = 400, n_init = 10, random_state = 0)
y_clusters = model.fit_predict(X)


# In[27]:


sns.countplot(y_clusters)


# In[28]:


print(X[y_clusters == 0,0][1])
print(X[y_clusters == 0,1][1])
print(X[y_clusters == 0,2][1])


# In[29]:


fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y_clusters == 0,0],X[y_clusters == 0,1],X[y_clusters == 0,2], s = 60 , color = 'blue', label = "cluster 0")
ax.scatter(X[y_clusters == 1,0],X[y_clusters == 1,1],X[y_clusters == 1,2], s = 60 , color = 'orange', label = "cluster 1")
ax.scatter(X[y_clusters == 2,0],X[y_clusters == 2,1],X[y_clusters == 2,2], s = 60 , color = 'green', label = "cluster 2")
ax.scatter(X[y_clusters == 3,0],X[y_clusters == 3,1],X[y_clusters == 3,2], s = 60 , color = '#D12B60', label = "cluster 3")
ax.scatter(X[y_clusters == 4,0],X[y_clusters == 4,1],X[y_clusters == 4,2], s = 60 , color = 'purple', label = "cluster 4")
ax.set_xlabel('Age of a customer-->')
ax.set_ylabel('Anual Income-->')
ax.set_zlabel('Spending Score-->')
ax.legend()
plt.show()


# In[30]:


import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py


# In[31]:


Scene = dict(xaxis = dict(title  = 'Age -->'),yaxis = dict(title  = 'Spending Score--->'),zaxis = dict(title  = 'Annual Income-->'))

# model.labels_ is nothing but the predicted clusters i.e y_clusters
labels = model.labels_
trace = go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers',marker=dict(color = labels, size= 10, line=dict(color= 'black',width = 10)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()


# In[32]:


import joblib
joblib.dump(model,"cusomer_segmentation")


# In[33]:


cluster_df = pd.concat([customer,pd.DataFrame({'Cluster':model.labels_})],axis=1)
cluster_df


# In[34]:


datamap_gender={
    "Male":1,
    "Female":0
}
cluster_df['Gender']=cluster_df['Gender'].map(datamap_gender)
cluster_df


# In[35]:


cluster_df.to_csv("Clustered_Customer_Data.csv")


# In[36]:


X = cluster_df.drop(['Cluster'],axis=1)
y= cluster_df[['Cluster']]
X_train, X_test, y_train, y_test =train_test_split(X, y_clusters, test_size=0.3,random_state=42)


# In[37]:


X_test


# In[38]:


clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[39]:


print(metrics.confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[40]:


accuracy_score(y_test, y_pred)


# In[ ]:





# In[ ]:




