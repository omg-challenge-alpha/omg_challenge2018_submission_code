
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import pandas as pd
from scipy.stats import zscore


# ## Read the models' files:

# In[2]:


musigma = pickle.load( open( "musigma.p", "rb" ) )
pcaF = pickle.load(open( "pcaF.p", "rb" ) )
modelF = pickle.load(open( "regF.p", "rb" ) )


# ## Read the pre-processed joint datasets

# In[3]:


dataF = pd.read_csv("testdataF.csv")


# ## Apply the model

# In[4]:


data2F = pd.DataFrame(pcaF.transform(dataF.iloc[:,3:]))
data2F.columns = ["PC1","PC2","PC3","PC4","PC5"]
data2F["Subject"]=np.array(dataF["Subject"])
data2F["Story"]=np.array(dataF["Story"])

preds = []
for s in dataF["Subject"].unique():
    pred = modelF.predict(data2F.loc[data2F["Subject"]==s].iloc[:,:5])
    mu = musigma[s][0]
    sigma = musigma[s][1]
    pred = zscore(pred)*sigma+mu
    pred[pred>1]=1
    pred[pred< -1]=-1
    pred = zscore(pred)*sigma+mu
    preds+=list(pred)
dataF["Valence"]=np.array(preds)
dataF.head()

