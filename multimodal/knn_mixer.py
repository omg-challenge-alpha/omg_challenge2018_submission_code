
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from scipy.stats import pearsonr,zscore
from scipy.signal import butter, lfilter, freqz
import pickle

def CCC(x,y):
    sx = np.std(x)
    sy = np.std(y)
    mx = np.mean(x)
    my = np.mean(y)
    rho = pearsonr(x,y)[0]
    return 2*rho*sx*sy/(sx**2+sy**2+(mx-my)**2)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter_b(data, cutoff=0.1, fs=25, order=1):
    y_first_pass = butter_lowpass_filter(data[::-1].flatten(), cutoff, fs, order)
    y_second_pass = butter_lowpass_filter(y_first_pass[::-1].flatten(), cutoff, fs, order)
    return y_second_pass


# In[2]:


data=pd.read_csv("testNF.csv")
data.head()

musigma = pickle.load( open( "musigma.p", "rb" ) )


# In[3]:


mrho = 0
for s in data["Subject"].unique():
    train = np.array(data.loc[data["Subject"]!=s].iloc[:,3:])
    test = np.array(data.loc[data["Subject"]==s].iloc[:,3:])
    train_e = np.hstack((train[:-20,:-1],train[5:-15,:-1],train[10:-10,:-1],train[15:-5,:-1],train[20:,:]))    
    test_e = np.hstack((test[:-20,:-1],test[5:-15,:-1],test[10:-10,:-1],test[15:-5,:-1],test[20:,:]))    
    knn = neighbors.KNeighborsRegressor(5, weights="distance")
    y_ = knn.fit(train_e[:,:-1], train_e[:,-1]).predict(test_e[:,:-1])
    y_ = butter_lowpass_filter_b(y_,.008)
    mu,sigma = musigma[s]
    y_ = mu+sigma*zscore(y_)
    mrho += CCC(y_,test_e[:,-1])
    print(s,CCC(y_,test_e[:,-1]),pearsonr(y_,test_e[:,-1]))
    plt.plot(y_)
    plt.plot(test_e[:,-1])
    plt.show()
print("Mean value:",mrho/10)


# In[17]:


data_e = np.transpose(np.array([[] for i in range(26)]))
for s in data["Subject"].unique():
    for st in data["Story"].unique():
        dd = np.array(data.loc[(data["Subject"]==s)&(data["Story"]==s)].iloc[:,3:])
        dd = np.hstack((dd[:-20,:-1],dd[5:-15,:-1],dd[10:-10,:-1],dd[15:-5,:-1],dd[20:,:])) 
        data_e=np.vstack((data_e,dd))


# In[18]:


knn = neighbors.KNeighborsRegressor(5, weights="distance")
knn = knn.fit(data_e[:,:-1], data_e[:,-1])


# In[27]:


PATH="/Users/yc00070/DATASETS/OMG_Empathy2019/FINALTESTS/results_knn/"

finald = pd.read_csv("testdataNF.csv")
final_e = np.transpose(np.array([[] for i in range(25)]))
finald["Valence"] = np.zeros((finald.shape[0],))
for s in finald["Subject"].unique():
    for st in finald["Story"].unique():
        final = np.array(finald.loc[(finald["Subject"]==s)&(finald["Story"]==st)].iloc[:,3:-1])
        final = np.hstack((final[:-20,:],final[5:-15,:],final[10:-10,:],final[15:-5,:],final[20:,:]))
        predi = knn.predict(final)
        predi = np.array(list(predi[0]+np.zeros((20,)))+list(predi))
        predi = zscore(butter_lowpass_filter_b(predi,.008))*musigma[s][1]+musigma[s][0]
        name = PATH+"Subject_%d_Story%d.csv"%(s,st)
        pd.DataFrame({"FrameNo":np.arange(1,predi.shape[0]+1),"Valence":predi}).to_csv(name,index=False,header=True)
        finald.loc[(finald["Subject"]==s)&(finald["Story"]==st),"Valence"]= predi
        print("Subject %d, Story %d"%(s,st))
        plt.plot(predi)
        plt.show()

