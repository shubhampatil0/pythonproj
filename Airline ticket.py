#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


os.chdir(r'H:\MTECH\IIT Kharagpur\NON CORE\1..Flight_Price--_ Machine Learning-20211221T171058Z-001')


# In[3]:


import pandas as pd


# In[4]:


train_data = pd.read_excel('Data_Train.xlsx',parse_dates=True)


# In[5]:


train_data.head()


# In[6]:


train_data.isnull().sum()


# In[7]:


train_data.shape


# In[8]:


train_data.dropna(inplace=True)


# In[9]:


train_data.dtypes


# In[10]:


def convert_into_datetime(col):
    train_data[col] = pd.to_datetime(train_data[col])


# In[11]:


train_data.columns


# In[12]:


for i in ['Date_of_Journey','Dep_Time', 'Arrival_Time']:
    convert_into_datetime(i)


# In[13]:


train_data.dtypes


# In[14]:


train_data['Journey_Day'] = train_data['Date_of_Journey'].dt.day
train_data['Journey_month'] = train_data['Date_of_Journey'].dt.month


# In[15]:


train_data.head()


# In[16]:


train_data.drop('Date_of_Journey',axis=1,inplace=True)


# In[17]:


def extract_hour(df,col):
    df[col+'_hour'] = df[col].dt.hour
    
def extract_min(df,col):
    df[col+'_min'] = df[col].dt.minute
    
def drop_col(df,col):
    df.drop(col,axis=1,inplace=True)


# In[18]:


extract_hour(train_data,'Dep_Time')
extract_min(train_data,'Dep_Time')
drop_col(train_data,'Dep_Time')


# In[19]:


extract_hour(train_data,'Arrival_Time')
extract_min(train_data,'Arrival_Time')
drop_col(train_data,'Arrival_Time')


# In[20]:


train_data.head()


# In[21]:


duration = list(train_data['Duration'])


# In[22]:


for i in range(len(duration)):
    if len(duration[i].split(' '))==2:
        pass
    else:
        if 'h' in duration[i]:
            duration[i] = duration[i]+' 0m'
        else:
            duration[i] = '0h '+ duration[i]


# In[23]:


train_data['Duration'] = duration


# In[24]:


train_data.head()


# In[25]:


def flight_dep_time(x):
    
    if (x>4)and(x<=8):
        return 'early morning'
    
    elif(x>8)and(x<=12):
        return ' morning'
    elif(x>12)and(x<=16):
        return 'noon'
    elif(x>16)and(x<=20):
        return ' evening'
    elif(x>20)and(x<=24):
        return 'night'
    else:
        return'late night'
    


# In[26]:


train_data['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind='bar')


# In[27]:


pip install plotly


# In[ ]:





# In[28]:


def preprocess_duration(x):
    if 'h' not in x:
        x='0h'+x
    elif 'm' not in x:
            x= x+'0m'
            return x


# In[29]:


train_data['Duration'].apply(preprocess_duration)


# In[30]:


train_data['Duration']


# In[31]:


int(train_data['Duration'][0].split(' ')[0][0:-1])


# In[32]:


train_data['Duration_hour']=train_data['Duration'].apply(lambda x:int(x.split()[0][0:-1]))


# In[33]:


train_data['Duration_min']=train_data['Duration'].apply(lambda x:int(x.split()[1][0:-1]))


# In[34]:


train_data.head()


# In[35]:


train_data['Duration_totalmin']=train_data['Duration'].str.replace('h','*60').str.replace(' ','+').str.replace('m','*1').apply(eval)


# In[36]:


import seaborn as sns


# In[37]:


sns.lmplot(x='Duration_totalmin',y='Price',data=train_data)


# In[38]:


import matplotlib.pyplot as plt


# In[39]:


plt.figure(figsize=(15,5))
sns.boxenplot(x='Airline',y='Price',data=train_data)
plt.xticks(rotation='vertical')


# In[40]:


train_data.drop(columns=['Route','Additional_Info'],axis=1,inplace=True)


# In[41]:


cat_col=[col for col in train_data.columns if train_data[col].dtype=='object']


# In[42]:


num_col=[col for col in train_data.columns if train_data[col].dtype!='object']


# In[43]:


for cat in train_data['Source'].unique():
    train_data['Source'+cat]=train_data['Source'].apply(lambda x:1 if x==cat else 0)


# In[44]:


train_data.head(2)


# In[45]:


airline=train_data.groupby(['Airline'])['Price'].mean().sort_values().index


# In[46]:


d1={key:ind for ind,key in enumerate(airline,0)}


# In[47]:


train_data['Airline']=train_data['Airline'].map(d1)


# In[48]:


train_data['Airline']


# In[49]:


train_data['Destination'].replace('New Delhi',"Delhi")


# In[50]:


dest=train_data.groupby(['Destination'])['Price'].mean().sort_values().index


# In[51]:


d2={key:ind for ind,key in enumerate(dest,0)}


# In[52]:


train_data['Destination']=train_data['Destination'].map(d2)


# In[53]:


train_data.head(3)


# In[54]:


train_data['Total_Stops'].unique()


# In[55]:


d3={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}


# In[56]:


train_data['Total_Stops']=train_data['Total_Stops'].map(d3)


# In[57]:


train_data.head()


# In[58]:


def plot(df,col):
    fig,(ax1,ax2,ax3)=plt.subplots(3,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)
    sns.distplot(df[col],ax=ax3,kde=False)


# In[59]:


plot(train_data,'Price')


# In[60]:


import numpy as np


# In[61]:


train_data['Price']=np.where(train_data['Price']>=35000,train_data['Price'].median(),train_data['Price'])


# In[62]:


plot(train_data,'Price')


# In[63]:


train_data.drop(columns=['Duration','Source'],axis=1,inplace=True)


# In[64]:


from sklearn.feature_selection import mutual_info_regression


# In[65]:


x=train_data.drop(['Price'],axis=1)


# In[66]:


y=train_data['Price']


# In[ ]:





# In[67]:


mutual_info_regression(x,y)


# In[68]:


imp=pd .DataFrame(mutual_info_regression(x,y),index=x.columns)
imp


# In[69]:


imp.columns=['importance']


# In[70]:


imp.sort_values(by='importance',ascending=False)


# In[71]:


from sklearn.model_selection import train_test_split


# In[72]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


# In[73]:


from sklearn.ensemble import RandomForestRegressor as rf


# In[74]:


ml_model=rf()


# In[75]:


model=ml_model.fit(x_train,y_train)


# In[104]:


y_pred=model.predict(x_test)


# In[106]:


from sklearn import metrics
metrics.r2_score(y_test,y_pred)


# In[77]:


import pickle


# In[78]:


file=open(r'H:\MTECH\IIT Kharagpur\NON CORE\1..Flight_Price--_ Machine Learning-20211221T171058Z-001/rf_rand.pkl','wb')


# In[79]:


pickle.dump(model,file)


# In[80]:


def mape(true,pred):
    true,pred=np.array(true),np.array(pred)
    return np.mean(np.abs(((true-pred)/true)))*100


# In[81]:


mape(y_test,y_pred)


# 
# 

# In[90]:


#ml pipeline
def predict(ml_model):
    model=ml_model.fit(x_train,y_rain)
    print('training score is : {}'.format(model.score(x_train,y_rain)))
    y_prediction=model.predict(x_test)
    print('predictions are: {}'.format(y_prediction))
    print('\n')
    
    from sklearn import metrics
    r2_score=metrics.r2_score(y_test,y_prediction)
    print('r2 score is : {}'.format(r2_score))
    print('MSE:{}'.format(metrics.mean_squared_error(y_test,y_prediction)))
    print('MAE:{}'.format(metrics.mean_absolute_error(y_test,y_prediction)))
    print('RMSE:{}'.format(np.sqrt(metrics.mean_squared_error(y_test,y_prediction)))
    print('MAPE:{}'.format(mape(y_test,y_prediction)))
    sns.distplot(y_test,y_prediction)
                              


# In[ ]:


predict(rf())


# In[84]:


from sklearn.model_selection import RandomizedSearchCV


# In[86]:


reg_rf=rf()


# In[89]:


n_estimators=[int(x) for x in np.linspace(start=1000,stop=1200,num=6)]
max_features= ['auto','sqrt']
max_depth=[int(x) for x in np.linspace(start=5,stop=30,num=4)]
min_samples_split=[5,10,15,100]


# In[96]:


random_grid={'n_estimators' : n_estimators,
             'max_features':max_features,
            'max_depth':max_depth,
            'min_samples_split':min_samples_split
}


# In[97]:


rf_random=RandomizedSearchCV(reg_rf,param_distributions=random_grid,cv=3)


# In[98]:


rf_random.fit(x_train,y_train)


# In[99]:


pred2=rf_random.predict(x_test)


# In[100]:


from sklearn import metrics
metrics.r2_score(y_test,pred2)


# In[ ]:




