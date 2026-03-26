# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

from sklearn.metrics import r2_score,mean_squared_error

# %%
data = pd.read_csv("C:/Users/hp/Downloads/archive.zip")

# %% [markdown]
# # 1- Explore Data

# %%
data.head(10)

# %%
data.shape

# %%
data.info()

# %%
data.describe()

# %%
data.duplicated().sum()

# %%
data.drop_duplicates(inplace=True)

# %%
data.shape

# %%
data.isnull().sum()

# %% [markdown]
# # 2- Analysis

# %%
data["Manufacturer"].unique()

# %%
for col in data.columns:
    print(col , ":" , data[col].nunique())

# %%
data.hist(bins=15,figsize=(15,10))
plt.show

# %%
top10cares=data["Manufacturer"].value_counts().sort_values(ascending=False)[:10]
top10cares

# %%
top10cares.plot(figsize=(10,4))
plt.show()

# %%
top10meanPrices=[data[data["Manufacturer"]==i]["Price"].mean() for i in list(top10cares.index)] 

# %%
top10meanPrices

# %%
plt.figure(figsize=(12,6))
plt.plot(top10cares.index,top10meanPrices)
plt.show()

# %%
cor=data.corr(numeric_only=True)
cor

# %%
sns.heatmap(cor , lw='0.5',annot=True , fmt = '.3f')
plt.show()

# %%
data_object=data.select_dtypes(include="object")

# %%
data_object.head()

# %%
data_object.info()

# %%
for col in data_object:
    plt.figure(figsize=(15,5))
    top10=data[col].value_counts()[:10]
    top10.plot(kind = 'bar')
    plt.title('top 10' + " " + col )
    plt.show()

# %% [markdown]
# # 3- Data processing

# %%
data=data.drop(["ID" , "Doors"],axis=1)

# %%
data.head()

# %% [markdown]
# # Data

# %%
import datetime
dtime=datetime.datetime.now()

# %%
data["Age"]=dtime.year-data["Prod. year"]

# %%
data=data.drop(["Prod. year"],axis=1)

# %%
data.head()

# %% [markdown]
# # Levy

# %%
data["Levy"]=data["Levy"].replace("-" , "0")
data["Levy"]=data["Levy"].astype(int)

# %%
data.info()

# %% [markdown]
# # Mileage

# %%
data.columns

# %%
data["Mileage"]=data["Mileage"].str.replace("km"," ")

# %%
data["Mileage"]

# %%
data["Engine volume"].unique()

# %%
data["Engine volume"]=data["Engine volume"].str.replace("Turbo"," ")
data["Engine volume"]=data["Engine volume"].astype(float)

# %%
data["Engine volume"]

# %%
data.info()

# %% [markdown]
# ## Detect OutLier

# %%
data_numeric=data.select_dtypes(exclude="object")
for col in data_numeric:
    q1=data[col].quantile(0.25)
    q3=data[col].quantile(0.75)
    iqr=q3-q1
    low=q1-1.5*iqr
    high=q3+1.5*iqr
    outlier=((data_numeric[col]>high)|(data_numeric[col]<low)).sum()
    total=data_numeric[col].shape[0]
    print(f"Total Outliers in {col} are :{outlier}-{round(100*(outlier)/total,2)}")
    if outlier>0:
        data=data.loc[(data[col]<=high) & (data[col]>=low)]

# %% [markdown]
# ## Transform Data

# %%
dobject=data.select_dtypes(include="object")
dnumeric=data.select_dtypes(exclude="object")

# %%
la=LabelEncoder()

# %%
for i in range(0,dobject.shape[1]):
    dobject.iloc[:,i]=la.fit_transform(dobject.iloc[:,i])

# %%
data=pd.concat([dobject,dnumeric],axis=1)
data[dobject.columns]=data[dobject.columns].astype(int)

# %%
data.info()

# %% [markdown]
# # 4-Model

# %%
x=data.drop("Price" , axis=1)
y= data["Price"]

# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# %%
Algorithm=["LinearRegression" , "DecisionTreeRegressor" , "RandomForestRegressor" , "GradientBoostingRegressor" , "XGBRegressor" , "SVR"]
R2=[]
RMSE=[]

# %%
def models(model):
    model.fit(x_train,y_train)
    pre=model.predict(x_test)
    r2=r2_score(y_test,pre)
    rmse=np.sqrt(mean_squared_error(y_test,pre))
    R2.append(r2)
    RMSE.append(rmse)
    score=model.score(x_test,y_test)
    print(f"The Score of Model is :{score}")
    

# %%
model1=LinearRegression()
model2=DecisionTreeRegressor()
model3=RandomForestRegressor()
model4=GradientBoostingRegressor()
model5=XGBRegressor()
model6=SVR()

# %%
models(model1)
models(model2)
models(model3)
models(model4)
models(model5)
models(model6)

# %%
df=pd.DataFrame({"Algorithm":Algorithm,"R2_score":R2,"RMSE":RMSE})
df

# %%
figz,sx=plt.subplots(figsize=(20,5))
plt.plot(df.Algorithm,df.R2_score,label="R2_score",marker="v")
plt.legend()
plt.show()

# %%
figz,sx=plt.subplots(figsize=(20,5))
plt.plot(df.Algorithm,df.RMSE,label="RMSE",c="r",marker="o")
plt.legend()
plt.show()

# %% [markdown]
# # 5-Using My Moedl To Predict New Data
# 

# %%
import pickle

# %%
fil_name="Cars_Predictions.sav"

# %%
pickle.dump(model3,open(fil_name,"wb"))

# %%
data.Manufacturer.unique()

# %%
            


