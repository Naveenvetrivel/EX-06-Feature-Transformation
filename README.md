# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
Developed by : Naveen Kumar.V<br>
Regsiter no  : 212221230068

Titanic dataset:
```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

df=pd.read_csv("titanic_dataset.csv")
df.info()

df.isnull().sum()

df['Cabin']=df['Cabin'].fillna(df['Cabin'].mode()[0])
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()

df.skew()
df1=df.copy()
df1=df.info()
df1.skew()
df1["Sibsp_1"]=np.sqrt(df1.SibSp)
df1.SibSp.hist()
df1.skew()
df

del df['Name']
del df['Cabin']
del df['Ticket']
df.isnull().sum()

from sklearn.preprocessing import
OrdinalEncoder
embark=["C","S","Q"]
emb=OrdinalEncoder (categories =[embark])
df["Embarked"]=emb.fit_transform(df[["Embarked"]])
df

from category_encoders import BinaryEncoder
be1=BinaryEncoder()
df['Sex']=be1.fit_transform(df[["Sex"]])
df

#Function Transformation:

#Log Tranformation:
np.log(df["Age"])

#Reciprocal Transformation
np.reciprocal (df[["Fare"]])

#sqrt transformation
np.sqrt(df["Embarked"])

#power transformation
df["Age_boxcox"],parameters=stats.boxcox(df["Age"])
df
df["Pclass_boxcox"],parameters=stats.boxcox(df["Pclass"])
df
df["Fare_yeojohnson"],parameters = stats.yeojohnson(df["Fare"])
df
df["Parch_yeojohnson"],parameters = stats.yeojohnson(df["Parch"])
df
df.skew()

#Quantile transformation

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution ='normal',n_quantiles=891)

df["Age_1"]=qt.fit_transform(df[["Age"]])
sm.qqplot(df['Age'],line='45')

sm.qqplot(df['Age_1'],line='45')

df["Fare_1"]=qt.fit_transform(df[["Fare"]])
sm.qqplot(df["Fare"],line='45')
sm.qqplot(df['Fare_1'],line='45')

df["Parch_1"]=qt.fit_transform(df[["Parch"]])
sm.qqplot(df['Parch'],line='45')
sm.qqplot(df['Parch_1'],line='45')
df
```
Data_for_Transform :
```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

df=pd.read_csv("Data_To_Transform.csv")
df

df.skew()

#Function Transformation 
#Log Transformation 
np.log(df["Highly Positive Skew"])
np.reciprocal(df["Moderate Positive Skew"])
np.sqrt(df["Highly Positive Skew"])

df["Highly positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])

df["Moderate Positive Skew_yeojohnson"],parameters=stats.boxcox(df["Moderate Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df

df.skew()
#Quantile Transformation 
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution ='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')

df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])
sm.qqplot(df["Highly Positive Skew"],line='45')

df
```


# OUTPUT:
![168438989-825cf576-580b-42dd-85f1-ab57b3c6aebe](https://user-images.githubusercontent.com/94165322/168634126-09421e41-ec51-4fbb-87c6-8ef1e1a2379f.png)
![168438996-9f00fc86-5484-45c0-bc61-6e7aaf3f6359](https://user-images.githubusercontent.com/94165322/168634150-7fb96ac8-84a2-4f01-bc11-749cbf16a221.png)
![168439002-43915bc6-f451-40b6-94fa-3d1a583d37cf](https://user-images.githubusercontent.com/94165322/168634165-b817c0af-a9dd-490e-bcb3-5648b94b0a84.png)

![168439016-a0cde39c-de3e-41d0-beae-b6d386c55289](https://user-images.githubusercontent.com/94165322/168634191-153d13e4-b2c1-49f5-815b-1f631ab08250.png)
![168439024-cc5cc903-0e82-4955-9987-7729c284ba42](https://user-images.githubusercontent.com/94165322/168634213-7cdc2dac-f4d7-45a6-9ba3-758282e7d7ee.png)
![168439028-5cf05af5-5a53-4745-a912-56c697db4f5e](https://user-images.githubusercontent.com/94165322/168634245-3d88e92d-3ccf-4054-80d7-979262b02bc5.png)
![168439031-8a3aaf3b-c314-4e61-99eb-2cad2da485b7](https://user-images.githubusercontent.com/94165322/168634263-0c627194-2edb-40bc-817b-154a86dfdfab.png)
### Result :
The various feature transformation techniques on a dataset and save the data to a file has been performed successfully.
