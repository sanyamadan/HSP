
# coding: utf-8

# In[1]:

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

#locating the excel file to be uploaded into python
wb_model = pd.ExcelFile("C:/Users/SA383640/Desktop/Model.xlsx")


# In[3]:

#reading the sheets in excel file
sheet_1 = pd.read_excel("C:/Users/SA383640/Desktop/Model.xlsx",sheetname=0)
sheet_2 = pd.read_excel("C:/Users/SA383640/Desktop/Model.xlsx",sheetname=1)


# In[4]:

#seeing the output of sheet which containd our data
sheet_2


# In[5]:

# the above output confirms that our excel file has been read
#importing the scikit learn libraries for performing random forest classification algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,classification_report


# In[7]:

# convertig the data into data variable type int
sheet_3 = sheet_2.astype(int)


# In[8]:

#mentioning the dependent varibale as Y and predictor/dependent variable as X
x = sheet_3.iloc[:,1:]
y = sheet_3['Major Injury Key']


# In[9]:

#checking whether the assignment for all the predictors and dependent variable has happened
x.describe()


# In[10]:

y.describe()


# In[11]:

#splitting the entore dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)


# In[13]:

#performing the random forest algorithm
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[14]:

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[15]:

confusion_matrix1 = confusion_matrix(y_test, y_pred)
print(confusion_matrix1)


# In[16]:

print(classification_report(y_test, y_pred))


# In[17]:

from sklearn.metrics import precision_score


# In[19]:

print(precision_score(y_test, y_pred))


# In[20]:

#counting the data which was taken for testing and training
y_train.value_counts()


# In[21]:

y_test.value_counts()


# In[22]:

#giving every feauture/predictor importance in descending order
feature_importances = pd.DataFrame(clf.feature_importances_,index = X_train.columns,
                                   columns=['importance']).sort_values('importance',ascending=False)
feature_importances


# In[ ]:

# this tells us that Nurse which was attending the patient was the most significant factor in determining the fall of the patient leading to a major injury

