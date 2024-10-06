import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
data_fake=pd.read_csv('Fake.csv')
data_true=pd.read_csv('True.csv')
data_fake.head()
data_true.head()
data_fake.shape
data_fake["class"]=0
data_true["class"]=1
data_true.shape,data_fake.shape
data_fake_manualtesting=data_fake.tail(10)
for i in range(23480,23470,-1):
    data_fake.drop(i,axis=0,inplace=True)

data_true_manualtesting=data_true.tail(10)
for i in range(21416,234706,-1):
    data_true.drop(i,axis=0,inplace=True)
data_true.shape,data_fake.shape
data_fake_manualtesting["class"]=0
data_true_manualtesting["class"]=1
data_fake_manualtesting.head(10)
data_true_manualtesting.head(10)
data_merge=pd.concat([data_fake,data_true],axis=0)
data_merge.head(10)
data_merge.columns
data=data_merge.drop(["title","subject","date"],axis=1)
data.isnull().sum()
data=data.sample(frac=1)
data.head()
data.reset_index(inplace=True)
data.drop(["index"],axis=1,inplace=True)
data.columns
data.head()
def wordopt(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\\W"," ",text)
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text
 data["text"]=data["text"].apply(wordopt)
x=data["text"]
y=data["class"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
#vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization=TfidfVectorizer()
xv_train=vectorization.fit_transform(x_train)
xv_test=vectorization.transform(x_test)
#using Logistic regression
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(xv_train,y_train)
pred_Lr=LR.predict(xv_test)
LR.score(xv_test,y_test)
print(classification_report(y_test,pred_Lr))
#using decission tree classifier 
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(xv_train,y_train)
pred_dt=DT.predict(xv_test)
DT.score(xv_test,y_test)
print(classification_report(y_test,pred_dt))
#using Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
GBC=GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train,y_train)
pred_db=GBC.predict(xv_test)
GBC.score(xv_test,y_test)
print(classification_report(y_test,pred_db))
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(random_state=0)
RFC.fit(xv_train,y_train)
pred_rfc=RFC.predict(xv_test)
RFC_score=RFC.score(xv_test,y_test)
print(classification_report(y_test,pred_rfc))

def output_label(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "True News"
def manual_testing(news):
    testing_news={"text":[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(wordopt)
    new_x_test=new_def_test["text"]
    new_xv_test=vectorization.transform(new_x_test)
    pred_LR=LR.predict(new_xv_test)
    pred_DT=DT.predict(new_xv_test)
    pred_GBC=GBC.predict(new_xv_test)
    pred_RFC=RFC.predict(new_xv_test)
    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_label(pred_LR[0]),
                                                                                                              output_label(pred_dt[0]),
                                                                                                              output_label(pred_GBC[0]),
                                                                                                              output_label(pred_RFC[0])))
news=str(input())
manual_testing(news)








    
