import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re
import string
fake_news = pd.read_csv('Fake.csv')
true_news = pd.read_csv('True.csv')
fake_news.head()
fake_news['isTrue'] = 0
true_news['isTrue'] = 1
# Combine both the Datasets
df = pd.concat([fake_news, true_news], axis=0)
#Dropping unnecessary columns and
df = df.drop(['title', 'subject', 'date'], axis=1)
df.head()
# Visualisation
label_counts = df['isTrue'].value_counts()
plt.bar(label_counts.index, label_counts.values, color=['red', 'blue'])
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.xticks([0, 1], ['Fake', 'True'])
plt.show()
    
#cleaning text to remove any unwanted strings

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]|\W|https?://\S+|www\.\S+|<.*?>+|\n|\w*\d\w*', '', text)
    return text
df["text"] = df["text"].apply(preprocess_text)

#prepare Data for Modeling
x = df["text"]
y = df["isTrue"]

#Divide the dataset into 80:20 for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

#Before training model converting text data to vectors
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

#Training a LogisticRegression Model
lr = LogisticRegression()
lr.fit(xv_train,y_train)
pred_lr = lr.predict(xv_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_lr))
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred_lr)