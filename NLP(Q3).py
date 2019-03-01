from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import re
import numpy as np


data = pd.read_csv('train.csv')
train_data = pd.DataFrame(data)
#print (train_data.shape)
#print(train_data.groupby('sentiment').count())
"""
The data is made of user id, sentiment and review of a movie with sentiment 1 for positive sentiments and 0 for negative.
There are total 997 valid sentiments, 517 out of which are negative while others are positive; thus the overall opinion
about this movie is neutral.   
"""
"""
Data cleaning and preprocessing will include 1)Drop NaN, 2)Clean all, leave only spaces and alphanumerics.
3) turn all into lowercase, 4)ignore stopwords, most and least frequent words， 5）Lemmatization， 
"""
train_data=train_data.dropna()
train_data['review'] = train_data['review'].map(lambda x: re.sub(r'([^\s\w]|_)+', '', x))
train_data['review'] = train_data['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
from nltk.corpus import stopwords
stop = stopwords.words('english')
train_data['review'] = train_data['review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
freq = pd.Series(' '.join(train_data['review']).split()).value_counts()[:10]
freq = freq.drop('good') #'good' is a relevant most freq word
train_data['review'] = train_data['review'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
freq = pd.Series(' '.join(train_data['review']).split()).value_counts()[-50:]
train_data['review'] = train_data['review'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
from textblob import Word
train_data['review'] = train_data['review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#split train and validation data
train=[]
validation=[]
Y=[]
for a in range (0,800):
    b = train_data['review'][a]
    train.append(b)
for a in range(800,996):
    b = train_data['review'][a]
    validation.append(b)
for c in range(0,800): #add sentiments to training data
    label = train_data['sentiment'][c]
    Y.append(label)

#bag of words tool
vectorizer = CountVectorizer(analyzer='word',tokenizer=None, preprocessor=None, max_features=1000)
X = vectorizer.fit_transform(train)
X = X.toarray()
clf = MultinomialNB(alpha=0.01)
clf.fit(X,np.array(Y))

tX = vectorizer.transform(validation).toarray()
tX = clf.predict(tX) #prediction
Y2 = []
for a in tX:
    Y2.append(a)
Y=[]
for c in range (800,996):
    label = train_data['sentiment'][c]
    Y.append(label)
sum = 0.00

for d in range (0,196): #to calculate accuracy
    sum = sum+abs(Y[d]-Y2[d])
print (sum)

acc = 100-((sum/200)*100)
print ('accuracy = ',acc)



