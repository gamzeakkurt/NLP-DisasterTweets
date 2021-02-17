import pandas as pd
import numpy as np
import re
from FeatureExtraction import *
from CrossValidation import *
import warnings
from Preprocessing import *
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings(action = 'ignore')

#read train file
train=pd.read_csv('Data/train.csv')

#read test file
test=pd.read_csv('Data/test.csv')


#combine train and test file to clean data
df = pd.concat([train, test], ignore_index=True)

#count most frequent word in frame
most_word=Counter(" ".join(df["text"]).split()).most_common(20)

#plot word 
x=[]
y=[]
for name,count in most_word[:20]:
	x.append(name)
	y.append(count)


sns.barplot(x=x,y=y)
plt.savefig('frequent_words.png')
plt.show()

#clean text

df=clean(df)
    
clean_messages = []
for message in df['text']:
    clean_messages.append(text_to_wordlist(
        message, remove_stopwords=True, return_list=False))


clean_message = []
for message in clean_messages:
    clean_message.append(abb(message))

lemmanization = []
for message in clean_message:
    lemmanization.append(lemma(message))
    
stemming=[]
for message in lemmanization:
    stemming.append(stem(message)) 
 
#convert text to vector using bag of words method
data_features,data_features_name=BagOfWords(stemming)

#data features convert data frame
data=pd.DataFrame(data_features,columns=data_features_name)

#remove text column from data frame
df.pop('text')

#index start from 0
df = df.reset_index(drop=True)

df=pd.concat([data,df],axis=1)


train = df.loc[df['Class'].notna()]
test = df.loc[df['Class'].isna()]

test.pop('Class')


predictors=train.drop(['Class','id1'],axis=1)
target=train['Class']



