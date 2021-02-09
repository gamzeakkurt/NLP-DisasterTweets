import nltk
from nltk.corpus import stopwords
import pandas as pd
import re
#english stop words
stops = set(stopwords.words('english'))
def clean(df):

    #change column names
    df=df.rename({'target':'Class'},axis=1)
    df=df.rename({'keyword':'keyword1'},axis=1)
    df=df.rename({'id':'id1'},axis=1)
    df=df.rename({'location':'location1'},axis=1)

    #preprocessing for location column
    df['location1'] = df['location1'].str.replace('\d+', '')
    df['location1'] = df['location1'].str.replace('[^\w\s]','')
    df['location1'] = df['location1'].str.lower()
    df['location1'] = pd.factorize(df.location1)[0]

    df['keyword1'] = pd.factorize(df.keyword1)[0]
    #clean text column

    df['text'] = df['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    df['text'] = df['text'].apply(lambda x: re.split('http:\/\/.*', str(x))[0])
    df['text'] = df['text'].str.lower()
    #remove user tag name
    df['text'] = df['text'].apply(lambda x: re.sub('@[\w]+','',str(x)))
    #remove digits
    df['text'] = df['text'].str.replace('\d+', '')
    #remove hashtag
    df['text'] = df['text'].apply(lambda x: re.sub('#[\w]+','',str(x)))
    #remove punctuation
    df["text"] = df['text'].str.replace('[^\w\s]','')
    #remove new line
    df['text']=df['text'].replace('\n',' ',regex=True)

    return df
#remove stop words
def text_to_wordlist(text, remove_stopwords=False, return_list=False):

    
    #split
    wordlist = text.split()
    
    #remove stopwords
    if remove_stopwords:
        wordlist = [w for w in wordlist if w not in stops]
        
    if return_list:
        return wordlist
    else:
        return ' '.join(wordlist)
