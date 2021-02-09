from sklearn.feature_extraction.text import CountVectorizer

def BagOfWords(x_data):
    vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None)


    #training set in the construction of the vocabulary 
    data_features = vectorizer.fit_transform(x_data)

    #convert it to numpy array since it is easier to work with
    data_features =data_features.toarray()
    data_features_name=vectorizer.get_feature_names()

    return data_features,data_features_name
