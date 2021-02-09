# NLP-DisasterTweets
Natural Language Processing for Disaster Tweets 


In this project, we used disaster tweet data which are available in **[Kaggle](https://www.kaggle.com)**. 

We have 2 files in the data folder. These are train.csv  and test.csv.
In the train.csv file, there are two classes (0 and 1). The number of samples is **4342** and **3271** respectively. 
The columns of the data are id, text, location, keyword, and target. Data types are  (id, int64), (text, object),(location, object), (keyword, object), and (target, int64).
In the test.csv file, only the target column is not available.

Keyword, location, and text columns are applied preprocessing techniques to clean data. Texts are converted from upper case to lower case and removed URL address, hashtags, punctuations, and digits. The 20 most frequent words are in **[Figure 1.0](https://user-images.githubusercontent.com/37912287/107418303-8ab5d100-6b27-11eb-9659-81ea700bef14.png)**. Also, these words are English stop words.  For removing stop words, we used the **NLTK** library. You can see more information in **preprocessing.py**


<p align="center"><img src="https://user-images.githubusercontent.com/37912287/107418303-8ab5d100-6b27-11eb-9659-81ea700bef14.png" /></p>
<p align="center">
  <b>Figure 1.0</b>
</p>

For feature extraction, we used the **bag of words** technique that is a pre-trained model to convert vector representation. Bag of words is a method that is used in natural language preprocessing. In this method, the frequency of each word is calculated, convert vector representation, and then is used for training data.

We used **10 fold** cross-validation technique. The only disadvantage is that there are not class labels in test data. To solve this problem, we used both train and test data. 

We used four different machine learning algorithms to train and test the dataset. These algorithms are **Random Forest**, **Naive Bayes**, **Decision Tree**, and **Gradient Boosting**. The model results are available in **Table 1.0**. As you can see in the table, the Random Forest algorithm has good classification results than other algorithms.
 We used the prediction result of Random Forest in order to submit Kaggle.
<p align="center"> <b> Table 1.0 </b></p>
<table border="1" cellspacing="0" cellpadding="1" align="center">
    <thead>
        <tr>
            <th align="left">Classifications</th>
            <th align="center">Results</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="left">Random Forest</td>
            <td align="center">78.71</td>
        </tr>
         <tr>
            <td align="left">Naive Bayes</td>
            <td align="center">75.55</td>
        </tr>
         <tr>
            <td align="left">Decision Tree</td>
            <td align="center">73.19</td>
        </tr>
         <tr>
            <td align="left">Gradient Boosting</td>
            <td align="center">71.88</td>
        </tr>
    </tbody>
</table>
</p>
