from Classifications import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from CrossValidation import *

#split data using cross validation
x_train,x_valid,y_train,y_valid=CV(predictors,target)

#randomforest
y_pred_R,model_R=RandomForest(x_train,x_valid,y_train)
#calculate accuracy
acc_randomforest = round(accuracy_score(y_pred_R, y_valid) * 100, 2)
print('Random Forest: ', acc_randomforest)


#DecisionTree
y_pred_D,model_D=DecisionTree(x_train,x_valid,y_train)
acc_decisiontree = round(accuracy_score(y_pred_D, y_valid) * 100, 2)
print('Decision Tree: ', acc_decisiontree)

#GradientBoosting
y_pred_G,model_G=GradientBoosting(x_train,x_valid,y_train)
acc_gbk = round(accuracy_score(y_pred_G, y_valid) * 100, 2)
print('Gradient Boosting: ',acc_gbk)

#NaiveBayes
y_pred_N,model_N=NaiveBayes(x_train,x_valid,y_train)
acc_naive = round(accuracy_score(y_pred_N, y_valid) * 100, 2)
print('Naive Bayes: ', acc_naive)

#submission csv file to kaggle
ids = test['id1']
predictions_R = model_R.predict(test.drop('id1', axis=1))
output = pd.DataFrame({ 'id' : ids, 'target': predictions_R })
output.to_csv('submission.csv', index=False)
