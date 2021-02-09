from sklearn.model_selection import StratifiedKFold
import pandas as pd

skf = StratifiedKFold(n_splits=10, random_state=48, shuffle=True)

def CV(predictors,target):
    
    for fold, (train_index, test_index) in enumerate(skf.split(predictors, target)):
        x_train, x_valid = pd.DataFrame(predictors.iloc[train_index]), pd.DataFrame(predictors.iloc[test_index])
        y_train, y_valid = target.iloc[train_index], target.iloc[test_index]

    return x_train, x_valid, y_train, y_valid    
