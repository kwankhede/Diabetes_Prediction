import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Load and preprocess your dataset
# Replace this with your actual data loading and preprocessing

def load_data(filename):
    original_data = pd.read_csv(filename, sep=",")
    return original_data


def replace_zeros_with_nan(data):
    columns_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    data[columns_to_replace] = data[columns_to_replace].replace(0, np.NaN)
    return data


def fill_missing_values(data, key):
    for var in data.columns[:-1]:
        temp = data[data[var].notnull()]
        temp = temp.groupby([key])[var].median().reset_index()

        data.loc[(data[key] == 0) & (data[var].isnull()), var] = temp[temp[key] == 0][
            var
        ].values[0]
        data.loc[(data[key] == 1) & (data[var].isnull()), var] = temp[temp[key] == 1][
            var
        ].values[0]
    return data


original_data = load_data("diabetes.csv")
original_data = replace_zeros_with_nan(original_data)
original_data = fill_missing_values(original_data, "Outcome")

X = original_data.iloc[:, :-1]
y = original_data.iloc[:, -1]


# Create and train the models
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X, y)

gradient_boosting_model = GradientBoostingClassifier()
gradient_boosting_model.fit(X, y)

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X, y)

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X, y)

naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X, y)

knn_model = KNeighborsClassifier()
knn_model.fit(X, y)

svm_model = SVC()
svm_model.fit(X, y)

xgboost_model = XGBClassifier()
xgboost_model.fit(X, y)

# Save the models to files
joblib.dump(random_forest_model, "Random_Forest_model.pkl")
joblib.dump(gradient_boosting_model, "Gradient_Boosting_model.pkl")
joblib.dump(logistic_regression_model, "Logistic_Regression_model.pkl")
joblib.dump(decision_tree_model, "Decision_Tree_model.pkl")
joblib.dump(naive_bayes_model, "Naive_Bayes_model.pkl")
joblib.dump(knn_model, "KNN_model.pkl")
joblib.dump(svm_model, "SVM_model.pkl")
joblib.dump(xgboost_model, "XGBoost_model.pkl")
