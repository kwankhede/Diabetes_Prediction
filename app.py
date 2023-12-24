import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve

# Load and preprocess your dataset
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
        data.loc[(data[key] == 0) & (data[var].isnull()), var] = temp[temp[key] == 0][var].values[0]
        data.loc[(data[key] == 1) & (data[var].isnull()), var] = temp[temp[key] == 1][var].values[0]
    return data

# Function to train and evaluate classifiers
@st.cache(allow_output_mutation=True, ttl=12*60*60)  # Set TTL to 12 hours
def train_and_evaluate(classifier, X_train, y_train, X_test, y_test):
    # Train the selected classifier
    classifier.fit(X_train, y_train)

    # Predict on the testing dataset
    y_pred = classifier.predict(X_test)

    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    return report, fpr, tpr, roc_auc

original_data = load_data("diabetes.csv")
original_data = replace_zeros_with_nan(original_data)
original_data = fill_missing_values(original_data, "Outcome")

X = original_data.iloc[:, :-1]
y = original_data.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a dictionary of the classifiers
classifiers_dict = {
    "Logistic Classifier": LogisticRegression(class_weight="balanced"),
    "Decision_Tree Classifier": DecisionTreeClassifier(class_weight="balanced"),
    "Random_Forest Classifier": RandomForestClassifier(class_weight="balanced"),
    "SVM Classifier": SVC(probability=True, gamma="scale"),
    "GaussianNB Classifier": GaussianNB(),
    "KNN Classifiers": KNeighborsClassifier(),
    "GB Classifier": GradientBoostingClassifier(loss="deviance"),
    "XGB Classifier": XGBClassifier(scale_pos_weight=2),
}

# Streamlit app
st.title("Diabetes Prediction Using Machine Learning")
st.markdown("This app predicts whether a person has diabetes or not based on user's input features.")
# ... (rest of your code)

# Sidebar with user input features
st.sidebar.header("User Input Features")
st.sidebar.markdown("Enter the following patient information:")
# ... (rest of your code)

if st.button("Train and Submit"):
    st.write("### Classification Results and ROC Curve")

    for classifier_name in classifiers_selected:
        selected_classifier = classifiers_dict[classifier_name]

        # Train and evaluate the classifier
        report, fpr, tpr, roc_auc = train_and_evaluate(selected_classifier, X_train, y_train, X_test, y_test)

        # Display classification results and ROC curve
        st.write(f"## {classifier_name} Results")
        st.write("Accuracy:", round(report["accuracy"], 3))
        st.write("Precision:", round(report["1"]["precision"], 3))
        st.write("Recall:", round(report["1"]["recall"], 3))
        st.write("F1:", round(report["1"]["f1-score"], 3))
        st.write(f"ROC AUC: {roc_auc:.3f}")

        # Plot interactive ROC curve
        fig = px.area(
            x=fpr,
            y=tpr,
            title=f"{classifier_name} ROC Curve",
            labels=dict(x="False Positive Rate", y="True Positive Rate"),
            width=700,
            height=500,
        )
        fig.update_layout(
            shapes=[dict(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)]
        )
        st.plotly_chart(fig)
