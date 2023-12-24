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

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

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
st.markdown(
    "This app predicts whether a person has diabetes or not based on user's input features."
)

st.markdown(
"""
**Objective:**

This project aims to develop a predictive model for determining whether an individual is diabetic or not, 
utilizing the Pima-Indians diabetes dataset from the UCI repository. Given the binary nature of our target outcome (Yes or No), 
we will employ supervised classification algorithms. Our goal, informed by existing research and studies 
(as cited in [this paper](https://www.ijedr.org/papers/IJEDR1703069.pdf)), is to achieve a model classification accuracy of 75% or higher, 
with a specific emphasis on minimizing false negatives (high recall) in the model.

The focus on minimizing false negatives is crucial to ensure that no individuals with diabetes are overlooked. While some tolerance for false positives is acceptable
(instances where non-diabetic individuals are misclassified), the priority is to avoid missing any patients with diabetes. 
In the event of misclassifying a non-diabetic person as diabetic, they would undergo further testing, ultimately confirming their non-diabetic status.
"""
)


# Sidebar with user input features
st.sidebar.header("User Input Features")
st.sidebar.markdown("Enter the following patient information:")

classifiers_selected = st.sidebar.multiselect(
    "Select Two Classifiers", list(classifiers_dict.keys()), default=["Logistic Classifier"]
)

pregnancies = st.sidebar.number_input(
    "Pregnancies (Range: 0 - 20)", min_value=0, max_value=20, value=0, step=1
)
glucose = st.sidebar.number_input(
    "Glucose (Range: 44 - 300 mg/dL)", min_value=44, max_value=300, value=120, step=1
)
blood_pressure = st.sidebar.number_input(
    "Blood Pressure (Range: 24 - 222 mm Hg)",
    min_value=24,
    max_value=222,
    value=120,
    step=1,
)
skin_thickness = st.sidebar.number_input(
    "Skin Thickness (Range: 7 - 199 mm)", min_value=7, max_value=199, value=20, step=1
)
insulin = st.sidebar.number_input(
    "Insulin (Range: 14 - 1200 mu U/ml)", min_value=14, max_value=1200, value=79, step=1
)
bmi = st.sidebar.number_input(
    "BMI (Range: 18.2 - 67.1)", min_value=18.2, max_value=67.1, value=32.0, step=0.1
)
diabetes_pedigree_function = st.sidebar.number_input(
    "Diabetes Pedigree Function (Range: 0.078 - 3.42)",
    min_value=0.078,
    max_value=3.42,
    value=0.25,
    step=0.001,
)
age = st.sidebar.number_input(
    "Age (Range: 21 - 110 years)", min_value=21, max_value=110, value=30, step=1
)



if st.button("Train and Submit"):
    st.write("### Classification Results and ROC Curve")

    for classifier_name in classifiers_selected:
        selected_classifier = classifiers_dict[classifier_name]

        # Train the selected classifier
        selected_classifier.fit(X_train, y_train)

        # Predict on the testing dataset
        y_pred = selected_classifier.predict(X_test)

        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, selected_classifier.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

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
